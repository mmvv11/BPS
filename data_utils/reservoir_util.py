from scipy.special import softmax
from scipy.stats import wasserstein_distance
import random
import numpy as np
from data_utils.utils import *
from scipy.sparse import csr_matrix
from scipy.spatial.distance import cdist
import tensorflow as tf


def lists_to_pairs(lists):
    pair_list = []
    for u in range(len(lists)):
        pair_list += [(u, i) for i in lists[u]]
    random.shuffle(pair_list)
    return pair_list


def weighted_sampling_from_sparse_matrix(input_matrix, output_size, weight=None, per_user=0):
    n_input = len(input_matrix.nonzero()[0])
    # assert output_size <= n_input

    if weight is None:
        weight = input_matrix / input_matrix.sum()

    if per_user == 0:
        rng = np.random.default_rng()
        sampled_idx = rng.choice(np.arange(n_input), size=output_size, replace=True,
                                 p=weight.data)  # uniform sample case

        rows = np.array(input_matrix.nonzero()[0][sampled_idx])
        cols = np.array(input_matrix.nonzero()[1][sampled_idx])
        data = np.ones(output_size)
        sample_matrix = csr_matrix((data, (rows, cols)), shape=(input_matrix.shape[0], input_matrix.shape[1]))
        # assert (input_matrix-sample_matrix).data.sum() + output_size == input_matrix.data.sum()
    else:
        # TODO: handle
        pass
    return sample_matrix


def weighted_sample_from_lists(input_lists, output_size, n_user, n_item, weight=None):
    input_matrix = generate_sparse_adj_matrix(input_lists, n_user, n_item)
    sample_matrix = weighted_sampling_from_sparse_matrix(input_matrix, output_size, weight=weight)
    sample_lists = convert_adj_matrix_to_lists(sample_matrix)

    return sample_lists


def bps_random_sample(matrix, k_values):
    num_rows = matrix.shape[0]
    sampled_rows = []

    for row_idx in range(num_rows):
        row = matrix.getrow(row_idx) 
        nonzero_indices = row.indices 

        if len(nonzero_indices) > k_values[row_idx]: 
            sampled_indices = np.random.choice(nonzero_indices, size=int(k_values[row_idx]), replace=False)
        else: 
            sampled_indices = nonzero_indices 

        sampled_rows.append(sampled_indices)  

    sampled_data = np.ones(sum(len(indices) for indices in sampled_rows), dtype=int)  
    sampled_indices = np.concatenate(sampled_rows) 
    sampled_indptr = np.cumsum([0] + [len(indices) for indices in sampled_rows]) 

    # Create CSR matrix using (sampled_data, (row, col), shape)
    sampled_csr_matrix = csr_matrix((sampled_data, sampled_indices, sampled_indptr), shape=matrix.shape)
    return sampled_csr_matrix


def bps_compute_average_item_embedding(pred, old_i_emb, k=2):
    """
    
    :param pred: user recommended items (u, 100) 
    :param old_i_emb: past item emb (i, emb_size)
    :param k: topk
    :return: 
    """
    # Get the top k recommended item for each user
    top_k = pred[:, :k]  # (u, k)

    # Extract the corresponding item embeddings
    top_k_item_embeddings = tf.nn.embedding_lookup(old_i_emb, top_k)

    # Compute the average embedding for each user
    avg_embeddings = tf.reduce_mean(top_k_item_embeddings, axis=1)
    return avg_embeddings


def bps_sample_items_by_l2_distance(avg_topk_item_emb_for_each_user, old_i_emb, reservoir_matrix, sample_size_for_each_user):
    """
    avg_topk_item_emb_for_each_user: (u, emb_size)
    old_i_emb: (i, emb_size)
    reservoir_matrix: (u, i)
    sample_size_for_each_user: sample_size_for_each_user
    """
    num_users, num_items = reservoir_matrix.shape 
    old_num_users = avg_topk_item_emb_for_each_user.shape[0]
    old_num_items = old_i_emb.shape[0]
    sampled_items = []

    for user_idx in range(old_num_users): 
        user_interacted_items = reservoir_matrix[:, :old_num_items].getrow(user_idx).indices 
        user_avg_emb = avg_topk_item_emb_for_each_user[user_idx] 

        # Compute L2 distances between user_avg_emb and old_i_emb for all items
        old_i_emb[user_interacted_items]
        distances = cdist([user_avg_emb], old_i_emb[user_interacted_items], metric='euclidean')[0]

        # Sort items by L2 distance and sample top-k items
        sorted_indices = np.argsort(distances)  
        sampled_indices = user_interacted_items[sorted_indices[:sample_size_for_each_user[user_idx]]] 

        sampled_items.append(sampled_indices)

    # Create a new CSR matrix with sampled items
    sampled_data = np.ones(sum(len(indices) for indices in sampled_items), dtype=int)
    sampled_indices = np.concatenate(sampled_items)
    sampled_indptr = np.cumsum([0] + [len(indices) for indices in sampled_items])
    sampled_csr_matrix = csr_matrix((sampled_data, sampled_indices, sampled_indptr), shape=reservoir_matrix.shape)

    return sampled_csr_matrix


class Reservoir(object):
    def __init__(self, base_block, reservoir_mode, replay_size, sample_mode, merge_mode='snu', sample_per_user=0,
                 logger=None, reservoir_size=None, sample_counts=None, njsd_list=None, item_sim_matrix=None,
                 mapping_data=None, ablation_mode=None):
        self.reservoir_mode = reservoir_mode
        self.reservoir_size = reservoir_size  # reservoir_size is used when reservoir_mode=reservoir_sampling
        self.replay_size = replay_size  # replay_size is the amount of old data to mix with new data
        self.sample_per_user = sample_per_user

        self.ablation_mode = ablation_mode

        self.sample_mode = sample_mode
        self.merge_mode = merge_mode

        self.logger = logger

        # pre-data
        self.sample_counts = sample_counts
        self.njsd_list = njsd_list
        self.item_sim_matrix = item_sim_matrix
        # mapping data
        self.u_mapping, self.i_mapping, self.inv_u_mapping, self.inv_i_mapping = mapping_data
        # training phase
        self.segment = 0
        # bps sampling data
        self.pred = None  
        self.old_u_emb = None 
        self.old_i_emb = None 

        self.reservoir, self.reservoir_matrix, self.n_reservoir_user, self.n_reservoir_item, self.full_data, self.full_data_matrix = self.create_first_reservoir(
            base_block)
        self.acc_data_size = get_list_of_lists_size(base_block['train'])

    def update_bps_data(self, pred, old_u_emb, old_i_emb):
        self.pred = pred
        self.old_u_emb = old_u_emb
        self.old_i_emb = old_i_emb

    def set_logger(self, logger):
        self.logger = logger

    def log(self, x):
        p = ''
        for s in x:
            p += str(s)
            p += ' '
        if self.logger is None:
            print(p)
        else:
            self.logger.write(p + '\n')

    def create_first_reservoir(self, base_block):
        base_block_user = base_block['n_user_train']
        base_block_item = base_block['n_item_train']
        if self.reservoir_mode == 'full':
            reservoir = base_block['train']
            reservoir_matrix = base_block['train_matrix']
        elif self.reservoir_mode == 'sliding':
            reservoir = base_block['sliding_lists']
            reservoir_matrix = base_block['sliding_matrix']
        else:
            reservoir = weighted_sample_from_lists(base_block['acc_train'], self.reservoir_size, base_block_user,
                                                   base_block_item, weight=None)
            reservoir_matrix = generate_sparse_adj_matrix(reservoir, base_block_user, base_block_item)

        full_data = base_block['train']
        full_data_matrix = base_block['train_matrix']

        return reservoir, reservoir_matrix, base_block_user, base_block_item, full_data, full_data_matrix

    def get_edge_weight(self, input_matrix):
        input_matrix = input_matrix.astype(np.float64)
        if self.sample_mode == 'uniform':
            weight = input_matrix / input_matrix.sum()

        elif self.sample_mode == 'prop_deg':
            pass

        elif self.sample_mode == 'inverse_deg':
            diag_deg, _ = np.histogram(input_matrix.nonzero()[0], np.arange(input_matrix.shape[0] + 1))
            diag_deg = diag_deg.astype(np.float64)
            mask = diag_deg != 0
            diag_deg = diag_deg.astype(np.float64)
            diag_deg[mask] = 1.0 / diag_deg[mask]
            weight = np.zeros(len(input_matrix.nonzero()[0]))
            source_node_idx = input_matrix.nonzero()[0]
            weight[0] = input_matrix.data[0] * diag_deg[source_node_idx[0]]
            for i in range(input_matrix.data.shape[0]):
                weight[i] = input_matrix.data[i] * diag_deg[source_node_idx[i]]
            weight /= weight.sum()

        else:
            raise NotImplementedError

        return weight

    def get_edge_weight_dense(self, input_matrix, predict_score=None, new_data_mat=None, top_k=0):
        input_matrix = input_matrix.astype(np.float64)

        if self.sample_mode == 'uniform':
            weight = input_matrix / input_matrix.sum()
        elif self.sample_mode == 'prop_deg':
            for r in range(len(input_matrix)):
                input_matrix[r] = input_matrix[r] * input_matrix[r].sum()
            weight = input_matrix / input_matrix.sum()

        elif self.sample_mode == 'inverse_deg':
            for r in range(len(input_matrix)):
                if input_matrix[r].sum() != 0:
                    input_matrix[r] = input_matrix[r] * (1 / input_matrix[r].sum())
            weight = input_matrix / input_matrix.sum()

        elif self.sample_mode == 'adp_inverse_deg':
            new_data_total_edge = new_data_mat.sum()
            old_data_total_edge = input_matrix.sum()
            for r in range(len(input_matrix)):
                if input_matrix[r].sum() != 0:
                    new_data_coef = new_data_mat[r].sum() / new_data_total_edge
                    old_data_coef = input_matrix[r].sum() / old_data_total_edge
                    # adp_coef = max(min(new_data_coef / old_data_coef, 5.0), 0.2) # caps (0.1, 10)
                    adp_coef = new_data_coef / old_data_coef
                    input_matrix[r] = input_matrix[r] * (1 / input_matrix[r].sum()) * adp_coef
            weight = input_matrix / input_matrix.sum()

        elif self.sample_mode == 'boosting_inner_product':
            predict_score = predict_score * input_matrix
            predict_score = -predict_score + np.max(predict_score)
            weight = predict_score / predict_score.sum()

        elif self.sample_mode == 'boosting_wasserstein':
            wasserstein = np.zeros(input_matrix.shape)
            for u in range(len(predict_score)):
                u_hat = softmax(predict_score[u])
                u_approx = softmax(input_matrix[u])  # not real ground truth for u
                # because reservoir is only a
                # subset of all data. this
                # results in false negatives
                wasserstein[u] = wasserstein_distance(u_hat, u_approx)
            weight = wasserstein * input_matrix
            # weight = weight ** 1.2 # sharpen the distribution
            weight = weight / weight.sum()

        elif self.sample_mode == 'mse_distill_score':
            masked_predict_score = input_matrix * predict_score
            # # top-k
            # if the input predict_score is a complete score matrix, do top-k selection
            # if the input predict_score is a row, do weighted sampling according to weights
            if predict_score.shape[0] != 1:
                score_shape = masked_predict_score.shape
                flatten_score = masked_predict_score.reshape(-1)
                flatten_argsort = flatten_score.argsort()[:-top_k]
                flatten_score[flatten_argsort] = 0
                flatten_score[flatten_score.nonzero()] = 1
                masked_predict_score = flatten_score.reshape(score_shape)
            weight = masked_predict_score / masked_predict_score.sum()

        else:
            raise NotImplementedError

        return weight

    def sample_and_union(self, new_data_lists, predict_score, new_data_mat=None):
        # sample part
        if self.sample_mode == 'BPS':
            sample_size_for_each_user = np.zeros(self.reservoir_matrix.shape[0], dtype=np.int32)
            cur_sample_counts = self.sample_counts[self.segment]
            for u, s in cur_sample_counts.items():
                map_u = self.u_mapping[u] 
                size = int(s * self.replay_size)  
                sample_size_for_each_user[map_u] = size

            bps_random = True if self.ablation_mode == 1 or self.ablation_mode == 3 else False
            k = 10
            if bps_random:
                sample_reservoir_mat = bps_random_sample(self.reservoir_matrix, sample_size_for_each_user)
                self.log(["bps randome sampling, sum of sampled pairs:", sample_reservoir_mat.sum()])
                self.log(['....................................'])
            else:
                avg_topk_item_emb_for_each_user = bps_compute_average_item_embedding(self.pred, self.old_i_emb, k=k)

                sample_reservoir_mat = bps_sample_items_by_l2_distance(avg_topk_item_emb_for_each_user,
                                                                         old_i_emb=self.old_i_emb,
                                                                         reservoir_matrix=self.reservoir_matrix,
                                                                         sample_size_for_each_user=sample_size_for_each_user)
                self.log(["bps item emb sampling, sum of sampled pairs:", sample_reservoir_mat.sum()])
                self.log(['....................................'])

            # sample without replacement
            sampled_reservoir_list = convert_adj_matrix_to_lists(sample_reservoir_mat)
            # union part
            result_lists = union_lists_of_list(sampled_reservoir_list, new_data_lists)
            # return None
            return result_lists  # , sample_reservoir_mat


        else:
            weight = self.get_edge_weight(self.reservoir_matrix)  

            self.log(['............printing weights.........'])
            self.log(["sample size:", self.replay_size])
            self.log(['mode:', self.sample_mode])

            weight_nonzero = weight[weight.nonzero()]
            self.log(['max, min, mean, std:', weight_nonzero.max(), weight_nonzero.min(), weight_nonzero.mean(),
                      weight_nonzero.std()])


            sample_reservoir_mat = weighted_sampling_from_sparse_matrix(self.reservoir_matrix, self.replay_size,
                                                                        weight=weight, per_user=self.sample_per_user)

            self.log(["sum of sampled pairs:", sample_reservoir_mat.sum()])
            self.log(['....................................'])

            # sample without replacement
            sampled_reservoir_list = convert_adj_matrix_to_lists(sample_reservoir_mat)

            # union part
            result_lists = union_lists_of_list(sampled_reservoir_list, new_data_lists)

            return result_lists  # , sample_reservoir_mat

    def update(self, new_data_lists, n_new_user, n_new_item, pre_computed_reservoir_lists=None,
               pre_computed_reservoir_matrix=None):

        self.full_data = union_lists_of_list(self.full_data, new_data_lists)
        self.full_data_matrix = generate_sparse_adj_matrix(self.full_data, n_new_user, n_new_item)
        if self.reservoir_mode == 'full':
            self.reservoir = self.full_data
            self.reservoir_matrix = self.full_data_matrix
            self.n_reservoir_user = n_new_user
            self.n_reservoir_item = n_new_item
        elif self.reservoir_mode == 'sliding':
            self.reservoir = pre_computed_reservoir_lists
            self.reservoir_matrix = pre_computed_reservoir_matrix
            assert pre_computed_reservoir_matrix.shape[0] == n_new_user
            assert pre_computed_reservoir_matrix.shape[1] == n_new_item
            self.n_reservoir_user = n_new_user
            self.n_reservoir_item = n_new_item
        elif self.reservoir_mode == 'reservoir_sampling':
            # for case there is a fixed sized reservoir - reservoir sampling algo
            # https://en.wikipedia.org/wiki/Reservoir_sampling
            # used in https://arxiv.org/pdf/2007.02747.pdf (potential baseline)
            for i in range(n_new_user - len(self.reservoir)):
                self.reservoir.append([])

            new_pair_list = []
            for u in range(len(new_data_lists)):
                new_pair_list += [(u, i) for i in new_data_lists[u]]
            random.shuffle(new_pair_list)

            for i in range(self.acc_data_size, self.acc_data_size + len(new_pair_list)):
                j = np.random.randint(0, i)
                if j < self.reservoir_size:
                    rand_u = np.random.randint(0, len(self.reservoir))
                    while len(self.reservoir[rand_u]) <= 0:
                        rand_u = np.random.randint(0, len(self.reservoir))
                    rand_i = np.random.randint(0, len(self.reservoir[rand_u]))
                    cur_new_pair = new_pair_list[i - self.acc_data_size]
                    self.reservoir[rand_u].pop(rand_i)
                    self.reservoir[cur_new_pair[0]].append(cur_new_pair[1])

            self.reservoir_matrix = generate_sparse_adj_matrix(self.reservoir, n_new_user, n_new_item)
            self.acc_data_size += len(new_pair_list)
            self.n_reservoir_user = n_new_user
            self.n_reservoir_item = n_new_item
        else:
            raise NotImplementedError

    def get_inc_train_data(self, new_data_lists, predict_score=None, n_new_user=None, n_new_item=None,
                           new_data_mat=None, cur_block_train_size=0):
        if self.merge_mode == 'snu':
            return self.sample_and_union(new_data_lists, predict_score, new_data_mat=new_data_mat)
        elif self.merge_mode == 'uns':
            assert n_new_user is not None and n_new_item is not None
            assert cur_block_train_size != 0
            # union_matrix_dense = np.array(generate_sparse_adj_matrix(new_data_lists, n_new_user, n_new_item).todense())
            # weight = self.get_edge_weight(union_matrix_dense, predict_score)
            # sample_reservoir_mat = weighted_sampling_from_dense_matrix(union_matrix_dense, self.replay_size+cur_block_train_size, weight=weight, per_user=self.sample_per_user)

            union_matrix_sparse = generate_sparse_adj_matrix(new_data_lists, n_new_user, n_new_item)
            weight = self.get_edge_weight(union_matrix_sparse, predict_score)
            sample_reservoir_mat = weighted_sampling_from_sparse_matrix(union_matrix_sparse,
                                                                        self.replay_size + cur_block_train_size,
                                                                        weight=weight, per_user=self.sample_per_user)
            return convert_adj_matrix_to_lists(sample_reservoir_mat)
        else:
            raise NotImplementedError