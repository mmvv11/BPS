import os
import sys
import pickle

import numpy as np
import pandas as pd

from scipy.spatial import distance # jsd
from scipy.special import rel_entr, kl_div # kld

def load_pickle(path):
    with open(path, 'rb') as f:
        return pickle.load(f, encoding='latin1')
    
def softmax(x):
    exp_x = np.exp(x)
    sum_exp_x = np.sum(exp_x, axis=1, keepdims=True)
    softmax_x = exp_x / sum_exp_x
    return softmax_x

def get_div(softmax_emb_list, df_list, user2id):
    all_jsd = [0] * 5
    all_kld = [0] * 5

    for i in range(1, 4):
        jsd = {}
        kld = {}
        emb = softmax_emb_list[i]
        users = df_list[i]['user'].unique()
        for u in users:
            for j in range(i-1, -1, -1):
                prev_users = df_list[j]['user'].unique()
                if u in prev_users:
                    prev_emb = softmax_emb_list[j]
                    """
                    JSD(t, t-1) range
                    JSD의 범위 0~1
                    
                    KL(t, t-1) range 
                    KL의 범위 0~무한
                    
                    lambda range
                    추후 적용
                    """
                    jsd[u] = distance.jensenshannon(emb[user2id[u]], prev_emb[user2id[u]])
                    kld[u] = sum(kl_div(emb[user2id[u]], prev_emb[user2id[u]]))
                    break
        all_jsd[i] = jsd
        all_kld[i] = kld
    return all_jsd, all_kld

def get_normalized_div_sample_counts(all_divergence):
    njsd_list = [0] * 5
    sample_counts = [0] * 5
    w = 1
    
    for i in range(1, 4):
        min_val = np.min(list(all_divergence[i].values()))
        max_val = np.max(list(all_divergence[i].values()))
        
        njsd = (list(all_divergence[i].values())-min_val)/(max_val-min_val) 
        exp = np.exp(-w*njsd)
        prob_dist_njsd = exp / exp.sum()
        count = list(prob_dist_njsd)
        sample_counts[i] = dict(zip(all_divergence[i].keys(), count))
        njsd_list[i] = dict(zip(all_divergence[i].keys(), njsd))    
    return njsd_list, sample_counts

def save_pre_data(njsd_list, njsd_sample_counts, nkld_list, nkld_sample_counts, model="NGCF", dataset=None):
    data = {
        "njsd_list": njsd_list,
        "njsd_sample_counts": njsd_sample_counts,
        "nkld_list": nkld_list,
        "nkld_sample_counts": nkld_sample_counts,
    }
    
    os.makedirs("./preprocess/jsd", exist_ok = True)
    with open(f"./preprocess/jsd/{model}-{dataset}-pre-data.pkl", 'wb') as f:
        pickle.dump(data, f)

if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise ValueError("Usage: python jsd_preprocess.py <data_type> <model_type>")
    else:
        data = sys.argv[1] #"ml-1m"
        model = sys.argv[2] #"NGCF"

    path = "./preprocess"
    user_emb_list = load_pickle(os.path.join(path, "model", "{0}-{1}-{2}".format(model, data, "user_emb_list.pkl")))
    item_emb_list = load_pickle(os.path.join(path, "model", "{0}-{1}-{2}".format(model, data, "item_emb_list.pkl")))
    user2id = load_pickle(os.path.join(path, "model", "{0}-{1}-{2}".format(model, data, "user2id.pkl")))
    blocks = load_pickle(os.path.join(path, data, "blocks.pkl"))

    softmax_user_emb_list = [softmax(x) for x in user_emb_list]
    jsd, kld = get_div(softmax_user_emb_list, blocks, user2id)
    njsd, njsd_sample_counts = get_normalized_div_sample_counts(jsd)
    nkld, nkld_sample_counts = get_normalized_div_sample_counts(kld)
    save_pre_data(njsd, njsd_sample_counts, nkld, nkld_sample_counts, dataset = data, model = model)
    
    print("finish")