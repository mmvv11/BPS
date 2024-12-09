import os
import sys
import pickle

import csv
import pandas as pd
import numpy as np

from math import floor
from libreco.data import (
    random_split,
    split_by_num,
    split_by_num_chrono,
    split_by_ratio,
    split_by_ratio_chrono,
)

from foursquare import *

def load_data(path):
    mode = os.path.basename(path)
    assert mode in ("gowalla", "ml-1m", "foursquare")
    
    if mode == "gowalla":
        cols = ['user', 'time', 'lat', 'long', 'item']
        df = pd.read_csv(os.path.join(path, "Gowalla_totalCheckins.txt"), header=None, sep='\t', names=cols)
    elif mode == "ml-1m":
        cols = ["user", "item", "rating", "time"]
        df = pd.read_csv(os.path.join(path, "ratings.dat"), sep="::", usecols=[0,1,2,3], names=cols)
    else: #foursquare
        csv_path = os.path.join(path, 'data.csv')
        if not os.path.exists(csv_path):
            out = open(csv_path, 'w', newline='')
            csv_writer = csv.writer(out, dialect='excel')
            f = open(os.path.join(path, "dataset_WWW_Checkins_anonymized.txt"), "r")
            for line in f.readlines():
                list = line.split()
                csv_writer.writerow(list)
        df = pd.read_csv(csv_path)
        df = df.rename(columns={"822121":"user_id", "4b4b87b5f964a5204a9f26e3":"item_id"})
        df = df.rename(columns={"Apr":"month", "2012":"year", "03":"day"})
        
        df['new_month'] = df['month'].apply(lambda x:month_transfer(x))
        df['new_year'] = df['year'].apply(lambda x:year_transfer(x))
        df = df[["user_id", "item_id", "month", "day", "year"]]
        df = df[df['day'] <= 40]
        df['new_day'] = df['day'].apply(lambda x:day_transfer(x))
        df = df[['user_id', 'item_id', 'new_year', 'new_month', 'new_day']]
        df['date'] = df.apply(lambda x: date_new_create(x.new_year, x.new_month, x.new_day), axis=1)
        df = df.rename(columns={"new_year":"year", "new_month":"month", "new_day":"day", "date":"timestamp"})
    if not mode == "foursquare":
        df = k_core(df.drop_duplicates(['user','item']), user_col = 'user', item_col = 'item', k = 10)
        df = df.sort_values('time')
        df['label'] = 1
        df = df[['user', 'item', 'label', 'time']]
    else:
        df = k_core_foursquare(df, 20, 20)
    return df

def save_data(blocks, path):
    mode = os.path.basename(path)
    assert mode in ("gowalla", "ml-1m", "foursquare")
    
    save_path = os.path.join(path, "blocks.pkl")
    os.makedirs(path, exist_ok = True)
    with open(save_path, 'wb') as f:
        pickle.dump(blocks, f)
    return save_path
    
def k_core(df, user_col='user', item_col='item', k = 20):
    while True:
        user_interactions = df[user_col].value_counts()
        item_interactions = df[item_col].value_counts()
        few_user_interactions = user_interactions[user_interactions < k].index.tolist()
        few_item_interactions = item_interactions[item_interactions < k].index.tolist()
        
        if len(few_user_interactions) == 0 and len(few_item_interactions) == 0:
            break

        df = df[~df[user_col].isin(few_user_interactions)]
        df = df[~df[item_col].isin(few_item_interactions)]
    return df

def split_block(df, ratio = [0.6, 0.1, 0.1, 0.1, 0.1]):
    total_len = len(df)
    start = 0
    result = []
    
    for r in ratio:
        end = start + floor(total_len * r)
        result.append(df.iloc[start:end].reset_index(drop = True))
        start = end
    return result

if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise ValueError("Usage: python data_preprocess.py <data_path>")
    else:
        data_path = sys.argv[1] #"./data/ml-1m"
    
    df = load_data(data_path)
    blocks = split_block(df)
    save_path = save_data(blocks, os.path.join("./preprocess", os.path.basename(data_path)))
    
    print("finish: {0}".format(save_path))