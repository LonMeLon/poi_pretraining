import enum
import torch
import numpy as np
import os, csv, random, re
from tqdm import tqdm


import os
import csv
from tqdm import tqdm
import torch
import faiss
import numpy as np
from datetime import datetime
import pickle

### [CLS]:101, [SEP]: 102, [MASK]: 103
def load_data2dict(file_path, begin_row, id_col, other_list_cols):
    poi_set = {}
    with open(file_path, "r") as file:
        reader = csv.reader(file, delimiter='\01')

        for index, row in tqdm(enumerate(reader)):
            if index >= begin_row:
                poi_id = row[id_col]
                other_list_attr = [row[col] for col in other_list_cols]
                poi_set[poi_id] = other_list_attr
              
    return poi_set

def load_data2list(file_path, begin_row, list_cols):
    poi_set = []
    with open(file_path, "r") as file:
        reader = csv.reader(file, delimiter='\01')

        for index, row in tqdm(enumerate(reader)):
            if index >= begin_row:
                list_attr = [row[col] for col in list_cols]
                poi_set.append(list_attr)
              
    return poi_set


def save_obj(path, data):
    file = open(path, 'wb')
    pickle.dump(data, file)
    file.close()

def load_obj(path):
    file = open(path, 'rb')
    return pickle.load(file)

def load_geohash(args):
    dict_geohash = {}
    with open(args.geohash_path, "r") as file:
        reader = csv.reader(file)
        for row in reader:
            hashcode = row[0]
            dict_geohash.setdefault(hashcode, len(dict_geohash))
    return dict_geohash


def load_poi_data(args, begin_row):
    # ['0:poi_id', '1:displayname', '2:address', '3:geohash']
    poi_set = {}
    with open(args.poi_path, "r") as file:
        reader = csv.reader(file)
        for index, row in tqdm(enumerate(reader)):
            if index >= begin_row:
                poi_id, displayname, address, geohash = row
                special_geohash = ''.join(['['+cd+']' for cd in geohash])
                poi_set[poi_id] = [displayname, address, special_geohash]
            #if index > 30000:
            #    break    
    return poi_set

def load_user_log_data(args, begin_row):
    # ['0:session_query', '1:session_rec_poid_list', '2:session_pos_poid', '3:session_geohash']
    poi_set = {}
    with open(args.poi_path, "r") as file:
        reader = csv.reader(file)
        for index, row in tqdm(enumerate(reader)):
            if index >= begin_row:
                poi_id, displayname, address, geohash = row
                special_geohash = ''.join(['['+cd+']' for cd in geohash])
                poi_set[poi_id] = [displayname, address, special_geohash]
            #if index > 30000:
            #    break    
    return poi_set
