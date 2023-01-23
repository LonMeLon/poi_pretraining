# coding:utf8

from ast import arg
import enum
import os, json
import csv
from statistics import mean
import string
import random
from datetime import datetime
from argparse import ArgumentParser
from tqdm import tqdm
from copy import copy, deepcopy

import sys, networkx
sys.path.append('/nfs/volume-93-2/meilang/myprojects/pretrain_poi_user_behavior/')
from util import function

def jishu(dict_start_end_num):
    mmm = 0
    for start in dict_start_end_num:
        for end in dict_start_end_num[start]:
            mmm += dict_start_end_num[start][end]
    return mmm



def capture_data(args):
    dict_start_end_num = {}
    dict_end_start_num = {}

    file = open(args.path_poi_task_log, mode='r')
    reader = csv.reader(file, delimiter='\01')
    for index, row in tqdm(enumerate(reader)):
        if index >= 1:
            start_poiid, end_poiid = row
            
            dict_start_end_num.setdefault(start_poiid, {})
            dict_start_end_num[start_poiid].setdefault(end_poiid, 0)
            dict_start_end_num[start_poiid][end_poiid] += 1
            
            dict_end_start_num.setdefault(end_poiid, {})
            dict_end_start_num[end_poiid].setdefault(start_poiid, 0)
            dict_end_start_num[end_poiid][start_poiid] += 1
    file.close()
    
    print(index, len(dict_start_end_num), len(dict_end_start_num))
    print(jishu(dict_start_end_num), jishu(dict_end_start_num))

    return dict_start_end_num, dict_end_start_num


def walk_one(args, dict_start_end_num, start_poid):
    nodes = []
    current_poid = copy(start_poid)
    nodes.append(current_poid)
    for step in range(args.sample_walk_length):
        if current_poid in dict_start_end_num:
            neigbor_poids_data = dict_start_end_num[current_poid]
            if len(neigbor_poids_data) > 0:
                neigbor_poids = [pppoiddd for pppoiddd in neigbor_poids_data]
                neigbor_poids_prob = [neigbor_poids_data[pppoiddd] for pppoiddd in neigbor_poids_data]
                
                current_poid = random.choices(neigbor_poids, weights=neigbor_poids_prob, k=1)[0]
                nodes.append(current_poid)
        else:
            break
    
    return nodes

def randwalk_sample(args, dict_start_end_num, ):
    all_paths = {}
    for start_poid in tqdm(dict_start_end_num):
        for time in range(args.sample_times):
            path = walk_one(args, dict_start_end_num, start_poid)
            if len(path) > float(args.sample_walk_length) * 0.35:
                all_paths.setdefault(json.dumps(path), 0)

    all_paths = [json.loads(key) for key in all_paths]
    all_lengths = [len(path) for path in all_paths]

    print(len(all_paths), sum(all_lengths) / len(all_lengths))



def get_topk_minq_next(end_poid, dict_end_start_num):
    dict_poid_num = dict_end_start_num[end_poid]
    list_poid_num = sorted(list(dict_poid_num.items()), key=lambda x:x[1], reverse=True)
    #average_num = sum([pair[1] for pair in list_poid_num]) / len(list_poid_num)

    next_poids_path = []
    for pair in list_poid_num[0:3]:
        if pair[1] > list_poid_num[0][1] / 3:
            next_poids_path.append((pair[0], end_poid))

    return next_poids_path

def neigbor2p(args, dict_end_start_num):
    dict_end_2p_edges = {}

    for end_poid in tqdm(dict_end_start_num):
        dict_end_2p_edges.setdefault(end_poid, {})

        path_poids_1p = get_topk_minq_next(end_poid, dict_end_start_num)
        for sepair_1p in path_poids_1p:
            dict_end_2p_edges[end_poid].setdefault(sepair_1p, 0)
            if sepair_1p[0] in dict_end_start_num:
                path_poids_2p = get_topk_minq_next(sepair_1p[0], dict_end_start_num)
                for sepair_2p in path_poids_2p:
                    dict_end_2p_edges[end_poid].setdefault(sepair_2p, 0)

        if len(dict_end_2p_edges[end_poid]) < 6:
            dict_end_2p_edges.pop(end_poid, None) 


    all_lengths = [len(dict_end_2p_edges[end_poid]) for end_poid in dict_end_2p_edges]

    print(len(dict_end_2p_edges), sum(all_lengths) / len(all_lengths))

    function.save_obj(args.path_sample_subgraph, dict_end_2p_edges)
    

if __name__ == "__main__":
    # parameter
    parser = ArgumentParser()

    parser.add_argument('--path_sample_subgraph', type=str, default='8_neigbor2p_userlog_sample_subgraph.dict')
    
    parser.add_argument('--path_poi_task_log', type=str, default='7_userlog_pretrain_poi_task.csv')
    parser.add_argument('--sample_walk_length', type=int, default=12)
    parser.add_argument('--sample_times', type=int, default=6)
    

    args = parser.parse_args()


    # function to sample data
    dict_start_end_num, dict_end_start_num = capture_data(args)
    #randwalk_sample(args, dict_start_end_num)
    neigbor2p(args, dict_end_start_num)