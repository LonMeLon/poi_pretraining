# coding:utf8

import os, json
import csv
import string
import random
import pandas as pd
import numpy as np
from datetime import datetime
from argparse import ArgumentParser
from tqdm import tqdm
from transformers import BertTokenizer, BertTokenizerFast

import matplotlib.pyplot as plt


import sys
sys.path.append('/nfs/volume-93-2/meilang/myprojects/pretrain_poi_user_behavior/')
from util import function




def capture_data(args):
    
    f_query = open(args.path_query_task, 'w')
    w_query = csv.writer(f_query, delimiter='\01')
    
    file = open(args.path_filter_log, mode='r')
    reader = csv.reader(file, delimiter='\01')

    for index, row in tqdm(enumerate(reader)):
        if index == 0:
            new_row = ['source_geohash', 'source_query', 'source_recpoid', 'target_query']
            w_query.writerow(new_row)
        if index >= 1:
            query, geohash, clk_poiid, sess_query_list, filter_sess_poilist_list, start_poiid = row
            # construction
            sess_query_list = json.loads(sess_query_list)
            filter_sess_poilist_list = json.loads(filter_sess_poilist_list)
            assert len(sess_query_list) == len(filter_sess_poilist_list)
            for iii in range(len(sess_query_list)):
                if (sess_query_list[iii] != query) and (len(sess_query_list[iii]) < len(query)):
                    marks = False
                    if sess_query_list[iii] in query:
                        if len(sess_query_list[iii]) < len(query) * 0.7:
                            marks = True
                    else:
                        if (len(sess_query_list[iii]) > 1):
                            marks = True
                    
                    if marks:
                        for pppoiddd in filter_sess_poilist_list[iii][0:2]:
                            if pppoiddd != clk_poiid:
                                new_row = [geohash, sess_query_list[iii], pppoiddd, query]
                                w_query.writerow(new_row)

    file.close()
    f_query.close()
    
    '''
    dict_poi_set = function.load_poi_data(args, 1, 0, [1,2,3])

    file = open(args.path_filter_log, mode='r')
    reader = csv.reader(file, delimiter='\01')

    for index, row in tqdm(enumerate(reader)):
        if index == 0:
            new_row = ['source_geohash', 'source_query', 'source_recpoid', 'target_query']
        if index >= 1:
            query, geohash, clk_poiid, sess_query_list, filter_sess_poilist_list, start_poiid = row
            # construction
            sess_query_list = json.loads(sess_query_list)
            filter_sess_poilist_list = json.loads(filter_sess_poilist_list)
            assert len(sess_query_list) == len(filter_sess_poilist_list)
            for iii in range(len(sess_query_list)):
                if (sess_query_list[iii] != query) and (len(sess_query_list[iii]) < len(query)):
                    if (len(sess_query_list[iii]) > 1) or ((sess_query_list[iii] in query) and (len(sess_query_list[iii]) < len(query) * 0.7)):
                        print('---------')
                        print('source_query', sess_query_list[iii])
                        print('source_recpoid', )
                        for pppoiddd in filter_sess_poilist_list[iii]:
                            print(pppoiddd, dict_poi_set[pppoiddd])
                        print('clk_poiid', clk_poiid, dict_poi_set[clk_poiid])
                        print('target_query', query)

    file.close()
    '''


if __name__ == "__main__":
    
    # parameter
    parser = ArgumentParser()

    parser.add_argument('--path_filter_log', type=str, default="3_userlog_need_filtering.csv")
    
    parser.add_argument('--path_query_task', type=str, default="6_userlog_pretrain_query_task.csv")

    parser.add_argument('--poi_path', type=str, default="1_poi_need_attr.csv")



    args = parser.parse_args()


    # function to sample data
    capture_data(args)