# filter chinese query / POI
# filter user log with existing poi
# filter user log with searching the close start position (distance < 100 m)


# coding:utf8

import os, json
import csv
import string
import random
import pandas as pd
from datetime import datetime
from argparse import ArgumentParser
from tqdm import tqdm
from transformers import BertTokenizer, BertTokenizerFast

def time_stat(start_time, end_time):
    return ((end_time - start_time).seconds)


def tensorize_text(text, tokenizer):
    return 0

def tensorize_category(cate, dict_cate):
    return dict_cate[cate]

def tensorize_geohash(geohash, dict_geohash):
    return [dict_geohash[char] for char in geohash]


def sample_split_data(args):
    
    tokenizer = BertTokenizerFast.from_pretrained(args.vocab_path)

    poi_set = {}
    poi_file = open(args.path_poi, 'r')
    reader_poi = csv.reader(poi_file, delimiter='\01')
    for index, row in tqdm(enumerate(reader_poi)):
        if index >= 1:
            # poi_id
            poid, name, address, geohash = row

            # dict:  3 type
            poi_set[poid] = (name, address, geohash)
    
    print('poi_set', len(poi_set))
    poi_file.close()


    count = [0, 0, 0]

    file = open(args.path_click_log, "r")
    reader = csv.reader(file, delimiter='\01')

    count___ = 0


    f_filter = open(args.path_filter_log, 'w')
    w_filter = csv.writer(f_filter, delimiter='\01')

    for index, row in tqdm(enumerate(reader)):
        if index == 0:
            print(row)
            w_filter.writerow(row)
            
        if index >= 1:
            query, geohash, clk_poiid, sess_query_list, sess_poilist_list, start_poiid = row
            
            
            filter_query = tokenizer(query, padding=True)['input_ids']
                
            if (index + 1) % 100000 == 0:
                print(index, count)

            if (len(filter_query) > 0) and (clk_poiid in poi_set) and (start_poiid in poi_set):
                count[0] += 1
                
                sess_poilist_list = json.loads(sess_poilist_list)
                
                filter_sess_poilist_list = []
                mark_ = True
                for one_poilist_list in sess_poilist_list:
                    filter_one_poilist_list = [pppoiddd for pppoiddd in one_poilist_list if pppoiddd in poi_set]
                    filter_sess_poilist_list.append(filter_one_poilist_list)
                    if len(filter_one_poilist_list) == 0: #< len(one_poilist_list):
                        mark_ = False
                        break
                
                if mark_ == True:
                    filter_sess_poilist_list = json.dumps(filter_sess_poilist_list)
                    w_filter.writerow([query, geohash, clk_poiid, sess_query_list, filter_sess_poilist_list, start_poiid])
                    count[1] += 1
                    
    file.close() 
    f_filter.close()
    print(count, count___)

def sample_data(args):
    sample_split_data(args)

if __name__ == "__main__":
    
    # parameter
    parser = ArgumentParser()
    parser.add_argument('--vocab_path', type=str, default="/nfs/volume-93-2/meilang/myprojects/poi_pretrain_dense/download_pretrained_model/bert-base-chinese/", help='')
    
    parser.add_argument('--path_filter_log', type=str, default="3_userlog_need_filtering.csv", help='')
    
    parser.add_argument('--path_poi', type=str, default="1_poi_need_attr.csv")
    parser.add_argument('--path_click_log', type=str, default="2_userlog_need.csv")
    

    parser.add_argument('--least_rec_poi_num', type=int, default=10)

    

    args = parser.parse_args()


    # function to sample data
    sample_data(args)