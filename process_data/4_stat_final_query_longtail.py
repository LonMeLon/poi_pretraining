# coding:utf8

import os
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


def stat_query(args):
    file = open(args.path_filter_log, "r")
    reader = csv.reader(file, delimiter='\01')

    dict_query = {}
    for index, row in tqdm(enumerate(reader)):
        if index == 0:
            print(row)
        if index >= 1:
            # text_feature: query
            query = row[0].strip()
            dict_query[query] = 1 + dict_query.get(query, 0)
        
    print(len(dict_query), index)
    sorted_queries = sorted(dict_query.items(), key=lambda x:x[1], reverse=True)

    
    f_query = open(args.path_stat_queries, "w")
    w_query = csv.writer(f_query, delimiter='\01')
    for pair in sorted_queries:
        #query, num = pair
        w_query.writerow(pair)
    f_query.close()



def sample_data(args):
    stat_query(args)


if __name__ == "__main__":
    
    # parameter
    parser = ArgumentParser()

    parser.add_argument('--path_filter_log', type=str, default="3_userlog_need_filtering.csv")
    parser.add_argument('--path_stat_queries', type=str, default="4_stat_query_frequency.csv")

    args = parser.parse_args()


    # function to sample data
    sample_data(args)