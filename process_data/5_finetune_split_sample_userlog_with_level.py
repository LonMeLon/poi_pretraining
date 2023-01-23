# coding:utf8

import os
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


def load_query(args):
    file = open(args.path_stat_queries, "r")
    reader = csv.reader(file, delimiter='\01')

    list_query_num = []
    for index, row in tqdm(enumerate(reader)):
        query, number = row[0], int(row[1])
        if number >= args.least_query_num:
            list_query_num.append([query, int(number)])
    print(len(list_query_num))
    return list_query_num


def sampled_split_data(args, dict_query_level, dict_level_ratio):

    f_train = open(args.path_clicklog_sample + 'train.csv', 'w')
    w_train = csv.writer(f_train, delimiter='\01')
    f_test = open(args.path_clicklog_sample + 'test.csv', 'w')
    w_test = csv.writer(f_test, delimiter='\01')
    

    file = open(args.path_filter_log, mode='r')
    reader = csv.reader(file, delimiter='\01')

    count_train = [[0, 0] for i in range(len(dict_level_ratio))] 
    count_test = [[0, 0] for i in range(len(dict_level_ratio))] 
    for index, row in tqdm(enumerate(reader)):
        if index == 0:
            print(row)

            w_train.writerow(row)
            w_test.writerow(row)
        if index >= 1:
            query_content = row[0].strip()
            query_content_level = dict_query_level[query_content]
            level_ratio = dict_level_ratio[query_content_level]
            # whether sample (1:1)
            prob = random.random()
            if prob <= args.sample_prob / (len(dict_level_ratio) * level_ratio):
                # split train test
                seed = random.random() # 0-1
                if seed <= 0.8:
                    count_train[query_content_level][0] += 1
                    w_train.writerow(row)
                else:
                    if seed <= 0.9:
                        count_test[query_content_level][0] += 1
                        w_test.writerow(row) 

    f_train.close()
    f_test.close()

    print(count_train, count_test)

def sample_data(args):
    # load and filter (least query >= )
    list_query_num = load_query(args)
    full_log_num = np.sum([pair[1] for pair in list_query_num])
    dict_query_num = {}
    for pair in list_query_num:
        dict_query_num[pair[0]] = pair[1]
    print(len(list_query_num), len(dict_query_num))
    
    
    #frequency = [3, 10, 30, 100, 300, 1000, 3000, 10000, 30000]
    frequency = [3, 10, 30, 100, 300, 1000, 3000, 10000]
    dict_level_query = {}
    dict_query_level = {}
    for query in tqdm(dict_query_num):
        level = np.searchsorted(frequency, dict_query_num[query])
        dict_level_query.setdefault(level, {})
        dict_level_query[level].setdefault(query, 0)
        dict_query_level[query] = level
    print(len(dict_level_query), len(dict_query_level))    

    dict_level_ratio = {}
    #dict_level_sample_prob = {}
    for level in dict_level_query:
        queries = dict_level_query[level]
        sum_num = np.sum([dict_query_num[qq] for qq in queries])
        print(level, len(queries), sum_num, sum_num / len(queries), sum_num / full_log_num)
        dict_level_ratio[level] = sum_num / full_log_num



    sampled_split_data(args, dict_query_level, dict_level_ratio)



    



if __name__ == "__main__":
    
    # parameter
    parser = ArgumentParser()
    parser.add_argument('--path_clicklog_sample', type=str, default="5_sampled_userlog_")

    parser.add_argument('--path_filter_log', type=str, default="3_userlog_need_filtering.csv")
    parser.add_argument('--path_stat_queries', type=str, default="4_stat_query_frequency.csv")
    
    parser.add_argument('--sample_prob', type=float, default=0.06, help='')

    parser.add_argument('--top_med_split_num', type=int, default=100, help='')
    parser.add_argument('--med_tail_split_num', type=int, default=10, help='')
    parser.add_argument('--least_query_num', type=int, default=-1, help='')
    

    

    args = parser.parse_args()


    # function to sample data
    sample_data(args)