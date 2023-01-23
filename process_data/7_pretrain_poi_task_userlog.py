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

from util import function


def capture_data(args):

    f_query = open(args.path_query_task, 'w')
    w_query = csv.writer(f_query, delimiter='\01')
    
    file = open(args.path_filter_log, mode='r')
    reader = csv.reader(file, delimiter='\01')

    for index, row in tqdm(enumerate(reader)):
        if index == 0:
            new_row = ['start_poid', 'clk_poid']
            w_query.writerow(new_row)
        if index >= 1:
            query, geohash, clk_poiid, sess_query_list, filter_sess_poilist_list, start_poiid = row
            # construction
            new_row = [start_poiid, clk_poiid]
            w_query.writerow(new_row)

    file.close()
    f_query.close()



if __name__ == "__main__":
    
    # parameter
    parser = ArgumentParser()

    parser.add_argument('--path_filter_log', type=str, default="3_userlog_need_filtering.csv")
    
    parser.add_argument('--path_query_task', type=str, default="7_userlog_pretrain_poi_task.csv")

    args = parser.parse_args()


    # function to sample data
    capture_data(args)