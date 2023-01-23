# coding:utf8

import json, ast
import os
import csv
import pandas as pd
from datetime import datetime
from argparse import ArgumentParser
from tqdm import tqdm

def time_stat(start_time, end_time):
    return ((end_time - start_time).seconds)

def get_filepaths(args):
    # paths of target 
    paths = []
    for root, dirs, files in os.walk(args.path_click_log): 
        for file in files:  
            if os.path.splitext(file)[1] == '.csv':  
                paths.append(os.path.join(root, file)) 
    #print(paths)
    paths.sort()
    return paths

def get_sample_click_data(args, path_files):
    start_time = datetime.now()
    # mark
    count_err, count_pf = 0, 0
    #click_log_city = []
    # visit file
    with open(args.sample_city_path, "w") as file_city:
        csv_write = csv.writer(file_city, delimiter='\01')
        #[0:Query, 
        # 1:cityid, 
        # 2:lng, 
        # 3:lat, 
        # 4:geohash, 
        # 5:clk_poiid, 
        # 6:clk_name, 
        # 7:clk_address, 
        # 8:clk_lng, 
        # 9:clk_lat, 
        # 10:clk_geohash,
        # 11:poi_list, 
        # 12:sess_time_list, 
        # 13:sess_query_list, 
        # 14:sess_poilist_list, 
        # 15:sess_lng_list, 
        # 16:sess_lat_list, 
        # 17:sess_geohash_list, 
        # 18:start_poiid, 
        # 19:start_name, 
        # 20:start_addr, 
        # 21:start_lng, 
        # 22:start_lat, 
        # 23:start_geohash]
        cols_value = ['query', 'cityid', 'lng', 'lat', 'geohash', 'clk_poiid', 'clk_name', 'clk_address', 'clk_lng', 'clk_lat', 'clk_geohash', 'poi_list', 'sess_time_list', 'sess_query_list', 'sess_poilist_list', 'sess_lng_list', 'sess_lat_list', 'sess_geohash_list', 'start_poiid', 'start_name', 'start_addr', 'start_lng', 'start_lat', 'start_geohash']
        
        # 0:Query, 
        # 4:geohash, 
        # 5:clk_poiid, 
        # 13:sess_query_list, 
        # 14:sess_poilist_list,
        # 18:start_poiid, 
        need_cols = [0, 4, 5, 13, 14, 18] # session_list

        need_col_value = [cols_value[col] for col in need_cols]
        csv_write.writerow(need_col_value)

        for jishu, pf in tqdm(enumerate(path_files)):
            print(jishu + 1, pf)
            with open(pf, "r") as file:
                csv_reader = csv.reader(file)
                for index, row in tqdm(enumerate(csv_reader)):
                    if index >= 1:
                        row_value = "".join(row).split('\01')
                        #print(len(row_value))
                        if len(row_value) == 24 and row_value[1] == args.sample_city:
                            count_pf += 1
                            #for kkk, vvv in enumerate(cols_value):
                            #    print(vvv, row_value[kkk])

                            query, geohash, clk_poiid, sess_query_list, sess_poilist_list, start_poiid = [row_value[col] for col in need_cols]

                            try:
                                sess_query_list = json.loads(json.loads(sess_query_list.replace(' ', ',')))
                                
                                sess_poilist_list__ = json.loads(json.loads(sess_poilist_list.replace(' ', ',')))
                                sess_poilist_list = []
                                for onepoilist in sess_poilist_list__:
                                    topk_onepoilist = onepoilist[0:3]
                                    sess_poilist_list.append([pppoi[0] for pppoi in topk_onepoilist])
    
                                assert len(sess_query_list) == len(sess_poilist_list)

                                #print(sess_query_list)
                                #print(sess_poilist_list)

                                sess_query_list = json.dumps(sess_query_list)
                                sess_poilist_list = json.dumps(sess_poilist_list)

                                csv_write.writerow([query, geohash, clk_poiid, sess_query_list, sess_poilist_list, start_poiid])
                            
                            except:
                                count_err += 1
                        else:
                            count_err += 1
                            #print(len(row_value), "\n", row_value, "\n", row)
                    #if index >= 1:
                    #    break
            file.close()
            #break
    file_city.close()
    print(count_err, count_pf)
        

def sample_data(args):
    # get file in path_log
    path_files = get_filepaths(args)
    
    # low memory read data, and get sample index
    get_sample_click_data(args, path_files)

if __name__ == "__main__":
    # parameter
    parser = ArgumentParser()
    parser.add_argument('--sample_city', type=str, default="1", help='北京市:1, 上海市：4')
    parser.add_argument('--sample_city_path', type=str, default="/nfs/volume-93-2/meilang/myprojects/pretrain_poi_user_behavior/dataset/1_processed_data/2_userlog_need.csv")
    parser.add_argument('--path_click_log', type=str, default="/nfs/volume-93-2/meilang/myprojects/pretrain_poi_user_behavior/dataset/userlog/user_click_data_with_sess_query_list_v3/")

    args = parser.parse_args()


    # function to sample data
    sample_data(args)