from dataclasses import dataclass
import enum
#import datasets
import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer, BatchEncoding
from transformers import DefaultDataCollator, DataCollatorWithPadding, DataCollatorForWholeWordMask
import numpy as np
import os, csv, random, re
from tqdm import tqdm
from copy import copy
import func

class OurBertPretrainedDataset(Dataset):
    def __init__(self, args, tokenizer, tasks,):
        self.args = args
        self.tokenizer = tokenizer
        self.dict_poi_set, \
            self.pretrain_query_data, \
            self.pretrain_poi_data = tasks
        # need poi data: poid, name, address, geohash
        self.list_poi_set = list(self.dict_poi_set.keys())
        # construct query task data: 'source_geohash', 'source_query', 'source_recpoid', 'target_query'
        self.pretrain_query_data_index = [kkk for kkk in range(len(self.pretrain_query_data))]
        # construct poi task data
        self.pretrain_poi_data_index = [kkk for kkk in range(len(self.pretrain_poi_data))]

    def __len__(self):
        return len(self.list_poi_set)

    def __getitem__(self, item):
        # get mlm pre_train data
        poi_index = self.list_poi_set[item]
        pretrain_mlm_poi_data = self.dict_poi_set[poi_index]
        # sample query task
        rand_index = random.sample(self.pretrain_query_data_index, 1)[0]
        pretrain_query_task = self.pretrain_query_data[rand_index]
        # sample poi task
        rand_index = random.sample(self.pretrain_poi_data_index, 1)[0]
        pretrain_poi_task = self.pretrain_poi_data[rand_index]
        
        return [
            pretrain_mlm_poi_data, 
            pretrain_query_task,
            pretrain_poi_task,
        ]
    

@dataclass
class OurCollator(DataCollatorWithPadding):
    def __init__(self, tokenizer, data_args, tasks):
        super().__init__(tokenizer)
        self.args = data_args
        self.tokenizer = tokenizer
        self.dict_poi_set, \
            self.pretrain_query_data, \
            self.pretrain_poi_data = tasks
        self.vocab_list = [iii for iii in range(105, len(self.tokenizer))]
        self.list_poi_set = list(self.dict_poi_set.keys())
        self.tokenize_keys = ['input_ids', 'attention_mask', 'token_type_ids']

    def __call__(self, features):
        batch_size = len(features)
        dict_features = {}
        # batch-form
        batch_pretrain_mlm_data = []
        batch_pretrain_query_task = []
        batch_pretrain_poi_task = []
        for item in features:
            pretrain_mlm_poi_data, \
                pretrain_query_task, \
                pretrain_poi_task = item
            batch_pretrain_mlm_data.append(pretrain_mlm_poi_data)
            batch_pretrain_query_task.append(pretrain_query_task)
            batch_pretrain_poi_task.append(pretrain_poi_task)
        # construct_pretrain_mlml_input
        batch_tok_poi_mlm_geo_name_add, \
            batch_label_mlm, \
            batch_tok_poi_name_add, \
            batch_neg_tok_poi_name_add, \
            batch_tok_poi_geo_add, \
            batch_neg_tok_poi_geo_add = self.construct_pretrain_mlm_input(batch_pretrain_mlm_data)
        # construct pretrain_query_task
        batch_source_request, \
            batch_target_query = self.construct_pretrain_query_task(batch_pretrain_query_task)
        # construct pretrain_poi_task
        batch_mask_nodes_context, \
            batch_mask_nodes_TF, \
            batch_pos_poi_context, \
            batch_neg_poi_context, \
            batch_mask_indexes, \
            batch_edges = self.construct_pretrain_poi_task(batch_pretrain_poi_task)
        
        # reform dict, shape [batch_size, 1]
        dict_features['labels'] = torch.LongTensor([[1000] for _ in range(batch_size)])
        dict_features['input_ids'] = torch.LongTensor([[1000] for _ in range(batch_size)])
        ### mlm, shape [batch_size, max_length]
        self.give_tokkey(dict_features, batch_tok_poi_mlm_geo_name_add, 'tok_poi_mlm_geo_name_add', self.tokenize_keys)
        dict_features['label_mlm'] = batch_label_mlm
        ### name-addess, shape [batch_size, max_length]
        self.give_tokkey(dict_features, batch_tok_poi_name_add, 'tok_poi_name_add', self.tokenize_keys)
        self.give_tokkey(dict_features, batch_neg_tok_poi_name_add, 'neg_tok_poi_name_add', self.tokenize_keys)
        ### geohash-address, shape [batch_size, max_length]
        self.give_tokkey(dict_features, batch_tok_poi_geo_add, 'tok_poi_geo_add', self.tokenize_keys)
        self.give_tokkey(dict_features, batch_neg_tok_poi_geo_add, 'neg_tok_poi_geo_add', self.tokenize_keys)
        ### query task, shape [batch_size, max_length]
        self.give_tokkey(dict_features, batch_source_request, 'source_request', self.tokenize_keys)
        self.give_tokkey(dict_features, batch_target_query, 'target_query', self.tokenize_keys)
        ### poi task
        ##### shape [batch_size, max_nodes, max_length]
        self.give_tokkey(dict_features, batch_mask_nodes_context, 'mask_node_context', self.tokenize_keys)
        dict_features['mask_nodes_TF'] = torch.LongTensor(batch_mask_nodes_TF)
        ##### shape [batch_size, max_length]
        self.give_tokkey(dict_features, batch_pos_poi_context, 'pos_node_context', self.tokenize_keys)
        ##### shape [batch_size, max_length] reuse
        self.give_tokkey(dict_features, batch_neg_poi_context, 'neg_node_context', self.tokenize_keys)
        ##### shape [batch_size, 1]
        dict_features['mask_index'] = torch.LongTensor(batch_mask_indexes)
        ##### shape [batch_size, 2, max_edge_length]
        dict_features['edge'] = torch.LongTensor(batch_edges)

        return dict_features

    def give_tokkey(self, dict_features, batch_tok, base_id, tokkeys):
        for kk in tokkeys:
            dict_features[kk + '_' + base_id] = batch_tok[kk]

    def construct_pretrain_mlm_input(self, batch_pretrain_mlm_data):
        batch_had_poids = set([item[0] for item in batch_pretrain_mlm_data])
        ### neg_poi
        neg_poi_data = self.sample_neg_nonoverlap(batch_had_poids, self.list_poi_set, 1)
        ### other batch
        batch_tok_poi_mlm_geo_name_add = []
        batch_label_mlm = []
        batch_tok_poi_name_add = []
        batch_neg_tok_poi_name_add = []
        batch_tok_poi_geo_add = []
        batch_neg_tok_poi_geo_add = []
        for pretrain_mlm_poi_data in batch_pretrain_mlm_data:
            poi_id, poi_name, poi_address, poi_geohash = pretrain_mlm_poi_data
            ### batch_tok_poi_mlm_geo_name_add
            batch_tok_poi_mlm_geo_name_add.append(
                self.geo_spec_tok(poi_geohash) + '[SEP]' + \
                poi_name + '[SEP]' + \
                poi_address
            )
            ### batch_tok_poi_name_add, batch_tok_poi_geo_add
            batch_tok_poi_name_add.append(
                poi_name + '[SEP]' + \
                poi_address
            )
            batch_tok_poi_geo_add.append(
                self.geo_spec_tok(poi_geohash) + '[SEP]' + \
                poi_address
            )
            ### batch_neg_tok_poi_name_add, batch_neg_tok_poi_geo_add
            for iii in range(len(neg_poi_data)):
                neg_poi_id, neg_poi_name, neg_poi_address, neg_poi_geohash = self.dict_poi_set[neg_poi_data[iii]]
                batch_neg_tok_poi_name_add.append(
                    poi_name + '[SEP]' + \
                    neg_poi_address
                )
                batch_neg_tok_poi_geo_add.append(
                    self.geo_spec_tok(poi_geohash) + '[SEP]' + \
                    neg_poi_address
                )
        # tokenize
        batch_tok_poi_mlm_geo_name_add = self.tokenizer(batch_tok_poi_mlm_geo_name_add, padding=True, return_tensors='pt')
        batch_tok_poi_name_add = self.tokenizer(batch_tok_poi_name_add, padding=True, return_tensors='pt')
        batch_neg_tok_poi_name_add = self.tokenizer(batch_neg_tok_poi_name_add, padding=True, return_tensors='pt')
        batch_tok_poi_geo_add = self.tokenizer(batch_tok_poi_geo_add, padding=True, return_tensors='pt')
        batch_neg_tok_poi_geo_add = self.tokenizer(batch_neg_tok_poi_geo_add, padding=True, return_tensors='pt')
        # masked prediction
        batch_label_mlm = -100 * torch.ones_like(batch_tok_poi_mlm_geo_name_add['input_ids'], dtype=torch.long)
        for iii in range(batch_tok_poi_mlm_geo_name_add['input_ids'].shape[0]):
            for mindex, tid in enumerate(batch_tok_poi_mlm_geo_name_add['input_ids'][iii]):
                if tid != 101 and tid != 102 and tid != 0:
                    if random.random() < self.args.mlm_probability:
                        batch_label_mlm[iii][mindex] = tid
                        prob = random.random()
                        # mask
                        if prob < 0.8:
                            batch_tok_poi_mlm_geo_name_add['input_ids'][iii][mindex] = 103
                        elif prob < 0.9:
                            batch_tok_poi_mlm_geo_name_add['input_ids'][iii][mindex] = random.sample(self.vocab_list, 1)[0]
                        # else: keep unchanged

        return [batch_tok_poi_mlm_geo_name_add,
                batch_label_mlm,
                batch_tok_poi_name_add,
                batch_neg_tok_poi_name_add,
                batch_tok_poi_geo_add,
                batch_neg_tok_poi_geo_add,
            ]

    def construct_pretrain_query_task(self, batch_pretrain_query_task):
        batch_source_request = []
        batch_target_query = []
        for item in batch_pretrain_query_task:
            source_geohash, \
                source_query, \
                source_recpoid, \
                target_query = item
            source_recpoi_id, \
                source_recpoi_name, \
                source_recpoi_address, \
                source_recpoi_geohash = self.dict_poi_set[source_recpoid]
            batch_source_request.append(
                self.geo_spec_tok(source_geohash) + '[SEP]' + \
                source_query + '[SEP]' + \
                self.geo_spec_tok(source_recpoi_geohash) + '[SEP]' + \
                source_recpoi_name + '[SEP]' + \
                source_recpoi_address
            )
            batch_target_query.append(target_query)
        # tokenize
        batch_source_request = self.tokenizer(batch_source_request, padding=True, return_tensors='pt')
        batch_target_query = self.tokenizer(batch_target_query, padding=True, return_tensors='pt')

        return [batch_source_request, batch_target_query]

    def construct_pretrain_poi_task(self, batch_pretrain_poi_task):
        batch_size = len(batch_pretrain_poi_task)

        batch_had_poids = []
        batch_neg_poi_context = []
        for item in batch_pretrain_poi_task:
            for nnn in item[0]:
                batch_had_poids.append(nnn)
        batch_had_poids = set(batch_had_poids)
        # negative
        neg_poi_data = self.sample_neg_nonoverlap(batch_had_poids, self.list_poi_set, batch_size)
        for iii in range(len(neg_poi_data)):
            _, neg_poi_name, neg_poi_address, neg_poi_geohash = self.dict_poi_set[neg_poi_data[iii]]
            batch_neg_poi_context.append(
                self.geo_spec_tok(neg_poi_geohash) + '[SEP]' + \
                neg_poi_name + '[SEP]' + \
                neg_poi_address
            )
        
        batch_mask_nodes_context = []
        batch_mask_nodes_TF = []
        batch_pos_poi_context = []
        batch_edges = []
        batch_mask_indexes = []

        
        max_node_len = max([len(item[0]) for item in batch_pretrain_poi_task])
        max_edge_len = max([len(item[1][0]) for item in batch_pretrain_poi_task])
        for index, item in enumerate(batch_pretrain_poi_task):
            node_poids, edges, posmask_poid = item
            # padding edge
            one_edges = [    
                edges[0] + [-100 for _ in range(max_edge_len - len(edges[0]))],
                edges[1] + [-100 for _ in range(max_edge_len - len(edges[1]))],
            ]
            batch_edges.append(one_edges)
            # padding nodes
            one_mask_nodes_context = []
            one_mask_nodes_TF = []
            for ppp, poi_nod in enumerate(node_poids):
                one_mask_nodes_TF.append(1)
                if poi_nod == posmask_poid:
                    this_id, this_name, this_add, this_geo = self.dict_poi_set[poi_nod]
                    one_mask_nodes_context.append('[MASK]')

                    batch_pos_poi_context.append(
                        self.geo_spec_tok(this_geo) + '[SEP]' + \
                        this_name + '[SEP]' + \
                        this_add
                    )
                    batch_mask_indexes.append([ppp])
                else:
                    this_id, this_name, this_add, this_geo = self.dict_poi_set[poi_nod]
                    one_mask_nodes_context.append(
                        self.geo_spec_tok(this_geo) + '[SEP]' + \
                        this_name + '[SEP]' + \
                        this_add
                    )
            assert len(node_poids) == len(one_mask_nodes_context) == len(one_mask_nodes_TF)
            for _ in range(max_node_len - len(node_poids)):
                one_mask_nodes_context.append('[CLS]')
                one_mask_nodes_TF.append(0)
            assert len(one_mask_nodes_context) == max_node_len
            batch_mask_nodes_context += one_mask_nodes_context
            batch_mask_nodes_TF.append(one_mask_nodes_TF)

        # shape: [batch, max_node_len, max_tok_len]
        batch_mask_nodes_context = self.tokenizer(batch_mask_nodes_context, padding=True, return_tensors='pt')
        max_tok_len_ = batch_mask_nodes_context['input_ids'].shape[1]
        for key in self.tokenize_keys:
            batch_mask_nodes_context[key] = batch_mask_nodes_context[key].reshape(batch_size, -1, max_tok_len_)
        # shape: [batch, max_tok_len]
        batch_pos_poi_context = self.tokenizer(batch_pos_poi_context, padding=True, return_tensors='pt')
        # shape: [batch, max_tok_len]
        batch_neg_poi_context = self.tokenizer(batch_neg_poi_context, padding=True, return_tensors='pt')
        
        return [batch_mask_nodes_context, batch_mask_nodes_TF, batch_pos_poi_context, batch_neg_poi_context, batch_mask_indexes, batch_edges]


    def geo_spec_tok(self, geohash):
        return ''.join(['['+cd+']' for cd in geohash])
    
    def sample_neg_nonoverlap(self, had_items_set, all_candidate, sample_num):
        rand_count = 0
        res = []
        while rand_count < sample_num:
            sub_samples = random.sample(all_candidate, 2 * sample_num)
            for rand_id in sub_samples:
                if rand_count == sample_num:
                    break
                if rand_id not in had_items_set:
                    rand_count += 1
                    res.append(rand_id) 
        
        assert len(res) == sample_num
        return res


    

    