from distutils.command.config import config
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import nn
from transformers import (
    BertForPreTraining, 
    BertModel, 
    BertGenerationEncoder, 
    BertGenerationDecoder, 
    EncoderDecoderModel,
    EncoderDecoderConfig,
)

from transformers.models.bert.modeling_bert import BertForPreTrainingOutput

import torch.nn.functional as F
from torch_scatter import scatter_max, segment_csr, scatter_max
from torch_scatter import scatter

from typing import Callable

class Our_pretrain(BertForPreTraining):
    def __init__(self, config, data_args, tokenizer, config_decoder,):
        super().__init__(config)
        '''hyper-parameter'''
        self.data_args = data_args
        self.config_decoder = config_decoder
        self.tokenizer_size = len(tokenizer)

        self.CrossEntropyLoss = nn.CrossEntropyLoss()
        self.MarginRankingLoss = nn.MarginRankingLoss(margin=1.0)

    def build_network(self, ):
        '''
        shared bert encoder; # hyper bert, cls (config)
        query task: thin transformer decoder; 
        poi task: thin graph transformer (mask adjacency matrices to self-attention)
        '''
        # query task: decoder
        decoder_transformer = BertGenerationDecoder(self.config_decoder)
        decoder_transformer.resize_token_embeddings(self.tokenizer_size)
        self.enc2dec = EncoderDecoderModel(encoder=self.bert, decoder=decoder_transformer)
        # poi task: graph transformer (self-defined adjacency-matrices-based self-attention)
        self.graph_transformer = graph_transformer(
            d_model=768,
            nhead=12,
            num_encoder_layers=2,
            dim_feedforward=768,
            dropout=0.1,
            activation=nn.LeakyReLU(),
        )
        self.map_linear = nn.Linear(self.config.hidden_size, self.data_args.dimension, bias=False)

    def compute_bertoutput(self, input_ids_token_res, attention_mask_token_res, token_type_ids_token_res):
        return self.bert(
            input_ids=input_ids_token_res, 
            attention_mask=attention_mask_token_res, 
            token_type_ids=token_type_ids_token_res,
        )
    
    def compute_cls(self, bertoutput):
        seq_output, pooled_output = bertoutput[:2]
        pred_scores, seq_relation_score = self.cls(seq_output, pooled_output)
        return pred_scores, seq_relation_score

    def compute_pairloss(self, positive_score, negative_score):
        assert positive_score.shape == negative_score.shape
        pos_rel = positive_score[:, 0]
        neg_rel = negative_score[:, 0]
        pairwise_label = torch.ones_like(pos_rel)
        
        return self.MarginRankingLoss(pos_rel, neg_rel, pairwise_label)
        
    def forward(
        # basic
        self, labels, input_ids, 
        # mlm task: token mask prediction
        input_ids_tok_poi_mlm_geo_name_add, attention_mask_tok_poi_mlm_geo_name_add, token_type_ids_tok_poi_mlm_geo_name_add,
        label_mlm,
        # name-address, match
        input_ids_tok_poi_name_add, attention_mask_tok_poi_name_add, token_type_ids_tok_poi_name_add,
        input_ids_neg_tok_poi_name_add, attention_mask_neg_tok_poi_name_add, token_type_ids_neg_tok_poi_name_add,
        # geohash-address, match
        input_ids_tok_poi_geo_add, attention_mask_tok_poi_geo_add, token_type_ids_tok_poi_geo_add, 
        input_ids_neg_tok_poi_geo_add, attention_mask_neg_tok_poi_geo_add,token_type_ids_neg_tok_poi_geo_add,
        # query task: retrieval augmentation query generation
        #'input_ids_', 'attention_mask_', 'token_type_ids_'
        input_ids_source_request, attention_mask_source_request, token_type_ids_source_request, 
        input_ids_target_query, attention_mask_target_query, token_type_ids_target_query, 
        # poi task: masked POI node prediction, graph transformer
        input_ids_mask_node_context, attention_mask_mask_node_context, token_type_ids_mask_node_context, 
        input_ids_pos_node_context, attention_mask_pos_node_context, token_type_ids_pos_node_context, 
        input_ids_neg_node_context, attention_mask_neg_node_context, token_type_ids_neg_node_context, 
        mask_nodes_TF, mask_index, edge,
    ):
        # mlm task: token mask prediction
        mlm_output_geo_name_add = self.compute_bertoutput(
            input_ids_tok_poi_mlm_geo_name_add, 
            attention_mask_tok_poi_mlm_geo_name_add, 
            token_type_ids_tok_poi_mlm_geo_name_add,
        )
        geo_name_add_pred_scores, _ = self.compute_cls(mlm_output_geo_name_add)
        self.loss_mlm = self.CrossEntropyLoss(
            geo_name_add_pred_scores.view(-1, self.config.vocab_size), 
            label_mlm.view(-1),
        )

        # name-address, next sentence prediction
        positive_output_name_add = self.compute_bertoutput(
            input_ids_tok_poi_name_add, 
            attention_mask_tok_poi_name_add, 
            token_type_ids_tok_poi_name_add,
        )
        negative_output_name_add = self.compute_bertoutput(
            input_ids_neg_tok_poi_name_add, 
            attention_mask_neg_tok_poi_name_add, 
            token_type_ids_neg_tok_poi_name_add,
        )
        _, positive_name_add_seq_rel_score = self.compute_cls(positive_output_name_add)
        _, negative_name_add_seq_rel_score = self.compute_cls(negative_output_name_add)
        self.loss_name_add = self.compute_pairloss(
            positive_name_add_seq_rel_score, 
            negative_name_add_seq_rel_score,
        )
        
        # geohash-address, next sentence prediction
        positive_output_geo_add = self.compute_bertoutput(
            input_ids_tok_poi_geo_add, 
            attention_mask_tok_poi_geo_add, 
            token_type_ids_tok_poi_geo_add,
        )
        negative_output_geo_add = self.compute_bertoutput(
            input_ids_neg_tok_poi_geo_add, 
            attention_mask_neg_tok_poi_geo_add,
            token_type_ids_neg_tok_poi_geo_add,
        )
        _, positive_geo_add_seq_rel_score = self.compute_cls(positive_output_geo_add)
        _, negative_geo_add_seq_rel_score = self.compute_cls(negative_output_geo_add)
        self.loss_geo_add = self.compute_pairloss(
            positive_geo_add_seq_rel_score, 
            negative_geo_add_seq_rel_score,
        )

        # query task: retrieval augmentation query generation
        self.loss_query_task = self.enc2dec(
            input_ids=input_ids_source_request, 
            attention_mask=attention_mask_source_request,
            decoder_input_ids=input_ids_target_query, 
            decoder_attention_mask=attention_mask_target_query,
            labels=input_ids_target_query,
        ).loss
        
        
        # poi task: masked POI node prediction, graph transformer 
        assert input_ids_mask_node_context.shape[:2] == mask_nodes_TF.shape
        
        mask_nodes_position = torch.where(mask_nodes_TF > 0)
        input_ids_mask_node_context = input_ids_mask_node_context[mask_nodes_position]
        attention_mask_mask_node_context = attention_mask_mask_node_context[mask_nodes_position]
        token_type_ids_mask_node_context = token_type_ids_mask_node_context[mask_nodes_position]
        assert input_ids_mask_node_context.shape == attention_mask_mask_node_context.shape == token_type_ids_mask_node_context.shape
        assert input_ids_mask_node_context.shape[0] == mask_nodes_position[0].shape[0]
        
        # mask_nodes_length
        mask_poi_embs = self.compute_bertoutput(
            input_ids_mask_node_context, 
            attention_mask_mask_node_context, 
            token_type_ids_mask_node_context,
        )[1]
        
        ### remove pad edge [batch, 2, max_edge_len], some -100
        nodes_shu = scatter(torch.ones_like(mask_nodes_position[0]), mask_nodes_position[0], dim=0, reduce="sum",)
        nodes_cum_shu = torch.cumsum(nodes_shu, dim=0)
        nodes_cum_shu = torch.cat([torch.tensor([0], device=nodes_cum_shu.device).long(), nodes_cum_shu[0:-1]], dim=0)
        edge_position = torch.where(edge>-1)
        edge = edge + nodes_cum_shu.reshape(-1, 1, 1).repeat(1, 2, 1)
        edge = edge[edge_position].reshape(2, -1)

        mask_index = mask_index.reshape(-1) + nodes_cum_shu
        
        assert mask_poi_embs.shape[0] > mask_index.reshape(-1).max()
        mask_graph_poi_emb = self.graph_transformer(mask_poi_embs, edge)
        
        mask_graph_poi_emb = mask_graph_poi_emb[mask_index]
        mask_graph_poi_emb = self.map_linear(mask_graph_poi_emb)
        mask_graph_poi_emb = F.normalize(mask_graph_poi_emb, p=2, dim=1)
        # positive
        pos_poi_emb = self.compute_bertoutput(
            input_ids_pos_node_context, 
            attention_mask_pos_node_context, 
            token_type_ids_pos_node_context,
        )[1]
        pos_poi_emb = self.map_linear(pos_poi_emb)
        pos_poi_emb = F.normalize(pos_poi_emb, p=2, dim=1)
        
        # negative
        neg_poi_emb = self.compute_bertoutput(input_ids_neg_node_context, attention_mask_neg_node_context, token_type_ids_neg_node_context,)[1]
        neg_poi_emb = self.map_linear(neg_poi_emb)
        neg_poi_emb = F.normalize(neg_poi_emb, p=2, dim=1)

        ### contrastive learning loss
        score_pos = torch.mul(mask_graph_poi_emb, pos_poi_emb).sum(1)
        score_neg = torch.matmul(mask_graph_poi_emb, neg_poi_emb.T) 
        score = torch.cat([score_pos.reshape(-1, 1), score_neg], dim=1)
        label_score = torch.zeros_like(score[:, 0], dtype=torch.long)
        
        self.loss_poi_task = self.CrossEntropyLoss(score / self.data_args.temp, label_score)
        
        # loss for training
        self.total_loss = self.loss_mlm + \
                        self.loss_name_add + \
                        self.loss_geo_add + \
                        self.loss_query_task + \
                        self.loss_poi_task
        

        return BertForPreTrainingOutput(
            loss=self.total_loss,
        )

class graph_transformer(nn.Module):
    def __init__(
        self, 
        d_model, 
        nhead, 
        num_encoder_layers,
        dim_feedforward, 
        dropout, 
        activation,
    ):
        super(graph_transformer, self).__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.encoder_layers = nn.ModuleList(
            [
                GraphEncoderLayer(
                    d_model, nhead, dim_feedforward=dim_feedforward, dropout=dropout, activation=activation,
                )
                for i in range(num_encoder_layers)
            ]
        )
        self.encoder_norm = torch.nn.LayerNorm(d_model)
        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src, edge_index: torch.Tensor):
        if src.shape[1] != self.d_model:
            raise RuntimeError("the feature number of src must be equal to d_model")
        # Encode
        memory = src
        for mod in self.encoder_layers:
            memory = mod(memory, edge_index)
        memory = self.encoder_norm(memory)

        return memory


class GraphEncoderLayer(nn.Module):
    def __init__(
        self, 
        d_model, 
        nhead, 
        dim_feedforward, 
        dropout, 
        activation,
    ):
        super(GraphEncoderLayer, self).__init__()
        self.attn = MultiheadAttention(d_model, num_heads=nhead, dropout=dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = nn.BatchNorm1d(d_model)
        self.activation = activation
        self.norm1 = nn.BatchNorm1d(d_model)
        self.dropout1 = nn.Dropout(dropout)

    def forward(
        self, 
        src: torch.Tensor, 
        edge_index: torch.Tensor, 
    ):
        # src: shape [N, d_model], (unique) node embedding
        # edge_index: shape [2,E], node-node edge connect
        src2 = self.attn(
            src, src, src, 
            torch.stack([edge_index[1], edge_index[0]], dim=0)
        )
        # Residual 1
        src = src + self.dropout1(src2)  # Residual 1
        src = self.norm1(src)
        # mlp
        src2 = self.linear2(
            self.dropout(
                self.activation(
                    self.linear1(src)
                )
            )
        )
        # Residual 2
        src = src + self.dropout2(src2)  # Residual 2
        src = self.norm2(src)
        
        return src


class MultiheadAttention(nn.Module):
    # Sparse MultiheadAttention
    def __init__(
        self, 
        embed_dim, 
        num_heads, 
        dropout, 
        bias=True, 
        add_bias_kv=False,
    ):
        super(MultiheadAttention, self).__init__()

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = nn.Dropout(p=dropout, inplace=False)
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"

        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=add_bias_kv)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=add_bias_kv)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=True)

        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.q_proj.weight)
        nn.init.xavier_uniform_(self.k_proj.weight)
        nn.init.xavier_uniform_(self.v_proj.weight)
        if self.q_proj.bias is not None:
            nn.init.constant_(self.q_proj.bias, 0.)
        if self.k_proj.bias is not None:
            nn.init.xavier_normal_(self.k_proj.bias)
        if self.v_proj.bias is not None:
            nn.init.xavier_normal_(self.v_proj.bias)
        if self.out_proj.bias is not None:
            nn.init.constant_(self.out_proj.bias, 0.)


    def sparse_softmax(self, src, index, num_nodes, dim=0):
        # Perform sparse softmax
        # scatter_max, same node (several score) to same group, softmax (max-normalize)
        src_max = scatter(src, index, dim=dim, reduce="max", dim_size=num_nodes) # 
        src_max = src_max.index_select(dim, index)
        out = (src - src_max).exp() # all edge node, un-normalized vector
        ### add the score ratio, normalize, according to same index
        out_sum = scatter(out, index, dim=dim, reduce="sum", dim_size=num_nodes)
        out_sum = out_sum.index_select(dim, index)

        return out / (out_sum + 1e-16)

    def forward(self, query, key, value, edge_index):
        r"""
        :param query: Tensor, shape [tgt_len, embed_dim]
        :param key: Tensor of shape [src_len, kdim]
        :param value: Tensor of shape [src_len, vdim]
        :param edge_index: Tensor of shape [2, E], a sparse matrix that has shape len(query)*len(key),
        :return Tensor of shape [tgt_len, embed_dim]
        """

        # Dimension checks
        assert edge_index.shape[0] == 2
        assert key.shape[0] == value.shape[0]
        # Dictionary size
        src_len, tgt_len, idx_len = key.shape[0], query.shape[0], edge_index.shape[1]

        assert query.shape[1] == self.embed_dim
        assert key.shape[1] == self.embed_dim
        assert value.shape[1] == self.embed_dim

        scaling = float(self.head_dim) ** -0.5
        q: torch.Tensor = self.q_proj(query) * scaling
        k: torch.Tensor = self.k_proj(key)
        v: torch.Tensor = self.v_proj(value)
        assert self.embed_dim == q.shape[1] == k.shape[1] == v.shape[1]

        # Split into heads
        q = q.contiguous().view(tgt_len, self.num_heads, self.head_dim)
        k = k.contiguous().view(src_len, self.num_heads, self.head_dim)
        v = v.contiguous().view(src_len, self.num_heads, self.head_dim)

        # Get score
        assert edge_index[0].shape == edge_index[1].shape
        attn_output_weights = (
            torch.index_select(q, 0, edge_index[0]) * torch.index_select(k, 0, edge_index[1])
        ).sum(dim=-1)
        
        # finite refered node
        assert list(attn_output_weights.size()) == [idx_len, self.num_heads]

        attn_output_weights = self.sparse_softmax(src=attn_output_weights, index=edge_index[0], num_nodes=tgt_len)
        attn_output_weights = self.dropout(attn_output_weights)

        # Get values
        # [edge in-node, n-head, dim]
        attn_output = attn_output_weights.unsqueeze(2) * torch.index_select(v, 0, edge_index[1])
        """Aggregation"""
        attn_output = scatter(attn_output, edge_index[0], dim=0, reduce="sum", dim_size=tgt_len)

        # all node in graph
        assert list(attn_output.size()) == [tgt_len, self.num_heads, self.head_dim]

        attn_output = self.out_proj(attn_output.contiguous().view(tgt_len, self.embed_dim))
        # all node in graph
        assert list(attn_output.size()) == [tgt_len, self.embed_dim]

        return attn_output
