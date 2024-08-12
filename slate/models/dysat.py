
# https://github.com/FeiGSSS/DySAT_pytorch/

import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.loss import BCEWithLogitsLoss, MSELoss
from torch_geometric.utils import coalesce,to_undirected

from slate.models.layers_dysat import StructuralAttentionLayer, TemporalAttentionLayer
from slate.models.reg_mlp import RegressionModel
#from utils.utilities import fixed_unigram_candidate_sampler

class DySat(nn.Module):
    def __init__(self, 
                 num_nodes: int,
                 num_features: int,
                 num_classes: int,
                 time_length: int,
                 window: int,
                 structural_head_config: list,
                 structural_layer_config: int,
                 temporal_head_config: int,
                 temporal_layer_config: int,
                 spatial_drop: float,
                 temporal_drop: float,
                 residual: bool,
                 neg_weight : float,
                 undirected : bool = False,
                 task_name: str = 'node_classif',
                 aggr: str = 'mean',
                 one_hot: bool = True,):
        """[summary]
            time_length (int): Total timesteps in dataset.
        """
        super(DySat, self).__init__()
        if window < 0:
            self.num_time_steps = time_length  
        else:
            self.num_time_steps = min(time_length, window + 1)  # window = 0 => only self.
        
        self.num_nodes = num_nodes
        self.num_features = num_features
        self.num_classes = num_classes
        self.structural_head_config = structural_head_config
        self.structural_layer_config = structural_layer_config
        self.temporal_head_config = temporal_head_config
        self.temporal_layer_config = temporal_layer_config
        self.spatial_drop = spatial_drop
        self.temporal_drop = temporal_drop
        self.residual = residual
        self.task = task_name
        self.aggr = aggr
        self.neg_weight = neg_weight
        self.undirected = undirected
        self.one_hot = one_hot
        self.dim_features = num_nodes if one_hot else int(num_features)
        self.structural_attn, self.temporal_attn = self.build_model()

        if self.task == 'link_pred':
            self.bceloss = BCEWithLogitsLoss()
        elif self.task == 'node_reg':
            self.mseloss = MSELoss()
        elif self.task == 'node_classif':
            pass
        else:
            raise NotImplementedError
        
        assert aggr.lower() in ['mean','sum','max','last'], 'Aggregation must be in [mean, sum, max, last]'
    
    @property
    def device(self):
        return next(self.parameters()).device
    
    def set_device(self,device):
        self.device = device
    
    def forward(self, graphs):

        # Structural Attention forward
        structural_out = []
        for t in range(len(graphs)):
            if graphs[t].x.device != self.device:
                graphs[t].x = graphs[t].x.to(self.device)
            if self.undirected:
                if graphs[t].edge_index.device != self.device:
                    graphs[t].edge_index = to_undirected(graphs[t].edge_index).to(self.device)
            else:
                if graphs[t].edge_index.device != self.device:
                    graphs[t].edge_index = graphs[t].edge_index.to(self.device)
            structural_out.append(self.structural_attn(graphs[t]))
        structural_outputs = [g.x[:,None,:] for g in structural_out] # list of [Ni, 1, F]

        # padding outputs along with Ni
        maximum_node_num = structural_outputs[-1].shape[0]
        out_dim = structural_outputs[-1].shape[-1]
        structural_outputs_padded = []
        for out in structural_outputs:
            zero_padding = torch.zeros(maximum_node_num-out.shape[0], 1, out_dim).to(out.device)
            padded = torch.cat((out, zero_padding), dim=0)
            structural_outputs_padded.append(padded)
        structural_outputs_padded = torch.cat(structural_outputs_padded, dim=1) # [N, T, F]
        
        # Temporal Attention forward
        temporal_out = self.temporal_attn(structural_outputs_padded)
        
        return temporal_out

    def build_model(self):
        input_dim = self.dim_features # one hot

        # 1: Structural Attention Layers
        structural_attention_layers = nn.Sequential()
        for i in range(len(self.structural_layer_config)):
            layer = StructuralAttentionLayer(input_dim=input_dim,
                                             output_dim=self.structural_layer_config[i],
                                             n_heads=self.structural_head_config[i],
                                             attn_drop=self.spatial_drop,
                                             ffd_drop=self.spatial_drop,
                                             residual=self.residual,
                                             num_nodes=self.num_nodes,)
            structural_attention_layers.add_module(name="structural_layer_{}".format(i), module=layer)
            input_dim = self.structural_layer_config[i]
        
        # 2: Temporal Attention Layers
        input_dim = self.structural_layer_config[-1]
        temporal_attention_layers = nn.Sequential()
        for i in range(len(self.temporal_layer_config)):
            layer = TemporalAttentionLayer(input_dim=input_dim,
                                           n_heads=self.temporal_head_config[i],
                                           num_time_steps=self.num_time_steps,
                                           attn_drop=self.temporal_drop,
                                           residual=self.residual)
            temporal_attention_layers.add_module(name="temporal_layer_{}".format(i), module=layer)
            input_dim = self.temporal_layer_config[i]
        
        if self.task == 'node_classif':
            input_dim = self.temporal_layer_config[-1]
            h_dim = input_dim // 2
            output_size = 1 if self.num_classes == 2 else self.num_classes
            self.classif  = nn.Sequential(
                                    nn.Linear(input_dim, output_size),
                                    #nn.ReLU(),
                                    #nn.Linear(h_dim, self.num_classes)
                                )
        elif self.task == 'node_reg':
            self.pred_reg = RegressionModel(input_dim,input_dim)

        return structural_attention_layers, temporal_attention_layers

    def get_loss_link_pred(self, feed_dict,graphs):

        node_1, node_2, node_2_negative, _, _, _, time  = feed_dict.values()
        # run gnn
        final_emb = self.forward(graphs) # [N, T, F]
        emb_source = final_emb[node_1,time,:]
        emb_pos  = final_emb[node_2,time,:]
        emb_neg = final_emb[node_2_negative,time,:]
        pos_score = torch.sum(emb_source*emb_pos, dim=1)
        neg_score = torch.sum(emb_source*emb_neg, dim=1)
        pos_loss = self.bceloss(pos_score, torch.ones_like(pos_score))
        neg_loss = self.bceloss(neg_score, torch.zeros_like(neg_score))
        graphloss = pos_loss + self.neg_weight*neg_loss
            
        return graphloss, pos_score.detach().sigmoid(), neg_score.detach().sigmoid()
    
    def get_loss_node_pred(self,feed_dict,graphs):
        node, y, time  = feed_dict.values()
        y = y.view(-1, 1)
        # run gnn
        final_emb = self.forward(graphs) # [N, T, F]
        emb_node = final_emb[node,time,:]
        pred = self.pred_reg(emb_node)
        graphloss = self.mseloss(pred, y)
        return graphloss
    
    def score_eval(self,feed_dict,graphs):
        with torch.no_grad():
            node_1, node_2, node_2_negative, _, _, _, time  = feed_dict.values()
            # run gnn
            final_emb = self.forward(graphs) # [N, T, F]
            emb_source = final_emb[node_1, time-1 ,:]
            emb_pos  = final_emb[node_2, time-1 ,:]
            emb_neg = final_emb[node_2_negative, time-1 ,:]
            pos_score = torch.sum(emb_source*emb_pos, dim=1)
            neg_score = torch.sum(emb_source*emb_neg, dim=1)
            return pos_score.sigmoid(),neg_score.sigmoid()
        
    def score_eval_node_reg(self,feed_dict,graphs):
        node, y, time  = feed_dict.values()
        y = y.view(-1, 1)
        # run gnn
        final_emb = self.forward(graphs) # [N, T, F]
        emb_node = final_emb[node,time-1,:]
        pred = self.pred_reg(emb_node)
        
        return pred.squeeze()
    
    
