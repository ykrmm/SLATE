from typing import Dict, List, Any, Optional

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

"""
   SLATE Layers and modules
"""
class EdgeAggregation(nn.Module):
    def __init__(self,
                 aggr="mean") -> None:
        super().__init__()
        self.aggr = aggr 
        
    def forward(self, edge_emb: torch.Tensor):
        """
        Arguments:
            edge_emb: Tensor, shape ``[B, W, dim_emb]``
        """
        if self.aggr == "mean":
            x = torch.mean(edge_emb,dim=1)
        elif self.aggr == "sum":
            x = torch.sum(edge_emb,dim=1)
        elif self.aggr == "max":
            x = torch.max(edge_emb,dim=1)[0]
        else: #last
            x = edge_emb[:,-1,:]
        return x
    
class NodeAggregation(nn.Module):
    def __init__(self,
                 aggr="mean") -> None:
        super().__init__()
        self.aggr = aggr 
        
    def forward(self, final_emb: torch.Tensor, node: torch.LongTensor, time: torch.LongTensor):
        """
        Arguments:
            final_emb: Tensor, shape ``[N, W, dim_emb]``
            node: Tensor, shape ``[B, 1]``
            time: Tensor, shape ``[B, 1]``
            Returns:
            x: Tensor, shape ``[batch_size, dim_emb]``
            """
        time = final_emb.shape[1]
        if self.aggr == "mean":
            x = final_emb.cumsum(dim=1)[node,-1,:]/ time
        elif self.aggr == "sum":
            x = final_emb.cumsum(dim=1)[node,-1,:]
        elif self.aggr == "max":
            x = final_emb.cummax(dim=1)[0][node, -1, :]
        else:
            x = final_emb[node,-1,:]
        
        return x

        
class LightMultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.embed_dim = embed_dim

        # Define weight matrices for the query and key only
        self.wq = nn.Linear(embed_dim, embed_dim)
        self.wk = nn.Linear(embed_dim, embed_dim)

        self.multihead_output = nn.Linear(embed_dim, embed_dim)

    def forward(self, query, key, value):
        batch_size = query.size(0)

        # Linear transformations
        query = self.wq(query)
        key = self.wk(key)

        # Compute attention scores
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.embed_dim)

        # Apply softmax to get attention weights
        attn_weights = F.softmax(scores, dim=-1)

        # Multiply attention weights by value to get new values
        output = torch.matmul(attn_weights, value)

        # Concatenate multiple heads
        output = output.transpose(1, 2).contiguous().view(batch_size, -1, self.embed_dim)

        # Final linear layer
        output = self.multihead_output(output)

        return output, attn_weights 

class CrossAttention(nn.Module):
    def __init__(self,
                 dim_emb: int,
                 num_heads: int,
                 dropout: float = 0.1,
                 bias: bool = True,
                 add_bias_kv: bool = False,
                 add_zero_attn: bool = False,
                 light: bool = True,
                 ) -> None:
        super().__init__()
        
        if light: 
            self.multihead_attn = LightMultiHeadAttention(dim_emb,num_heads)
        else:
            self.multihead_attn = nn.MultiheadAttention(dim_emb,
                                                        num_heads,
                                                        dropout,
                                                        bias,
                                                        add_bias_kv,
                                                        add_zero_attn,
                                                        batch_first=True)
        self.norm1 = nn.LayerNorm(dim_emb)
        self.norm2 = nn.LayerNorm(dim_emb)
        self.norm3 = nn.LayerNorm(dim_emb)
        self.mlp = nn.Sequential(
            nn.Linear(dim_emb, dim_emb*4),
            nn.ReLU(inplace=True),
            nn.Linear(dim_emb*4, dim_emb),
        )
    
    def forward(self,
                src: torch.Tensor,
                tgt: torch.Tensor,
                attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Arguments:
            src: Tensor, shape ``[batch_size, W, dim_emb]``
            tgt: Tensor, shape ``[batch_size, W, dim_emb]``
            v: Tensor, shape ``[batch_size, seq_len, dim_emb]``
            attn_mask: Tensor, shape ``[batch_size, seq_len, seq_len]``
        Returns:
            output: Tensor, shape ``[batch_size, seq_len, dim_emb]``
        """
        #src = self.norm1(src)
        #tgt = self.norm2(tgt)
        output, _ = self.multihead_attn(src, tgt, tgt)
        output = src + output
        output = self.norm3(output)
        output = self.mlp(output) + output
        #output = self.norm1(output)
        return output


class PositionalEncoding(nn.Module):

    def __init__(self,
                 d_model: int,
                 dropout: float = 0.1,
                 dim_te: int = 12 ,
                 max_len: int = 5000,
                 linear: bool = False,
                 learn: bool = False):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.linear = linear
        self.dim_te = dim_te
        self.d_model = d_model
        self.learn = learn
        
        if  self.learn: # Learn the positional embedding
            self.pos_emb = nn.Embedding(max_len, dim_te)

        else: #cos sin (Attention is all you need) 
            if self.linear:
                self.lin = nn.Linear(d_model,dim_te,bias=False)
            
            # Compute position vector
            position = torch.arange(max_len).unsqueeze(1)
            div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
            pe = torch.zeros(max_len, 1, d_model)
            pe[:, 0, 0::2] = torch.sin(position * div_term)
            pe[:, 0, 1::2] = torch.cos(position * div_term)
            self.register_buffer('pe', pe)
        
    def forward(self, x: torch.Tensor, pos: int) -> torch.Tensor:
        """
        Arguments:
            x: Tensor, shape ``[ N , embedding_dim - dim_te ]``

        """
        n = x.shape[0]
        if self.learn: 
            pe_temp = self.pos_emb(pos)
        else:
            pos = int(pos)
            pe_temp = self.pe[pos] 
            if self.linear:
                pe_temp = self.lin(pe_temp)
            elif self.dim_te != self.d_model:
                pe_temp = F.interpolate(pe_temp, size=self.dim_te, mode='linear', align_corners=True)
        pe_temp = pe_temp.repeat(n,1) # Node share the same temporal encoding because they are in the same snapshot
        #pe_temp = self.dropout(pe_temp)
        return pe_temp

class LinkPredScore(nn.Module):
    def __init__(self, dim_emb: int,
                 dropout: float = 0.1,
                 edge: bool = False) -> None:
        """
        Arguments:
            dim_emb: int, dimension of the embedding
            score: bool, whether to use the score
        """

        super().__init__()
        self.edge = edge
        hidden_dim = dim_emb 
        if edge: 
            in_dim =  dim_emb 
        else:    
            in_dim = dim_emb * 2
        self.lin1 = nn.Linear(in_dim , hidden_dim, bias=True)
        self.dropout = nn.Dropout(dropout)
        self.lin2 = nn.Linear(hidden_dim, 1, bias=True)
        self.act = nn.ReLU()

    def forward(self, emb1, emb2 = None, emb3 = None):
        """
        Arguments:
            emb1: Tensor, shape ``[batch_size, dim_emb]``
            emb2: Tensor, shape ``[batch_size, dim_emb]``
            s: Tensor, shape ``[batch_size, 1]``
        Returns:
            sim_score: Tensor, shape ``[batch_size, 1]``   
        """
        if self.edge:
            x = emb1  
        else:
            x = torch.cat([emb1,emb2],dim=-1)            
        
        x = self.act(self.lin1(x))
        x = self.dropout(x)
        sim_score = self.lin2(x)
        return sim_score
