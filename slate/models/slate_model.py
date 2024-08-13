import numpy as np
import torch
import torch.nn as nn
from torch.backends.cuda import sdp_kernel, SDPBackend
from torch.nn.modules.loss import BCEWithLogitsLoss
from slate.models.slate_layers import LinkPredScore, CrossAttention, NodeAggregation, EdgeAggregation
from slate.lib import graphs_to_supra, reindex_edge_index, AddSupraLaplacianPE 
from performer_pytorch import Performer

class SLATE(nn.Module):
    """
    SLATE model with a full spatio temporal attention and a full spatio temporal encoding. 
    """
    def __init__(
        self,
        num_nodes: int,
        num_features: int,
        num_classes: int,
        time_length: int,
        window: int,
        pred_next: bool,
        decision: str,
        use_cross_attn: bool,
        flash: bool,
        use_performer: bool,
        light_ca: bool,
        nhead_ca: int,
        dropout_ca: float,
        bias_ca : bool,
        add_bias_kv: bool,
        add_zero_attn: bool,
        dropout_dec: float,
        dim_emb: int,
        dim_pe: int,
        norm_lap: str,
        add_eig_vals: bool,
        remove_isolated: bool,
        add_vn: bool,
        isolated_in_transformer: bool,
        add_time_connection: bool,
        p_self_time: float,
        add_lin_pe: bool,
        bias_lin_pe: bool,
        dim_feedforward: int,
        nhead: int,
        num_layers_trsf: int,
        aggr: str,
        one_hot: bool = True,
        norm_first: bool = False,
        undirected: bool = True, 
    ):
        
        super().__init__()
        # Graphs info 
        self.num_nodes = num_nodes
        self.num_classes = num_classes
        self.num_features = num_features
        self.time_length = time_length
        self.undirected = undirected
        # Training parameters
        self.window = window 
        self.pred_next = pred_next
        self.decision = decision
        self.aggr = aggr
        # Transformer and cross attn parameters
        self.use_cross_attn = use_cross_attn
        self.light_ca = light_ca
        self.flash = flash
        self.use_performer = use_performer
        self.nhead_ca = nhead_ca
        self.dropout_ca = dropout_ca
        self.bias_ca = bias_ca
        self.add_bias_kv = add_bias_kv
        self.add_zero_attn = add_zero_attn
        self.dropout_dec = dropout_dec
        self.dim_emb = dim_emb
        self.num_layers_trsf = num_layers_trsf
        self.one_hot = one_hot
        self.norm_first = norm_first
        # SupraLaplacian PE
        self.dim_pe = dim_pe
        self.add_lin_pe = add_lin_pe
        self.bias_lin_pe = bias_lin_pe
        self.dim_feedforward = dim_feedforward
        self.nhead = nhead
        self.norm_lap = norm_lap
        self.add_eig_vals = add_eig_vals
        self.add_vn = add_vn
        self.remove_isolated = remove_isolated
        self.isolated_in_transformer = isolated_in_transformer
        self.add_time_connection = add_time_connection
        self.p_self_time = p_self_time
        
        self.bceloss = BCEWithLogitsLoss()
        self.build_model()
        if self.use_performer:
            self.flash = False # Performer does not support flash (not tested )
        assert self.aggr in ["mean","sum","max","last"], "Aggregation must be either last, mean, sum or max"
        assert self.decision in ["mlp","dot"], "Decision must be either mlp or dot"

            
    def set_device(self,device):
        self.device = device

    def build_model(self):
        self.num_nodes_embedding = self.num_nodes + 1 if self.add_vn else self.num_nodes
        # Initialize node embedding
        self.node_embedding = nn.Embedding(num_embeddings=self.num_nodes_embedding, embedding_dim=self.dim_emb)

        # Initialize spatio temporal PE
        self.use_edge_attr = False
        self.supralaplacianPE = AddSupraLaplacianPE(k=self.dim_pe,
                                               normalization=self.norm_lap,
                                               is_undirected=self.undirected,
                                               add_eig_vals=self.add_eig_vals)
        
        self.in_dim_pe = 2 * self.dim_pe if self.add_eig_vals else self.dim_pe
               
        # Initialize linear PE
        self.lin_pe = nn.Linear(self.in_dim_pe, self.in_dim_pe, bias=self.bias_lin_pe)
        
        # Initialize linear input 
        self.lin_input = nn.Linear(self.dim_emb + self.in_dim_pe, self.dim_emb, bias=True)
        
        # Initialize projection layer (deprecated)
        self.proj = nn.Linear(self.dim_emb, self.dim_emb, bias=True)
        
        # Decision function
        if self.decision == "mlp" or self.use_cross_attn:
            self.pred = LinkPredScore(dim_emb = self.dim_emb,
                                      dropout = self.dropout_dec,
                                      edge = self.use_cross_attn)
        
        # Initialize spatio-temporal attention

        norm = nn.LayerNorm(self.dim_emb)
        
        if self.use_performer: 
            # PROTOTYPE with naive parameters for rebuttal
            self.spatio_temp_attn = Performer(
                    dim=self.dim_emb,
                    depth=self.num_layers_trsf,
                    heads=self.nhead,
                    causal=False,           # Set to True for autoregressive tasks
                    dim_head=self.dim_emb // self.nhead, # Dimension of each attention head
                    ff_mult=self.dim_feedforward // self.dim_emb   # Feedforward network multiplier
                )
        else:
            encoder_layer = nn.TransformerEncoderLayer(d_model = self.dim_emb,
                                                    nhead=self.nhead,
                                                    dim_feedforward=self.dim_feedforward,
                                                    batch_first=True,
                                                    norm_first=self.norm_first,
                                                    )     
                
            self.spatio_temp_attn = nn.TransformerEncoder(encoder_layer,
                                                        num_layers = self.num_layers_trsf,
                                                        norm = norm)
            
        if self.use_cross_attn: 
            self.cross_attn = CrossAttention(dim_emb=self.dim_emb,
                                             num_heads=self.nhead_ca,
                                             dropout=self.dropout_ca,
                                             bias=self.bias_ca,
                                             add_bias_kv=self.add_bias_kv,
                                             add_zero_attn=self.add_zero_attn,
                                             light=self.light_ca)
            
        # Aggregation  (Temporal aggregation in our Figure 2 .d )
        if self.use_cross_attn:
            self.aggregation = EdgeAggregation(self.aggr)
        else:
            self.aggregation = NodeAggregation(self.aggr)
        



    
    def compute_st_pe(self, graphs):
        """
        Arguments:
            graphs: List of torch_geometric.data.Data
        """
        # First : from graphs, construct a supra adjacency matrix.
        w = len(graphs)
        edge_index, edge_weight, mask = graphs_to_supra(graphs,
                                                  self.num_nodes,
                                                  self.add_time_connection,
                                                  self.remove_isolated,
                                                  add_vn=self.add_vn,
                                                  p=self.p_self_time)
        
        # Second : if we remove isolated nodes, reindex the edge to compute the laplacian 
        if self.remove_isolated: 
            # makes the graph connected by snapshot
            edge_index = reindex_edge_index(edge_index) # reindex node from 0 to len(torch.unique(edge_index))
        
        edge_index = edge_index.to(self.device)
        edge_weight = edge_weight.to(self.device)
        mask = mask.to(self.device)
        
        # The number of nodes in the supra graph
        num_nodes_supra_adj = len(torch.unique(edge_index)) if self.remove_isolated else self.num_nodes*w
        
        # Third : compute the supralaplacian PE
        pe = self.supralaplacianPE(edge_index, edge_weight, num_nodes_supra_adj)
        if self.add_lin_pe:
            pe = self.lin_pe(pe)

        # Fourth : Construct the token for the transformer
        if not self.isolated_in_transformer:
            raise NotImplementedError #
            # TODO : TEST WITHOUT ISOLATED NODES IN TRANSFORMER
        else:
            all_pe = torch.zeros((self.num_nodes_embedding*w,self.in_dim_pe)).to(self.device)
            all_pe[mask == 1] = pe
            node_emb = self.node_embedding(torch.arange(self.num_nodes_embedding).to(self.device))
            tokens = node_emb[:-1,:].repeat(w,1)
            vn_emb = node_emb[-1].repeat(w,1)
            tokens = torch.cat((tokens,vn_emb),dim=0) # Add the virtual nodes at the end of the tokens matrix (easyer to process later)
            # Add the positional encoding to the tokens
            tokens = torch.cat((tokens,all_pe),dim=1)
        # Fifth : Project linearly the tokens containing node emb and supraPE
        tokens = self.lin_input(tokens) # [N',dim_emb] or [N*W,dim_emb] if integrate isolated node for transformer
        
        return tokens, mask

    def forward(self, graphs,eval=False):
        """
        Arguments:
            graphs: List of torch_geometric.data.Data
            eval: bool, if True, the model is in evaluation mode (for debug)

        Returns:
            final_emb: Tensor, shape [N, T, F]
        """
        w = len(graphs)
        # compute the spatio temporal positional encoding
        tokens, mask = self.compute_st_pe(graphs) # tokens: [N',F]  mask: [W,N]
        
        # Perform a spatio-temporal full attention # We flat all nodes at w snapshots  
        # The idea of SLATE is to consider same node at different snapshot as independant token
        z_tokens = self.spatio_temp_attn(tokens.unsqueeze(0)).squeeze(0) # [N',F] # careful, with isolated nodes N' != N*len(graphs)
        
        if not self.isolated_in_transformer: 
            # Remove vn of the token matrix
            # We need to proj the isolated nodes in the same emb space as the z_tokens
            final_emb = self.proj(final_emb) # [N, W, F]
            raise NotImplementedError # TODO : TEST WITHOUT
        else:
            if self.add_vn:
                z_tokens = z_tokens[:-w] # We dont need virtual nodes in the final embedding for predictions 
            final_emb = z_tokens.reshape(self.num_nodes, w, self.dim_emb) # [N, W, F]
        return final_emb
    

    def get_loss_link_pred(self, feed_dict, graphs):
        """
        Arguments:
            feed_dict: dict, keys are node_1, node_2, node_2_negative, time
            graphs: List of torch_geometric.data.Data
        """
        with sdp_kernel(enable_flash=self.flash, enable_math=not(self.flash), enable_mem_efficient=False):
            # s_pos, s_neg is deprecated: TODO REMOVE FROM DATALOADER
            node_1, node_2, node_2_negative, _, s_pos, s_neg, time  = feed_dict.values()
            # FORWARD
            tw = max(0,len(graphs)-self.window)        
            final_emb = self.forward(graphs[tw:]) # [N, W, F]

            if self.use_cross_attn: # EDGE WISE REPRESENTATIONS
                # Cross Attention
                pos_edge = self.cross_attn(final_emb[node_1],final_emb[node_2])
                neg_edge = self.cross_attn(final_emb[node_1],final_emb[node_2_negative])
                # Aggregation
                pos_edge = self.aggregation(pos_edge).squeeze()
                neg_edge = self.aggregation(neg_edge).squeeze()
                # Decision
                pos_score = self.pred(pos_edge, s_pos)
                neg_score = self.pred(neg_edge, s_neg)
                
            else: # NODE WISE REPRESENTATIONS
                # Node Aggregation
                emb_source = self.aggregation(final_emb,node_1,time)
                emb_pos = self.aggregation(final_emb,node_2,time)
                emb_neg = self.aggregation(final_emb,node_2_negative,time)

                # Decision 
                if self.decision == "mlp":
                    pos_score = self.pred(emb_source, emb_pos, s_pos)
                    neg_score = self.pred(emb_source, emb_neg, s_neg)
                else:      
                    pos_score = torch.sum(emb_source*emb_pos, dim=1) 
                    neg_score = torch.sum(emb_source*emb_neg, dim=1)
            # LOSS
            pos_loss = self.bceloss(pos_score, torch.ones_like(pos_score))
            neg_loss = self.bceloss(neg_score, torch.zeros_like(neg_score))
            graphloss = pos_loss + neg_loss   
            return graphloss, pos_score.detach().sigmoid(), neg_score.detach().sigmoid()
    

    def score_eval(self, feed_dict, graphs):
        """
        Arguments:
            feed_dict: dict, keys are node_1, node_2, node_2_negative, time
            graphs: List of torch_geometric.data.Data
        """
        with torch.no_grad():
            with sdp_kernel(enable_flash=self.flash, enable_math=False, enable_mem_efficient=False):
                node_1, node_2, node_2_negative, _, s_pos, s_neg, time  = feed_dict.values()
                
                # FORWARD
                tw = max(0,len(graphs)-self.window)        
                final_emb = self.forward(graphs[tw:]) # [N, W, F]

                if self.use_cross_attn: # EDGE WISE REPRESENTATIONS
                    # Cross Attention
                    pos_edge = self.cross_attn(final_emb[node_1],final_emb[node_2])
                    neg_edge = self.cross_attn(final_emb[node_1],final_emb[node_2_negative])
                    # Aggregation
                    pos_edge = self.aggregation(pos_edge).squeeze()
                    neg_edge = self.aggregation(neg_edge).squeeze()
                    # Decision
                    pos_score = self.pred(pos_edge, s_pos)
                    neg_score = self.pred(neg_edge, s_neg)
                    
                else: # NODE WISE REPRESENTATIONS
                    # Node Aggregation
                    emb_source = self.aggregation(final_emb,node_1,time)
                    emb_pos = self.aggregation(final_emb,node_2,time)
                    emb_neg = self.aggregation(final_emb,node_2_negative,time)
                    # Decision 
                    if self.decision == "mlp":
                        pos_score = self.pred(emb_source, emb_pos, s_pos)
                        neg_score = self.pred(emb_source, emb_neg, s_neg)
                    else:      
                        pos_score = torch.sum(emb_source*emb_pos, dim=1) 
                        neg_score = torch.sum(emb_source*emb_neg, dim=1)
                
                return pos_score.sigmoid() ,neg_score.sigmoid()