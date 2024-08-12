import torch
import torch.nn as nn
from torch.nn.modules.loss import BCEWithLogitsLoss
from torch_geometric.data import Data
from torch_geometric.transforms import AddLaplacianEigenvectorPE,AddRandomWalkPE
from torch_geometric.utils import to_undirected
#from fast_transformers.builders import TransformerEncoderBuilder
from slate.models.slate_layers import PositionalEncoding

class LSTMGT(nn.Module):
    def __init__(
        self,
        num_nodes: int,
        num_features: int,
        num_classes: int,
        time_length: int,
        window: int,
        spatial_pe: str,
        dim_emb: int,
        dim_pe: int,
        dim_feedforward: int,
        nhead: int,
        num_layers_lstm: int,
        one_hot: bool = True,
        norm_first: bool = False,
        undirected: bool = True, 

    ):
        super().__init__()
        self.num_nodes = num_nodes
        self.num_classes = num_classes
        self.num_features = num_features
        self.time_length = time_length
        self.window = window 
        self.spatial_pe = spatial_pe
        self.dim_emb = dim_emb
        self.dim_pe = dim_pe
        self.dim_feedforward = dim_feedforward
        self.nhead = nhead
        self.lstm_num_layers = num_layers_lstm
        self.one_hot = one_hot
        self.norm_first = norm_first
        self.undirected = undirected
        self.bceloss = BCEWithLogitsLoss()
        self.in_channels = self.dim_emb - self.dim_pe
        self.build_model()
        assert self.spatial_pe in ["lpe","rwpe"], "PE must be either lpe or rwpe"
        
    def set_device(self,device):
        self.device = device

    def build_model(self):
        # One hot encoding transformation
        self.lin = nn.Linear(self.num_nodes, self.in_channels, bias=False)
        # Linear for Positional / Structural Encoding
        self.lin_pe = nn.Linear(self.dim_pe, self.dim_pe, bias=False)
        # Linear for input
        self.lin_input = nn.Linear(self.dim_emb, self.dim_emb, bias=False)
        # Spatial Full Attention
        self.spatial_attn = nn.TransformerEncoderLayer(
                                                        d_model=self.dim_emb,
                                                        dim_feedforward=self.dim_feedforward,
                                                        nhead=self.nhead,
                                                        batch_first=True,
                                                        norm_first=self.norm_first
                                                        )
        # Temporal Update LSTM
        self.temporal_updtate = nn.LSTM(
            input_size=self.dim_emb,
            hidden_size=self.dim_emb,
            num_layers=self.lstm_num_layers,
        )
        
        # Initalize the positional encoding method
        if self.spatial_pe == "lpe":
            self.pe_enc = AddLaplacianEigenvectorPE(k=self.dim_pe,is_undirected=self.undirected)
        elif self.spatial_pe == "rwpe":
            self.pe_enc = AddRandomWalkPE(walk_length=self.dim_pe)
        
        
        
    def construct_graph_pe(self,edge_index,x):
        """
        Arguments:
            x: Tensor, shape [N, F]
            edge_index: Tensor, shape [2, E]
        """
        # Construct Pytorch geometric data object of the graph snapshot.
        if self.spatial_pe == 'lpe' or self.spatial_pe == 'rwpe':
            data = Data(x=x,edge_index=edge_index)
            data.num_nodes = self.num_nodes

        if self.spatial_pe == 'lpe':
            graph_pe = self.pe_enc(data).laplacian_eigenvector_pe
        elif self.spatial_pe == 'rwpe':
            graph_pe = self.pe_enc(data).random_walk_pe
        elif self.spatial_pe == 'gnn':
            graph_pe = self.pe_enc(x,edge_index)
        else: 
            graph_pe = torch.ones((self.num_nodes,self.dim_pe))
        return graph_pe.to(self.device)

    
    def forward_snapshot(self, x, edge_index, h, c):
        """
        Arguments:
            x: Tensor, shape [N, F]
            edge_index: Tensor, shape [2, E]
        """
        if self.undirected:
            edge_index = to_undirected(edge_index)
            
        # Get the node embedding
        if self.one_hot:
            x = self.lin(x)
        
        # Get the positional encoding
        graph_pe = self.construct_graph_pe(edge_index,x)
        
        # Linear for Positional / Structural Encoding       
        graph_pe = self.lin_pe(graph_pe)
           
        # Concatenate the node embedding with PE
        x = torch.cat([x,graph_pe],dim=-1)
        
        # Project linearly the token containing node emb, pos emb and temp emb   
        tokens = self.lin_input(x)
        
        # Spatial Full Attention 
        output = self.spatial_attn(tokens.unsqueeze(0)).squeeze(0) # [1, N, F]
        
        if h is None and c is None:
            output, (h, c) = self.temporal_updtate(output)
        elif h is not None and c is not None:
            #h = h[None, :, :]
            #h = c[None, :, :]
            output, (h, c) = self.temporal_updtate(output, (h, c))
        else:
            raise ValueError("Invalid hidden state and cell matrices.")
        
        return output, (h,c) # [N, F]

    def forward(self, graphs,eval=False):
        """
        Arguments:
            graphs: List of torch_geometric.data.Data
            eval: bool, if True, the model is in evaluation mode (for debug)

        Returns:
            final_emb: Tensor, shape [N, T, F]
        """
        h = None 
        c = None
        for t in range(len(graphs)):
            final_emb, (h,c) = self.forward_snapshot(
                                            graphs[t].x.to(self.device),
                                            graphs[t].edge_index.to(self.device),
                                            h,
                                            c,
                                            )            
        
        return final_emb # [N,F]

    def get_loss_link_pred(self, feed_dict,graphs):
        node_1, node_2, node_2_negative, _, _, _, time  = feed_dict.values()
        # run gnn
        final_emb = self.forward(graphs) # [N, T, F]
        emb_source = final_emb[node_1]
        emb_pos  = final_emb[node_2]
        emb_neg = final_emb[node_2_negative]
        pos_score = torch.sum(emb_source*emb_pos, dim=1)
        neg_score = torch.sum(emb_source*emb_neg, dim=1)
        pos_loss = self.bceloss(pos_score, torch.ones_like(pos_score))
        neg_loss = self.bceloss(neg_score, torch.zeros_like(neg_score))
        graphloss = pos_loss + neg_loss  
        return graphloss, pos_score.detach().sigmoid(), neg_score.detach().sigmoid()
    
    def score_eval(self,feed_dict,graphs):
        with torch.no_grad():
            node_1, node_2, node_2_negative, _, _, _, time  = feed_dict.values()
            # run gnn
            final_emb = self.forward(graphs) # [N, T, F]
            # time-1 because we want to predict the next time step in eval
            emb_source = final_emb[node_1 ,:]
            emb_pos  = final_emb[node_2 ,:]
            emb_neg = final_emb[node_2_negative ,:]
            pos_score = torch.sum(emb_source*emb_pos, dim=1)
            neg_score = torch.sum(emb_source*emb_neg, dim=1)
            
            return pos_score.sigmoid(),neg_score.sigmoid()