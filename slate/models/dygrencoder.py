import torch
from torch.nn import LSTM
from torch.nn.modules.loss import BCEWithLogitsLoss
from torch_geometric.nn import GatedGraphConv
from torch_geometric.utils import to_undirected


class DyGrEncoder(torch.nn.Module):
    r"""An implementation of the integrated Gated Graph Convolution Long Short
    Term Memory Layer. For details see this paper: `"Predictive Temporal Embedding
    of Dynamic Graphs." <https://ieeexplore.ieee.org/document/9073186>`_

    Args:
        conv_out_channels (int): Number of output channels for the GGCN.
        conv_num_layers (int): Number of Gated Graph Convolutions.
        conv_aggr (str): Aggregation scheme to use
            (:obj:`"add"`, :obj:`"mean"`, :obj:`"max"`).
        lstm_out_channels (int): Number of LSTM channels.
        lstm_num_layers (int): Number of neurons in LSTM.
    """

    def __init__(
        self,
        num_nodes: int,
        num_features: int,
        in_channels: int,
        conv_num_layers: int,
        conv_aggr: str,
        lstm_num_layers: int,
        time_length: int,
        one_hot: bool = False,
        undirected: bool = True,
    ):
        super(DyGrEncoder, self).__init__()
        assert conv_aggr in ["mean", "add", "max"], "Wrong aggregator."
        self.num_nodes = num_nodes
        self.num_features = num_features
        self.conv_out_channels = in_channels
        self.conv_num_layers = conv_num_layers
        self.conv_aggr = conv_aggr
        self.lstm_out_channels = in_channels
        self.lstm_num_layers = lstm_num_layers
        self.one_hot = one_hot
        self.undirected = undirected
        self.bceloss = BCEWithLogitsLoss()
        self._create_layers()

    def set_device(self,device):
        self.device = device

    def _create_layers(self):
        if self.one_hot:
            self.pool_layer = torch.nn.Linear(self.num_nodes,self.conv_out_channels,bias=False)
        else: 
            self.pool_layer = torch.nn.Linear(self.num_features,self.conv_out_channels,bias=False)

        self.conv_layer = GatedGraphConv(
            out_channels=self.conv_out_channels,
            num_layers=self.conv_num_layers,
            aggr=self.conv_aggr,
            bias=True,
        )

        self.recurrent_layer = LSTM(
            input_size=self.conv_out_channels,
            hidden_size=self.lstm_out_channels,
            num_layers=self.lstm_num_layers,
        )

    def forward_snapshot(
        self,
        X: torch.FloatTensor,
        edge_index: torch.LongTensor,
        edge_weight: torch.FloatTensor = None,
        H: torch.FloatTensor = None,
        C: torch.FloatTensor = None,
    ) -> torch.FloatTensor:
        """
        Making a forward pass. If the hidden state and cell state matrices are
        not present when the forward pass is called these are initialized with zeros.

        Arg types:
            * **X** *(PyTorch Float Tensor)* - Node features.
            * **edge_index** *(PyTorch Long Tensor)* - Graph edge indices.
            * **edge_weight** *(PyTorch Float Tensor, optional)* - Edge weight vector.
            * **H** *(PyTorch Float Tensor, optional)* - Hidden state matrix for all nodes.
            * **C** *(PyTorch Float Tensor, optional)* - Cell state matrix for all nodes.

        Return types:
            * **H_tilde** *(PyTorch Float Tensor)* - Output matrix for all nodes.
            * **H** *(PyTorch Float Tensor)* - Hidden state matrix for all nodes.
            * **C** *(PyTorch Float Tensor)* - Cell state matrix for all nodes.
        """
        X = self.pool_layer(X)
        H_tilde = self.conv_layer(X, edge_index, edge_weight)
        H_tilde = H_tilde[None, :, :]
        if H is None and C is None:
            H_tilde, (H, C) = self.recurrent_layer(H_tilde)
        elif H is not None and C is not None:
            H = H[None, :, :]
            C = C[None, :, :]
            H_tilde, (H, C) = self.recurrent_layer(H_tilde, (H, C))
        else:
            raise ValueError("Invalid hidden state and cell matrices.")
        H_tilde = H_tilde.squeeze()
        H = H.squeeze()
        C = C.squeeze()
        return H_tilde, H, C
    
    def forward(self,graphs):
        output = []
        H = None
        C = None 
        for t in range(len(graphs)):
            edge_index = to_undirected(graphs[t].edge_index).to(self.device) if self.undirected else graphs[t].edge_index.to(self.device)
            edge_weight = torch.ones_like(edge_index[0]).to(self.device)
            H_tilde,H,C = self.forward_snapshot(graphs[t].x.to(self.device), edge_index, edge_weight,H,C)
            output.append(H_tilde)
        final_emb = torch.stack(output, dim=1)
        return final_emb
    
    def get_loss_link_pred(self, feed_dict,graphs):

        node_1, node_2, node_2_negative, _, _, _, time  = feed_dict.values()
        # run gnn
        self.final_emb = self.forward(graphs) # [N, T, F]
        #import ipdb; ipdb.set_trace()
        emb_source = self.final_emb[node_1,time,:]
        emb_pos  = self.final_emb[node_2,time,:]
        emb_neg = self.final_emb[node_2_negative,time,:]
        pos_score = torch.sum(emb_source*emb_pos, dim=1)
        neg_score = torch.sum(emb_source*emb_neg, dim=1)
        pos_loss = self.bceloss(pos_score, torch.ones_like(pos_score))
        neg_loss = self.bceloss(neg_score, torch.zeros_like(neg_score))
        graphloss = pos_loss + neg_loss
        return graphloss
    
    def score_eval(self,feed_dict,graphs):
        with torch.no_grad():
            node_1, node_2, node_2_negative, _, _, _, time  = feed_dict.values()
            # run gnn
            final_emb = self.forward(graphs) # [N, T, F]
            # for the eval we get the last embedding
            emb_source = final_emb[node_1, time-1 ,:]
            emb_pos  = final_emb[node_2, time-1 ,:]
            emb_neg = final_emb[node_2_negative, time-1 ,:]
            pos_score = torch.sum(emb_source*emb_pos, dim=1)
            neg_score = torch.sum(emb_source*emb_neg, dim=1)        
            return pos_score.sigmoid(),neg_score.sigmoid()
    