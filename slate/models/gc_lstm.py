import torch
from torch.nn import Parameter
from torch.nn.modules.loss import BCEWithLogitsLoss, MSELoss
from torch_geometric.nn import ChebConv
from torch_geometric.nn.inits import glorot, zeros
from torch_geometric.utils import to_undirected
from slate.models.reg_mlp import RegressionModel
class GCLSTM(torch.nn.Module):
    r"""An implementation of the the Integrated Graph Convolutional Long Short Term
    Memory Cell. For details see this paper: `"GC-LSTM: Graph Convolution Embedded LSTM
    for Dynamic Link Prediction." <https://arxiv.org/abs/1812.04206>`_

    Args:
        in_channels (int): Number of input features.
        out_channels (int): Number of output features.
        K (int): Chebyshev filter size :math:`K`.
        normalization (str, optional): The normalization scheme for the graph
            Laplacian (default: :obj:`"sym"`):

            1. :obj:`None`: No normalization
            :math:`\mathbf{L} = \mathbf{D} - \mathbf{A}`

            2. :obj:`"sym"`: Symmetric normalization
            :math:`\mathbf{L} = \mathbf{I} - \mathbf{D}^{-1/2} \mathbf{A}
            \mathbf{D}^{-1/2}`

            3. :obj:`"rw"`: Random-walk normalization
            :math:`\mathbf{L} = \mathbf{I} - \mathbf{D}^{-1} \mathbf{A}`

            You need to pass :obj:`lambda_max` to the :meth:`forward` method of
            this operator in case the normalization is non-symmetric.
            :obj:`\lambda_max` should be a :class:`torch.Tensor` of size
            :obj:`[num_graphs]` in a mini-batch scenario and a
            scalar/zero-dimensional tensor when operating on single graphs.
            You can pre-compute :obj:`lambda_max` via the
            :class:`torch_geometric.transforms.LaplacianLambdaMax` transform.
        bias (bool, optional): If set to :obj:`False`, the layer will not learn
            an additive bias. (default: :obj:`True`)
    """

    def __init__(
        self,
        num_nodes: int,
        num_features: int,
        in_channels: int,
        time_length: int,
        K: int,
        normalization: str = "sym",
        bias: bool = True,
        one_hot: bool = False,
        undirected: bool = False,
        task_name: str = "link_pred",
    ):
        super(GCLSTM, self).__init__()
        self.num_nodes = num_nodes
        self.num_features = num_features
        self.in_channels = in_channels
        self.out_channels = in_channels
        self.K = K
        self.normalization = normalization
        self.bias = bias
        self.one_hot = one_hot
        self.undirected = undirected
        self.task_name = task_name
        if self.task_name == "link_pred":
            self.bceloss = BCEWithLogitsLoss()
        elif self.task_name == "node_reg":
            self.mseloss = MSELoss()
        else:
            raise ValueError("Task name not recognized")
        self._create_parameters_and_layers()
        self._set_parameters()

    def set_device(self,device):
        self.device = device

    def _create_input_gate_parameters_and_layers(self):

        self.conv_i = ChebConv(
            in_channels=self.out_channels,
            out_channels=self.out_channels,
            K=self.K,
            normalization=self.normalization,
            bias=self.bias,
        )

        self.W_i = Parameter(torch.Tensor(self.in_channels, self.out_channels))
        self.b_i = Parameter(torch.Tensor(1, self.out_channels))

    def _create_forget_gate_parameters_and_layers(self):

        self.conv_f = ChebConv(
            in_channels=self.out_channels,
            out_channels=self.out_channels,
            K=self.K,
            normalization=self.normalization,
            bias=self.bias,
        )

        self.W_f = Parameter(torch.Tensor(self.in_channels, self.out_channels))
        self.b_f = Parameter(torch.Tensor(1, self.out_channels))

    def _create_cell_state_parameters_and_layers(self):

        self.conv_c = ChebConv(
            in_channels=self.out_channels,
            out_channels=self.out_channels,
            K=self.K,
            normalization=self.normalization,
            bias=self.bias,
        )

        self.W_c = Parameter(torch.Tensor(self.in_channels, self.out_channels))
        self.b_c = Parameter(torch.Tensor(1, self.out_channels))

    def _create_output_gate_parameters_and_layers(self):

        self.conv_o = ChebConv(
            in_channels=self.out_channels,
            out_channels=self.out_channels,
            K=self.K,
            normalization=self.normalization,
            bias=self.bias,
        )

        self.W_o = Parameter(torch.Tensor(self.in_channels, self.out_channels))
        self.b_o = Parameter(torch.Tensor(1, self.out_channels))

    def _create_parameters_and_layers(self):
        if self.one_hot:
            self.pool_layer = torch.nn.Linear(self.num_nodes,self.in_channels,bias=False)
        else: 
            self.pool_layer = torch.nn.Linear(self.num_features,self.in_channels,bias=False)
        self._create_input_gate_parameters_and_layers()
        self._create_forget_gate_parameters_and_layers()
        self._create_cell_state_parameters_and_layers()
        self._create_output_gate_parameters_and_layers()

        if self.task_name == 'node_reg':
            self.pred_reg = RegressionModel(self.in_channels,self.in_channels)

    def _set_parameters(self):
        glorot(self.W_i)
        glorot(self.W_f)
        glorot(self.W_c)
        glorot(self.W_o)
        zeros(self.b_i)
        zeros(self.b_f)
        zeros(self.b_c)
        zeros(self.b_o)

    def _set_hidden_state(self, X, H):
        if H is None:
            H = torch.zeros(X.shape[0], self.out_channels).to(X.device)
        return H

    def _set_cell_state(self, X, C):
        if C is None:
            C = torch.zeros(X.shape[0], self.out_channels).to(X.device)
        return C

    def _calculate_input_gate(self, X, edge_index, edge_weight, H, C, lambda_max):
        I = torch.matmul(X, self.W_i)
        I = I + self.conv_i(H, edge_index, edge_weight, lambda_max=lambda_max)
        I = I + self.b_i
        I = torch.sigmoid(I)
        return I

    def _calculate_forget_gate(self, X, edge_index, edge_weight, H, C, lambda_max):
        F = torch.matmul(X, self.W_f)
        F = F + self.conv_f(H, edge_index, edge_weight, lambda_max=lambda_max)
        F = F + self.b_f
        F = torch.sigmoid(F)
        return F

    def _calculate_cell_state(self, X, edge_index, edge_weight, H, C, I, F, lambda_max):
        T = torch.matmul(X, self.W_c)
        T = T + self.conv_c(H, edge_index, edge_weight, lambda_max=lambda_max)
        T = T + self.b_c
        T = torch.tanh(T)
        C = F * C + I * T
        return C

    def _calculate_output_gate(self, X, edge_index, edge_weight, H, C, lambda_max):
        O = torch.matmul(X, self.W_o)
        O = O + self.conv_o(H, edge_index, edge_weight, lambda_max=lambda_max)
        O = O + self.b_o
        O = torch.sigmoid(O)
        return O

    def _calculate_hidden_state(self, O, C):
        H = O * torch.tanh(C)
        return H

    def forward_snapshot(
        self,
        X: torch.FloatTensor,
        edge_index: torch.LongTensor,
        edge_weight: torch.FloatTensor = None,
        H: torch.FloatTensor = None,
        C: torch.FloatTensor = None,
        lambda_max: torch.Tensor = None,
    ) -> torch.FloatTensor:
        """
        Making a forward pass. If edge weights are not present the forward pass
        defaults to an unweighted graph. If the hidden state and cell state
        matrices are not present when the forward pass is called these are
        initialized with zeros.

        Arg types:
            * **X** *(PyTorch Float Tensor)* - Node features.
            * **edge_index** *(PyTorch Long Tensor)* - Graph edge indices.
            * **edge_weight** *(PyTorch Long Tensor, optional)* - Edge weight vector.
            * **H** *(PyTorch Float Tensor, optional)* - Hidden state matrix for all nodes.
            * **C** *(PyTorch Float Tensor, optional)* - Cell state matrix for all nodes.
            * **lambda_max** *(PyTorch Tensor, optional but mandatory if normalization is not sym)* - Largest eigenvalue of Laplacian.

        Return types:
            * **H** *(PyTorch Float Tensor)* - Hidden state matrix for all nodes.
            * **C** *(PyTorch Float Tensor)* - Cell state matrix for all nodes.
        """
        X = self.pool_layer(X)
        H = self._set_hidden_state(X, H)
        C = self._set_cell_state(X, C)
        I = self._calculate_input_gate(X, edge_index, edge_weight, H, C, lambda_max)
        F = self._calculate_forget_gate(X, edge_index, edge_weight, H, C, lambda_max)
        C = self._calculate_cell_state(X, edge_index, edge_weight, H, C, I, F, lambda_max)
        O = self._calculate_output_gate(X, edge_index, edge_weight, H, C, lambda_max)
        H = self._calculate_hidden_state(O, C)
        return H, C
    
    def forward(self,graphs):
        output = []
        H = None
        C = None 
        for t in range(len(graphs)):
            edge_index = graphs[t].edge_index.to(self.device) if self.undirected else graphs[t].edge_index.to(self.device)
            edge_weight = torch.ones_like(edge_index[0], dtype=torch.float).to(self.device)
            H,C = self.forward_snapshot(graphs[t].x.to(self.device), edge_index, edge_weight,H,C)
            output.append(H)
        final_emb = torch.stack(output, dim=1)
        return final_emb
    
    def get_loss_link_pred(self, feed_dict,graphs):

        node_1, node_2, node_2_negative, _, _, _, time  = feed_dict.values()
        # run gnn
        self.final_emb = self.forward(graphs) # [N, T, F]
        emb_source = self.final_emb[node_1,time,:]
        emb_pos  = self.final_emb[node_2,time,:]
        emb_neg = self.final_emb[node_2_negative,time,:]
        pos_score = torch.sum(emb_source*emb_pos, dim=1)
        neg_score = torch.sum(emb_source*emb_neg, dim=1)
        pos_loss = self.bceloss(pos_score, torch.ones_like(pos_score))
        neg_loss = self.bceloss(neg_score, torch.zeros_like(neg_score))
        graphloss = pos_loss + neg_loss
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
            # time-1 because we want to predict the next time step in eval
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