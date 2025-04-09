import torch
from torch.nn import Parameter
from torch.nn.modules.loss import BCEWithLogitsLoss
from torch_geometric.nn import ChebConv
from torch_geometric.nn.inits import glorot, zeros
from torch_geometric.utils import to_undirected


class GConvLSTM(torch.nn.Module):
    r"""An implementation of the Chebyshev Graph Convolutional Long Short Term Memory
    Cell. For details see this paper: `"Structured Sequence Modeling with Graph
    Convolutional Recurrent Networks." <https://arxiv.org/abs/1612.07659>`_

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
    ):
        super(GConvLSTM, self).__init__()
        self.num_nodes = num_nodes
        self.num_features = num_features
        self.in_channels = in_channels
        self.out_channels = in_channels
        self.K = K
        self.normalization = normalization
        self.bias = bias
        self.one_hot = one_hot
        self.undirected = undirected
        self.bceloss = BCEWithLogitsLoss()
        self._create_parameters_and_layers()
        self._set_parameters()

    def _create_input_gate_parameters_and_layers(self):
        self.conv_x_i = ChebConv(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            K=self.K,
            normalization=self.normalization,
            bias=self.bias,
        )

        self.conv_h_i = ChebConv(
            in_channels=self.out_channels,
            out_channels=self.out_channels,
            K=self.K,
            normalization=self.normalization,
            bias=self.bias,
        )

        self.w_c_i = Parameter(torch.Tensor(1, self.out_channels))
        self.b_i = Parameter(torch.Tensor(1, self.out_channels))

    def _create_forget_gate_parameters_and_layers(self):
        self.conv_x_f = ChebConv(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            K=self.K,
            normalization=self.normalization,
            bias=self.bias,
        )

        self.conv_h_f = ChebConv(
            in_channels=self.out_channels,
            out_channels=self.out_channels,
            K=self.K,
            normalization=self.normalization,
            bias=self.bias,
        )

        self.w_c_f = Parameter(torch.Tensor(1, self.out_channels))
        self.b_f = Parameter(torch.Tensor(1, self.out_channels))

    def _create_cell_state_parameters_and_layers(self):
        self.conv_x_c = ChebConv(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            K=self.K,
            normalization=self.normalization,
            bias=self.bias,
        )

        self.conv_h_c = ChebConv(
            in_channels=self.out_channels,
            out_channels=self.out_channels,
            K=self.K,
            normalization=self.normalization,
            bias=self.bias,
        )

        self.b_c = Parameter(torch.Tensor(1, self.out_channels))

    def _create_output_gate_parameters_and_layers(self):
        self.conv_x_o = ChebConv(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            K=self.K,
            normalization=self.normalization,
            bias=self.bias,
        )

        self.conv_h_o = ChebConv(
            in_channels=self.out_channels,
            out_channels=self.out_channels,
            K=self.K,
            normalization=self.normalization,
            bias=self.bias,
        )

        self.w_c_o = Parameter(torch.Tensor(1, self.out_channels))
        self.b_o = Parameter(torch.Tensor(1, self.out_channels))

    def _create_parameters_and_layers(self):
        if self.one_hot:
            self.pool_layer = torch.nn.Linear(
                self.num_nodes, self.in_channels, bias=False
            )
        else:
            self.pool_layer = torch.nn.Linear(
                self.num_features, self.in_channels, bias=False
            )
        self._create_input_gate_parameters_and_layers()
        self._create_forget_gate_parameters_and_layers()
        self._create_cell_state_parameters_and_layers()
        self._create_output_gate_parameters_and_layers()

    def _set_parameters(self):
        glorot(self.w_c_i)
        glorot(self.w_c_f)
        glorot(self.w_c_o)
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
        input_gate = self.conv_x_i(X, edge_index, edge_weight, lambda_max=lambda_max)
        input_gate = input_gate + self.conv_h_i(
            H, edge_index, edge_weight, lambda_max=lambda_max
        )
        input_gate = input_gate + (self.w_c_i * C)
        input_gate = input_gate + self.b_i
        input_gate = torch.sigmoid(input_gate)
        return input_gate

    def _calculate_forget_gate(self, X, edge_index, edge_weight, H, C, lambda_max):
        F = self.conv_x_f(X, edge_index, edge_weight, lambda_max=lambda_max)
        F = F + self.conv_h_f(H, edge_index, edge_weight, lambda_max=lambda_max)
        F = F + (self.w_c_f * C)
        F = F + self.b_f
        F = torch.sigmoid(F)
        return F

    def _calculate_cell_state(
        self, X, edge_index, edge_weight, H, C, input_gate, F, lambda_max
    ):
        T = self.conv_x_c(X, edge_index, edge_weight, lambda_max=lambda_max)
        T = T + self.conv_h_c(H, edge_index, edge_weight, lambda_max=lambda_max)
        T = T + self.b_c
        T = torch.tanh(T)
        C = F * C + input_gate * T
        return C

    def _calculate_output_gate(self, X, edge_index, edge_weight, H, C, lambda_max):
        output_gate = self.conv_x_o(X, edge_index, edge_weight, lambda_max=lambda_max)
        output_gate = output_gate + self.conv_h_o(
            H, edge_index, edge_weight, lambda_max=lambda_max
        )
        output_gate = output_gate + (self.w_c_o * C)
        output_gate = output_gate + self.b_o
        output_gate = torch.sigmoid(output_gate)
        return output_gate

    def _calculate_hidden_state(self, output_gate, C):
        H = output_gate * torch.tanh(C)
        return H

    def set_device(self, device):
        self.device = device

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
        X = X.to(self.device)
        edge_index = edge_index.to(self.device)

        if self.undirected:
            edge_index = to_undirected(edge_index)
        edge_weight = torch.ones_like(edge_index[0], dtype=torch.float).to(
            self.device
        )  # all edge weights are 1
        X = self.pool_layer(X)
        H = self._set_hidden_state(X, H)
        C = self._set_cell_state(X, C)
        input_gate = self._calculate_input_gate(
            X, edge_index, edge_weight, H, C, lambda_max
        )
        F = self._calculate_forget_gate(X, edge_index, edge_weight, H, C, lambda_max)
        C = self._calculate_cell_state(
            X, edge_index, edge_weight, H, C, input_gate, F, lambda_max
        )
        output_gate = self._calculate_output_gate(
            X, edge_index, edge_weight, H, C, lambda_max
        )
        H = self._calculate_hidden_state(output_gate, C)
        return H, C

    def forward(self, graphs):
        output = []
        H = None
        C = None
        for t in range(len(graphs)):
            H, C = self.forward_snapshot(
                graphs[t].x, graphs[t].edge_index, graphs[t].edge_weight, H, C
            )
            output.append(H)
        final_emb = torch.stack(output, dim=1)
        return final_emb

    def get_loss_link_pred(self, feed_dict, graphs):
        node_1, node_2, node_2_negative, _, _, _, time = feed_dict.values()
        # run gnn
        self.final_emb = self.forward(graphs)  # [N, T, F]
        emb_source = self.final_emb[node_1, time, :]
        emb_pos = self.final_emb[node_2, time, :]
        emb_neg = self.final_emb[node_2_negative, time, :]
        pos_score = torch.sum(emb_source * emb_pos, dim=1)
        neg_score = torch.sum(emb_source * emb_neg, dim=1)
        pos_loss = self.bceloss(pos_score, torch.ones_like(pos_score))
        neg_loss = self.bceloss(neg_score, torch.zeros_like(neg_score))
        graphloss = pos_loss + neg_loss
        return graphloss

    def score_eval(self, feed_dict, graphs):
        with torch.no_grad():
            node_1, node_2, node_2_negative, _, _, _, time = feed_dict.values()
            # run gnn
            final_emb = self.forward(graphs)  # [N, T, F]
            emb_source = final_emb[node_1, time - 1, :]
            emb_pos = final_emb[node_2, time - 1, :]
            emb_neg = final_emb[node_2_negative, time - 1, :]
            pos_score = torch.sum(emb_source * emb_pos, dim=1)
            neg_score = torch.sum(emb_source * emb_neg, dim=1)
            return pos_score.sigmoid(), neg_score.sigmoid()
