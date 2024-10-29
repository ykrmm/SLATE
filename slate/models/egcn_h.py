import math
import torch
import torch.nn as nn
from torch.nn.modules.loss import BCEWithLogitsLoss, MSELoss
from torch_geometric.utils import to_undirected
from torch_geometric_temporal.nn.recurrent.evolvegcno import GCNConv_Fixed_W

from slate.models.reg_mlp import RegressionModel
# Original source code from :
# https://pytorch-geometric-temporal.readthedocs.io/en/latest/_modules/torch_geometric_temporal/nn/recurrent/evolvegcnh.html#EvolveGCNH
# This is the modified version for the link prediction task with the same training protocol as the other models (DySAT, DGT…)


def pad_with_last_val(vect, k):
    device = "cuda" if vect.is_cuda else "cpu"
    pad = torch.ones(k - vect.size(0), dtype=torch.long, device=device) * vect[-1]
    vect = torch.cat([vect, pad])
    return vect


class mat_GRU_cell(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.update = mat_GRU_gate(
            self.in_channels, self.out_channels, torch.nn.Sigmoid()
        )

        self.reset = mat_GRU_gate(
            self.in_channels, self.out_channels, torch.nn.Sigmoid()
        )

        self.htilda = mat_GRU_gate(self.in_channels, self.out_channels, torch.nn.Tanh())

        self.choose_topk = TopK(feats=self.in_channels, k=self.out_channels)

    def forward(self, prev_W, prev_Z):
        z_topk = self.choose_topk(prev_Z)

        update = self.update(z_topk, prev_W)
        reset = self.reset(z_topk, prev_W)

        h_cap = reset * prev_W
        h_cap = self.htilda(z_topk, h_cap)

        new_W = (1 - update) * prev_W + update * h_cap

        return new_W


class mat_GRU_gate(torch.nn.Module):
    def __init__(self, rows, cols, activation):
        super().__init__()
        self.activation = activation
        # the k here should be in_feats which is actually the rows
        self.W = nn.Parameter(torch.Tensor(rows, rows))
        self.reset_param(self.W)

        self.U = nn.Parameter(torch.Tensor(rows, rows))
        self.reset_param(self.U)

        self.bias = nn.Parameter(torch.zeros(rows, cols))

    def reset_param(self, t):
        # Initialize based on the number of columns
        stdv = 1.0 / math.sqrt(t.size(1))
        t.data.uniform_(-stdv, stdv)

    def forward(self, x, hidden):
        out = self.activation(self.W.matmul(x) + self.U.matmul(hidden) + self.bias)

        return out


class TopK(torch.nn.Module):
    def __init__(self, feats, k):
        super().__init__()
        self.scorer = nn.Parameter(torch.Tensor(feats, 1))
        self.reset_param(self.scorer)

        self.k = k

    def reset_param(self, t):
        # Initialize based on the number of rows
        stdv = 1.0 / math.sqrt(t.size(0))
        t.data.uniform_(-stdv, stdv)

    def forward(self, node_embs):
        scores = node_embs.matmul(self.scorer) / self.scorer.norm()
        # scores = scores + mask

        vals, topk_indices = scores.view(-1).topk(self.k)
        topk_indices = topk_indices[vals > -float("Inf")]

        if topk_indices.size(0) < self.k:
            topk_indices = pad_with_last_val(topk_indices, self.k)

        tanh = torch.nn.Tanh()

        if isinstance(node_embs, torch.sparse.FloatTensor) or isinstance(
            node_embs, torch.cuda.sparse.FloatTensor
        ):
            node_embs = node_embs.to_dense()

        out = node_embs[topk_indices] * tanh(scores[topk_indices].view(-1, 1))

        # we need to transpose the output
        return out.t()


class EvolveGCNH(torch.nn.Module):
    r"""An implementation of the Evolving Graph Convolutional Hidden Layer.
    For details see this paper: `"EvolveGCN: Evolving Graph Convolutional
    Networks for Dynamic Graph." <https://arxiv.org/abs/1902.10191>`_

    Args:
        num_of_nodes (int): Number of vertices.
        in_channels (int): Number of filters.
        improved (bool, optional): If set to :obj:`True`, the layer computes
            :math:`\mathbf{\hat{A}}` as :math:`\mathbf{A} + 2\mathbf{I}`.
            (default: :obj:`False`)
        cached (bool, optional): If set to :obj:`True`, the layer will cache
            the computation of :math:`\mathbf{\hat{D}}^{-1/2} \mathbf{\hat{A}}
            \mathbf{\hat{D}}^{-1/2}` on first execution, and will use the
            cached version for further executions.
            This parameter should only be set to :obj:`True` in transductive
            learning scenarios. (default: :obj:`False`)
        normalize (bool, optional): Whether to add self-loops and apply
            symmetric normalization. (default: :obj:`True`)
        add_self_loops (bool, optional): If set to :obj:`False`, will not add
            self-loops to the input graph. (default: :obj:`True`)
    """

    def __init__(
        self,
        num_nodes: int,
        num_features: int,
        in_channels: int,
        num_layers_rnn: int,
        time_length: int,
        improved: bool = False,
        cached: bool = False,
        normalize: bool = True,
        add_self_loops: bool = True,
        use_edge_weight: bool = False,
        undirected: bool = True,
        neg_weight: float = 1.0,
        one_hot: bool = True,
        task_name: str = "link_pred",
    ):
        super(EvolveGCNH, self).__init__()
        self.num_nodes = num_nodes
        self.num_features = num_features
        self.in_channels = in_channels
        self.improved = improved
        self.cached = cached
        self.normalize = normalize
        self.add_self_loops = add_self_loops
        self.neg_weight = neg_weight
        self.one_hot = one_hot
        self.num_layers_rnn = num_layers_rnn
        self.use_edge_weight = (
            use_edge_weight if not undirected else False
        )  # if undirected, use_edge_weight is always False
        self.undirected = undirected
        self.task_name = task_name
        self.initial_weight = torch.nn.Parameter(torch.Tensor(in_channels, in_channels))
        if self.task_name == "link_pred":
            self.bceloss = BCEWithLogitsLoss()
        elif self.task_name == "node_reg":
            self.mseloss = MSELoss()
        else:
            raise ValueError("Task name not recognized")
        self._create_layers()
        self.reset_param(self.initial_weight)

    def set_device(self, device):
        self.device = device

    def reset_param(self, t):
        # Initialize based on the number of columns
        stdv = 1.0 / math.sqrt(t.size(1))
        t.data.uniform_(-stdv, stdv)

    def _create_layers(self):
        self.act = nn.RReLU()
        if self.one_hot:
            self.pool_layer = torch.nn.Linear(
                self.num_nodes, self.in_channels, bias=False
            )
        else:
            self.pool_layer = torch.nn.Linear(
                self.num_features, self.in_channels, bias=False
            )

        self.evolve_weights = mat_GRU_cell(self.in_channels, self.in_channels)

        self.conv_layer = GCNConv_Fixed_W(
            in_channels=self.in_channels,
            out_channels=self.in_channels,
            improved=self.improved,
            cached=self.cached,
            normalize=self.normalize,
            add_self_loops=self.add_self_loops,
        )
        if self.task_name == "node_reg":
            self.pred_reg = RegressionModel(self.in_channels, self.in_channels)

    def forward_snapshot(
        self,
        X: torch.FloatTensor,
        W: torch.FloatTensor,
        edge_index: torch.LongTensor,
        edge_weight: torch.FloatTensor = None,
    ) -> torch.FloatTensor:
        """
        Making a forward pass.

        Arg types:
            * **X** *(PyTorch Float Tensor)* - Node embedding.
            * **edge_index** *(PyTorch Long Tensor)* - Graph edge indices.
            * **edge_weight** *(PyTorch Float Tensor, optional)* - Edge weight vector.

        Return types:
            * **X** *(PyTorch Float Tensor)* - Output matrix for all nodes.
        """

        X_tilde = self.pool_layer(X)
        W = self.evolve_weights(W, X_tilde)
        if self.use_edge_weight:
            edge_weight = edge_weight.to(self.device)
            X = self.act(
                self.conv_layer(W.squeeze(), X_tilde.squeeze(), edge_index, edge_weight)
            )
        else:
            X = self.act(self.conv_layer(W.squeeze(), X_tilde.squeeze(), edge_index))
        return X, W

    def forward(self, graphs):
        output = []
        W = self.initial_weight
        for t in range(len(graphs)):
            edge_index = (
                to_undirected(graphs[t].edge_index).to(self.device)
                if self.undirected
                else graphs[t].edge_index.to(self.device)
            )
            X, W = self.forward_snapshot(
                graphs[t].x.to(self.device), W, edge_index, graphs[t].edge_weight
            )
            output.append(X)
        final_emb = torch.stack(output, dim=1)
        return final_emb

    def get_loss_node_pred(self, feed_dict, graphs):
        node, y, time = feed_dict.values()
        y = y.view(-1, 1)
        # run gnn
        final_emb = self.forward(graphs)  # [N, T, F]
        emb_node = final_emb[node, time, :]
        pred = self.pred_reg(emb_node)
        graphloss = self.mseloss(pred, y)
        return graphloss

    def get_loss_link_pred(self, feed_dict, graphs):
        node_1, node_2, node_2_negative, _, _, _, time = feed_dict.values()
        # run gnn
        final_emb = self.forward(graphs)  # [N, T, F]
        emb_source = final_emb[node_1, time, :]
        emb_pos = final_emb[node_2, time, :]
        emb_neg = final_emb[node_2_negative, time, :]
        pos_score = torch.sum(emb_source * emb_pos, dim=1)
        neg_score = torch.sum(emb_source * emb_neg, dim=1)
        pos_loss = self.bceloss(pos_score, torch.ones_like(pos_score))
        neg_loss = self.bceloss(neg_score, torch.zeros_like(neg_score))
        graphloss = pos_loss + self.neg_weight * neg_loss

        return graphloss, pos_score.detach().sigmoid(), neg_score.detach().sigmoid()

    def score_eval(self, feed_dict, graphs):
        with torch.no_grad():
            node_1, node_2, node_2_negative, _, _, _, time = feed_dict.values()
            # run gnn
            final_emb = self.forward(graphs)  # [N, T, F]
            # time-1 because we want to predict the next time step in eval
            emb_source = final_emb[node_1, time - 1, :]
            emb_pos = final_emb[node_2, time - 1, :]
            emb_neg = final_emb[node_2_negative, time - 1, :]
            pos_score = torch.sum(emb_source * emb_pos, dim=1)
            neg_score = torch.sum(emb_source * emb_neg, dim=1)
            return pos_score.sigmoid(), neg_score.sigmoid()

    def score_eval_node_reg(self, feed_dict, graphs):
        node, y, time = feed_dict.values()
        y = y.view(-1, 1)
        # run gnn
        final_emb = self.forward(graphs)  # [N, T, F]
        emb_node = final_emb[node, time - 1, :]
        pred = self.pred_reg(emb_node)

        return pred.squeeze()
