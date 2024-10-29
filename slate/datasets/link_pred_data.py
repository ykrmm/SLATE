from os.path import join
import random
from typing import Optional, Callable

import numpy as np
import torch
from torch.utils.data import Dataset

from slate.datasets import process_data as ld
from slate.datasets import Discrete_graph


class ListLinkPredDataset(Dataset):
    def __init__(
        self,
        graphs: list[Discrete_graph],
        node_train: torch.LongTensor,
        ldg: list[torch.LongTensor],
        t_max: int,
        sampling: str = "random",
        node_history: torch.LongTensor = None,
        pred_next: bool = False,
    ) -> None:
        super().__init__()
        self.graphs = graphs
        self.node_train = node_train
        self.ldg = ldg
        self.sampling = sampling
        self.node_history = node_history
        self.pred_next = pred_next
        self.t_max = t_max

    def get_dataset_t(self, t):
        """
        Return the dataset at time t
        """
        assert t < self.t_max, "t must be smaller than t_max"
        dg_at_t = self.ldg[t]
        edge_index = dg_at_t[:, :2].T
        weights = dg_at_t[:, 2]
        time = torch.ones(len(weights), dtype=torch.long) * t
        if self.pred_next:
            graphs = self.graphs[:t]
        else:
            graphs = self.graphs[: t + 1]

        dts = LinkPredDataset(
            edge_index,
            weights,
            time,
            graphs,
            self.node_train,
            sampling=self.sampling,
            node_history=self.node_history,
        )

        return dts


class LinkPredDataset(Dataset):
    """
    Class giving the Dataset of a temporal graph for training set and test set
    Edge index is the complete list of edges in the set
    x is the list of features (one-hot or not)
    Neg_edge is the list of negative edges at time t with the appropriate sampling strategies
    """

    def __init__(
        self,
        edge_index: torch.LongTensor,
        weights: torch.FloatTensor,
        time: torch.LongTensor,
        graphs: list[Discrete_graph],
        node_train: torch.LongTensor,
        sampling: str = "random",
        node_history: torch.LongTensor = None,
    ) -> None:
        self.graphs = graphs
        self.edge_index = edge_index
        self.edge_weight = weights
        self.time = time
        self.sampling = sampling
        self.node_history = node_history
        self.node_train = node_train

    def __len__(self):
        return len(self.edge_index[0])

    def __getitem__(self, idx: int):
        """
        Return the graph at time t no matter if evolving or not
        """
        t = self.time[idx]
        src_node = self.edge_index[0][idx]
        connected_nodes = (
            self.edge_index[1][self.edge_index[0] == src_node].tolist()
            + self.edge_index[0][self.edge_index[1] == src_node].tolist()
        )
        if self.sampling == "random":
            assert len(np.unique(connected_nodes)) != len(
                self.node_train
            ), "All nodes are connected to the source node"
            while True:
                neg_edge = random.choice(self.node_train)
                if neg_edge.item() not in connected_nodes:
                    break
        elif self.sampling == "historical":
            assert (
                self.node_history is not None
            ), "node_history must be given for historical sampling"
            if (
                len(set(connected_nodes))
                >= len(set(self.node_history[src_node.item()]))
                or len(set(self.node_history[src_node.item()])) == 0
            ):
                """
                    If the number of connected nodes is greater or equal than the number of nodes in the history 
                    of the source node we sample random nodes. 
                """
                while True:
                    neg_edge = np.random.choice(self.node_train)
                    if neg_edge.item() not in connected_nodes:
                        break
            else:
                while True:
                    neg_edge = np.random.choice(self.node_history[src_node.item()])
                    if neg_edge.item() not in connected_nodes:
                        break
        elif self.sampling == "inductive":
            """
                Inductive sampling of negative edges.
                Negative edge are sampled from all possible edges except the edge in the history of the source node
            """
            assert (
                self.node_history is not None
            ), "node_history must be given for inductive sampling"
            while True:
                neg_edge = np.random.choice(self.node_train)
                if (
                    neg_edge.item() not in connected_nodes
                    and neg_edge.item() not in self.node_history[src_node.item()]
                ):
                    break
        else:
            raise ValueError("Sampling must be random, historical or inductive")

        if t == 0:
            score_pos = torch.tensor(-1.0)
            score_neg = torch.tensor(-1.0)
        else:
            try:
                score_pos = self.graphs[t - 1].score_mat[
                    src_node, self.edge_index[1][idx]
                ]
                score_neg = self.graphs[t - 1].score_mat[src_node, neg_edge]
            except Exception:
                # TODO: Unused in SLATE finally
                score_pos = torch.tensor(-1.0)
                score_neg = torch.tensor(-1.0)

        feed_dict = {
            "src_nodes": src_node,
            "pos_nodes": self.edge_index[1][idx],
            "neg_nodes": neg_edge,
            "weights": self.edge_weight[idx],
            "score_pos": score_pos,
            "score_neg": score_neg,
            "time": self.time[idx],
        }
        return feed_dict

    def get_graphs(self):
        return self.graphs


class LinkPredData:
    """
    Link Pred data return a dataset (Discrete_graph_link_pred) for each timestamp
    datadir: log dir of DPPIN/TBE directory
    dataname: name of the DPPIN/TBE dataset to load
    dgb: If True, the dataset loaded is from DGB benchmark and not DPPIN.
    transform: Transformation functions for the dynamic graph
    mode: 'train' or 'test'
    evolving: If False, at a time t , considers that the previous links have disappeared
    n_nodes : number of nodes in the dataset
    one_hot: If True, features are one-hot of the adjacency matrix
    sampling: 'random' or 'historical'. If 'historical', negative edges are sampled from the previous timestamps
    split: 'last'. If last, the last timestamp will be for test and the others for train
    """

    def __init__(
        self,
        datadir: str,
        dataname: str,
        dgb: bool = False,
        transform: Optional[Callable] = None,
        evolving: bool = True,
        score: str = "none",
        n_nodes: int = None,
        one_hot: bool = True,
        sampling: str = "random",
        train_ratio: float = 0.7,
        split: str = "lastk",
        k_test: int = 1,
        k_val: int = 1,
        pred_next: bool = False,
    ):
        self.n_nodes = n_nodes
        self.evolving = evolving
        self.score = score
        self.one_hot = one_hot
        self.sampling = sampling
        self.split = split
        self.k_test = k_test
        self.k_val = k_val
        self.pred_next = pred_next
        self.train_ratio = train_ratio
        if dgb:
            self.dynamic_graph, self.weights = ld.read_dynamic_dgb(
                join(datadir, dataname, "ml_" + str(dataname) + ".csv")
            )
        else:
            self.dynamic_graph, self.weights = ld.read_dynamic(
                join(datadir, dataname, "Dynamic_PPIN.txt")
            )

        self.T = (
            int(max(self.dynamic_graph[:, 2])) + 1
        )  # Number of timestamps in the dynamic graphs (start at 0)
        if dgb:
            self.dynamic_features = torch.eye(
                self.n_nodes
            )  # One-hot features (no features in dgb dataset)
        else:
            self.dynamic_features = ld.read_features(
                join(datadir, dataname, "Node_Features.txt"), self.T
            )
        self.split_train_test()

        assert self.sampling.lower() in [
            "random",
            "historical",
            "inductive",
        ], "Sampling must be historical, random or inductive"
        assert self.split in ["lastk", "temporal"], "Split must be lastk or temporal"
        assert self.score in [
            "none",
            "common_neighbors",
            "jaccard",
            "adamic",
            "preferential",
        ], "Score must be none, common_neighbors, jaccard, adamic or preferential"
        self.history = self.sampling.lower() in [
            "historical",
            "inductive",
        ]  # If True, construct a dictionnary of users for historical sampling
        if self.history:
            self.node_history = dict()
        else:
            self.node_history = None
        if transform is not None:
            self.dynamic_graph = transform(self.dynamic_graph)

    def split_train_test(self):
        """
        Split the dynamic graph into train and test set
        Deleting edges in the test set that evolving nodes that are not in the train set
        Updating self.weights and self.dynamic_graph
        """
        if self.split == "lastk":
            self.T_train = self.T - self.k_test - self.k_val
            self.T_val = self.T - self.k_test
            self.T_test = self.T
        else:
            self.T_train = int(self.T * self.train_ratio)
            self.T_val = int(self.T * (self.train_ratio + (1 - self.train_ratio) / 2))
            self.T_test = self.T

        dynamic_graph_train = self.dynamic_graph[
            self.dynamic_graph[:, 2] < self.T_train
        ]
        dynamic_graph_val = self.dynamic_graph[
            (self.dynamic_graph[:, 2] >= self.T_train)
            & (self.dynamic_graph[:, 2] < self.T_val)
        ]
        dynamic_graph_test = self.dynamic_graph[self.dynamic_graph[:, 2] >= self.T_val]
        self.edge_index_train = dynamic_graph_train[:, :2].T
        self.weights_train = self.weights[self.dynamic_graph[:, 2] < self.T_train]
        self.time_train = dynamic_graph_train[:, 2]
        self.node_train = torch.unique(self.edge_index_train).long()
        weights_val = self.weights[
            (self.dynamic_graph[:, 2] >= self.T_train)
            & (self.dynamic_graph[:, 2] < self.T_val)
        ]
        weights_test = self.weights[self.dynamic_graph[:, 2] >= self.T_val]

        # delete nodes in dynamic_graph_test that are not in node_train and update weights
        filtered_dynamic_graph_test = dynamic_graph_test[
            np.isin(dynamic_graph_test[:, :2], self.node_train).all(axis=1)
        ]
        self.weights_test = weights_test[
            np.isin(dynamic_graph_test[:, :2], self.node_train).all(axis=1)
        ]
        self.edge_index_test = filtered_dynamic_graph_test[:, :2].T
        self.time_test = filtered_dynamic_graph_test[:, 2]

        # delete nodes in dynamic_graph_val that are not in node_train and update weights
        filtered_dynamic_graph_val = dynamic_graph_val[
            np.isin(dynamic_graph_val[:, :2], self.node_train).all(axis=1)
        ]
        self.weights_val = weights_val[
            np.isin(dynamic_graph_val[:, :2], self.node_train).all(axis=1)
        ]
        self.edge_index_val = filtered_dynamic_graph_val[:, :2].T
        self.time_val = filtered_dynamic_graph_val[:, 2]

        # New dynamic graph with all the snapshots and no nodes that are not in node_train
        self.dynamic_graph = torch.LongTensor(
            np.concatenate(
                (
                    dynamic_graph_train,
                    filtered_dynamic_graph_val,
                    filtered_dynamic_graph_test,
                )
            )
        )
        self.weights = torch.FloatTensor(
            np.concatenate((self.weights_train, self.weights_val, self.weights_test))
        )
        # assertion
        source_nodes = self.dynamic_graph[:, 0]
        target_nodes = self.dynamic_graph[:, 1]
        source_mask = torch.isin(source_nodes, self.node_train)
        target_mask = torch.isin(target_nodes, self.node_train)
        assert (
            source_mask.all() and target_mask.all()
        ), "Some nodes in the test set are not in the train set"
        assert not self.evolving or bool(
            torch.all(self.dynamic_graph[:, 2][:-1] <= self.dynamic_graph[:, 2][1:])
        ), "dynamic_graph must be sorted by time"
        assert (
            self.dynamic_graph.shape[0] == self.weights.shape[0]
        ), "dynamic_graph and weights must have the same length"
        assert (
            self.edge_index_train.shape[1]
            == self.weights_train.shape[0]
            == self.time_train.shape[0]
        ), "edge_index_train, time_train and weights_train must have the same length"
        assert (
            self.edge_index_test.shape[1]
            == self.weights_test.shape[0]
            == self.time_test.shape[0]
        ), "edge_index_test, time_test and weights_test must have the same length"
        assert (
            self.edge_index_val.shape[1]
            == self.weights_val.shape[0]
            == self.time_val.shape[0]
        ), "edge_index_val , time_val and weights_val must have the same length"

    def construct_node_history(self, edge_index: torch.LongTensor):
        """
        Construct a dictionnary of users for historical sampling
        """
        unique_src_nodes = torch.unique(edge_index[0])
        unique_dest_nodes = torch.unique(edge_index[1])

        # Updating node_history
        for src_node in unique_src_nodes:
            connected_nodes = edge_index[1][edge_index[0] == src_node]
            if src_node.item() not in self.node_history:
                self.node_history[src_node.item()] = connected_nodes.tolist()
            else:
                self.node_history[src_node.item()].extend(connected_nodes.tolist())

        for dest_node in unique_dest_nodes:
            connected_nodes = edge_index[0][edge_index[1] == dest_node]
            if dest_node.item() not in self.node_history:
                self.node_history[dest_node.item()] = connected_nodes.tolist()
            else:
                self.node_history[dest_node.item()].extend(connected_nodes.tolist())

    def get_dynamic_graph_at_t(self, t: int):
        """
        Return the dynamic graph at time t.

        """
        dg_at_t = []
        weights_at_t = []
        indices = torch.nonzero(self.dynamic_graph[:, 2] == t).view(-1)
        if self.evolving:
            ind_max = max(indices)
            dg_at_t = self.dynamic_graph[: ind_max + 1]
            weights_at_t = self.weights[: ind_max + 1]
        else:
            dg_at_t = self.dynamic_graph[indices]
            weights_at_t = self.weights[indices]
        return dg_at_t, weights_at_t

    def get_features_at_t(self, t: int):
        return self.dynamic_features[:, :t]  # Get node features until time t

    def construct_graph_snapshot(self):
        """
        Construct a list of Discrete_graph objects, each object representing a graph at a time t
        This graphs are used for the link prediction model.
        """
        graphs = []
        list_dg = []  # list of dynamic graphs at time t no regarding if evolving or not
        if self.one_hot:
            x = torch.eye(self.n_nodes)
        else:
            x = self.get_features_at_t(self.T_train)

        for t in range(self.T):
            dg, weights = self.get_dynamic_graph_at_t(t)
            edge_index = dg[:, :2].T
            time = dg[:, 2]
            sc = None

            # Construct the Discrete graph object
            graphs.append(Discrete_graph(edge_index, weights, time, x, sc))
            list_dg.append(self.dynamic_graph[self.dynamic_graph[:, 2] == t])
        graphs_train = graphs[: self.T_train]
        graphs_val = graphs[: self.T_val]
        graphs_test = graphs[: self.T_test]
        assert (
            len(list_dg) == self.T
        ), "list_dg must have the same length as the number of snapshots"
        assert (
            len(graphs_train) == self.T_train
        ), "graphs_train must have the same length as the number of snapshots in the training set"
        assert (
            len(graphs_val) == self.T_val
        ), "graphs_val must have the same length as the number of snapshots in the validation set"
        assert (
            len(graphs_test) == self.T_test
        ), "graphs_test must have the same length as the number of snapshots in the test set"

        return graphs_train, graphs_val, graphs_test, list_dg

    def get_datasets(self, seed):
        """
        Return the list of 'mode' datasets (Discrete_graph_link_pred) for each timestamp
        """
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        graphs_train, graphs_val, graphs_test, list_dg = self.construct_graph_snapshot()
        if self.history:
            self.construct_node_history(self.edge_index_train)
            assert (
                len(self.node_history.keys()) == len(self.node_train)
            ), "The number of nodes in the node_history dictionnary is not equal to the number of nodes in the training set"

        train_datasets = ListLinkPredDataset(
            graphs_train,
            self.node_train,
            list_dg,
            t_max=self.T_train,
            sampling="random",
            node_history=self.node_history,
            pred_next=self.pred_next,
        )
        val_datasets = ListLinkPredDataset(
            graphs_val,
            self.node_train,
            list_dg,
            t_max=self.T_val,
            sampling=self.sampling,
            node_history=self.node_history,
            pred_next=True,
        )
        # Pred next = true for evaluation to avoid data leakage
        test_datasets = ListLinkPredDataset(
            graphs_test,
            self.node_train,
            list_dg,
            t_max=self.T_test,
            sampling=self.sampling,
            node_history=self.node_history,
            pred_next=True,
        )

        return train_datasets, val_datasets, test_datasets
