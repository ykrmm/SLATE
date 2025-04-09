import copy
import torch


def g_to_device(graphs, device):
    """
    Move a list of graphs snapshot to a device.
    """
    graphs_to_device = copy.deepcopy(graphs)
    for i in range(len(graphs)):
        graphs_to_device[i].edge_index = graphs[i].edge_index.to(device)
        graphs_to_device[i].x = graphs[i].x.to(device)
    return graphs_to_device


def feed_dict_to_device(feed_dict, device):
    """
    Move a feed_dict to a device.
    """
    feed_dict_to_device = copy.deepcopy(feed_dict)
    for key in feed_dict_to_device.keys():
        feed_dict_to_device[key] = feed_dict_to_device[key].to(device)
    return feed_dict_to_device


def edge_index_to_adj_matrix(edge_index, num_nodes):
    """
    Convert an edge index to an adjacency matrix.

    Parameters:
    edge_index (Tensor): [2, num_edges]
    num_nodes (int): Number of distinct nodes in the dynamic graphs.

    Returns:
    Tensor: Adjacency Matrix [num_nodes, num_nodes].
    """

    adj_matrix = torch.zeros((num_nodes, num_nodes), dtype=torch.float32)
    edge_index = edge_index.long()
    adj_matrix[edge_index[0], edge_index[1]] = 1

    return adj_matrix
