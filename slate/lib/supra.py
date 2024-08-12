import torch 
import numpy as np 
from torch_geometric.utils import to_undirected




def graphs_to_supra(graphs,
                    num_nodes,
                    add_time_connection=False,
                    remove_isolated=True,
                    add_vn=False,
                    p: float = 0.5):
    """
    Args:
        graphs (List[Data]): A list of graphs
        num_nodes (int): The number of nodes in each graph
        add_time_connection (bool, optional): Whether to add self time connections between nodes
        add_vn : add virtual node
    """
    # Construct supra adjacency
    num_graphs = len(graphs)
    edge_index = []
    edge_weight = []
    for i, graph in enumerate(graphs):
        edge_index.append(graph.edge_index + i * num_nodes)
        edge_weight.append(graph.edge_weight)
        # add a virtual node connected to all nodes in the snapshot 
        if add_vn: 
            vn_dst = torch.unique(graph.edge_index + i * num_nodes)
            vn_src = torch.ones_like(vn_dst) * (num_graphs * num_nodes + i)
            vn_co = torch.vstack((vn_src,vn_dst))
            vn_weight = torch.ones_like(vn_dst)
            edge_index.append(vn_co)
            edge_index.append(vn_weight)
            
            
    edge_index = torch.cat(edge_index, dim=1).to(graph.edge_index.device)
    edge_weight = torch.ones_like(edge_index[0], dtype=edge_weight[0].dtype, device=graph.edge_weight[0].device)

    # Compute mask for isolated nodes
    total_nodes = num_nodes * num_graphs + num_graphs if add_vn else num_nodes * num_graphs
    if remove_isolated:
        mask = torch.zeros(total_nodes, dtype=torch.bool, device=edge_index.device)
        mask[torch.unique(edge_index)] = 1
    else:
        mask = torch.ones(total_nodes, dtype=torch.bool, device=edge_index.device)
    
    # Add time connections
    if num_graphs > 1:
        if add_time_connection and remove_isolated:
            time_connection = []
            for i in range(num_graphs - 1):
                mask1 = mask[i * num_nodes:(i + 1) * num_nodes]
                n1 = torch.arange(start=i * num_nodes, end=(i + 1) * num_nodes)
                n1 = n1[mask1]
                n2 = n1 + num_nodes
                time_connection.append(torch.stack((n1, n2)))
            time_connection = torch.cat(time_connection, dim=1)
            weight_connection = torch.ones(time_connection.shape[1], dtype=edge_weight.dtype, device=edge_weight.device) * p
            edge_index = torch.cat([edge_index, time_connection], dim=1)
            mask[torch.unique(edge_index)] = 1
            edge_weight = torch.cat([edge_weight, weight_connection], dim=0) 
            
        elif add_time_connection and not remove_isolated:
            time_connection = []
            for i in range(num_graphs - 1):
                n1 = torch.arange(start=i * num_nodes, end=(i + 1) * num_nodes)
                n2 = n1 + num_nodes
                time_connection.append(torch.stack((n1, n2)))
            time_connection = torch.cat(time_connection, dim=1)
            weight_connection = torch.ones(time_connection.shape[1], dtype=edge_weight.dtype, device=edge_weight.device) * p
            edge_index = torch.cat([edge_index, time_connection], dim=1)
            edge_weight = torch.cat([edge_weight, weight_connection], dim=0)

    # undirected graphs necessary for the laplacian
    edge_index, edge_weight = to_undirected(edge_index, edge_weight)
    return edge_index, edge_weight, mask

def reindex_edge_index(edge_index):
    """
    Args:
        edge_index (Tensor): Edge index
        mask (Tensor): Mask
    """
    # Determine the number of unique nodes
    num_nodes = edge_index.max().item() + 1

    # Create a tensor representing the old indices
    old_indices = torch.arange(num_nodes)

    # Determine which nodes are present in the edge index
    unique_nodes = torch.unique(edge_index)

    # Create a mapping from old indices to new indices
    new_indices = torch.arange(unique_nodes.size(0))

    # Create the node mapping using PyTorch's indexing
    node_mapping = torch.zeros_like(old_indices)
    node_mapping[unique_nodes] = new_indices

    # Apply the node mapping to the edge index
    edge_index = node_mapping[edge_index]
    
    return edge_index