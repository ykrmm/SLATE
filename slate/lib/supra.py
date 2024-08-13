import torch 
import numpy as np 
from torch_geometric.utils import to_undirected




def graphs_to_supra(graphs,
                    num_nodes,
                    add_time_connection=False,
                    remove_isolated=True,
                    add_vn=False,
                    p: float = 1.0):
    """
    Args:
        graphs (List[Data]): A list of graphs
        num_nodes (int): The number of nodes in each graph
        add_time_connection (bool, optional): Whether to add self time connections between nodes
        add_vn : add virtual node
        p: weight of the temporal connections (not explored yet but interesting to do (see NaturePaper))
    """

    num_graphs = len(graphs)
    edge_index = []
    edge_weight = []
    # Supra graph creation

    for i in range(num_graphs): 
        ei = graphs[i].edge_index + i * num_nodes # IMPORTANT: We considere nodes in different snapshots as different nodes
        ew = graphs[i].edge_weight
        
        if add_vn:
            id_vn = num_nodes*num_graphs + i # Assign an id to the virtual node
            nodes_snapshot = torch.unique(ei) # Get the connected nodes in the snapshot
            # Add connections between the virtual node and the nodes (deg > 0) in the snapshot
            # We do not connect the virtual node to isolated nodes
            vn_connections = torch.cat((torch.tensor([id_vn]*len(nodes_snapshot)).\
                view(1,-1),nodes_snapshot.view(1,-1)),dim=0).to(graphs[0].edge_index.device)
            ei = torch.cat((ei,vn_connections),dim=1)
            ew = torch.cat((ew,torch.ones(len(nodes_snapshot))))
            
        if add_time_connection: 
            # Add temporal connections between identical nodes in different snapshots
            if i < num_graphs - 1:
                ei_i = graphs[i].edge_index 
                ei_next = graphs[i+1].edge_index 
                nodes_snapshot = torch.unique(ei_i.view(-1))
                nodes_snapshot_next = torch.unique(ei_next.view(-1))
                # Intersection 
                common_nodes = torch.LongTensor(np.intersect1d(nodes_snapshot,nodes_snapshot_next))
                # Add temporal connections
                src = common_nodes + i*num_nodes
                dst = common_nodes + (i+1)*num_nodes
                time_co = torch.vstack((src,dst)).to(graphs[0].edge_index.device)
                ei = torch.cat((ei,time_co),dim=1)
                ew = torch.cat((ew,p*torch.ones(len(common_nodes))))
                
                
        edge_index.append(ei)
        edge_weight.append(ew)
    edge_index = torch.cat(edge_index,dim=1).to(graphs[0].edge_index.device)
    edge_weight = torch.cat(edge_weight).to(graphs[0].edge_index.device)

    # Now we have to create a mask to remove the isolated nodes
    total_nodes = num_nodes * num_graphs + num_graphs if add_vn else num_nodes * num_graphs
    if remove_isolated:
        mask = torch.zeros(total_nodes, dtype=torch.bool, device=edge_index.device)
        mask[torch.unique(edge_index)] = 1
    else:
        mask = torch.ones(total_nodes, dtype=torch.bool, device=edge_index.device)
        
    # Make the graph undirected
    edge_index, edge_weight = to_undirected(edge_index,edge_weight)
    # in edge_weight i want to have a max value of 1. 
    edge_weight[edge_weight > 1] = 1 # Keep 1 everywhere except for the temporal connections
    # We return the edge_index, edge_weight, and the mask list of isolated nodes
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