
import torch

import copy
from torch_geometric.utils import to_undirected

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


def snap_to_supra_adj(
    graphs,
    num_nodes,
    num_time_steps,
    undirected = True,
    use_edge_attr = False,
    direction = 'pf'):
    
        """
        Convert a list of snapshots to a spatio-temporal edge index (supra-adjacency matrix)
        Arguments:
            graphs : List of torch_geometric.data.Data
            num_nodes : int
            num_time_steps : int total timestepts in the dataset (train+val+test)
            undirected : bool, whether to use undirected graph
            use_edge_attr : bool, whether to use edge attributes
            direction : str, 'pf' for past to future, 'fp' for future to past, undirected otherwise
        Returns:
            x : torch.Tensor, one-hot encoding of nodes 
            edge_index : torch.Tensor, supra-adjacency edge index
        """
        
        edge_index = []
        edge_attr = []
        x = []
        total_nodes = num_nodes * num_time_steps
        x = torch.eye(num_nodes) # one-hot encoding of temporal nodes
        x = x.repeat(num_time_steps,1)
        for i,t in enumerate(range(len(graphs))):
            ei = graphs[t].edge_index.cpu().clone()
            ei = to_undirected(ei) if undirected else ei
            ei = ei + (i * num_nodes)
            ea = torch.zeros(ei.shape[1])
            if i != len(graphs)-1:
                n1 = torch.arange(start=i*num_nodes,end=(i+1)*num_nodes) 
                n2=  torch.arange(start=(i+1)*num_nodes,end=(i+2)*num_nodes)
                
                # Adding temporal edges
                if direction.lower() == 'pf':
                    st_connection = torch.stack((n1,n2))
                elif direction.lower() == 'fp':
                    st_connection = torch.stack((n2,n1))
                else:
                    st_connection = torch.stack((n1,n2))
                    st_connection = to_undirected(st_connection)
                st_attr = torch.ones(st_connection.shape[1])
                ei = torch.cat((ei,st_connection),dim=-1)
                ea = torch.cat((ea,st_attr),dim=-1)
            edge_index.append(ei)
            edge_attr.append(ea)
            
        # Construct edge index and edge attributes for the supra-adjacency matrix
        edge_index = torch.hstack(edge_index)
        edge_attr = torch.hstack(edge_attr).long() if use_edge_attr else None
        return x[:len(graphs)*num_nodes], edge_index, edge_attr
    
    
def compute_score_linkpred(edge_index: torch.Tensor, score: str, n_nodes: int):
    """
    Compute the score of the predicted edges
    Arguments:
        edge_index : torch.Tensor, edge index of the predicted edges
        score : str, score function to use
        n_nodes : int, number of nodes in the graph
    """
    
    with torch.no_grad():
        adj = torch.zeros((n_nodes,n_nodes))
        adj[edge_index[0],edge_index[1]] = 1
        adj[edge_index[1],edge_index[0]] = 1
        
        if score == 'common_neighbors':
            sc = torch.mm(adj,adj)
            sc.fill_diagonal_(0)
            return sc
        
        elif score == 'jaccard':
            eps = 1e-3
            sc = torch.mm(adj,adj)
            degrees = torch.sum(adj,dim=1)
            den = (degrees.unsqueeze(1) + degrees.unsqueeze(0) - sc)
            sc = sc / (den + eps)
            sc.fill_diagonal_(0)            
            return sc
            
        elif score == 'adamic':
            eps = 1e-3
            sc = torch.mm(adj,adj)
            degrees = torch.sum(adj,dim=1)
            den = torch.log(degrees.unsqueeze(1)) + torch.log(degrees.unsqueeze(0)) - torch.log(sc + 1)
            sc = sc / (den  + eps)
            sc.fill_diagonal_(0)
            return sc
        
        elif score == 'preferential':
            degrees = torch.sum(adj,dim=1)
            sc = degrees.unsqueeze(1) * degrees.unsqueeze(0)
            sc.fill_diagonal_(0)
            return sc
        
        else:
            raise NotImplementedError
        