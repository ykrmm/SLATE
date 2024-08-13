from typing import Any, Tuple
import numpy as np 
from scipy.sparse.linalg import eigs, eigsh
import torch 
import torch.nn.functional as F
from torch import Tensor
from torch_sparse import SparseTensor
from torch_geometric.utils import (
    get_laplacian,
    get_self_loop_attr,
    to_scipy_sparse_matrix,
)


class AddSupraRWPE:
    def __init__(
        self,
        k: int,
        **kwargs: Any,
    ) -> None:
        
        self.walk_length = k
        self.kwargs = kwargs

    def __call__(self, edge_index, edge_weight, num_nodes) -> Tuple[torch.LongTensor, torch.FloatTensor]:
        num_nodes = num_nodes
        edge_index, edge_weight = edge_index, edge_weight

        adj = SparseTensor.from_edge_index(edge_index, edge_weight,
                                           sparse_sizes=(num_nodes, num_nodes))

        # Compute D^{-1} A:
        deg_inv = 1.0 / adj.sum(dim=1)
        deg_inv[deg_inv == float('inf')] = 0
        adj = adj * deg_inv.view(-1, 1)

        out = adj
        row, col, value = out.coo()
        pe_list = [get_self_loop_attr((row, col), value, num_nodes)]
        for _ in range(self.walk_length - 1):
            out = out @ adj
            row, col, value = out.coo()
            pe_list.append(get_self_loop_attr((row, col), value, num_nodes))
        pe = torch.stack(pe_list, dim=-1)
        return pe
    

class AddSupraLaplacianPE:
    r"""Supra Laplacian positional encoding from the eigenvectors of the Laplacian.
    

    Args:
        k (int): The number of non-trivial eigenvectors to consider.
        attr_name (str, optional): The attribute name of the data object to add
            positional encodings to. If set to :obj:`None`, will be
            concatenated to :obj:`data.x`.
            (default: :obj:`"laplacian_eigenvector_pe"`)
        is_undirected (bool, optional): If set to :obj:`True`, this transform
            expects undirected graphs as input, and can hence speed up the
            computation of eigenvectors. (default: :obj:`False`)
        **kwargs (optional): Additional arguments of
            :meth:`scipy.sparse.linalg.eigs` (when :attr:`is_undirected` is
            :obj:`False`) or :meth:`scipy.sparse.linalg.eigsh` (when
            :attr:`is_undirected` is :obj:`True`).
    """
    # Number of nodes from which to use sparse eigenvector computation:
    SPARSE_THRESHOLD: int = 100

    def __init__(
        self,
        k: int,
        is_undirected: bool = False,
        normalization: str = 'sym',
        add_eig_vals: bool = False,
        **kwargs: Any,
    ) -> None:
        
        self.k = k
        self.is_undirected = is_undirected
        self.kwargs = kwargs
        self.normalization = normalization
        self.add_eig_vals = add_eig_vals
        self.ncv = None
        assert self.normalization in [None,'sym', 'rw'], 'Invalid normalization'
    

    def __call__(self, edge_index, edge_weight, num_nodes) -> Tuple[torch.LongTensor, torch.FloatTensor]:
        """
        Args:
            torch.LongTensor : edge_index representing the supra-adjacency matrix
            torch.FloatTensor : edge_weight representing the supra-adjacency matrix
            int : num_nodes representing the number of nodes in the graph
        """
        if self.ncv is None:
            self.ncv = num_nodes if num_nodes < 500 else min(num_nodes, max(2*self.k + 1, 20))
            self.ncv = min(200, self.ncv)
            
        edge_index, edge_weight = get_laplacian(
            edge_index,
            edge_weight,
            normalization=self.normalization,
            num_nodes=num_nodes,
        )

        L = to_scipy_sparse_matrix(edge_index, edge_weight, num_nodes)

        
        eig_fn = eigs if not self.is_undirected else eigsh
        max_attempts = 10  # Nombre maximum de tentatives
        attempts = 0  # Compteur de tentatives

        while attempts < max_attempts:
            try:
                eig_vals, eig_vecs = eig_fn(  # type: ignore
                    L,
                    k=self.k+1,
                    which='SR' if not self.is_undirected else 'SA',
                    return_eigenvectors=True,
                    ncv=self.ncv,
                    **self.kwargs,
                )
                break  
            except Exception as e: 
                attempts += 1 
                self.ncv += 100  
                if attempts == max_attempts:
                    print("Decomposition failed after 10 attempts.")
                    raise  e
            
        eig_vecs = np.real(eig_vecs[:, eig_vals.argsort()])
        pe = torch.from_numpy(eig_vecs[:, 1:self.k + 1])
        sign = -1 + 2 * torch.randint(0, 2, (self.k, ))
        pe *= sign
        if self.add_eig_vals:
            eig_vals = torch.from_numpy(eig_vals).repeat(num_nodes, 1)
            pe = torch.cat([pe, eig_vals], dim=1)
        pe = pe.to(edge_index.device)
        pe = F.normalize(pe, p=2, dim=-1)
        return pe
    