from slate.lib.utils import g_to_device,feed_dict_to_device,snap_to_supra_adj,edge_index_to_adj_matrix,compute_score_linkpred
from slate.lib.logger import LOGGER
from slate.lib.encoding import AddSupraLaplacianPE,AddSupraRWPE
from slate.lib.supra import graphs_to_supra, reindex_edge_index