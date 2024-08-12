import torch 
from torch_geometric.nn import GCN,GAT,GIN, GAE, VGAE
from torch_geometric.nn import GCNConv
from torch_geometric.utils import to_undirected
from torch.nn.modules.loss import BCEWithLogitsLoss, MSELoss
from slate.models.reg_mlp import RegressionModel

class StaticGNN(torch.nn.Module):
    def __init__(self,
                 model_name: str,
                 num_nodes: int,
                 features_dim: int,
                 layers: int,
                 hidden_dim: int,
                 one_hot: bool = True,
                 num_classes: int=2,
                 undirected: bool = True,
                 task_name: str = 'link_pred',
                 *args,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.model_name = model_name.upper()
        self.num_nodes = num_nodes
        self.features_dim = features_dim
        self.num_classes = num_classes
        self.layers = layers
        self.hidden_dim = hidden_dim
        self.one_hot = one_hot
        self.undirected = undirected
        self.task_name = task_name

        if self.task_name == 'link_pred':
            self.bceloss = BCEWithLogitsLoss()
        elif self.task_name == 'node_reg':
            self.mseloss = MSELoss()
        else:
            raise ValueError("Task name not recognized")
        self.build_model()

    def build_model(self):
        out_channels = self.hidden_dim
        in_channels = self.num_nodes if self.one_hot else self.features_dim
        if self.one_hot: 
            self.x = torch.eye(self.num_nodes)
        if self.model_name == 'GCN':
            self.gnn = GCN(in_channels,\
                hidden_channels=self.hidden_dim,\
                    num_layers=self.layers,\
                        out_channels=out_channels)
        elif self.model_name == 'GAT':
            self.gnn = GAT(in_channels,\
                hidden_channels=self.hidden_dim,\
                    num_layers=self.layers,\
                        out_channels=out_channels)
        elif self.model_name == 'GIN':
            self.gnn = GIN(in_channels,\
                hidden_channels=self.hidden_dim,\
                    num_layers=self.layers,\
                        out_channels=out_channels)
        else:
            raise NotImplementedError("Model not implemented")
        
        if self.task_name == 'node_reg':
            self.pred_reg = RegressionModel(out_channels,out_channels)
        
    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, (GCN, GAT, GIN, GAE, VGAE)):
                m.reset_parameters()

    def set_device(self,device):
        self.device = device
        self.x = self.x.to(device)   

    def forward_snapshot(self,x,edge_index):
        return self.gnn(x.to(self.device),edge_index.to(self.device))
    
    def forward(self,graphs):
        output = []
        for t in range(len(graphs)):
            edge_index = to_undirected(graphs[t].edge_index) if self.undirected else graphs[t].edge_index
            x = graphs[t].x
            output.append(self.forward_snapshot(x,edge_index))
        return torch.stack(output, dim=1)

    def forward_node_classif(self, graphs, nodes):
        out = self.forward(graphs)
        return out[nodes]
    
    def get_loss_link_pred(self, feed_dict,graphs):
        node_1, node_2, node_2_negative, _, _, _, time  = feed_dict.values()
        # run gnn
        self.final_emb = self.forward(graphs) # [N,T, F]
        emb_source = self.final_emb[node_1,time,:]
        emb_pos  = self.final_emb[node_2,time,:]
        emb_neg = self.final_emb[node_2_negative,time,:]
        pos_score = torch.sum(emb_source*emb_pos, dim=1)
        neg_score = torch.sum(emb_source*emb_neg, dim=1)
        pos_loss = self.bceloss(pos_score, torch.ones_like(pos_score))
        neg_loss = self.bceloss(neg_score, torch.zeros_like(neg_score))
        graphloss = pos_loss + neg_loss
        return graphloss, pos_score.sigmoid(), neg_score.sigmoid()
    
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
            emb_source = final_emb[node_1,time-1,:]
            emb_pos  = final_emb[node_2 ,time-1,:]
            emb_neg = final_emb[node_2_negative ,time-1,:]
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
    
    def __repr__(self):
        return self.__class__.__name__
    

