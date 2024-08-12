import torch 

class EdgeBank(torch.nn.Module):
    """
    EdgeBank model for link prediction
    mode: 'tw' or 'infinity', if infinity then the model will consider all the history of a node
    """
    def __init__(self, mode='tw'):
        super(EdgeBank, self).__init__()
        self.lin = torch.nn.Linear(1,1) # avoid error for optimizer
        self.mode = mode
        assert self.mode in ['tw','infinity'], 'mode must be tw or infinity'
        
    def construct_node_history(self,graph_train,tw):
        self.node_history = {}
        t_train = len(graph_train)
        b = 0 if self.mode == 'infinity' else t_train-tw    
        for g in graph_train[b:]:
            ei = g.edge_index
            for i in range(ei.shape[1]):
                node_1 = ei[0,i].item()
                node_2 = ei[1,i].item()
                if node_1 not in self.node_history.keys():
                    self.node_history[node_1] = [node_2]
                else:
                    self.node_history[node_1].append(node_2)
                if node_2 not in self.node_history.keys():
                    self.node_history[node_2] = [node_1]
                else:
                    self.node_history[node_2].append(node_1)
        for key in self.node_history.keys():
            self.node_history[key] = list(set(self.node_history[key]))             
    def set_device(self,device):
        self.device = device

    def score_eval(self,feed_dict,graphs):
        node_1, node_2, node_2_negative, _, _, _, time  = feed_dict.values()
        pos_score = []
        neg_score = []
        for i in range(node_1.shape[0]):
            s = node_1[i].item()    
            p = node_2[i].item() 
            n = node_2_negative[i].item()
            try:
                if p in self.node_history[s]:
                    pos_score.append(1)
                else:
                    pos_score.append(0)
            except:
                pos_score.append(0) # if no history, then the score is 0 (case mode='tw')
            try:
                if n in self.node_history[s]:
                    neg_score.append(1)
                else:
                    neg_score.append(0)
            except:
                neg_score.append(0)
        pos_score = torch.Tensor(pos_score).to(self.device)
        neg_score = torch.Tensor(neg_score).to(self.device)
        return pos_score,neg_score

    def __repr__(self):
        return self.__class__.__name__