import copy

import torch
import torch.nn as nn
import random
#from utils.utilities import fixed_unigram_candidate_sampler

class RandomModel(nn.Module):
    def __init__(self, num_features: int,
                 num_classes: int):
        super(RandomModel, self).__init__()
        self.num_classes = num_classes
        self.num_features = num_features
        self.linear = nn.Linear(num_features,num_classes) # Dummy parameters to avoid errors
                 
    
    def forward_node_classif(self, graphs,nodes):
        device = next(self.parameters()).device
        out = torch.rand((len(nodes),self.num_classes),device=device)
        return out
