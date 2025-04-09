import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, features_dim, num_classes):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(features_dim, 15),
            nn.ReLU(),
            nn.Linear(15, 1 if num_classes == 2 else num_classes),
        )

    def forward_node_classif(self, graphs, nodes):
        x = graphs[-1].x  # only using node features (x)
        output = self.layers(x)
        return output[nodes]
