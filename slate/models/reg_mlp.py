import torch.nn as nn


# Define your regression model
class RegressionModel(nn.Module):
    def __init__(self, embedding_dim, hidden_dim):
        super(RegressionModel, self).__init__()
        self.fc1 = nn.Linear(embedding_dim, hidden_dim)
        self.fc2 = nn.Linear(
            hidden_dim, 1
        )  # Output layer with single neuron for regression

    def forward(self, x):
        x = nn.ReLU()(self.fc1(x))
        x = self.fc2(x)
        return x
