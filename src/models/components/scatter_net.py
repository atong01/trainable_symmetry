from torch import nn
from torch_geometric.nn import global_mean_pool

from src.models.scatter import Scatter


class ScatterSVM(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.scatter = Scatter(input_dim)
        self.linear = nn.Linear(self.scatter.out_shape(), output_dim)

    def forward(self, data):
        x = self.scatter(data)
        x = self.linear(x)
        return x


class LR(nn.Linear):
    def forward(self, data):
        x = data.x
        return super().forward(x)
