from typing import Optional

from torch import nn

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


class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dims: Optional[list] = None):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [64, 64]
        hd = [input_dim, *hidden_dims, output_dim]
        layers = []
        for i in range(len(hd) - 1):
            layers.append(nn.Linear(hd[i], hd[i + 1]))
            layers.append(nn.ReLU())
        self.model = nn.Sequential(*layers)

    def forward(self, data):
        x = data.x
        return self.model(x)
