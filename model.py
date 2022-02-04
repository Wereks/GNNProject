import torch
import torch.nn as nn
import torch.nn.functional as F


class GCNLayer(torch.nn.Module):
    def __init__(self, input_shape):
        super().__init__()
        nodes_shape, adj_shape = input_shape, (input_shape[0], input_shape[0])
        weights = torch.ones((nodes_shape[1], nodes_shape[1])).to(dtype=torch.float64)
        self.weights = nn.Parameter(weights)

    def forward(self, inputs):
        nodes, adj = inputs
        degree = torch.sum(adj, dim=-1)

        new_nodes = torch.einsum("bi,bij,bjk,kl->bil", 1 / degree, adj, nodes, self.weights)
        out = torch.nan_to_num(new_nodes, nan=0)

        return out, adj


class GCN(nn.Module):
    def __init__(self, input_shape, num_layers=1):
        super().__init__()

        #Too much layers will cause weird errors, probably bcs reaching the limit of float32
        self.GCNLayers = torch.nn.ModuleList([GCNLayer(input_shape) for _ in range(num_layers)])

        self.readout = torch.nn.Sequential(
            nn.Linear(input_shape[1], 16),
            nn.Tanh(),
            nn.Linear(16, 1)
        )

    def forward(self, data):
        x = data

        for gcn in self.GCNLayers:
            x = gcn(x)
        x = torch.mean(F.relu(x[0]), dim=1).to(dtype=torch.float32)
        x = self.readout(x)

        return x