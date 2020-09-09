"""Define the graph neural network
"""

from __future__ import absolute_import
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.nn import global_mean_pool
from dataset import NUM_ATTACK_TYPES


class TCPNet(torch.nn.Module):
    """Simple neural network for the TCP network
    """

    def __init__(self, hidden_channels=100):
        super().__init__()

        self.conv1 = GCNConv(1, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.conv3 = GCNConv(hidden_channels, hidden_channels)
        self.lin = torch.nn.Linear(hidden_channels, NUM_ATTACK_TYPES)

    def forward(self, data):
        """Forward propagation

        Args:
            data : input data
        """
        out, edge_index, batch = data.x, data.edge_index, data.batch

        out = self.conv1(out, edge_index)
        out = F.relu(out)
        out = self.conv2(out, edge_index)
        out = F.relu(out)
        out = self.conv3(out, edge_index)

        out = global_mean_pool(out, batch)
        out = self.lin(out)

        return out
