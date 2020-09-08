"""Define the graph neural network
"""

from __future__ import absolute_import
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from dataset import NUM_ATTACK_TYPES


class TCPNet(torch.nn.Module):
    """Simple neural network for the TCP network
    """

    def __init__(self):
        super().__init__()
        self.conv1 = GCNConv(1, 20)
        self.conv2 = GCNConv(20, NUM_ATTACK_TYPES)

    def forward(self, data):
        """Forward propagation

        Args:
            data : input data
        """
        out, edge_index = data.x, data.edge_index
        out = self.conv1(out, edge_index)
        out = F.relu(out)
        out = F.dropout(out, training=self.training)
        out = self.conv2(out, edge_index)

        return out
