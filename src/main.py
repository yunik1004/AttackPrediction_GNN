"""Main file for the example project
"""

from __future__ import absolute_import
from torch_geometric.data import DataLoader
from dataset import TCPNetworkTrainDataset


if __name__ == "__main__":
    dataset = TCPNetworkTrainDataset()
    loader = DataLoader(dataset, batch_size=2, shuffle=True)

    for batch in loader:
        print(batch)
