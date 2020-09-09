"""Main file for the example project
"""

from __future__ import absolute_import
import torch
from torch_geometric.data import DataLoader
from dataset import TCPNetworkTrainDataset
from gnn import TCPNet


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    BATCH_TRAIN = 2

    model = TCPNet().to(device)

    train_dataset = TCPNetworkTrainDataset()
    train_loader = DataLoader(train_dataset, batch_size=BATCH_TRAIN, shuffle=True)

    for data in train_loader:
        data = data.to(device)
        out = model(data)

        break
