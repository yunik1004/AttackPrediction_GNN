"""Main file for the example project
"""

from __future__ import absolute_import, division
import matplotlib.pyplot as plt
import torch
from torch_geometric.data import DataLoader
from dataset import TCPNetworkTrainDataset
from gnn import TCPNet


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    HIDDEN_CHANNELS = 256
    BATCH_SIZE = 8
    NUM_EPOCHS = 100
    LEARNING_RATE = 0.01

    model = TCPNet(hidden_channels=HIDDEN_CHANNELS).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = torch.nn.BCEWithLogitsLoss()

    train_dataset = TCPNetworkTrainDataset()
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    loss_list = list()

    for epoch in range(1, NUM_EPOCHS + 1):
        total_loss = 0

        model.train()
        for data in train_loader:
            data = data.to(device)
            out = model(data)

            loss = criterion(out, data.y)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            total_loss += loss.item()

        average_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch}: loss = {average_loss}")
        loss_list.append(average_loss)

    plt.plot(list(range(1, NUM_EPOCHS + 1)), loss_list)
    plt.show()
