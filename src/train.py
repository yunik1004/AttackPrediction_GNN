"""Main file for the example project
"""

from __future__ import absolute_import, division
import os
from pathlib import Path
import matplotlib.pyplot as plt
import torch
from torch_geometric.data import DataLoader
from dataset import TCPNetworkTrainDataset, TCPNetworkValidDataset
from gnn import TCPNet


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    HIDDEN_CHANNELS = 256
    BATCH_SIZE = 8
    NUM_EPOCHS = 1000
    LEARNING_RATE = 0.01
    VALID_FREQ = 100

    OUTPUT_DIR = os.path.join(Path(__file__).parent.parent.absolute(), "out")

    model = TCPNet(hidden_channels=HIDDEN_CHANNELS).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = torch.nn.BCEWithLogitsLoss()

    train_dataset = TCPNetworkTrainDataset()
    valid_dataset = TCPNetworkValidDataset()

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=1, shuffle=False)

    train_loss_list = list()
    valid_loss_list = list()

    for epoch in range(1, NUM_EPOCHS + 1):
        total_train_loss = 0

        model.train()
        for data in train_loader:
            data = data.to(device)
            out = model(data)

            loss = criterion(out, data.y)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            total_train_loss += loss.item()

        average_train_loss = total_train_loss / len(train_loader)
        print(f"Epoch {epoch}: train loss = {average_train_loss}")
        train_loss_list.append(average_train_loss)

        if epoch % VALID_FREQ == 0:
            total_valid_loss = 0

            model.eval()
            for data in valid_loader:
                data = data.to(device)
                out = model(data)

                loss = criterion(out, data.y)
                total_valid_loss += loss.item()

            average_valid_loss = total_valid_loss / len(valid_loader)
            print(f"valid loss = {average_valid_loss}")
            valid_loss_list.append(average_valid_loss)

            torch.save(model, os.path.join(OUTPUT_DIR, f"model_{epoch}.pth"))

    plt.plot(
        list(range(1, NUM_EPOCHS + 1)),
        train_loss_list,
        label="Train loss",
        color="blue",
    )
    plt.plot(
        list(map(lambda x: x * VALID_FREQ, list(range(1, len(valid_loss_list) + 1)))),
        valid_loss_list,
        label="Valid loss",
        color="red",
    )
    plt.title("Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(os.path.join(OUTPUT_DIR, "loss.svg"))
    plt.show()
