"""File to load the network dataset
"""

from __future__ import absolute_import
import os
from pathlib import Path
import pandas as pd
import torch
from torch_geometric.data import Data, Dataset


TCPNETWORK_NUM_NODES = 23398

ATTACK_TYPES = [
    "apache2",
    "back",
    "dict",
    "guest",
    "httptunnel-e",
    "ignore",
    "ipsweep",
    "mailbomb",
    "mscan",
    "neptune",
    "neptunettl",
    "nmap",
    "pod",
    "portsweep",
    "processtable",
    "rootkit",
    "saint",
    "satan",
    "smurf",
    "smurfttl",
    "snmpgetattack",
    "snmpguess",
    "teardrop",
    "warez",
    "warezclient",
]

NUM_ATTACK_TYPES = len(ATTACK_TYPES)
ATTACK_DICT = dict(zip(ATTACK_TYPES, list(range(NUM_ATTACK_TYPES))))


class TCPNetworkTrainDataset(Dataset):
    """Train dataset class for loading the TCP networks
    """

    def __init__(self):
        root = os.path.join(Path(__file__).parent.parent.absolute(), "dataset")
        super().__init__(root, None, None, None)

    @property
    def raw_dir(self):
        return os.path.join(self.root, "raw", "train")

    @property
    def processed_dir(self):
        return os.path.join(self.root, "processed", "train")

    @property
    def raw_file_names(self):
        return os.listdir(self.raw_dir)

    @property
    def processed_file_names(self):
        return list(map(lambda x: f"{Path(x).stem}.pt", self.raw_file_names))

    def download(self):
        pass

    def process(self):
        for raw_path in self.raw_paths:
            dataframe = pd.read_csv(raw_path, sep="\t", header=None)

            edge_index = torch.from_numpy(dataframe.iloc[:, 0:2].values).long()
            node_features = torch.ones(TCPNETWORK_NUM_NODES, 1, dtype=torch.float)
            graph_feature = torch.zeros(NUM_ATTACK_TYPES, 1, dtype=torch.float)

            attack_list = list(set(dataframe.iloc[:, 4].values).difference({"-"}))
            for attack in attack_list:
                graph_feature[ATTACK_DICT[attack]] = 1

            data = Data(
                x=node_features, edge_index=edge_index.t().contiguous(), y=graph_feature
            )

            torch.save(
                data, os.path.join(self.processed_dir, f"{Path(raw_path).stem}.pt")
            )

    def len(self):
        return len(self.raw_file_names)

    def get(self, idx):
        data = torch.load(os.path.join(self.processed_dir, f"train_{idx:03d}.pt"))
        return data
