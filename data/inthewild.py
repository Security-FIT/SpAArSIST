"""In-the-Wild dataset (out-of-domain evaluation).

Expects ``meta.csv`` with columns ``file, speaker, label`` under ``root_dir``.
Labels: 0 = bona fide, 1 = spoof.
"""

import os

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from torchaudio import load


class InTheWildDataset(Dataset):
    def __init__(self, root_dir: str, protocol_file_name: str = "meta.csv", variant: str = "eval"):
        self.root_dir = root_dir
        self.protocol_df = pd.read_csv(os.path.join(root_dir, protocol_file_name))

    def __len__(self):
        return len(self.protocol_df)

    def get_labels(self) -> np.ndarray:
        return self.protocol_df["label"].map({"bona-fide": 0, "spoof": 1}).to_numpy()

    def get_class_weights(self) -> torch.FloatTensor:
        counts = np.bincount(self.get_labels())
        return torch.FloatTensor(1.0 / counts)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        name = str(self.protocol_df.loc[idx, "file"])
        waveform, _ = load(os.path.join(self.root_dir, name))
        label = 0 if self.protocol_df.loc[idx, "label"] == "bona-fide" else 1
        return name, waveform, label
