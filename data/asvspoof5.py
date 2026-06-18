"""ASVspoof 5 Track 1 dataset (in-domain train / dev / eval).

Expects the standard ASVspoof 5 layout under ``root_dir``:
    flac_T/ flac_D/ flac_E_eval/   and the space-separated protocol .tsv files.
Labels: 0 = bona fide, 1 = spoof.
"""

import os
from typing import Literal

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from torchaudio import load

from augmentation.Augment import Augmentor


class ASVspoof5Dataset(Dataset):
    def __init__(
        self,
        root_dir: str,
        protocol_file_name: str,
        variant: Literal["train", "dev", "eval"] = "train",
        augment: bool = False,
        rir_root: str = "",
    ):
        self.variant = variant
        self.augment = augment and variant == "train"
        if self.augment:
            self.augmentor = Augmentor(rir_root=rir_root)

        self.root_dir = root_dir
        self.protocol_df = pd.read_csv(os.path.join(root_dir, protocol_file_name), sep=" ", header=None)
        self.protocol_df.columns = [
            "SPEAKER_ID", "AUDIO_FILE_NAME", "GENDER", "CODEC", "CODEC_Q",
            "CODEC_SEED", "ATTACK_TAG", "ATTACK_LABEL", "KEY", "-",
        ]
        subdir = {"train": "flac_T", "dev": "flac_D", "eval": "flac_E_eval"}[variant]
        self.rec_dir = os.path.join(root_dir, subdir)

    def __len__(self):
        return len(self.protocol_df)

    def get_labels(self) -> np.ndarray:
        return self.protocol_df["KEY"].map({"bonafide": 0, "spoof": 1}).to_numpy()

    def get_class_weights(self) -> torch.FloatTensor:
        counts = np.bincount(self.get_labels())
        return torch.FloatTensor(1.0 / counts)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        name = self.protocol_df.loc[idx, "AUDIO_FILE_NAME"]
        path = os.path.join(self.rec_dir, f"{name}.flac")
        try:
            waveform, _ = load(path)
        except Exception as e:
            print(f"Error loading {path}: {e}")
            return None
        label = 0 if self.protocol_df.loc[idx, "KEY"] == "bonafide" else 1
        if self.augment:
            waveform = self.augmentor.augment(waveform)
        return name, waveform, label
