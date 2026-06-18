#!/usr/bin/env python3

from typing import Literal
import pandas as pd
import torch
import torchaudio
import os
import numpy as np


class RIRDataset:
    """
    Class for RIR augmentation. Contains a pandas dataframe with the filepaths. Can be randomly sampled from.
    """

    def __init__(self, rir_root):
        self.rir_root = rir_root
        # Pointsource noises are the sounds and not the impulse responses
        pointsource_df = pd.read_csv(
            os.path.join(rir_root, "RIRS_NOISES", "pointsource_noises", "noise_list"), sep=" ", header=None
        ).iloc[:, -1]  # Get the last column that contains the filepaths
        isotropic_df = pd.read_csv(
            os.path.join(rir_root, "RIRS_NOISES", "real_rirs_isotropic_noises", "noise_list"), sep=" ", header=None
        ).iloc[:, -1]  # Get the last column that contains the filepaths
        rir_df = pd.read_csv(
            os.path.join(rir_root, "RIRS_NOISES", "real_rirs_isotropic_noises", "rir_list"), sep=" ", header=None
        ).iloc[:, -1]  # Get the last column that contains the filepaths

        # Remove RWCP from the isotropic noises (is not mono)
        self.df_rir = rir_df
        self.df_rir = self.df_rir[~self.df_rir.str.contains("RWCP")]
        self.df_noise = pd.concat([pointsource_df, isotropic_df], ignore_index=True)

    def __len__(self):
        return len(self.df_rir) + len(self.df_noise)

    def get_random_rir(self, which_augmentation: Literal["rir", "noise", None] = None):
        if which_augmentation is None:
            which_augmentation = "rir" if np.random.rand() < 0.5 else "noise"
        random_df = self.df_rir if which_augmentation == "rir" else self.df_noise
        path = os.path.join(self.rir_root, random_df.sample(1).iloc[0])
        # print(f"Loading {which_augmentation} from {path}.")
        try:
            rir, sr = torchaudio.load(path)
        except Exception as e:
            print(f"Failed to load RIR from {path}.")
            raise e
        if rir.size(0) > 1:
            rir = rir.mean(0)
        return rir, which_augmentation


class RIRAugmentations:
    """
    Class for RIR augmentations.
    """

    def __init__(
        self, rir_root: str, sample_rate: int = 16000, device="cuda" if torch.cuda.is_available() else "cpu"
    ):
        self.device = device
        self.sample_rate = sample_rate
        self.rir_dataset = RIRDataset(rir_root)

    def apply_rir(
        self,
        waveform: torch.Tensor,
        which_augmentation: Literal["rir", "noise", None] = None,
        scale_factor: float | torch.Tensor = 0.5,
    ) -> torch.Tensor:
        """
        Apply a random RIR to the audio waveform.

        param waveform: The audio waveform to apply the RIR to.
        param which_augmentation: The type of augmentation to apply. Can be "rir" or "noise".
        param scale_factor: The scale factor to apply to the RIR, should be between 0.2 and 0.8.

        return: The audio waveform with the RIR applied.
        """
        waveform = waveform.to(self.device)
        rir, which_augmentation = self.rir_dataset.get_random_rir(which_augmentation)
        rir = rir.to(self.device)
        if which_augmentation == "rir":
            rir = rir / torch.linalg.vector_norm(rir, ord=2)
            rir = rir.squeeze()
            T = waveform.shape[-1]
            wf = torchaudio.functional.fftconvolve(waveform, rir)
            wf = wf[..., :T]
        elif which_augmentation == "noise":
            rir = rir.squeeze()
            if len(rir) < len(waveform):
                rir = torch.cat([rir, torch.zeros(len(waveform) - len(rir), device=self.device)])
                # print("Zero-padded RIR to match waveform length. Waveform length:", waveform.shape, "RIR length:", rir.shape)
            else:
                rir = rir[:len(waveform)]
                # print("Trimmed RIR to match waveform length. Waveform length:", waveform.shape, "RIR length:", rir.shape)
            wf = torchaudio.functional.add_noise(waveform, rir, torch.tensor(scale_factor))
        wf = wf.squeeze()
        return wf
    


if __name__ == "__main__":
    rir_root = "/mnt/d/VUT/Deepfakes/Datasets/rirs_noises/"
    rir_augment = RIRAugmentations(rir_root)
    waveform, sr = torchaudio.load("fake.flac")
    waveform = waveform.squeeze()
    print(f"Before RIR: {waveform.shape}")
    augmented_waveform = rir_augment.apply_rir(waveform, which_augmentation="rir", scale_factor=0.8)
    print(f"After RIR: {augmented_waveform.shape}")
    # torchaudio.save(os.path.join("augmentation", "rir.wav"), augmented_waveform.unsqueeze(0).to("cpu"), sr)
    print(f"Before noise: {waveform.shape}")
    augmented_waveform = rir_augment.apply_rir(waveform, which_augmentation="noise", scale_factor=0.8)
    print(f"After noise: {augmented_waveform.shape}")
    # torchaudio.save(os.path.join("augmentation", "noise.wav"), augmented_waveform.unsqueeze(0).to("cpu"), sr)
