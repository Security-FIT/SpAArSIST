"""Batch collation. Waveforms are standardised to a fixed length of 64,600
samples (~4 s at 16 kHz) by random crop (train) or centre crop / zero-pad."""

import numpy as np
import torch

TARGET_NUM_SAMPLES = 64600  # ~4 s at 16 kHz


def _crop_or_pad(waveform: torch.Tensor, target_len: int, deterministic_crop: bool) -> torch.Tensor:
    length = waveform.size(1)
    if length == target_len:
        return waveform
    if length > target_len:
        max_offset = length - target_len
        start = max_offset // 2 if deterministic_crop else int(torch.randint(0, max_offset + 1, (1,)).item())
        return waveform[:, start: start + target_len]
    return torch.nn.functional.pad(waveform, (0, target_len - length))


def collate_fn(batch: list, deterministic_crop: bool = False):
    """Collate (file_name, waveform, label) tuples into fixed-length batches."""
    batch = [b for b in batch if b is not None]
    if len(batch) == 0:
        return [], torch.empty(0), torch.empty(0)

    batch_size = len(batch)
    file_names = []
    padded = torch.zeros(batch_size, TARGET_NUM_SAMPLES)
    labels = torch.zeros(batch_size)
    for i, (name, waveform, label) in enumerate(batch):
        file_names.append(name)
        padded[i] = _crop_or_pad(waveform, TARGET_NUM_SAMPLES, deterministic_crop).squeeze(0)
        try:
            labels[i] = torch.tensor(label)
        except Exception:
            labels[i] = np.nan
    return file_names, padded, labels
