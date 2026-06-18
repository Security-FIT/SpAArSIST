"""Wav2Vec2.0 XLS-R (300M) self-supervised front-end.

Produces frame-level representations that the SpAArSIST backend pools into an
utterance embedding. Weights come from torchaudio's bundled pipeline.
"""

import torch
import torch.nn as nn
from torchaudio.pipelines import WAV2VEC2_XLSR_300M


class XLSR_300M(nn.Module):
    def __init__(self, finetune: bool = False):
        super().__init__()
        self.model = WAV2VEC2_XLSR_300M.get_model()
        self.finetune = finetune
        self.transformer_layers = 24
        self.feature_size = 1024

    def extract_features(self, input_data):
        """Args:
            input_data: waveforms (batch_size, seq_len) at 16 kHz.

        Returns:
            Tensor (24, batch_size, time_frames, 1024) — one slice per
            transformer layer; the backend consumes the last layer.
        """
        with torch.set_grad_enabled(self.finetune):
            emb = self.model.extract_features(input_data)[0]
            return torch.stack(emb)
