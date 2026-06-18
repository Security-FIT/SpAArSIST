"""End-to-end detector: XLS-R front-end + SpAArSIST backend + FF head.

The classification head matches the public XLS-R+AASIST baseline: two
projection blocks (Linear-BN-ReLU) followed by a 2-way output (bona fide /
spoof). Logit index 0 is bona fide, index 1 is spoof.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .frontend import XLSR_300M
from .backend import SpAArSIST


class SpAArSISTModel(nn.Module):
    def __init__(
        self,
        node_scoring: str = "magnitude",
        stack_aggregation: str = "mean",
        pool_ratio_train: float = 0.3,
        pool_ratio_infer: float | None = 0.1,
        finetune_frontend: bool = False,
        embedding_dim: int = 1024,
    ):
        super().__init__()
        self.extractor = XLSR_300M(finetune=finetune_frontend)
        self.feature_processor = SpAArSIST(
            inputs_dim=self.extractor.feature_size,
            outputs_dim=embedding_dim,
            node_scoring=node_scoring,
            stack_aggregation=stack_aggregation,
            pool_ratio_train=pool_ratio_train,
            pool_ratio_infer=pool_ratio_infer,
        )

        in_dim = embedding_dim
        self.classifier = nn.Sequential(
            nn.Linear(in_dim, in_dim // 2),
            nn.BatchNorm1d(in_dim // 2),
            nn.ReLU(),
            nn.Linear(in_dim // 2, in_dim // 4),
            nn.BatchNorm1d(in_dim // 4),
            nn.ReLU(),
            nn.Linear(in_dim // 4, 2),
        )

    def forward(self, waveforms, return_prob: bool = False):
        """Args:
            waveforms: (batch_size, seq_len) raw audio at 16 kHz.
        Returns:
            logits (B, 2), or (logits, probs) when ``return_prob`` is set.
            probs[:, 0] = P(bona fide), probs[:, 1] = P(spoof).
        """
        emb = self.extractor.extract_features(waveforms)
        emb = self.feature_processor(emb)
        logits = self.classifier(emb)
        if return_prob:
            return logits, F.softmax(logits, dim=1)
        return logits

    @property
    def inference_pool_ratio(self):
        return self.feature_processor.pool_S.k_infer
