# SpAArSIST: Sparsified AASIST for Efficient and Reliable Anti-Spoofing

> **Supplementary material** for the Interspeech 2026 paper *"SpAArSIST:
> Sparsified AASIST for Efficient and Reliable Anti-Spoofing"* (Firc, Staněk, Lička, Malinka, Perešíni;
> Security@FIT; Brno University of Technology). This repository accompanies the
> publication and provides the reference implementation, training, and
> evaluation code.

Reference implementation for **SpAArSIST**, a deployment-oriented refinement of
the AASIST graph-pooling backend for SSL-based audio anti-spoofing.

SpAArSIST keeps the XLS-R + AASIST pipeline and isolates three explicit,
lightweight changes to the backend:

1. **Separate train/inference pooling ratios (`k_tr`, `k_inf`).** Graph
   operations scale with the number of retained nodes, so `k_inf < k_tr` prunes
   more aggressively at inference without retraining.
2. **Magnitude node scoring (`Mag`).** The learned GraphPool scorer
   `s_i = σ(wᵀnᵢ + b)` is replaced by the parameter-free proxy
   `s_i = ‖nᵢ‖₂²`; pooling becomes pure top-k selection by feature energy.
3. **Mean stack aggregation (`Mean`).** Motivated by the public AASIST running
   the stack-node attention at temperature τ = 100 (already near-uniform), the
   HS-GAL attention is replaced by an attention-free, mean-equivalent linear
   path: nodes are aggregated through the "without-attention" projection and the
   stack node by a linear projection of the incoming master. This is the
   configuration benchmarked for the reported backend size and MACs (it removes
   the node-node *and* master attention projections, not only the stack-node
   scorer).

The best overall configuration (**AST-03-01-Mag**: `k_tr=0.3`, `k_inf=0.1`,
magnitude scoring) cuts backend compute by 20.7% and improves In-the-Wild
robustness to 2.82% EER / 0.078 minDCF, while remaining competitive in-domain on
ASVspoof 5.

## Layout

```
spaarsist/
  backend.py     AASIST backend with the Mag / Mean / k_tr,k_inf toggles
  frontend.py    Wav2Vec2.0 XLS-R (300M) front-end
  model.py       full detector: front-end + backend + FF head
  metrics.py     EER, minDCF, actDCF, Cllr, ECE
data/            ASVspoof 5 and In-the-Wild dataset loaders + collation
augmentation/    training-time waveform augmentation suite
train.py         two-stage training (10 epochs frozen + 5 epochs end-to-end)
eval.py          evaluation with eval-time Mag / Mean / k_inf reconfiguration
tools/efficiency.py   backend parameter / MAC accounting
config.py        dataset paths and the training recipe
```

## Install

```bash
# Install PyTorch + torchaudio for your platform first: https://pytorch.org/
pip install -r requirements.txt
```

Set the dataset location (parent folder of `ASVspoof5/` and
`release_in_the_wild/`):

```bash
export SPAARSIST_DATA_DIR=/path/to/datasets
```

## Quick start

```python
import torch
from spaarsist import SpAArSISTModel

model = SpAArSISTModel(node_scoring="magnitude", stack_aggregation="mean",
                       pool_ratio_train=0.3, pool_ratio_infer=0.1).eval()
logits, probs = model(torch.randn(1, 64600), return_prob=True)
# probs[:, 0] = P(bona fide), probs[:, 1] = P(spoof)
```

## Train

Models are trained with the baseline backend at a fixed `k_tr`; the Mag / Mean /
`k_inf` choices are applied at evaluation (the paper's protocol).

```bash
python train.py --pool_ratio_train 0.3 --augment --out runs/ast-03
```

## Evaluate

Reproduce the rank-1 system **AST-03-01-Mag** from a `k_tr=0.3` checkpoint:

```bash
# In-domain ASVspoof 5
python eval.py --ckpt runs/ast-03/best_model.pt --dataset asvspoof5 \
    --node_scoring magnitude --pool_ratio_train 0.3 --pool_ratio_infer 0.1

# Out-of-domain In-the-Wild
python eval.py --ckpt runs/ast-03/best_model.pt --dataset inthewild \
    --node_scoring magnitude --pool_ratio_train 0.3 --pool_ratio_infer 0.1
```

`eval.py` writes `scores.csv` (`file_name, score_bonafide, label`) and
`metrics.json`. A baseline checkpoint loads into any backend configuration: the
magnitude proxy is parameter-free, and mean aggregation reuses existing
projections.

Backend size / compute accounting:

```bash
python tools/efficiency.py
```

## Conventions

- Labels: `0 = bona fide`, `1 = spoof`. Score reported is `P(bona fide)`.
- Metrics use the ASVspoof cost model: `p_spoof = 0.05`, `C_miss = 1`,
  `C_fa = 10`.
- Audio is standardised to 64,600 samples (~4 s at 16 kHz).

## Citation

This work builds directly on **AASIST**. If you use this code, please cite the
SpAArSIST paper (Interspeech 2026) **and** the original AASIST works:

```bibtex
@inproceedings{jung2022aasist,
  title     = {{AASIST}: Audio Anti-Spoofing Using Integrated Spectro-Temporal
               Graph Attention Networks},
  author    = {Jung, Jee-weon and Heo, Hee-Soo and Tak, Hemlata and
               Shim, Hye-jin and Chung, Joon Son and Lee, Bong-Jin and
               Yu, Ha-Jin and Evans, Nicholas},
  booktitle = {ICASSP 2022 - 2022 IEEE International Conference on Acoustics,
               Speech and Signal Processing (ICASSP)},
  pages     = {6367--6371},
  year      = {2022},
}

@inproceedings{tak2022automatic,
  title     = {Automatic Speaker Verification Spoofing and Deepfake Detection
               Using Wav2vec 2.0 and Data Augmentation},
  author    = {Tak, Hemlata and Todisco, Massimiliano and Wang, Xin and
               Jung, Jee-weon and Yamagishi, Junichi and Evans, Nicholas},
  booktitle = {The Speaker and Language Recognition Workshop (Odyssey 2022)},
  pages     = {112--119},
  year      = {2022},
}
```

## Acknowledgements & license

The graph-attention backbone is the original AASIST design by NAVER corp.
([Jung et al., ICASSP 2022](https://arxiv.org/pdf/2110.01200);
code: [TakHemlata/SSL_Anti-spoofing](https://github.com/TakHemlata/SSL_Anti-spoofing),
[clovaai/aasist](https://github.com/clovaai/aasist)), distributed under the MIT
License. This repository is released under the MIT License (see `LICENSE`).
