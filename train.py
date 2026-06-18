#!/usr/bin/env python3
"""Two-stage training for SpAArSIST on ASVspoof 5 Track 1.

Recipe (Firc et al., Section 4.2):
  Stage 1 - 10 epochs, frozen XLS-R, batch size 64.
  Stage 2 -  5 epochs, end-to-end fine-tuning, batch size 32.
Adam, lr 1e-4, softmax cross-entropy. The best checkpoint per stage is selected
by development-set EER and reloaded before the next stage.

Models are trained with the baseline backend at a fixed k_tr; magnitude scoring,
mean aggregation, and inference-time sparsity (k_inf) are applied at evaluation
(see eval.py), matching the paper's protocol.

Example:
    python train.py --node_scoring learned --stack_aggregation attention \\
        --pool_ratio_train 0.3 --augment --out runs/ast-03
"""

import argparse
import os

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm

import config
from data import ASVspoof5Dataset, collate_fn
from spaarsist import SpAArSISTModel
from spaarsist.metrics import equal_error_rate


def build_loader(dataset, batch_size, shuffle, num_workers, weighted=False):
    sampler = None
    if weighted:
        labels = dataset.get_labels()
        class_weights = 1.0 / np.bincount(labels)
        sample_weights = class_weights[labels]
        sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)
        shuffle = False
    return DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, sampler=sampler,
        num_workers=num_workers, collate_fn=collate_fn, pin_memory=True,
        persistent_workers=num_workers > 0,
    )


def run_epoch(model, loader, optimizer, device):
    model.train()
    losses = []
    for _, wf, label in tqdm(loader, desc="train"):
        wf, label = wf.to(device), label.to(device).long()
        optimizer.zero_grad()
        logits = model(wf)
        loss = F.cross_entropy(logits, label)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    return float(np.mean(losses))


@torch.no_grad()
def validate(model, loader, device):
    model.eval()
    labels, scores = [], []
    for _, wf, label in tqdm(loader, desc="dev"):
        wf = wf.to(device)
        _, probs = model(wf, return_prob=True)
        scores.extend(probs[:, 0].tolist())  # P(bona fide)
        labels.extend(label.tolist())
    eer, _ = equal_error_rate(np.array(labels), np.array(scores))
    return eer


def train_stage(model, train_loader, dev_loader, epochs, lr, device, ckpt_path, stage_name):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    best_eer = None
    for epoch in range(1, epochs + 1):
        loss = run_epoch(model, train_loader, optimizer, device)
        eer = validate(model, dev_loader, device)
        print(f"[{stage_name}] epoch {epoch}/{epochs}  train_loss={loss:.4f}  dev_EER={eer*100:.3f}%")
        if best_eer is None or eer < best_eer:
            best_eer = eer
            torch.save(model.state_dict(), ckpt_path)
            print(f"  -> new best (dev EER {eer*100:.3f}%), saved to {ckpt_path}")
    if os.path.exists(ckpt_path):
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
    return best_eer


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--node_scoring", choices=["learned", "magnitude"], default="learned",
                   help="Train with baseline learned scorer (default) or magnitude proxy.")
    p.add_argument("--stack_aggregation", choices=["attention", "mean"], default="attention")
    p.add_argument("--pool_ratio_train", type=float, default=0.3, help="k_tr")
    p.add_argument("--augment", action="store_true", help="Enable the augmentation suite.")
    p.add_argument("--out", default="runs/spaarsist", help="Checkpoint directory.")
    p.add_argument("--num_workers", type=int, default=config.TRAIN["num_workers"])
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.out, exist_ok=True)
    ckpt = os.path.join(args.out, "best_model.pt")

    cfg = config.DATASETS["asvspoof5"]
    train_set = ASVspoof5Dataset(cfg["root"], cfg["train_protocol"], "train",
                                 augment=args.augment, rir_root=config.RIR_ROOT)
    dev_set = ASVspoof5Dataset(cfg["root"], cfg["dev_protocol"], "dev")

    model = SpAArSISTModel(
        node_scoring=args.node_scoring,
        stack_aggregation=args.stack_aggregation,
        pool_ratio_train=args.pool_ratio_train,
        pool_ratio_infer=args.pool_ratio_train,  # matched sparsity during training
    ).to(device)

    t = config.TRAIN
    # Stage 1: frozen front-end.
    model.extractor.finetune = False
    s1_train = build_loader(train_set, t["stage1_batch_size"], True, args.num_workers, weighted=True)
    dev_loader = build_loader(dev_set, t["stage2_batch_size"], False, args.num_workers)
    print("\n=== Stage 1: frozen XLS-R ===")
    train_stage(model, s1_train, dev_loader, t["stage1_epochs"], t["lr"], device, ckpt, "stage1")

    # Stage 2: end-to-end fine-tuning.
    model.extractor.finetune = True
    s2_train = build_loader(train_set, t["stage2_batch_size"], True, args.num_workers, weighted=True)
    print("\n=== Stage 2: end-to-end fine-tuning ===")
    train_stage(model, s2_train, dev_loader, t["stage2_epochs"], t["lr"], device, ckpt, "stage2")

    print(f"\nDone. Best checkpoint: {ckpt}")


if __name__ == "__main__":
    main()
