#!/usr/bin/env python3
"""Evaluate a SpAArSIST checkpoint and report ASVspoof metrics.

Following the paper's protocol, the backend can be reconfigured at evaluation
time independently of how it was trained:
  --node_scoring magnitude   replace the learned GraphPool scorer with ||n||^2
  --stack_aggregation mean    replace the stack-node attention with a mean
  --pool_ratio_infer K        inference-time sparsity k_inf (e.g. 0.1)

A baseline checkpoint loads cleanly into any of these configurations: the
magnitude proxy is parameter-free (the saved scorer weights are simply unused)
and mean aggregation reuses the existing master projections.

Examples:
    # Best overall system AST-03-01-Mag (k_tr=0.3 trained, k_inf=0.1, magnitude)
    python eval.py --ckpt runs/ast-03/best_model.pt --dataset asvspoof5 \\
        --node_scoring magnitude --pool_ratio_infer 0.1

    python eval.py --ckpt runs/ast-03/best_model.pt --dataset inthewild \\
        --node_scoring magnitude --pool_ratio_infer 0.1
"""

import argparse
import json
import os

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

import config
from data import ASVspoof5Dataset, InTheWildDataset, collate_fn
from spaarsist import SpAArSISTModel
from spaarsist.metrics import compute_all


def build_eval_loader(dataset_name, batch_size, num_workers):
    if dataset_name == "asvspoof5":
        cfg = config.DATASETS["asvspoof5"]
        ds = ASVspoof5Dataset(cfg["root"], cfg["eval_protocol"], "eval")
    elif dataset_name == "inthewild":
        cfg = config.DATASETS["inthewild"]
        ds = InTheWildDataset(cfg["root"], cfg["eval_protocol"], "eval")
    else:
        raise ValueError(dataset_name)
    return DataLoader(
        ds, batch_size=batch_size, shuffle=False, num_workers=num_workers,
        collate_fn=lambda b: collate_fn(b, deterministic_crop=True), pin_memory=True,
    )


@torch.no_grad()
def run(model, loader, device):
    model.eval()
    names, labels, scores = [], [], []
    for batch_names, wf, label in tqdm(loader, desc="eval"):
        wf = wf.to(device)
        _, probs = model(wf, return_prob=True)
        names.extend(batch_names)
        scores.extend(probs[:, 0].tolist())  # P(bona fide)
        labels.extend(label.tolist())
    return names, np.array(labels), np.array(scores)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt", required=True, help="Path to a trained checkpoint (state_dict).")
    p.add_argument("--dataset", choices=["asvspoof5", "inthewild"], default="asvspoof5")
    p.add_argument("--node_scoring", choices=["learned", "magnitude"], default="magnitude")
    p.add_argument("--stack_aggregation", choices=["attention", "mean"], default="attention")
    p.add_argument("--pool_ratio_train", type=float, default=0.3, help="k_tr the checkpoint was trained with.")
    p.add_argument("--pool_ratio_infer", type=float, default=0.1, help="k_inf used at inference.")
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--out", default=None, help="Directory for scores.csv / metrics.json (default: alongside ckpt).")
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    out_dir = args.out or os.path.join(os.path.dirname(args.ckpt) or ".", f"eval_{args.dataset}")
    os.makedirs(out_dir, exist_ok=True)

    model = SpAArSISTModel(
        node_scoring=args.node_scoring,
        stack_aggregation=args.stack_aggregation,
        pool_ratio_train=args.pool_ratio_train,
        pool_ratio_infer=args.pool_ratio_infer,
    ).to(device)

    state = torch.load(args.ckpt, map_location=device, weights_only=True)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if unexpected:
        print(f"Note: {len(unexpected)} checkpoint tensors unused in this config "
              f"(expected when switching to magnitude scoring): e.g. {unexpected[:3]}")
    if missing:
        print(f"WARNING: {len(missing)} model tensors not found in checkpoint: e.g. {missing[:3]}")

    loader = build_eval_loader(args.dataset, args.batch_size, args.num_workers)
    names, labels, scores = run(model, loader, device)

    scores_path = os.path.join(out_dir, "scores.csv")
    with open(scores_path, "w") as f:
        f.write("file_name,score_bonafide,label\n")
        for n, s, l in zip(names, scores, labels):
            f.write(f"{n},{s},{int(l)}\n")

    metrics = compute_all(labels, scores)
    metrics["config"] = {
        "dataset": args.dataset,
        "node_scoring": args.node_scoring,
        "stack_aggregation": args.stack_aggregation,
        "k_tr": args.pool_ratio_train,
        "k_inf": args.pool_ratio_infer,
    }
    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\n=== {args.dataset} | scoring={args.node_scoring} "
          f"agg={args.stack_aggregation} k_tr={args.pool_ratio_train} k_inf={args.pool_ratio_infer} ===")
    print(f"  EER:    {metrics['eer']*100:.2f}%")
    print(f"  minDCF: {metrics['min_dcf']:.3f}")
    print(f"  actDCF: {metrics['act_dcf']:.3f}   (gap {metrics['gap']:.3f})")
    print(f"  Cllr:   {metrics['cllr']:.3f}")
    print(f"  ECE:    {metrics['ece']:.3f}")
    print(f"Scores -> {scores_path}")


if __name__ == "__main__":
    main()
