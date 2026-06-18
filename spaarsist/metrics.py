"""ASVspoof discrimination and calibration metrics.

Convention (matching the training pipeline): ``score = P(bona fide | x)`` from
the softmax, and ``label = 0`` for bona fide, ``1`` for spoof. Bona fide is the
target class (``pos_label=0``). Cost model follows ASVspoof: p_spoof = 0.05,
C_miss = 1, C_fa = 10.
"""

import numpy as np
from sklearn.metrics import det_curve


def _logit(scores, eps: float = 1e-6):
    scores = np.clip(scores, eps, 1.0 - eps)
    return np.log(scores / (1.0 - scores))


def equal_error_rate(labels, scores, pos_label: int = 0):
    """EER and its threshold; ``scores`` are P(bona fide)."""
    fpr, fnr, thresholds = det_curve(labels, scores, pos_label=pos_label)
    idx = np.nanargmin(np.abs(fnr - fpr))
    return (fpr[idx] + fnr[idx]) / 2, thresholds[idx]


def min_dcf(labels, scores, p_target: float = 0.95, c_miss: float = 1.0,
            c_fa: float = 10.0, pos_label: int = 0):
    far, frr, thresholds = det_curve(labels, scores, pos_label=pos_label)
    c_det = c_miss * frr * p_target + c_fa * far * (1.0 - p_target)
    idx = int(np.argmin(c_det))
    c_def = min(c_miss * p_target, c_fa * (1.0 - p_target))
    return c_det[idx] / c_def, thresholds[idx]


def act_dcf(labels, scores, p_target: float = 0.95, c_miss: float = 1.0,
            c_fa: float = 10.0, pos_label: int = 0):
    """Actual DCF at the cost-only Bayes threshold in posterior log-odds space."""
    z = _logit(np.asarray(scores, dtype=float))  # logit(P(bona fide))
    threshold = np.log(c_fa / c_miss)
    labels = np.asarray(labels)
    target = labels == pos_label
    p_miss = np.mean(z[target] < threshold) if np.any(target) else np.nan
    p_fa = np.mean(z[~target] >= threshold) if np.any(~target) else np.nan
    c_det = c_miss * p_miss * p_target + c_fa * p_fa * (1.0 - p_target)
    c_def = min(c_miss * p_target, c_fa * (1.0 - p_target))
    return c_det / c_def, threshold


def cllr(labels, scores, pos_label: int = 0):
    """Posterior cross-entropy in bits, averaged per class (Cllr proxy)."""
    llr = _logit(np.asarray(scores, dtype=float))
    labels = np.asarray(labels)
    tgt = llr[labels == pos_label]
    non = llr[labels != pos_label]
    c1 = np.mean(np.log1p(np.exp(-tgt))) if tgt.size else np.nan
    c2 = np.mean(np.log1p(np.exp(non))) if non.size else np.nan
    return 0.5 * (c1 + c2) / np.log(2)


def expected_calibration_error(labels, scores, n_bins: int = 15):
    """ECE on the spoof posterior P(spoof) = 1 - P(bona fide)."""
    p_spoof = 1.0 - np.asarray(scores, dtype=float)
    y_spoof = (np.asarray(labels) == 1).astype(float)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(p_spoof)
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (p_spoof > lo) & (p_spoof <= hi) if lo > 0 else (p_spoof >= lo) & (p_spoof <= hi)
        if not np.any(mask):
            continue
        conf = p_spoof[mask].mean()
        acc = y_spoof[mask].mean()
        ece += (mask.sum() / n) * abs(acc - conf)
    return ece


def compute_all(labels, scores, p_spoof: float = 0.05, c_miss: float = 1.0, c_fa: float = 10.0):
    """All paper metrics from labels (0 bona fide / 1 spoof) and P(bona fide) scores."""
    labels = np.asarray(labels, dtype=int)
    scores = np.asarray(scores, dtype=float)
    p_target = 1.0 - p_spoof
    eer, eer_t = equal_error_rate(labels, scores)
    mdcf, mdcf_t = min_dcf(labels, scores, p_target, c_miss, c_fa)
    adcf, adcf_t = act_dcf(labels, scores, p_target, c_miss, c_fa)
    return {
        "eer": float(eer),
        "eer_threshold": float(eer_t),
        "min_dcf": float(mdcf),
        "act_dcf": float(adcf),
        "gap": float(adcf - mdcf),
        "cllr": float(cllr(labels, scores)),
        "ece": float(expected_calibration_error(labels, scores)),
        "p_spoof": p_spoof,
        "c_miss": c_miss,
        "c_fa": c_fa,
    }
