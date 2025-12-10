#!/usr/bin/env python3
"""
Train a simple heteroassociative linear MHN-style matrix on VideoMAE embeddings.

This is a compatibility rewrite of your DINOv2 CLS trainer to support the
.pt dict format produced by extract_embedding_denser.py.

Expected per-video .pt keys (from your extractor):
  - tubelet_embeddings:        (T, D)
  - clip_tubelet_embeddings:   (W, t_per_window, D)
  - clip_video_embeddings:     (W, D)
  - delta_tubelet_embeddings:  (T-1, D)  [optional]
  - video_embedding:           (D,)
  - metadata: dict

Recommended keys for training:
  - tubelet_embeddings (default): best for fine temporal heteroassociative learning
  - clip_video_embeddings: faster/coarser debugging

Usage:
  python train_mhn_videomae.py \
    --processed_root /path/to/processed_video/train \
    --key tubelet_embeddings \
    --mode plain \
    --stride 1

  # Fast smoke test:
  python train_mhn_videomae.py \
    --processed_root /path/to/processed_video/train \
    --key clip_video_embeddings \
    --mode plain \
    --stride 1
"""

import argparse
import json
from pathlib import Path
import time
import numpy as np
import torch


# ---------------------------
# Data discovery / loading
# ---------------------------

def list_videos(processed_root: Path):
    """
    Returns list of records with:
      {
        "video_id": str,
        "pt_path": Path
      }
    """
    processed_root = Path(processed_root)
    pt_files = sorted(processed_root.rglob("*.pt"))

    records = []
    for p in pt_files:
        records.append({
            "video_id": p.stem,
            "pt_path": p
        })
    return records


def load_tensor(record, key="tubelet_embeddings"):
    """
    Load a tensor from a VideoMAE .pt file.

    Returns numpy float32 array shaped [T, D].

    Handles:
      - dict with the requested key
      - direct Tensor saves (rare)
      - clip_tubelet_embeddings flattening (W, t, D) -> (W*t, D)
      - 1D vectors (D,) -> (1, D)
    """
    obj = torch.load(record["pt_path"], map_location="cpu")

    if isinstance(obj, torch.Tensor):
        x = obj
    elif isinstance(obj, dict):
        if key not in obj:
            raise KeyError(
                f"Key '{key}' not found in {record['pt_path']}.\n"
                f"Available keys: {list(obj.keys())}"
            )
        x = obj[key]
    else:
        raise TypeError(f"Unsupported .pt type: {type(obj)} in {record['pt_path']}")

    # Normalize expected shapes for training code
    if x.ndim == 3 and key == "clip_tubelet_embeddings":
        # (W, t, D) -> flatten to long sequence
        x = x.reshape(-1, x.shape[-1])

    if x.ndim == 1:
        # (D,) -> (1, D)
        x = x.unsqueeze(0)

    if x.ndim != 2:
        raise ValueError(
            f"Expected 2D tensor for key '{key}'. Got shape {tuple(x.shape)} "
            f"in {record['pt_path']}."
        )

    return x.float().cpu().numpy().astype(np.float32)


def try_load_metadata(record):
    """
    Best-effort metadata read for naming/debug.
    """
    try:
        obj = torch.load(record["pt_path"], map_location="cpu")
        if isinstance(obj, dict):
            return obj.get("metadata", {})
    except Exception:
        pass
    return {}


# ---------------------------
# Math utils
# ---------------------------

def l2_normalize(x, axis=-1, eps=1e-6):
    norm = np.linalg.norm(x, axis=axis, keepdims=True)
    return x / (norm + eps)


def build_video_level_split(records, val_ratio=0.1, seed=0):
    rng = np.random.default_rng(seed)
    idx = np.arange(len(records))
    rng.shuffle(idx)

    n_val = max(1, int(round(len(records) * val_ratio)))
    val_idx = set(idx[:n_val].tolist())

    train = [r for i, r in enumerate(records) if i not in val_idx]
    val = [r for i, r in enumerate(records) if i in val_idx]
    return train, val


# ---------------------------
# MHN training
# ---------------------------

def accumulate_mhn(records, stride=1, mode="plain", use_unit_norm=True, key="tubelet_embeddings"):
    """
    Memory-efficient accumulation:
      plain:  M = (1/N) * sum_t x_t ⊗ x_{t+stride}
      delta:  M = (1/N) * sum_t x_t ⊗ (x_{t+stride} - x_t)

    Returns:
      M, pair_count, dim
    """
    assert mode in ("plain", "delta")

    M = None
    pair_count = 0
    dim = None

    for rec in records:
        cls = load_tensor(rec, key=key)
        T = cls.shape[0]
        if T <= stride:
            continue

        X = cls[:-stride]
        Y = cls[stride:]

        if mode == "delta":
            Y = Y - X

        if use_unit_norm:
            X = l2_normalize(X)
            Y = l2_normalize(Y)

        if dim is None:
            dim = X.shape[1]
            M = np.zeros((dim, dim), dtype=np.float32)

        M += X.T @ Y
        pair_count += X.shape[0]

    if pair_count > 0:
        M /= float(pair_count)

    return M, pair_count, dim


# ---------------------------
# 1-step evaluation
# ---------------------------

def predict_one_step(M, Xn, mode="plain"):
    """
    Xn: [N, D] assumed normalized if use_unit_norm
    Returns Yhat normalized.
    """
    assert mode in ("plain", "delta")

    pred = (M.T @ Xn.T).T  # [N, D]

    if mode == "plain":
        Yhat = pred
    else:
        Yhat = Xn + pred

    Yhat = l2_normalize(Yhat)
    return Yhat


def eval_mhn_one_step(records, M, stride=1, mode="plain",
                      use_unit_norm=True, max_pairs_per_video=None,
                      key="tubelet_embeddings"):
    """
    Measures cosine between MHN prediction and true x_{t+stride}.
    Baseline is identity: cosine(x_t, x_{t+stride}).

    Returns:
      mean_cos, std_cos, baseline_mean_cos
    """
    cos_vals = []
    base_vals = []

    for rec in records:
        cls = load_tensor(rec, key=key)
        T = cls.shape[0]
        if T <= stride:
            continue

        X = cls[:-stride]
        Y_true = cls[stride:]

        if max_pairs_per_video is not None and X.shape[0] > max_pairs_per_video:
            X = X[:max_pairs_per_video]
            Y_true = Y_true[:max_pairs_per_video]

        if use_unit_norm:
            Xn = l2_normalize(X)
            Yn = l2_normalize(Y_true)
        else:
            Xn, Yn = X, Y_true

        Yhat = predict_one_step(M, Xn, mode=mode)

        cos = np.sum(Yhat * Yn, axis=1)
        cos_vals.append(cos)

        base = np.sum(Xn * Yn, axis=1)
        base_vals.append(base)

    if len(cos_vals) == 0:
        return float("nan"), float("nan"), float("nan")

    cos_all = np.concatenate(cos_vals)
    base_all = np.concatenate(base_vals)

    return float(cos_all.mean()), float(cos_all.std()), float(base_all.mean())


# ---------------------------
# Multi-step rollout evaluation
# ---------------------------

def rollout_from_anchor(M, x0, steps, mode="plain"):
    """
    x0: [D] assumed normalized
    Returns:
      seq: [steps+1, D] including x0
    """
    assert mode in ("plain", "delta")

    seq = [x0]
    x = x0

    for _ in range(steps):
        pred = M.T @ x
        if mode == "plain":
            x = pred
        else:
            x = x + pred

        x = x / (np.linalg.norm(x) + 1e-6)
        seq.append(x)

    return np.stack(seq, axis=0)


def eval_mhn_rollout(records, M, stride=1, mode="plain",
                     use_unit_norm=True,
                     rollout_k=5,
                     anchors_per_video=5,
                     seed=0,
                     key="tubelet_embeddings"):
    """
    Multi-step evaluation aligned with replay:
      Start from anchors x_t, roll out k steps with stride s:
        x_t -> x_{t+s} -> ... -> x_{t+k*s}

    Reports per-step cosine vs true and identity baseline:
      baseline at step j is cosine(x_t, x_{t+j*s}).
    """
    rng = np.random.default_rng(seed)

    per_step_vals = [[] for _ in range(rollout_k)]
    per_step_base = [[] for _ in range(rollout_k)]

    for rec in records:
        cls = load_tensor(rec, key=key)
        T, D = cls.shape

        max_start = T - rollout_k * stride - 1
        if max_start <= 0:
            continue

        if anchors_per_video is None or anchors_per_video <= 0:
            anchor_idx = np.arange(0, max_start)
        else:
            anchor_idx = rng.integers(0, max_start, size=anchors_per_video)

        for t0 in anchor_idx:
            x0 = cls[t0]

            true_steps = []
            for j in range(1, rollout_k + 1):
                true_steps.append(cls[t0 + j * stride])
            true_steps = np.stack(true_steps, axis=0)  # [k, D]

            if use_unit_norm:
                x0n = l2_normalize(x0[None, :])[0]
                true_n = l2_normalize(true_steps)
            else:
                x0n = x0
                true_n = true_steps

            pred_seq = rollout_from_anchor(M, x0n, rollout_k, mode=mode)
            pred_steps = pred_seq[1:]  # [k, D]

            cos = np.sum(pred_steps * true_n, axis=1)
            base = np.sum(np.repeat(x0n[None, :], rollout_k, axis=0) * true_n, axis=1)

            for j in range(rollout_k):
                per_step_vals[j].append(cos[j])
                per_step_base[j].append(base[j])

    per_step_mean = []
    per_step_std = []
    per_step_base_mean = []
    per_step_impr = []
    all_cos = []

    for j in range(rollout_k):
        if len(per_step_vals[j]) == 0:
            per_step_mean.append(float("nan"))
            per_step_std.append(float("nan"))
            per_step_base_mean.append(float("nan"))
            per_step_impr.append(float("nan"))
            continue

        vals = np.array(per_step_vals[j], dtype=np.float32)
        bases = np.array(per_step_base[j], dtype=np.float32)

        per_step_mean.append(float(vals.mean()))
        per_step_std.append(float(vals.std()))
        per_step_base_mean.append(float(bases.mean()))
        per_step_impr.append(float(vals.mean() - bases.mean()))

        all_cos.append(vals)

    overall_mean = float(np.concatenate(all_cos).mean()) if len(all_cos) else float("nan")

    return {
        "per_step_mean": per_step_mean,
        "per_step_std": per_step_std,
        "per_step_baseline_mean": per_step_base_mean,
        "per_step_improvement": per_step_impr,
        "overall_mean": overall_mean
    }


# ---------------------------
# Saving
# ---------------------------

def save_model(models_dir: Path, name: str, M, meta: dict):
    models_dir.mkdir(parents=True, exist_ok=True)

    out_M = models_dir / f"{name}_M.npy"
    out_meta = models_dir / f"{name}_meta.json"

    np.save(out_M, M.astype(np.float32))
    with open(out_meta, "w") as f:
        json.dump(meta, f, indent=2)

    return out_M, out_meta


# ---------------------------
# Main
# ---------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Train heteroassociative MHN on VideoMAE .pt embeddings "
                    "with optional delta mode and multi-step rollout eval."
    )
    parser.add_argument("--processed_root", type=str, required=True,
                        help="Path to processed VideoMAE folder, e.g. data/processed_video/train")
    parser.add_argument("--models_dir", type=str, default="models",
                        help="Directory to save MHN model")
    parser.add_argument("--name", type=str, default=None,
                        help="Model name prefix. Default auto name.")
    parser.add_argument("--val_ratio", type=float, default=0.1,
                        help="Video-level validation ratio")
    parser.add_argument("--seed", type=int, default=0,
                        help="Random seed for split")

    parser.add_argument("--key", type=str, default="tubelet_embeddings",
                        help="Which tensor key to use from each .pt file. "
                             "Recommended: tubelet_embeddings or clip_video_embeddings.")

    parser.add_argument("--stride", type=int, default=1,
                        help="Temporal stride for pairs: x_t -> x_{t+stride}")
    parser.add_argument("--mode", type=str, default="plain", choices=["plain", "delta"],
                        help="plain: learn x_t -> x_{t+s}. "
                             "delta: learn x_t -> (x_{t+s}-x_t)")

    parser.add_argument("--no_unit_norm", action="store_true",
                        help="Disable unit-norm normalization for X/Y and eval")

    parser.add_argument("--max_val_pairs_per_video", type=int, default=None,
                        help="Limit 1-step eval pairs per video for faster validation")

    parser.add_argument("--rollout_k", type=int, default=5,
                        help="Number of MHN rollout steps to evaluate")
    parser.add_argument("--rollout_anchors_per_video", type=int, default=5,
                        help="Number of anchor starting points per video for rollout eval "
                             "(set 0 to use all possible anchors)")

    args = parser.parse_args()

    if args.stride < 1:
        raise ValueError("--stride must be >= 1")
    if args.rollout_k < 1:
        raise ValueError("--rollout_k must be >= 1")

    processed_root = Path(args.processed_root)
    models_dir = Path(args.models_dir)

    if not processed_root.exists():
        raise FileNotFoundError(f"processed_root not found: {processed_root}")

    records = list_videos(processed_root)
    if len(records) == 0:
        raise RuntimeError(f"No .pt videos found under: {processed_root}")

    train_recs, val_recs = build_video_level_split(
        records, val_ratio=args.val_ratio, seed=args.seed
    )

    use_unit_norm = not args.no_unit_norm

    # Auto name uses first file metadata if possible
    if args.name is None:
        meta0 = try_load_metadata(train_recs[0])
        model_name = meta0.get("model_name", "videomae")
        sampling_mode = meta0.get("sampling_mode", "unk")
        sample_fps = meta0.get("sample_fps", "unkfps")
        args.name = f"mhn_{model_name}_{args.key}_{sampling_mode}_{sample_fps}_{args.mode}_s{args.stride}"

    # ---- Train ----
    t0 = time.time()
    M, n_pairs, dim = accumulate_mhn(
        train_recs,
        stride=args.stride,
        mode=args.mode,
        use_unit_norm=use_unit_norm,
        key=args.key
    )
    train_time = time.time() - t0

    if M is None or n_pairs == 0:
        raise RuntimeError(
            "No training pairs found. "
            "Check your processed embeddings, key, and stride."
        )

    min_T_for_rollout = args.rollout_k * args.stride + 1

    def filter_by_min_T(recs, key, min_T):
        out = []
        for r in recs:
            x = load_tensor(r, key=key)
            if x.shape[0] > min_T:
                out.append(r)
        return out

    val_recs_roll = filter_by_min_T(val_recs, args.key, min_T_for_rollout)
    print("Val videos eligible for rollout:", len(val_recs_roll), "/", len(val_recs))

    # ---- 1-step Eval ----
    mean_cos, std_cos, baseline_mean = eval_mhn_one_step(
        val_recs, M,
        stride=args.stride, mode=args.mode,
        use_unit_norm=use_unit_norm,
        max_pairs_per_video=args.max_val_pairs_per_video,
        key=args.key
    )
    improvement = (
        mean_cos - baseline_mean
        if np.isfinite(mean_cos) and np.isfinite(baseline_mean)
        else float("nan")
    )

    # ---- Rollout Eval ----
    rollout_stats = eval_mhn_rollout(
        val_recs, M,
        stride=args.stride, mode=args.mode,
        use_unit_norm=use_unit_norm,
        rollout_k=args.rollout_k,
        anchors_per_video=args.rollout_anchors_per_video,
        seed=args.seed,
        key=args.key
    )

    meta = {
        "name": args.name,
        "processed_root": str(processed_root),
        "key": args.key,
        "num_videos_total": len(records),
        "num_videos_train": len(train_recs),
        "num_videos_val": len(val_recs),
        "dim": dim,
        "train_pairs": int(n_pairs),
        "unit_norm": use_unit_norm,
        "val_ratio": args.val_ratio,
        "seed": args.seed,
        "stride": args.stride,
        "mode": args.mode,
        "metrics": {
            "one_step": {
                "val_cosine_mean": mean_cos,
                "val_cosine_std": std_cos,
                "val_baseline_cosine_mean": baseline_mean,
                "val_improvement_over_baseline": improvement
            },
            "rollout": rollout_stats
        },
        "train_time_sec": train_time
    }

    out_M, out_meta = save_model(models_dir, args.name, M, meta)

    # ---- Print summary ----
    print("\n=== MHN Training Complete (VideoMAE) ===")
    print(f"Processed root: {processed_root}")
    print(f"Key: {args.key}")
    print(f"Videos: total={len(records)} train={len(train_recs)} val={len(val_recs)}")
    print(f"Embedding dim: {dim}")
    print(f"Training pairs: {n_pairs}")
    print(f"Unit norm: {use_unit_norm}")
    print(f"Mode: {args.mode}")
    print(f"Stride: {args.stride}")
    print(f"Train time: {train_time:.2f}s")

    print("\n--- 1-step Validation ---")
    print(f"MHN cosine mean: {mean_cos:.4f}  std: {std_cos:.4f}")
    print(f"Baseline cosine mean (x_t vs x_{{t+{args.stride}}}): {baseline_mean:.4f}")
    print(f"Improvement: {improvement:.4f}")

    print("\n--- Rollout Validation ---")
    for j in range(args.rollout_k):
        m_j = rollout_stats["per_step_mean"][j]
        b_j = rollout_stats["per_step_baseline_mean"][j]
        imp_j = rollout_stats["per_step_improvement"][j]
        print(f"Step {j+1}: MHN={m_j:.4f}  baseline={b_j:.4f}  improvement={imp_j:.4f}")
    print(f"Rollout overall mean cosine: {rollout_stats['overall_mean']:.4f}")

    print("\n--- Saved ---")
    print(f"M matrix: {out_M}")
    print(f"Metadata: {out_meta}")


if __name__ == "__main__":
    main()
