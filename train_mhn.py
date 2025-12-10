# #!/usr/bin/env python3
# import argparse
# import json
# import os
# from pathlib import Path
# import time
# import numpy as np


# def list_videos(processed_root: Path):
#     """
#     Returns list of records with:
#       {
#         "group": str,
#         "video_id": str,
#         "cls_path": Path,
#         "meta_path": Path
#       }
#     """
#     records = []
#     for group_dir in processed_root.iterdir():
#         if not group_dir.is_dir():
#             continue

#         # Each video has a sidecar json named <video_id>.json
#         for meta_path in sorted(group_dir.glob("*.json")):
#             try:
#                 meta = json.load(open(meta_path))
#             except Exception:
#                 continue

#             vid = meta.get("video_id", meta_path.stem)
#             cls_path = group_dir / f"{vid}_cls_fp16.npy"

#             if cls_path.exists():
#                 records.append({
#                     "group": group_dir.name,
#                     "video_id": vid,
#                     "cls_path": cls_path,
#                     "meta_path": meta_path
#                 })
#     return records


# def load_cls(record):
#     return np.load(record["cls_path"]).astype(np.float32)  # [T, D]


# def l2_normalize(x, axis=-1, eps=1e-6):
#     norm = np.linalg.norm(x, axis=axis, keepdims=True)
#     return x / (norm + eps)


# def build_video_level_split(records, val_ratio=0.1, seed=0):
#     rng = np.random.default_rng(seed)
#     idx = np.arange(len(records))
#     rng.shuffle(idx)

#     n_val = max(1, int(round(len(records) * val_ratio)))
#     val_idx = set(idx[:n_val].tolist())

#     train = [r for i, r in enumerate(records) if i not in val_idx]
#     val = [r for i, r in enumerate(records) if i in val_idx]
#     return train, val


# def accumulate_mhn(records, stride=1, use_unit_norm=True):
#     """
#     Memory-efficient accumulation:
#       M = (1/N) * sum_t x_t ⊗ x_{t+stride}
#     Returns:
#       M, pair_count, dim
#     """
#     M = None
#     pair_count = 0
#     dim = None

#     for rec in records:
#         cls = load_cls(rec)
#         T = cls.shape[0]
#         if T <= stride:
#             continue

#         X = cls[:-stride]
#         Y = cls[stride:]

#         if use_unit_norm:
#             X = l2_normalize(X)
#             Y = l2_normalize(Y)

#         if dim is None:
#             dim = X.shape[1]
#             M = np.zeros((dim, dim), dtype=np.float32)

#         M += X.T @ Y
#         pair_count += X.shape[0]

#     if pair_count > 0:
#         M /= float(pair_count)

#     return M, pair_count, dim


# def eval_mhn(records, M, stride=1, use_unit_norm=True, max_pairs_per_video=None):
#     """
#     Returns:
#       mean_cos, std_cos, baseline_mean_cos

#     Baseline = cosine(x_t, x_{t+stride}) without MHN
#     """
#     cos_vals = []
#     base_vals = []

#     for rec in records:
#         cls = load_cls(rec)
#         T = cls.shape[0]
#         if T <= stride:
#             continue

#         X = cls[:-stride]
#         Y = cls[stride:]

#         if max_pairs_per_video is not None and X.shape[0] > max_pairs_per_video:
#             X = X[:max_pairs_per_video]
#             Y = Y[:max_pairs_per_video]

#         if use_unit_norm:
#             Xn = l2_normalize(X)
#             Yn = l2_normalize(Y)
#         else:
#             Xn, Yn = X, Y

#         # MHN prediction
#         Yhat = (M.T @ Xn.T).T  # [T-stride, D]
#         Yhat = l2_normalize(Yhat)

#         # cosine(Yhat, Y)
#         cos = np.sum(Yhat * Yn, axis=1)
#         cos_vals.append(cos)

#         # baseline cosine(x_t, x_{t+stride})
#         base = np.sum(Xn * Yn, axis=1)
#         base_vals.append(base)

#     if len(cos_vals) == 0:
#         return float("nan"), float("nan"), float("nan")

#     cos_all = np.concatenate(cos_vals)
#     base_all = np.concatenate(base_vals)

#     return float(cos_all.mean()), float(cos_all.std()), float(base_all.mean())


# def save_model(models_dir: Path, name: str, M, meta: dict):
#     models_dir.mkdir(parents=True, exist_ok=True)

#     out_M = models_dir / f"{name}_M.npy"
#     out_meta = models_dir / f"{name}_meta.json"

#     np.save(out_M, M.astype(np.float32))
#     with open(out_meta, "w") as f:
#         json.dump(meta, f, indent=2)

#     return out_M, out_meta


# def main():
#     parser = argparse.ArgumentParser(description="Train heteroassociative MHN on DINOv2 CLS embeddings.")
#     parser.add_argument("--processed_root", type=str, required=True,
#                         help="Path to processed DINO folder, e.g. data/processed/dino/train")
#     parser.add_argument("--models_dir", type=str, default="models",
#                         help="Directory to save MHN model")
#     parser.add_argument("--name", type=str, default=None,
#                         help="Model name prefix. Default auto name.")
#     parser.add_argument("--val_ratio", type=float, default=0.1,
#                         help="Video-level validation ratio")
#     parser.add_argument("--seed", type=int, default=0,
#                         help="Random seed for split")
#     parser.add_argument("--stride", type=int, default=1,
#                         help="Temporal stride for pairs: x_t -> x_{t+stride}")
#     parser.add_argument("--no_unit_norm", action="store_true",
#                         help="Disable unit-norm normalization for X/Y")
#     parser.add_argument("--max_val_pairs_per_video", type=int, default=None,
#                         help="Limit eval pairs per video for faster validation")
#     args = parser.parse_args()

#     if args.stride < 1:
#         raise ValueError("--stride must be >= 1")

#     processed_root = Path(args.processed_root)
#     models_dir = Path(args.models_dir)

#     if not processed_root.exists():
#         raise FileNotFoundError(f"processed_root not found: {processed_root}")

#     records = list_videos(processed_root)
#     if len(records) == 0:
#         raise RuntimeError(f"No videos found under: {processed_root}")

#     train_recs, val_recs = build_video_level_split(
#         records, val_ratio=args.val_ratio, seed=args.seed
#     )

#     use_unit_norm = not args.no_unit_norm

#     # Auto name (include stride so you don't overwrite)
#     if args.name is None:
#         try:
#             m = json.load(open(train_recs[0]["meta_path"]))
#             fps = m.get("sampling_fps", "unkfps")
#             model = m.get("model", "dinov2")
#             repr_ = m.get("repr", "cls")
#             args.name = f"mhn_{model}_{repr_}_{fps}fps_s{args.stride}"
#         except Exception:
#             args.name = f"mhn_dino_cls_s{args.stride}"

#     t0 = time.time()
#     M, n_pairs, dim = accumulate_mhn(
#         train_recs, stride=args.stride, use_unit_norm=use_unit_norm
#     )
#     train_time = time.time() - t0

#     if M is None or n_pairs == 0:
#         raise RuntimeError("No training pairs found. Check your processed embeddings and stride.")

#     # Evaluate
#     mean_cos, std_cos, baseline_mean = eval_mhn(
#         val_recs, M, stride=args.stride, use_unit_norm=use_unit_norm,
#         max_pairs_per_video=args.max_val_pairs_per_video
#     )

#     improvement = (
#         mean_cos - baseline_mean
#         if np.isfinite(mean_cos) and np.isfinite(baseline_mean)
#         else float("nan")
#     )

#     meta = {
#         "name": args.name,
#         "processed_root": str(processed_root),
#         "num_videos_total": len(records),
#         "num_videos_train": len(train_recs),
#         "num_videos_val": len(val_recs),
#         "dim": dim,
#         "train_pairs": int(n_pairs),
#         "unit_norm": use_unit_norm,
#         "val_ratio": args.val_ratio,
#         "seed": args.seed,
#         "stride": args.stride,
#         "metrics": {
#             "val_cosine_mean": mean_cos,
#             "val_cosine_std": std_cos,
#             "val_baseline_cosine_mean": baseline_mean,
#             "val_improvement_over_baseline": improvement
#         },
#         "train_time_sec": train_time
#     }

#     out_M, out_meta = save_model(models_dir, args.name, M, meta)

#     print("\n=== MHN Training Complete ===")
#     print(f"Processed root: {processed_root}")
#     print(f"Videos: total={len(records)} train={len(train_recs)} val={len(val_recs)}")
#     print(f"Embedding dim: {dim}")
#     print(f"Training pairs: {n_pairs}")
#     print(f"Unit norm: {use_unit_norm}")
#     print(f"Stride: {args.stride}")
#     print(f"Train time: {train_time:.2f}s")

#     print("\n--- Validation ---")
#     print(f"MHN cosine mean: {mean_cos:.4f}  std: {std_cos:.4f}")
#     print(f"Baseline cosine mean (x_t vs x_{{t+{args.stride}}}): {baseline_mean:.4f}")
#     print(f"Improvement: {improvement:.4f}")

#     print("\n--- Saved ---")
#     print(f"M matrix: {out_M}")
#     print(f"Metadata: {out_meta}")


# if __name__ == "__main__":
#     main()


#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
import time
import numpy as np


# ---------------------------
# Data discovery / loading
# ---------------------------

def list_videos(processed_root: Path):
    """
    Returns list of records with:
      {
        "group": str,
        "video_id": str,
        "cls_path": Path,
        "meta_path": Path
      }
    """
    records = []
    for group_dir in processed_root.iterdir():
        if not group_dir.is_dir():
            continue

        # Each video has a sidecar json named <video_id>.json
        for meta_path in sorted(group_dir.glob("*.json")):
            try:
                meta = json.load(open(meta_path))
            except Exception:
                continue

            vid = meta.get("video_id", meta_path.stem)
            cls_path = group_dir / f"{vid}_cls_fp16.npy"

            if cls_path.exists():
                records.append({
                    "group": group_dir.name,
                    "video_id": vid,
                    "cls_path": cls_path,
                    "meta_path": meta_path
                })
    return records


def load_cls(record):
    return np.load(record["cls_path"]).astype(np.float32)  # [T, D]


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

def accumulate_mhn(records, stride=1, mode="plain", use_unit_norm=True):
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
        cls = load_cls(rec)
        T = cls.shape[0]
        if T <= stride:
            continue

        X = cls[:-stride]
        Y = cls[stride:]

        if mode == "delta":
            Y = Y - X  # target is delta

        if use_unit_norm:
            X = l2_normalize(X)
            # For delta targets, normalizing the target is optional.
            # We normalize for numeric stability but it is not strictly required.
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

    # predicted target
    pred = (M.T @ Xn.T).T  # [N, D]

    if mode == "plain":
        Yhat = pred
    else:
        # delta mode predicts change; add to current state
        Yhat = Xn + pred

    Yhat = l2_normalize(Yhat)
    return Yhat


def eval_mhn_one_step(records, M, stride=1, mode="plain",
                      use_unit_norm=True, max_pairs_per_video=None):
    """
    Measures cosine between MHN prediction and true x_{t+stride}.
    Baseline is identity: cosine(x_t, x_{t+stride}).

    Returns:
      mean_cos, std_cos, baseline_mean_cos
    """
    cos_vals = []
    base_vals = []

    for rec in records:
        cls = load_cls(rec)
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

        base = np.sum(Xn * Yn, axis=1)  # identity baseline
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
        pred = M.T @ x  # [D]
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
                     seed=0):
    """
    Multi-step evaluation aligned with replay:
      Start from anchors x_t, roll out k steps with stride s:
        x_t -> x_{t+s} -> ... -> x_{t+k*s}

    Reports per-step cosine vs true and identity baseline:
      baseline at step j is cosine(x_t, x_{t+j*s}).

    Returns dict:
      {
        "per_step_mean": [k],
        "per_step_std": [k],
        "per_step_baseline_mean": [k],
        "per_step_improvement": [k],
        "overall_mean": float
      }
    """
    rng = np.random.default_rng(seed)

    per_step_vals = [[] for _ in range(rollout_k)]
    per_step_base = [[] for _ in range(rollout_k)]

    for rec in records:
        cls = load_cls(rec)
        T, D = cls.shape

        max_start = T - rollout_k * stride - 1
        if max_start <= 0:
            continue

        # choose anchor indices
        if anchors_per_video is None or anchors_per_video <= 0:
            anchor_idx = np.arange(0, max_start)
        else:
            # uniform-ish random anchors
            anchor_idx = rng.integers(0, max_start, size=anchors_per_video)

        for t0 in anchor_idx:
            x0 = cls[t0]

            # true targets for steps 1..k
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

            # per-step cosine
            cos = np.sum(pred_steps * true_n, axis=1)  # [k]

            # identity baseline per step uses the anchor x0
            base = np.sum(np.repeat(x0n[None, :], rollout_k, axis=0) * true_n, axis=1)

            for j in range(rollout_k):
                per_step_vals[j].append(cos[j])
                per_step_base[j].append(base[j])

    # aggregate
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
        description="Train heteroassociative MHN on DINOv2 CLS embeddings "
                    "with optional delta mode and multi-step rollout eval."
    )
    parser.add_argument("--processed_root", type=str, required=True,
                        help="Path to processed DINO folder, e.g. data/processed/dino/train")
    parser.add_argument("--models_dir", type=str, default="models",
                        help="Directory to save MHN model")
    parser.add_argument("--name", type=str, default=None,
                        help="Model name prefix. Default auto name.")
    parser.add_argument("--val_ratio", type=float, default=0.1,
                        help="Video-level validation ratio")
    parser.add_argument("--seed", type=int, default=0,
                        help="Random seed for split")

    parser.add_argument("--stride", type=int, default=1,
                        help="Temporal stride for pairs: x_t -> x_{t+stride}")
    parser.add_argument("--mode", type=str, default="plain", choices=["plain", "delta"],
                        help="plain: learn x_t -> x_{t+s}. delta: learn x_t -> (x_{t+s}-x_t)")

    parser.add_argument("--no_unit_norm", action="store_true",
                        help="Disable unit-norm normalization for X/Y and eval")

    parser.add_argument("--max_val_pairs_per_video", type=int, default=None,
                        help="Limit 1-step eval pairs per video for faster validation")

    # rollout controls
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
        raise RuntimeError(f"No videos found under: {processed_root}")

    train_recs, val_recs = build_video_level_split(
        records, val_ratio=args.val_ratio, seed=args.seed
    )

    use_unit_norm = not args.no_unit_norm

    # Auto name includes model/fps if possible + mode + stride
    if args.name is None:
        try:
            m = json.load(open(train_recs[0]["meta_path"]))
            fps = m.get("sampling_fps", "unkfps")
            model = m.get("model", "dinov2")
            repr_ = m.get("repr", "cls")
            args.name = f"mhn_{model}_{repr_}_{fps}fps_{args.mode}_s{args.stride}"
        except Exception:
            args.name = f"mhn_dino_cls_{args.mode}_s{args.stride}"

    # ---- Train ----
    t0 = time.time()
    M, n_pairs, dim = accumulate_mhn(
        train_recs, stride=args.stride, mode=args.mode, use_unit_norm=use_unit_norm
    )
    train_time = time.time() - t0

    if M is None or n_pairs == 0:
        raise RuntimeError("No training pairs found. Check your processed embeddings and stride.")

    # ---- 1-step Eval ----
    mean_cos, std_cos, baseline_mean = eval_mhn_one_step(
        val_recs, M,
        stride=args.stride, mode=args.mode,
        use_unit_norm=use_unit_norm,
        max_pairs_per_video=args.max_val_pairs_per_video
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
        seed=args.seed
    )

    meta = {
        "name": args.name,
        "processed_root": str(processed_root),
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
    print("\n=== MHN Training Complete ===")
    print(f"Processed root: {processed_root}")
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
