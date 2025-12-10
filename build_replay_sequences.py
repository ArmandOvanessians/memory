#!/usr/bin/env python3
import argparse
from pathlib import Path
import json
import numpy as np
import torch

def l2_normalize(x, eps=1e-6):
    n = np.linalg.norm(x, axis=-1, keepdims=True)
    return x / (n + eps)

def load_tubelets(pt_path, key="tubelet_embeddings"):
    obj = torch.load(pt_path, map_location="cpu")
    x = obj[key]
    if x.ndim == 3 and key == "clip_tubelet_embeddings":
        x = x.reshape(-1, x.shape[-1])
    if x.ndim == 1:
        x = x.unsqueeze(0)
    return x.float().cpu().numpy()

def rollout_from_anchor(M, x0, steps, mode="plain"):
    seq = [x0]
    x = x0
    for _ in range(steps):
        pred = M.T @ x
        x = pred if mode == "plain" else (x + pred)
        x = x / (np.linalg.norm(x) + 1e-6)
        seq.append(x)
    return np.stack(seq, axis=0)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--processed_root", required=True)
    ap.add_argument("--mhn_M_path", required=True, help="Path to *_M.npy")
    ap.add_argument("--out_dir", required=True)

    ap.add_argument("--key", default="tubelet_embeddings")
    ap.add_argument("--mode", default="plain", choices=["plain", "delta"])
    ap.add_argument("--stride", type=int, default=1)

    ap.add_argument("--rollout_k", type=int, default=3)
    ap.add_argument("--anchors_per_video", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)

    ap.add_argument("--max_videos", type=int, default=None)
    ap.add_argument("--save_every", type=int, default=500)

    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    root = Path(args.processed_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    M = np.load(args.mhn_M_path).astype(np.float32)

    files = sorted(root.rglob("*.pt"))
    if args.max_videos:
        files = files[:args.max_videos]

    real_buf = []
    replay_buf = []
    mixed_buf = []
    meta_buf = []

    shard_idx = 0

    def flush():
        nonlocal shard_idx, real_buf, replay_buf, mixed_buf, meta_buf
        if not real_buf and not replay_buf and not mixed_buf:
            return
        out_path = out_dir / f"replay_shard_{shard_idx:04d}.npz"
        np.savez_compressed(
            out_path,
            real=np.array(real_buf, dtype=np.float32),
            replay=np.array(replay_buf, dtype=np.float32),
            mixed=np.array(mixed_buf, dtype=np.float32),
            meta=np.array(meta_buf, dtype=object),
        )
        print(f"Saved {out_path} with {len(real_buf)} sequences")
        shard_idx += 1
        real_buf, replay_buf, mixed_buf, meta_buf = [], [], [], []

    for i, p in enumerate(files):
        try:
            x = load_tubelets(p, key=args.key)  # [T, D]
            if x.shape[0] <= args.rollout_k * args.stride + 1:
                continue

            # normalize for stable MHN replay alignment
            x_n = l2_normalize(x)

            T = x_n.shape[0]
            max_start = T - args.rollout_k * args.stride - 1
            if max_start <= 0:
                continue

            # anchors
            anchors = rng.integers(0, max_start, size=args.anchors_per_video)

            for t0 in anchors:
                # real sequence of length k+1 (teacher forcing target)
                real_seq = [x_n[t0 + j * args.stride] for j in range(args.rollout_k + 1)]
                real_seq = np.stack(real_seq, axis=0)  # [k+1, D]

                # replay sequence using MHN
                x0 = x_n[t0]
                rep_seq = rollout_from_anchor(M, x0, args.rollout_k, mode=args.mode)  # [k+1, D]

                # mixed: first half real, second half replay (simple curriculum-friendly blend)
                cut = max(1, (args.rollout_k + 1) // 2)
                mix_seq = np.concatenate([real_seq[:cut], rep_seq[cut:]], axis=0)

                real_buf.append(real_seq)
                replay_buf.append(rep_seq)
                mixed_buf.append(mix_seq)
                meta_buf.append({"pt": str(p), "anchor": int(t0)})

        except Exception:
            continue

        if (i + 1) % args.save_every == 0:
            flush()

    flush()

    # Save config used
    cfg = vars(args)
    with open(out_dir / "replay_build_config.json", "w") as f:
        json.dump(cfg, f, indent=2)

if __name__ == "__main__":
    main()
