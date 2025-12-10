# #!/usr/bin/env python3
# """
# VideoMAE embedding extraction with robust temporal sampling + windowed inference
# (+ small upgrades for heteroassociative Modern Hopfield / sequential dynamics).

# Key upgrades vs earlier version:
# 1) Overlap by default:
#    - If --window_stride is not provided, we use 50% overlap.
#    - This increases usable transition pairs for heteroassociative MHN.

# 2) Tubelet-level timestamps:
#    - We compute timestamps per sampled frame (if fps available),
#      then aggregate into tubelet timestamps by averaging each tube-sized group.
#    - Helps interpret stride in real time.

# 3) Delta tubelets:
#    - We save delta_tubelet_embeddings = x_{t+1} - x_t
#    - Useful if you train MHN in "delta" mode.

# 4) Optional save-time unit normalization:
#    - --save_unit_norm to store normalized tubelet + clip embeddings.

# Outputs per video (.pt dict):
# - clip_tubelet_embeddings: (W, t_per_window, D)
# - tubelet_embeddings:      (W * t_per_window, D)  [flattened]
# - delta_tubelet_embeddings:(W * t_per_window - 1, D)  [optional]
# - clip_video_embeddings:   (W, D)
# - video_embedding:         (D,)
# - metadata (with tubelet_timestamps_sec)

# Usage examples:
#   # Legacy-ish (uniform-count)
#   python extract_embeddings_windowed.py --input_dir X --output_dir Y --device cuda

#   # Recommended for sequential pipelines
#   python extract_embeddings_windowed.py \
#     --input_dir X --output_dir Y --device cuda \
#     --sample_fps 2 --max_frames 256

#   # Denser transitions (overlap)
#   python extract_embeddings_windowed.py \
#     --input_dir X --output_dir Y --device cuda \
#     --sample_fps 2 --max_frames 256 \
#     --window_stride 8

#   # Save normalized embeddings
#   python extract_embeddings_windowed.py \
#     --input_dir X --output_dir Y --device cuda \
#     --sample_fps 2 --max_frames 256 \
#     --save_unit_norm
# """

# import argparse
# from pathlib import Path

# import numpy as np
# import torch
# import torch.nn.functional as F
# from decord import VideoReader
# from tqdm import tqdm
# from transformers import VideoMAEImageProcessor, VideoMAEModel


# def _safe_get_fps(vr: VideoReader):
#     try:
#         fps = float(vr.get_avg_fps())
#         if fps <= 0 or np.isnan(fps):
#             return None
#         return fps
#     except Exception:
#         return None


# def _make_indices_fps_based(total_frames, fps, sample_fps, max_frames=None):
#     stride = max(int(round(fps / sample_fps)), 1)
#     indices = np.arange(0, total_frames, stride, dtype=int)
#     if max_frames is not None:
#         indices = indices[:max_frames]
#     return indices


# def _make_indices_uniform_count(total_frames, num_frames):
#     if num_frames <= 0:
#         raise ValueError("num_frames must be > 0.")
#     return np.linspace(0, total_frames - 1, num_frames).astype(int)


# def _chunk_indices(indices, window_size, window_stride):
#     """
#     Make sliding windows over a 1D indices array.
#     Drops last partial window by default.
#     """
#     windows = []
#     n = len(indices)
#     start = 0
#     while start + window_size <= n:
#         windows.append(indices[start:start + window_size])
#         start += window_stride
#     return windows


# def _compute_window_embeddings(frames, model, processor, device):
#     """
#     frames: numpy array (T, H, W, 3) for exactly T == window_size
#     Returns:
#       tubelet_embeddings: (t_tokens, D)
#       clip_embedding: (D,)
#     """
#     inputs = processor(list(frames), return_tensors="pt")
#     pixel_values = inputs["pixel_values"]  # (T, C, H, W)

#     if pixel_values.ndim == 4:
#         pixel_values = pixel_values.unsqueeze(0)  # (1, T, C, H, W)

#     pixel_values = pixel_values.to(device)

#     with torch.no_grad():
#         outputs = model(pixel_values=pixel_values)

#     tokens = outputs.last_hidden_state[0]  # (seq_len, D)

#     tube = getattr(model.config, "tubelet_size", 2)
#     T = pixel_values.shape[1]

#     if T % tube != 0:
#         raise RuntimeError(f"Window T={T} not divisible by tubelet_size={tube}")

#     t_tokens = T // tube

#     image_size = getattr(model.config, "image_size", 224)
#     patch_size = getattr(model.config, "patch_size", 16)
#     num_patches_per_tubelet = (image_size // patch_size) ** 2

#     expected_seq = t_tokens * num_patches_per_tubelet
#     if tokens.shape[0] != expected_seq:
#         raise RuntimeError(
#             f"Unexpected seq_len={tokens.shape[0]}, expected {expected_seq}. "
#             f"(t_tokens={t_tokens}, patches={num_patches_per_tubelet})."
#         )

#     tokens = tokens.reshape(t_tokens, num_patches_per_tubelet, -1)
#     tubelet_embeddings = tokens.mean(dim=1)  # (t_tokens, D)
#     clip_embedding = tubelet_embeddings.mean(dim=0)  # (D,)

#     return tubelet_embeddings.detach().cpu(), clip_embedding.detach().cpu()


# def extract_videomae_embeddings(
#     video_path,
#     model,
#     processor,
#     device="cpu",
#     num_frames=None,
#     sample_fps=None,
#     max_frames=None,
#     window_size=None,
#     window_stride=None,
#     save_unit_norm=False,
# ):
#     """
#     Windowed extraction returning a richer sequential structure.

#     Returns dict with:
#       - clip_tubelet_embeddings: (W, t_per_window, D)
#       - tubelet_embeddings:      (W*t_per_window, D)
#       - delta_tubelet_embeddings:(W*t_per_window-1, D)  [if length > 1]
#       - clip_video_embeddings:   (W, D)
#       - video_embedding:         (D,)
#       - metadata
#     """
#     vr = VideoReader(str(video_path))
#     total_frames = len(vr)
#     if total_frames == 0:
#         raise ValueError(f"Video {video_path} has no frames.")

#     fps = _safe_get_fps(vr)

#     # Model expected window length
#     model_frames = getattr(model.config, "num_frames", 16)
#     if window_size is None:
#         window_size = model_frames

#     # Default overlap for better heteroassociative training signal
#     if window_stride is None:
#         window_stride = max(1, window_size // 2)  # 50% overlap by default

#     if window_size <= 0 or window_stride <= 0:
#         raise ValueError("window_size and window_stride must be > 0.")

#     # Build sampled indices
#     sampling_mode = None
#     if sample_fps is not None and fps is not None:
#         indices = _make_indices_fps_based(total_frames, fps, sample_fps, max_frames=max_frames)
#         sampling_mode = "fps"
#     else:
#         if num_frames is None:
#             # Default to enough frames for at least 1 window if possible
#             num_frames = max(model_frames, getattr(model.config, "num_frames", 16))
#         indices = _make_indices_uniform_count(total_frames, num_frames)
#         sampling_mode = "uniform_count"

#     indices = np.asarray(indices, dtype=int)

#     # If too short for even one window, fall back to uniform window-sized sampling
#     if len(indices) < window_size:
#         indices = _make_indices_uniform_count(total_frames, window_size)
#         indices = np.asarray(indices, dtype=int)
#         sampling_mode = sampling_mode + "_fallback_one_window"

#     windows = _chunk_indices(indices, window_size=window_size, window_stride=window_stride)

#     if len(windows) == 0:
#         raise ValueError(
#             f"No full windows could be formed for {video_path}. "
#             f"len(indices)={len(indices)}, window_size={window_size}."
#         )

#     clip_tubelets = []
#     clip_embs = []

#     for w_idx in windows:
#         frames = vr.get_batch(w_idx).asnumpy()
#         t_emb, c_emb = _compute_window_embeddings(frames, model, processor, device)
#         clip_tubelets.append(t_emb)
#         clip_embs.append(c_emb)

#     clip_tubelet_embeddings = torch.stack(clip_tubelets, dim=0)  # (W, t_per_window, D)
#     clip_video_embeddings = torch.stack(clip_embs, dim=0)        # (W, D)

#     # Flatten tubelets into one long sequence
#     tubelet_embeddings = clip_tubelet_embeddings.reshape(-1, clip_tubelet_embeddings.shape[-1])

#     # Optional normalize before saving
#     if save_unit_norm:
#         tubelet_embeddings = F.normalize(tubelet_embeddings, dim=-1)
#         clip_video_embeddings = F.normalize(clip_video_embeddings, dim=-1)
#         # also normalize clip tubelets consistently
#         clip_tubelet_embeddings = F.normalize(clip_tubelet_embeddings, dim=-1)

#     # Delta embeddings for "delta" MHN mode
#     delta_tubelet_embeddings = None
#     if tubelet_embeddings.shape[0] > 1:
#         delta_tubelet_embeddings = tubelet_embeddings[1:] - tubelet_embeddings[:-1]

#     # Global
#     video_embedding = tubelet_embeddings.mean(dim=0)

#     # Best-effort timestamps for each sampled index
#     sampled_timestamps_sec = (indices / fps).tolist() if fps is not None else None

#     # Tubelet-level timestamps (average within each tube-sized group)
#     tube = getattr(model.config, "tubelet_size", 2)
#     tubelet_timestamps_sec = None
#     if sampled_timestamps_sec is not None:
#         ts = np.array(sampled_timestamps_sec, dtype=np.float32)
#         n = (len(ts) // tube) * tube
#         ts = ts[:n]
#         if n > 0:
#             tubelet_timestamps_sec = ts.reshape(-1, tube).mean(axis=1).tolist()

#     out = {
#         "clip_tubelet_embeddings": clip_tubelet_embeddings,
#         "tubelet_embeddings": tubelet_embeddings,
#         "delta_tubelet_embeddings": delta_tubelet_embeddings,
#         "clip_video_embeddings": clip_video_embeddings,
#         "video_embedding": video_embedding,
#         "metadata": {
#             "video_path": str(video_path),
#             "total_frames_in_file": int(total_frames),
#             "fps_estimate": fps,
#             "sampling_mode": sampling_mode,
#             "sample_fps": float(sample_fps) if sample_fps is not None else None,
#             "max_frames": int(max_frames) if max_frames is not None else None,
#             "num_frames_arg": int(num_frames) if num_frames is not None else None,
#             "sampled_indices": indices.tolist(),
#             "sampled_timestamps_sec": sampled_timestamps_sec,
#             "tubelet_timestamps_sec": tubelet_timestamps_sec,
#             "window_size": int(window_size),
#             "window_stride": int(window_stride),
#             "num_windows": int(len(windows)),
#             "tubelet_size": int(tube),
#             "t_per_window": int(window_size // tube),
#             "image_size": int(getattr(model.config, "image_size", 224)),
#             "patch_size": int(getattr(model.config, "patch_size", 16)),
#             "hidden_size": int(tubelet_embeddings.shape[-1]),
#             "model_name": getattr(model.config, "_name_or_path", "unknown"),
#             "save_unit_norm": bool(save_unit_norm),
#         }
#     }
#     return out


# def get_all_videos(input_dir):
#     input_dir = Path(input_dir)
#     return sorted(input_dir.rglob("*.mp4"))


# def _safe_video_id(video_path: Path, input_dir: Path):
#     """
#     Collision-proof id: relative path with separators replaced.
#     """
#     rel = video_path.relative_to(input_dir).with_suffix("")
#     return str(rel).replace("/", "__")


# def main():
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--input_dir", required=True, type=str)
#     parser.add_argument("--output_dir", required=True, type=str)
#     parser.add_argument("--device", default="cuda", type=str)

#     # Sampling controls
#     parser.add_argument("--num_frames", type=int, default=None,
#                         help="Uniform-count sampling across the whole video (legacy/fallback).")
#     parser.add_argument("--sample_fps", type=float, default=None,
#                         help="Preferred for sequential work. Sampling rate in frames/sec.")
#     parser.add_argument("--max_frames", type=int, default=None,
#                         help="Cap the number of sampled frames (mainly for --sample_fps).")

#     # Window controls
#     parser.add_argument("--window_size", type=int, default=None,
#                         help="Frames per window. Default = model.config.num_frames (usually 16).")
#     parser.add_argument("--window_stride", type=int, default=None,
#                         help="Stride in sampled-index space. "
#                              "Default = 50%% overlap (window_size//2).")

#     # Save-time normalization
#     parser.add_argument("--save_unit_norm", action="store_true",
#                         help="Save unit-normalized tubelet + clip embeddings.")

#     args = parser.parse_args()

#     input_dir = Path(args.input_dir)
#     output_dir = Path(args.output_dir)
#     output_dir.mkdir(parents=True, exist_ok=True)

#     device = args.device if (args.device == "cuda" and torch.cuda.is_available()) else "cpu"

#     processor = VideoMAEImageProcessor.from_pretrained("MCG-NJU/videomae-base")
#     model = VideoMAEModel.from_pretrained("MCG-NJU/videomae-base").to(device).eval()

#     videos = get_all_videos(input_dir)
#     print(f"Found {len(videos)} videos in {input_dir}")

#     for video_path in tqdm(videos, desc="Extracting embeddings"):
#         video_id = _safe_video_id(video_path, input_dir)
#         save_path = output_dir / f"{video_id}.pt"
#         save_path.parent.mkdir(parents=True, exist_ok=True)

#         if save_path.exists():
#             continue

#         try:
#             emb = extract_videomae_embeddings(
#                 video_path=video_path,
#                 model=model,
#                 processor=processor,
#                 device=device,
#                 num_frames=args.num_frames,
#                 sample_fps=args.sample_fps,
#                 max_frames=args.max_frames,
#                 window_size=args.window_size,
#                 window_stride=args.window_stride,
#                 save_unit_norm=args.save_unit_norm,
#             )
#             torch.save(emb, save_path)

#         except Exception as e:
#             print(f"[ERROR] Failed on {video_path}: {e}")

#     print(f"Done. Saved embeddings to {output_dir}")


# if __name__ == "__main__":
#     main()

#!/usr/bin/env python3
import argparse
import json
import numpy as np
from pathlib import Path
import torch
import matplotlib.pyplot as plt

from train_transformer import TransformerPredictor


def l2_normalize(x, eps=1e-6):
    n = np.linalg.norm(x, axis=-1, keepdims=True)
    return x / (n + eps)


def load_tubelet_embeddings(pt_path, key="tubelet_embeddings"):
    obj = torch.load(pt_path, map_location="cpu")
    x = obj[key]  # shape [T, D]
    return x.float().cpu().numpy()


# --------------------------------------------------------------------
# TRANSFORMER rollout for step-by-step prediction
# --------------------------------------------------------------------
def rollout_transformer(model, x0, steps):
    """
    x0: numpy array of shape [D]
    returns array of shape [steps+1, D]
    """
    model.eval()

    x = torch.tensor(x0, dtype=torch.float32).unsqueeze(0).unsqueeze(0)  # [1,1,D]
    seq = [x]

    for _ in range(steps):
        inp = torch.cat(seq, dim=1)  # [1,t,D]

        with torch.no_grad():
            pred = model(inp)[0, -1]  # [D]

        pred = pred / (pred.norm() + 1e-6)
        seq.append(pred.unsqueeze(0).unsqueeze(0))

    out = torch.cat(seq, dim=1)[0].cpu().numpy()  # [steps+1, D]
    return out


# --------------------------------------------------------------------
# MHN rollout (teacher)
# --------------------------------------------------------------------
def rollout_mhn(M, x0, steps):
    seq = [x0]
    x = x0
    for _ in range(steps):
        nxt = M.T @ x
        nxt = nxt / (np.linalg.norm(nxt) + 1e-6)
        seq.append(nxt)
        x = nxt
    return np.stack(seq, axis=0)  # [steps+1, D]


# --------------------------------------------------------------------
# Cosine similarity (per step)
# --------------------------------------------------------------------
def cosine(a, b):
    a = np.asarray(a)
    b = np.asarray(b)

    if a.ndim == 1:
        a = a[None, :]
    if b.ndim == 1:
        b = b[None, :]

    num = (a * b).sum(axis=-1)
    den = np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1)
    return num / (den + 1e-8)


def summarize_curves(all_tr, all_mhn, rollout_k):
    """
    all_tr/all_mhn: [N, steps+1]
    """
    mean_tr = all_tr.mean(0)
    std_tr = all_tr.std(0)

    mean_mhn = all_mhn.mean(0)
    std_mhn = all_mhn.std(0)

    # Exclude step 0 because it's trivially 1.0 (x0 vs x0)
    steps = np.arange(0, rollout_k + 1)

    overall_tr = float(mean_tr[1:].mean())
    overall_mhn = float(mean_mhn[1:].mean())

    return {
        "steps": steps.tolist(),
        "transformer": {
            "per_step_mean": mean_tr.tolist(),
            "per_step_std": std_tr.tolist(),
            "overall_mean_step1_to_k": overall_tr,
        },
        "mhn": {
            "per_step_mean": mean_mhn.tolist(),
            "per_step_std": std_mhn.tolist(),
            "overall_mean_step1_to_k": overall_mhn,
        },
        "student_minus_teacher": {
            "per_step_mean_gap": (mean_tr - mean_mhn).tolist(),
            "overall_gap_step1_to_k": float(overall_tr - overall_mhn),
        }
    }


# --------------------------------------------------------------------
# Main
# --------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--val_root", required=True)
    ap.add_argument("--transformer_path", required=True)
    ap.add_argument("--mhn_path", required=True)
    ap.add_argument("--key", default="tubelet_embeddings")
    ap.add_argument("--rollout_k", type=int, default=20)
    ap.add_argument("--num_videos", type=int, default=20)

    # optional output controls
    ap.add_argument("--save_json", action="store_true")
    ap.add_argument("--json_path", default="rollout_generalization_metrics.json")
    ap.add_argument("--fig_path", default="rollout_generalization.png")

    args = ap.parse_args()

    # -------------------------- Load Transformer --------------------------
    model = TransformerPredictor(
        d_model=768,
        nhead=8,
        num_layers=2
    ).cpu()

    state = torch.load(args.transformer_path, map_location="cpu")
    model.load_state_dict(state)
    print("Loaded transformer:", args.transformer_path)

    # ----------------------------- Load MHN -------------------------------
    M = np.load(args.mhn_path)
    print("Loaded MHN:", args.mhn_path)

    # ----------------------- Load Validation Videos -----------------------
    val_paths = sorted(Path(args.val_root).rglob("*.pt"))[:args.num_videos]
    print(f"Found {len(val_paths)} candidate unseen videos")

    all_transformer = []
    all_mhn = []

    used = 0
    skipped = 0

    need = args.rollout_k + 1  # because you are indexing gt[:k+1]

    for p in val_paths:
        try:
            x = load_tubelet_embeddings(p, key=args.key)
            x = l2_normalize(x)

            if x.shape[0] < need:
                skipped += 1
                continue

            x0 = x[0]
            gt = x[:args.rollout_k + 1]

            tr = rollout_transformer(model, x0, args.rollout_k)
            mhn = rollout_mhn(M, x0, args.rollout_k)

            c_tr = cosine(tr, gt)     # [k+1]
            c_mhn = cosine(mhn, gt)

            all_transformer.append(c_tr)
            all_mhn.append(c_mhn)
            used += 1

        except Exception:
            skipped += 1
            continue

    if used == 0:
        print("ERROR: No valid videos for evaluation.")
        return

    all_transformer = np.stack(all_transformer)  # [N, k+1]
    all_mhn = np.stack(all_mhn)

    # -------------------------- Metrics --------------------------
    metrics = summarize_curves(all_transformer, all_mhn, args.rollout_k)

    print("\n=== ROLLOUT GENERALIZATION METRICS ===")
    print(f"Key: {args.key}")
    print(f"Rollout k: {args.rollout_k}")
    print(f"Used videos: {used}")
    print(f"Skipped videos: {skipped}")

    mean_tr = np.array(metrics["transformer"]["per_step_mean"])
    std_tr  = np.array(metrics["transformer"]["per_step_std"])
    mean_m  = np.array(metrics["mhn"]["per_step_mean"])
    std_m   = np.array(metrics["mhn"]["per_step_std"])

    print("\nPer-step cosine (mean ± std):")
    for s in range(args.rollout_k + 1):
        print(
            f"  step {s}: "
            f"Transformer={mean_tr[s]:.4f}±{std_tr[s]:.4f} | "
            f"MHN={mean_m[s]:.4f}±{std_m[s]:.4f}"
        )

    print("\nOverall mean cosine (steps 1..k):")
    print(f"  Transformer: {metrics['transformer']['overall_mean_step1_to_k']:.4f}")
    print(f"  MHN        : {metrics['mhn']['overall_mean_step1_to_k']:.4f}")
    print(f"  Gap (T - M): {metrics['student_minus_teacher']['overall_gap_step1_to_k']:.4f}")

    # -------------------------- Plot --------------------------
    plt.figure(figsize=(8, 5))
    plt.plot(mean_tr, label="Transformer (student)", marker="o")
    plt.plot(mean_m, label="MHN (teacher)", linestyle="--", marker="o")
    plt.xlabel("rollout step")
    plt.ylabel("cosine similarity to ground truth")
    plt.title("Generalization on Unseen Videos")
    plt.legend()
    plt.grid(True)
    plt.ylim(0.9, 1.0)  # tighter view for this regime
    plt.tight_layout()
    plt.savefig(args.fig_path)
    print(f"\nSaved figure: {args.fig_path}")

    # -------------------------- Save JSON --------------------------
    if args.save_json:
        out = {
            "val_root": args.val_root,
            "transformer_path": args.transformer_path,
            "mhn_path": args.mhn_path,
            "key": args.key,
            "rollout_k": args.rollout_k,
            "num_videos_requested": args.num_videos,
            "videos_used": used,
            "videos_skipped": skipped,
            "metrics": metrics,
        }
        with open(args.json_path, "w") as f:
            json.dump(out, f, indent=2)
        print(f"Saved metrics JSON: {args.json_path}")


if __name__ == "__main__":
    main()
