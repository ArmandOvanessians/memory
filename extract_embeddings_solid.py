#!/usr/bin/env python3

"""
VideoMAE embedding extraction with robust temporal sampling + windowed inference.

Why windowed?
- HF VideoMAE checkpoints (e.g., videomae-base) are trained with a fixed
  num_frames (typically 16). Positional embeddings assume that length.
- Feeding more frames than the checkpoint expects will cause shape mismatches.

What this script does:
- Recursively finds .mp4 files under --input_dir
- Samples frames either:
    (A) FPS-based: --sample_fps (recommended for sequential pipelines)
    (B) Uniform-count: --num_frames (legacy mode)
- Groups sampled frames into fixed-length windows equal to the model's
  expected num_frames (or --window_size override).
- Runs VideoMAE per window and concatenates temporal embeddings.

Saves one .pt per video containing:
- clip_tubelet_embeddings: (num_windows, t_per_window, D)
- tubelet_embeddings:      (num_windows * t_per_window, D)  [flattened]
- clip_video_embeddings:   (num_windows, D)
- video_embedding:         (D,)  [mean over all tubelets]
- metadata
"""

import argparse
from pathlib import Path

import numpy as np
import torch
from decord import VideoReader
from tqdm import tqdm
from transformers import VideoMAEImageProcessor, VideoMAEModel


def _safe_get_fps(vr: VideoReader):
    try:
        fps = float(vr.get_avg_fps())
        if fps <= 0 or np.isnan(fps):
            return None
        return fps
    except Exception:
        return None


def _make_indices_fps_based(total_frames, fps, sample_fps, max_frames=None):
    stride = max(int(round(fps / sample_fps)), 1)
    indices = np.arange(0, total_frames, stride, dtype=int)
    if max_frames is not None:
        indices = indices[:max_frames]
    return indices


def _make_indices_uniform_count(total_frames, num_frames):
    if num_frames <= 0:
        raise ValueError("num_frames must be > 0.")
    return np.linspace(0, total_frames - 1, num_frames).astype(int)


def _chunk_indices(indices, window_size, window_stride):
    """
    Make sliding windows over a 1D indices array.
    Drops last partial window by default.
    """
    windows = []
    n = len(indices)
    start = 0
    while start + window_size <= n:
        windows.append(indices[start:start + window_size])
        start += window_stride
    return windows


def _compute_window_embeddings(frames, model, processor, device):
    """
    frames: numpy array (T, H, W, 3) for exactly T == window_size
    Returns:
      tubelet_embeddings: (t_tokens, D)
      clip_embedding: (D,)
    """
    inputs = processor(list(frames), return_tensors="pt")
    pixel_values = inputs["pixel_values"]  # (T, C, H, W)

    if pixel_values.ndim == 4:
        pixel_values = pixel_values.unsqueeze(0)  # (1, T, C, H, W)

    pixel_values = pixel_values.to(device)

    with torch.no_grad():
        outputs = model(pixel_values=pixel_values)

    tokens = outputs.last_hidden_state[0]  # (seq_len, D)

    tube = getattr(model.config, "tubelet_size", 2)
    T = pixel_values.shape[1]

    if T % tube != 0:
        raise RuntimeError(f"Window T={T} not divisible by tubelet_size={tube}")

    t_tokens = T // tube

    image_size = getattr(model.config, "image_size", 224)
    patch_size = getattr(model.config, "patch_size", 16)
    num_patches_per_tubelet = (image_size // patch_size) ** 2

    expected_seq = t_tokens * num_patches_per_tubelet
    if tokens.shape[0] != expected_seq:
        raise RuntimeError(
            f"Unexpected seq_len={tokens.shape[0]}, expected {expected_seq}. "
            f"(t_tokens={t_tokens}, patches={num_patches_per_tubelet})."
        )

    tokens = tokens.reshape(t_tokens, num_patches_per_tubelet, -1)
    tubelet_embeddings = tokens.mean(dim=1)  # (t_tokens, D)
    clip_embedding = tubelet_embeddings.mean(dim=0)  # (D,)

    return tubelet_embeddings.detach().cpu(), clip_embedding.detach().cpu()


def extract_videomae_embeddings(
    video_path,
    model,
    processor,
    device="cpu",
    num_frames=None,
    sample_fps=None,
    max_frames=None,
    window_size=None,
    window_stride=None,
):
    """
    Windowed extraction returning a richer sequential structure.

    Returns dict with:
      - clip_tubelet_embeddings: (W, t_per_window, D)
      - tubelet_embeddings:      (W*t_per_window, D)
      - clip_video_embeddings:   (W, D)
      - video_embedding:         (D,)
      - metadata
    """
    vr = VideoReader(str(video_path))
    total_frames = len(vr)
    if total_frames == 0:
        raise ValueError(f"Video {video_path} has no frames.")

    fps = _safe_get_fps(vr)

    # Model expected window length
    model_frames = getattr(model.config, "num_frames", 16)
    if window_size is None:
        window_size = model_frames

    if window_stride is None:
        window_stride = window_size  # default: non-overlapping windows

    if window_size <= 0 or window_stride <= 0:
        raise ValueError("window_size and window_stride must be > 0.")

    # Build sampled indices
    sampling_mode = None
    if sample_fps is not None and fps is not None:
        indices = _make_indices_fps_based(total_frames, fps, sample_fps, max_frames=max_frames)
        sampling_mode = "fps"
    else:
        if num_frames is None:
            # Default to enough frames for at least 1 window if possible
            num_frames = max(model_frames, getattr(model.config, "num_frames", 16))
        indices = _make_indices_uniform_count(total_frames, num_frames)
        sampling_mode = "uniform_count"

    indices = np.asarray(indices, dtype=int)

    # If too short for even one window, fall back to uniform window-sized sampling
    if len(indices) < window_size:
        # resample exactly one window uniformly over the full video
        indices = _make_indices_uniform_count(total_frames, window_size)
        indices = np.asarray(indices, dtype=int)
        sampling_mode = sampling_mode + "_fallback_one_window"

    windows = _chunk_indices(indices, window_size=window_size, window_stride=window_stride)

    if len(windows) == 0:
        raise ValueError(
            f"No full windows could be formed for {video_path}. "
            f"len(indices)={len(indices)}, window_size={window_size}."
        )

    clip_tubelets = []
    clip_embs = []

    for w_idx in windows:
        frames = vr.get_batch(w_idx).asnumpy()
        t_emb, c_emb = _compute_window_embeddings(frames, model, processor, device)
        clip_tubelets.append(t_emb)
        clip_embs.append(c_emb)

    # Stack into tensors
    # Each tubelet embedding has shape (t_per_window, D)
    clip_tubelet_embeddings = torch.stack(clip_tubelets, dim=0)  # (W, t_per_window, D)
    clip_video_embeddings = torch.stack(clip_embs, dim=0)        # (W, D)

    # Flatten tubelets into one long sequence
    tubelet_embeddings = clip_tubelet_embeddings.reshape(-1, clip_tubelet_embeddings.shape[-1])

    # Global
    video_embedding = tubelet_embeddings.mean(dim=0)

    # Best-effort timestamps for each sampled index
    sampled_timestamps_sec = (indices / fps).tolist() if fps is not None else None

    out = {
        "clip_tubelet_embeddings": clip_tubelet_embeddings,
        "tubelet_embeddings": tubelet_embeddings,
        "clip_video_embeddings": clip_video_embeddings,
        "video_embedding": video_embedding,
        "metadata": {
            "video_path": str(video_path),
            "total_frames_in_file": int(total_frames),
            "fps_estimate": fps,
            "sampling_mode": sampling_mode,
            "sample_fps": float(sample_fps) if sample_fps is not None else None,
            "max_frames": int(max_frames) if max_frames is not None else None,
            "num_frames_arg": int(num_frames) if num_frames is not None else None,
            "sampled_indices": indices.tolist(),
            "sampled_timestamps_sec": sampled_timestamps_sec,
            "window_size": int(window_size),
            "window_stride": int(window_stride),
            "num_windows": int(len(windows)),
            "tubelet_size": int(getattr(model.config, "tubelet_size", 2)),
            "t_per_window": int(window_size // getattr(model.config, "tubelet_size", 2)),
            "image_size": int(getattr(model.config, "image_size", 224)),
            "patch_size": int(getattr(model.config, "patch_size", 16)),
            "hidden_size": int(tubelet_embeddings.shape[-1]),
            "model_name": getattr(model.config, "_name_or_path", "unknown"),
        }
    }
    return out


def get_all_videos(input_dir):
    input_dir = Path(input_dir)
    return sorted(input_dir.rglob("*.mp4"))


def _safe_video_id(video_path: Path, input_dir: Path):
    """
    Collision-proof id: relative path with separators replaced.
    """
    rel = video_path.relative_to(input_dir).with_suffix("")
    return str(rel).replace("/", "__")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", required=True, type=str)
    parser.add_argument("--output_dir", required=True, type=str)
    parser.add_argument("--device", default="cuda", type=str)

    # Sampling controls
    parser.add_argument("--num_frames", type=int, default=None,
                        help="Uniform-count sampling across the whole video (legacy/fallback).")
    parser.add_argument("--sample_fps", type=float, default=None,
                        help="Preferred for sequential work. Sampling rate in frames/sec.")
    parser.add_argument("--max_frames", type=int, default=None,
                        help="Cap the number of sampled frames (mainly for --sample_fps).")

    # Window controls
    parser.add_argument("--window_size", type=int, default=None,
                        help="Frames per window. Default = model.config.num_frames (usually 16).")
    parser.add_argument("--window_stride", type=int, default=None,
                        help="Stride in sampled-index space. Default = window_size (no overlap).")

    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = args.device if (args.device == "cuda" and torch.cuda.is_available()) else "cpu"

    processor = VideoMAEImageProcessor.from_pretrained("MCG-NJU/videomae-base")
    model = VideoMAEModel.from_pretrained("MCG-NJU/videomae-base").to(device).eval()

    videos = get_all_videos(input_dir)
    print(f"Found {len(videos)} videos in {input_dir}")

    for video_path in tqdm(videos, desc="Extracting embeddings"):
        video_id = _safe_video_id(video_path, input_dir)
        save_path = output_dir / f"{video_id}.pt"
        save_path.parent.mkdir(parents=True, exist_ok=True)

        if save_path.exists():
            continue

        try:
            emb = extract_videomae_embeddings(
                video_path=video_path,
                model=model,
                processor=processor,
                device=device,
                num_frames=args.num_frames,
                sample_fps=args.sample_fps,
                max_frames=args.max_frames,
                window_size=args.window_size,
                window_stride=args.window_stride,
            )
            torch.save(emb, save_path)

        except Exception as e:
            print(f"[ERROR] Failed on {video_path}: {e}")

    print(f"Done. Saved embeddings to {output_dir}")


if __name__ == "__main__":
    main()
