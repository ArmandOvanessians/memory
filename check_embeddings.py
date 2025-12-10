#!/usr/bin/env python3
"""
Sanity-check VideoMAE frame/tubelet embeddings saved as .pt.

What it prints:
- type + shape + dtype
- NaN/Inf checks
- min/max/mean/std
- first few rows (first 8 dims)
- count of consecutive identical embeddings (exact + allclose)
- adjacent cosine similarity stats + preview
- mean L2 step size between adjacent embeddings
- optional guess of repetition pattern

Usage:
  python check_embeddings.py video_00000.pt
  python check_embeddings.py video_00000.pt --show_rows 3 --show_dims 16
"""

import argparse
import math
from pathlib import Path

import torch
import torch.nn.functional as F


# def load_embedding(path: Path):
#     obj = torch.load(path, map_location="cpu")

#     # Handle common save formats:
#     # 1) direct Tensor
#     # 2) dict with plausible keys
#     if isinstance(obj, torch.Tensor):
#         return obj, None

#     if isinstance(obj, dict):
#         # Try common keys
#         for k in ["frame_embeddings", "emb", "embeddings", "tensor", "x"]:
#             if k in obj and isinstance(obj[k], torch.Tensor):
#                 return obj[k], k

#         # If dict of tensors, pick the first tensor-like item
#         for k, v in obj.items():
#             if isinstance(v, torch.Tensor):
#                 return v, k

#     raise TypeError(
#         f"Unsupported .pt content type: {type(obj)}. "
#         "Expected a Tensor or a dict containing a Tensor."
#     )
def load_embedding(path: Path, preferred_key=None):
    obj = torch.load(path, map_location="cpu")

    if isinstance(obj, torch.Tensor):
        return obj, None

    if isinstance(obj, dict):
        # If user explicitly requests a key
        if preferred_key is not None:
            if preferred_key in obj and isinstance(obj[preferred_key], torch.Tensor):
                return obj[preferred_key], preferred_key
            raise KeyError(f"Key '{preferred_key}' not found or not a Tensor.")

        # Try common keys (expanded for new format)
        for k in [
            "tubelet_embeddings",
            "frame_embeddings",
            "clip_tubelet_embeddings",
            "clip_video_embeddings",
            "video_embedding",
            "emb",
            "embeddings",
            "tensor",
            "x",
        ]:
            if k in obj and isinstance(obj[k], torch.Tensor):
                return obj[k], k

        # Fallback: first tensor item
        for k, v in obj.items():
            if isinstance(v, torch.Tensor):
                return v, k

    raise TypeError(
        f"Unsupported .pt content type: {type(obj)}. "
        "Expected a Tensor or a dict containing a Tensor."
    )



def consecutive_equal_counts(emb: torch.Tensor, atol=1e-6, rtol=1e-6):
    exact = 0
    close = 0
    pairs = emb.shape[0] - 1

    for i in range(pairs):
        a = emb[i]
        b = emb[i + 1]
        if torch.equal(a, b):
            exact += 1
        if torch.allclose(a, b, atol=atol, rtol=rtol):
            close += 1

    return exact, close, pairs


def adjacent_cosine_sims(emb: torch.Tensor):
    # Normalize to unit vectors3
    emb_n = F.normalize(emb, dim=-1)
    # cosine between consecutive rows
    sims = (emb_n[:-1] * emb_n[1:]).sum(dim=-1)
    return sims


def guess_repeat_group_size(emb: torch.Tensor, max_group=8):
    """
    Very rough heuristic:
    checks how often embeddings repeat in small groups.
    Returns the group size with the highest "repeat score".
    """
    T = emb.shape[0]
    if T < 4:
        return None, None

    best_g = None
    best_score = -1.0

    for g in range(2, max_group + 1):
        # Score how often positions i and i+1 within each group are identical
        # e.g. for g=2, check pairs (0,1), (2,3), ...
        matches = 0
        comparisons = 0
        for start in range(0, T - g + 1, g):
            for j in range(g - 1):
                a = emb[start + j]
                b = emb[start + j + 1]
                comparisons += 1
                if torch.allclose(a, b, atol=1e-6, rtol=1e-6):
                    matches += 1

        if comparisons == 0:
            continue
        score = matches / comparisons
        if score > best_score:
            best_score = score
            best_g = g

    return best_g, best_score


def main():
    parser = argparse.ArgumentParser(description="Sanity-check saved embeddings.")
    parser.add_argument("pt_path", type=str, help="Path to .pt file")
    parser.add_argument("--show_rows", type=int, default=5, help="How many time steps to preview")
    parser.add_argument("--show_dims", type=int, default=8, help="How many dims to print per preview row")
    parser.add_argument("--atol", type=float, default=1e-6, help="atol for allclose checks")
    parser.add_argument("--rtol", type=float, default=1e-6, help="rtol for allclose checks")
    parser.add_argument(
    "--key",
    type=str,
    default=None,
    help="Which tensor key to inspect (e.g., tubelet_embeddings).",
    )
    args = parser.parse_args()

    path = Path(args.pt_path)
    if not path.exists():
        raise FileNotFoundError(path)

    emb, key = load_embedding(path, preferred_key=args.key)


    print("=" * 80)
    print(f"File: {path}")
    if key is not None:
        print(f"Loaded tensor from dict key: '{key}'")
    print(f"Type: {type(emb)}")

    if emb.ndim != 2:
        print(f"WARNING: Expected 2D (T, D). Got shape {tuple(emb.shape)}")
    else:
        print("Shape (T, D):", tuple(emb.shape))

    print("Dtype:", emb.dtype)
    print("Device:", emb.device)

    # Basic validity checks
    nan_count = torch.isnan(emb).sum().item()
    inf_count = torch.isinf(emb).sum().item()
    print("NaN count:", nan_count)
    print("Inf count:", inf_count)

    # Stats
    emb_float = emb.float()
    print("Min:", emb_float.min().item())
    print("Max:", emb_float.max().item())
    print("Mean:", emb_float.mean().item())
    print("Std:", emb_float.std(unbiased=False).item())

    # Preview
    T = emb.shape[0]
    show_rows = min(args.show_rows, T)
    show_dims = min(args.show_dims, emb.shape[-1])

    print("-" * 80)
    print(f"Preview first {show_rows} time steps (first {show_dims} dims):")
    for i in range(show_rows):
        vals = emb_float[i, :show_dims].tolist()
        print(f"  t={i:02d}  {vals}")

    if T < 2:
        print("Not enough time steps for temporal checks.")
        print("=" * 80)
        return

    # Consecutive equality checks
    exact, close, pairs = consecutive_equal_counts(emb_float, atol=args.atol, rtol=args.rtol)
    print("-" * 80)
    print("Consecutive identical embeddings:")
    print(f"  Exact equal pairs:   {exact} / {pairs}")
    print(f"  Allclose equal pairs (atol={args.atol}, rtol={args.rtol}): {close} / {pairs}")

    # Adjacent cosine similarity
    sims = adjacent_cosine_sims(emb_float).cpu()
    sims_list = sims.tolist()
    print("-" * 80)
    print("Adjacent cosine similarity:")
    print("  First 10:", [round(x, 6) for x in sims_list[:10]])
    print("  Mean:", float(sims.mean().item()))
    print("  Std:", float(sims.std(unbiased=False).item()))
    print("  Min/Max:", float(sims.min().item()), float(sims.max().item()))

    # L2 step size
    diffs = emb_float[1:] - emb_float[:-1]
    step_l2 = diffs.norm(dim=-1)
    print("-" * 80)
    print("Adjacent L2 step size:")
    print("  Mean:", float(step_l2.mean().item()))
    print("  Std:", float(step_l2.std(unbiased=False).item()))
    print("  First 10:", [round(x, 6) for x in step_l2[:10].tolist()])

    # Heuristic group repetition guess
    g, score = guess_repeat_group_size(emb_float, max_group=8)
    print("-" * 80)
    if g is not None:
        print("Heuristic repetition check:")
        print(f"  Likely small repeat group size ~{g} (score={score:.3f})")
        print("  If you used VideoMAE tubelet expansion, a strong score near 1.0")
        print("  at group size 2 may suggest tubelet_size=2 repetition.")
    else:
        print("Heuristic repetition check: not enough data")

    print("=" * 80)
    print("Interpretation tips:")
    print("  - If many consecutive pairs are identical or allclose≈identical,")
    print("    your 'per-frame' embeddings may be repeated tubelet embeddings.")
    print("  - Cosine sims close to 1.0 everywhere => very slow or repeated change.")
    print("  - Non-zero L2 steps with varied cosine sims => temporal signal present.")
    print("  - This script can't prove semantic 'goodness', only consistency + variation.")


if __name__ == "__main__":
    main()
