import os
import argparse
from pathlib import Path
import torch
from transformers import VideoMAEVideoProcessor, VideoMAEModel
from einops import rearrange
from tqdm import tqdm


def extract_video_embedding(video_path, model, processor, device="cpu"):
    """
    Returns per-frame embeddings: shape [T, D]
    """
    # Load and preprocess video directly from .mp4
    inputs = processor(video_path, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model(**inputs)

    # tokens: [1, num_tokens, D]
    tokens = outputs.last_hidden_state.squeeze(0)

    # Extract T from pixel_values: [B, C, T, H, W]
    T = inputs["pixel_values"].shape[2]

    # Number of patches per frame
    patches = tokens.shape[0] // T

    # Rearrange: (T * patches, D) → (T, patches, D)
    tokens = rearrange(tokens, "(t p) d -> t p d", t=T, p=patches)

    # Mean-pool patches to get one embedding per frame
    frame_embeddings = tokens.mean(dim=1)  # [T, D]

    return frame_embeddings.cpu()  # save on CPU


def get_all_videos(input_dir):
    """
    Recursively find ALL .mp4 videos inside input_dir.
    """
    input_dir = Path(input_dir)
    return sorted(input_dir.rglob("*.mp4"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", required=True,
                        help="Directory containing video folders (e.g., data/raw/train)")
    parser.add_argument("--output_dir", required=True,
                        help="Directory where embeddings will be saved")
    parser.add_argument("--device", default="cuda", help="cuda or cpu")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load VideoMAE components
    device = args.device if torch.cuda.is_available() and args.device == "cuda" else "cpu"
    processor = VideoMAEVideoProcessor.from_pretrained("MCG-NJU/videomae-base")
    model = VideoMAEModel.from_pretrained("MCG-NJU/videomae-base").to(device)
    model.eval()

    # Get list of all videos
    videos = get_all_videos(input_dir)
    print(f"Found {len(videos)} videos in {input_dir}")

    # Process all videos
    for video_path in tqdm(videos, desc="Extracting embeddings"):
        video_id = video_path.stem  # e.g., video_00000
        save_path = output_dir / f"{video_id}.pt"

        if save_path.exists():
            continue  # skip already processed videos

        try:
            emb = extract_video_embedding(
                video_path, model, processor, device=device
            )
            torch.save(emb, save_path)
        except Exception as e:
            print(f"[ERROR] Failed on {video_path}: {e}")
            continue

    print(f"Done. Saved embeddings to {output_dir}")


if __name__ == "__main__":
    main()
