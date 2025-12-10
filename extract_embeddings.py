# import os
# import argparse
# from pathlib import Path
# import torch
# import decord
# from decord import VideoReader
# import numpy as np

# #from transformers import VideoMAEVideoProcessor, VideoMAEModel
# from transformers import VideoMAEImageProcessor, VideoMAEModel
# from einops import rearrange
# from tqdm import tqdm


# import decord
# from decord import VideoReader
# import numpy as np

# import numpy as np
# import torch
# from decord import VideoReader
# from einops import rearrange
# from transformers import VideoMAEImageProcessor, VideoMAEModel

# import numpy as np
# import torch
# from decord import VideoReader
# from transformers import VideoMAEImageProcessor, VideoMAEModel


# def extract_videomae_embeddings(
#     video_path,
#     model,
#     processor,
#     device="cpu",
#     num_frames=None,
# ):
#     """
#     Returns a dict with:
#       - frame_embeddings: (T, D)  [approx, by repeating tubelets]
#       - tubelet_embeddings: (T//tube, D)  [faithful to VideoMAE]
#       - video_embedding: (D,)  [global average over tubelets]
#       - metadata
#     """

#     vr = VideoReader(str(video_path))
#     total_frames = len(vr)
#     if total_frames == 0:
#         raise ValueError(f"Video {video_path} has no frames.")

#     # Use model default if not specified (videomae-base typically 16)
#     if num_frames is None:
#         num_frames = getattr(model.config, "num_frames", 16)

#     # Uniform sampling across the whole video
#     indices = np.linspace(0, total_frames - 1, num_frames).astype(int)
#     frames = vr.get_batch(indices).asnumpy()  # (T, H, W, 3)

#     # Preprocess with HF processor
#     inputs = processor(list(frames), return_tensors="pt")
#     pixel_values = inputs["pixel_values"]

#     # Ensure shape is (1, T, C, H, W)
#     if pixel_values.ndim == 4:
#         pixel_values = pixel_values.unsqueeze(0)

#     pixel_values = pixel_values.to(device)

#     with torch.no_grad():
#         outputs = model(pixel_values=pixel_values)

#     # tokens: (seq_len, D)
#     tokens = outputs.last_hidden_state[0]

#     # VideoMAE temporal grouping
#     tube = getattr(model.config, "tubelet_size", 2)
#     T = pixel_values.shape[1]
#     if T % tube != 0:
#         # If this ever happens, we can still handle it,
#         # but it's cleaner to keep T divisible by tube.
#         raise RuntimeError(f"T={T} not divisible by tubelet_size={tube}")

#     t_tokens = T // tube

#     image_size = getattr(model.config, "image_size", 224)
#     patch_size = getattr(model.config, "patch_size", 16)
#     num_patches_per_tubelet = (image_size // patch_size) ** 2

#     expected_seq = t_tokens * num_patches_per_tubelet
#     if tokens.shape[0] != expected_seq:
#         raise RuntimeError(
#             f"Unexpected seq_len={tokens.shape[0]}, expected {expected_seq}. "
#             f"(t_tokens={t_tokens}, patches={num_patches_per_tubelet}). "
#             f"Check num_frames vs checkpoint config."
#         )

#     # Reshape into (t_tokens, spatial_patches, D)
#     tokens = tokens.reshape(t_tokens, num_patches_per_tubelet, -1)

#     # Average spatial patches -> tubelet embeddings (true temporal units)
#     tubelet_embeddings = tokens.mean(dim=1)  # (t_tokens, D)

#     # Approximate per-frame by repeating each tubelet embedding 'tube' times
#     frame_embeddings = tubelet_embeddings.repeat_interleave(tube, dim=0)[:T]  # (T, D)

#     # Global video embedding (simple and robust)
#     video_embedding = tubelet_embeddings.mean(dim=0)  # (D,)

#     out = {
#         "frame_embeddings": frame_embeddings.detach().cpu(),
#         "tubelet_embeddings": tubelet_embeddings.detach().cpu(),
#         "video_embedding": video_embedding.detach().cpu(),
#         "metadata": {
#             "video_path": str(video_path),
#             "total_frames_in_file": int(total_frames),
#             "sampled_num_frames": int(T),
#             "sampled_indices": indices.tolist(),
#             "tubelet_size": int(tube),
#             "t_tokens": int(t_tokens),
#             "image_size": int(image_size),
#             "patch_size": int(patch_size),
#             "hidden_size": int(frame_embeddings.shape[-1]),
#             "model_name": getattr(model.config, "_name_or_path", "unknown"),
#         }
#     }
#     return out




# def get_all_videos(input_dir):
#     """
#     Recursively find ALL .mp4 videos inside input_dir.
#     """
#     input_dir = Path(input_dir)
#     return sorted(input_dir.rglob("*.mp4"))


# def main():
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--input_dir", required=True)
#     parser.add_argument("--output_dir", required=True)
#     parser.add_argument("--device", default="cuda")
#     parser.add_argument("--num_frames", type=int, default=None)
#     args = parser.parse_args()

#     input_dir = Path(args.input_dir)
#     output_dir = Path(args.output_dir)
#     output_dir.mkdir(parents=True, exist_ok=True)

#     device = args.device if torch.cuda.is_available() and args.device == "cuda" else "cpu"

#     processor = VideoMAEImageProcessor.from_pretrained("MCG-NJU/videomae-base")
#     model = VideoMAEModel.from_pretrained("MCG-NJU/videomae-base").to(device)
#     model.eval()

#     videos = get_all_videos(input_dir)
#     print(f"Found {len(videos)} videos in {input_dir}")

#     for video_path in tqdm(videos, desc="Extracting embeddings"):
#         video_id = video_path.stem
#         save_path = output_dir / f"{video_id}.pt"
#         if save_path.exists():
#             continue

#         try:
#             emb = extract_videomae_embeddings(
#                 video_path,
#                 model,
#                 processor,
#                 device=device,
#                 num_frames=args.num_frames
#             )
#             torch.save(emb, save_path)
#         except Exception as e:
#             print(f"[ERROR] Failed on {video_path}: {e}")
#             continue

#     print(f"Done. Saved embeddings to {output_dir}")



# if __name__ == "__main__":
#     main()

import argparse
from pathlib import Path

import torch
import numpy as np
from decord import VideoReader
from tqdm import tqdm
from transformers import VideoMAEImageProcessor, VideoMAEModel


def extract_videomae_embeddings(
    video_path,
    model,
    processor,
    device="cpu",
    num_frames=None,
):
    """
    Returns a dict with:
      - frame_embeddings: (T, D)  [approx, by repeating tubelets]
      - tubelet_embeddings: (T//tube, D)  [faithful to VideoMAE]
      - video_embedding: (D,)  [global average over tubelets]
      - metadata
    """
    vr = VideoReader(str(video_path))
    total_frames = len(vr)
    if total_frames == 0:
        raise ValueError(f"Video {video_path} has no frames.")

    if num_frames is None:
        num_frames = getattr(model.config, "num_frames", 16)

    indices = np.linspace(0, total_frames - 1, num_frames).astype(int)
    frames = vr.get_batch(indices).asnumpy()  # (T, H, W, 3)

    inputs = processor(list(frames), return_tensors="pt")
    pixel_values = inputs["pixel_values"]

    # Ensure shape is (1, T, C, H, W)
    if pixel_values.ndim == 4:
        pixel_values = pixel_values.unsqueeze(0)

    pixel_values = pixel_values.to(device)

    with torch.no_grad():
        outputs = model(pixel_values=pixel_values)

    tokens = outputs.last_hidden_state[0]  # (seq_len, D)

    tube = getattr(model.config, "tubelet_size", 2)
    T = pixel_values.shape[1]

    if T % tube != 0:
        raise RuntimeError(f"T={T} not divisible by tubelet_size={tube}")

    t_tokens = T // tube

    image_size = getattr(model.config, "image_size", 224)
    patch_size = getattr(model.config, "patch_size", 16)
    num_patches_per_tubelet = (image_size // patch_size) ** 2

    expected_seq = t_tokens * num_patches_per_tubelet
    if tokens.shape[0] != expected_seq:
        raise RuntimeError(
            f"Unexpected seq_len={tokens.shape[0]}, expected {expected_seq}. "
            f"(t_tokens={t_tokens}, patches={num_patches_per_tubelet}). "
            f"Check num_frames vs checkpoint config."
        )

    tokens = tokens.reshape(t_tokens, num_patches_per_tubelet, -1)

    tubelet_embeddings = tokens.mean(dim=1)  # (t_tokens, D)
    frame_embeddings = tubelet_embeddings.repeat_interleave(tube, dim=0)[:T]  # (T, D)
    video_embedding = tubelet_embeddings.mean(dim=0)  # (D,)

    return {
        "frame_embeddings": frame_embeddings.cpu(),
        "tubelet_embeddings": tubelet_embeddings.cpu(),
        "video_embedding": video_embedding.cpu(),
        "metadata": {
            "video_path": str(video_path),
            "total_frames_in_file": int(total_frames),
            "sampled_num_frames": int(T),
            "sampled_indices": indices.tolist(),
            "tubelet_size": int(tube),
            "t_tokens": int(t_tokens),
            "image_size": int(image_size),
            "patch_size": int(patch_size),
            "hidden_size": int(frame_embeddings.shape[-1]),
            "model_name": getattr(model.config, "_name_or_path", "unknown"),
        }
    }


def get_all_videos(input_dir):
    input_dir = Path(input_dir)
    return sorted(input_dir.rglob("*.mp4"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num_frames", type=int, default=None)
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
        video_id = video_path.stem
        save_path = output_dir / f"{video_id}.pt"

        if save_path.exists():
            continue

        try:
            emb = extract_videomae_embeddings(
                video_path,
                model,
                processor,
                device=device,
                num_frames=args.num_frames,
            )
            torch.save(emb, save_path)

        except Exception as e:
            print(f"[ERROR] Failed on {video_path}: {e}")

    print(f"Done. Saved embeddings to {output_dir}")


if __name__ == "__main__":
    main()
