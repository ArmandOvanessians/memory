CLEVERER Processing & Training Pipeline

This guide describes how to preprocess the CLEVERER dataset, extract VideoMAE embeddings, store them using a Modern Hopfield Network (MHN), and build replay sequences for training a Transformer.

Step 1 — Download the Dataset

Download the CLEVR/CLEVERER dataset:

🔗 https://cs.stanford.edu/people/jcjohns/clevr/

Place the dataset in your preferred directory.

Step 2 — Extract VideoMAE Embeddings (Denser)

Run the embedding extraction script on the directory containing the raw CLEVERER videos.

python extract_embedding_denser.py \
  --input_dir CLEV/data/raw/val \
  --output_dir CLEV/data/processed_video_denser_higher_temp/val \
  --device cuda \
  --sample_fps 10 \
  --max_frames 50 \
  --window_size 16 \
  --window_stride 4


Arguments:

--sample_fps: frames per second to sample

--max_frames: maximum number of frames per video

--window_size: temporal window size for VideoMAE

--window_stride: stride between windows

Step 3 — Store Frame Pairs with the Modern Hopfield Network

Use the processed video embeddings to build and store MHN memory patterns:

python train_mhn_videomae.py \
  --processed_root data/processed_video_denser_higher_temp_stitched/train \
  --key stitched_tubelet_embeddings \
  --mode plain \
  --stride 4 \
  --rollout_k 5 \
  --rollout_anchors_per_video 10 \
  --name mhn_videomae_denser_temp_stride4_k5_M.npy

Step 4 — Build Replay Sequences for Transformer Training

Generate replay sequences using the MHN memory matrix:

python build_replay_sequences.py \
  --processed_root data/processed_video_denser_higher_temp_stitched/train \
  --mhn_M_path models/mhn_videomae_denser_temp_stride4_k5_M.npy \
  --out_dir CLEV/replay_denser_stride4_stitched \
  --key stitched_tubelet_embeddings \
  --mode plain \
  --stride 4 \
  --rollout_k 5 \
  --anchors_per_video 5 \
  --save_every 500

Step 5 — Transformer Training

Generate replay sequences using the MHN memory matrix:

python train_transformer.py \                                                                                                              6s
  --replay_root /Entropy/Projects/memory/CLEV/replay_denser_stride4_stitched \
  --mode mixed \
  --out_dir models/transformer_mixed \
  --epochs 20
change mode to real: To evaluete transformer trained on real data. 
