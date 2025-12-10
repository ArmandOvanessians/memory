Step 1: 
Download the CLEVERER dataset from the following link: 
https://cs.stanford.edu/people/jcjohns/clevr/
Step 2: 
Run extract_embedding_denser.py on the folder where you saved the data: 
Example
> python extract_embedding_denser.py \                                                                                                                                                           6s
  --input_dir CLEV/data/raw/val \
  --output_dir CLEV/data/processed_video_denser_higher_temp/val \
  --device cuda \
  --sample_fps 10 \
  --max_frames 50 \
  --window_size 16 \
  --window_stride 4
Step 3: Store frames (pairs) in the modern hopfield implementation
> python train_mhn_videomae.py \                                                                                                                                                                 6s
  --processed_root data/processed_video_denser_higher_temp_stitched/train \
  --key stitched_tubelet_embeddings \
  --mode plain \
  --stride 4 \
  --rollout_k 5 \
  --rollout_anchors_per_video 10 \
  --name mhn_videomae_denser_temp_stride4_k5_M.npy
Step4: Train Transformer on either the
> python build_replay_sequences.py \                                                                                                                                                             6s
  --processed_root data/processed_video_denser_higher_temp_stitched/train \
  --mhn_M_path models/mhn_videomae_denser_temp_stride4_k5_M.npy \
  --out_dir CLEV/replay_denser_stride4_stitched \
  --key stitched_tubelet_embeddings \
  --mode plain \
  --stride 4 \
  --rollout_k 5 \
  --anchors_per_video 5 \
  --save_every 500
