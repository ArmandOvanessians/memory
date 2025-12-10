#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F


# ---------------------------------------------------------
# Dataset
# ---------------------------------------------------------
class ReplaySequenceDataset(Dataset):
    """
    Loads sequences of shape [k+1, D]
    For training, we use pairs (x_t, x_{t+1}).
    """
    def __init__(self, shard_paths, mode="real"):
        self.samples = []
        self.mode = mode

        for p in shard_paths:
            data = np.load(p, allow_pickle=True)
            seqs = data[mode]   # shape [N, k+1, D]
            self.samples.append(seqs)

        self.samples = np.concatenate(self.samples, axis=0)
        self.samples = self.samples.astype(np.float32)

    def __len__(self):
        return self.samples.shape[0]

    def __getitem__(self, idx):
        seq = self.samples[idx]  # [k+1, D]

        x_t = seq[:-1]  # [k, D]
        y_t = seq[1:]   # [k, D]

        return torch.tensor(x_t), torch.tensor(y_t)


# ---------------------------------------------------------
# Transformer model
# ---------------------------------------------------------
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=32):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2) * (-np.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)

        self.pe = pe

    def forward(self, x):
        return x + self.pe[:x.size(1)].to(x.device)


class TransformerPredictor(nn.Module):
    def __init__(self, d_model=768, nhead=8, num_layers=2, dim_feedforward=2048, dropout=0.1):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=dim_feedforward, dropout=dropout,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.pos = PositionalEncoding(d_model)

        # Map output of transformer to next-state prediction
        self.head = nn.Linear(d_model, d_model)

    def forward(self, x):
        x = self.pos(x)
        h = self.encoder(x)
        y = self.head(h)
        y = F.normalize(y, dim=-1)
        return y


# ---------------------------------------------------------
# Loss function: cosine similarity
# ---------------------------------------------------------
def cosine_loss(pred, target):
    pred_n = F.normalize(pred, dim=-1)
    tgt_n = F.normalize(target, dim=-1)
    cos = (pred_n * tgt_n).sum(dim=-1)
    return 1.0 - cos.mean()


# ---------------------------------------------------------
# Training loop
# ---------------------------------------------------------
def train_epoch(model, loader, opt, device):
    model.train()
    total = 0.0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        pred = model(x)
        loss = cosine_loss(pred, y)

        opt.zero_grad()
        loss.backward()
        opt.step()

        total += loss.item()
    return total / len(loader)


@torch.no_grad()
def eval_epoch(model, loader, device):
    model.eval()
    total = 0.0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        pred = model(x)
        loss = cosine_loss(pred, y)
        total += loss.item()
    return total / len(loader)


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--replay_root", required=True)
    ap.add_argument("--mode", default="real", choices=["real", "replay", "mixed"])
    ap.add_argument("--out_dir", required=True)

    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--val_split", type=float, default=0.1)

    ap.add_argument("--d_model", type=int, default=768)
    ap.add_argument("--nhead", type=int, default=8)
    ap.add_argument("--layers", type=int, default=2)

    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load dataset shards
    shards = sorted(Path(args.replay_root).glob("replay_shard_*.npz"))
    n_val = max(1, int(len(shards) * args.val_split))

    train_shards = shards[:-n_val]
    val_shards = shards[-n_val:]

    train_set = ReplaySequenceDataset(train_shards, mode=args.mode)
    val_set = ReplaySequenceDataset(val_shards, mode=args.mode)

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False)

    # Model
    model = TransformerPredictor(
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.layers
    ).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)

    # Training
    history = {"train_loss": [], "val_loss": []}

    for epoch in range(args.epochs):
        tr = train_epoch(model, train_loader, opt, device)
        val = eval_epoch(model, val_loader, device)
        history["train_loss"].append(tr)
        history["val_loss"].append(val)

        print(f"Epoch {epoch+1}/{args.epochs}  train={tr:.4f}  val={val:.4f}")

        # Save checkpoint each epoch
        torch.save(model.state_dict(), out_dir / f"model_epoch{epoch+1}.pt")

    # Save training metadata
    with open(out_dir / "train_config.json", "w") as f:
        json.dump(vars(args), f, indent=2)

    np.save(out_dir / "loss_history.npy", history)
    print("Training complete.")


if __name__ == "__main__":
    main()
