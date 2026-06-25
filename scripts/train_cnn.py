from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from minesweeper_ai.model import MinesweeperCNN


def masked_bce_loss(logits: torch.Tensor, target: torch.Tensor, pos_weight: torch.Tensor) -> torch.Tensor:
    b, h, w = logits.shape
    logits_flat = logits.reshape(b, h * w)
    mask = target != -100

    if mask.sum() == 0:
        return logits_flat.sum() * 0

    valid_logits = logits_flat[mask]
    valid_target = target[mask].float()
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    return loss_fn(valid_logits, valid_target)


@torch.no_grad()
def evaluate_loss(model: nn.Module, loader: DataLoader, device: torch.device, pos_weight: torch.Tensor) -> float:
    model.eval()
    losses = []
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = masked_bce_loss(logits, y, pos_weight)
        losses.append(float(loss.item()))
    return float(np.mean(losses)) if losses else 0.0


def load_resume_model(model: MinesweeperCNN, resume_path: str, device: torch.device) -> bool:
    path = Path(resume_path)
    if not path.exists():
        print(f'resume model not found: {path}; training from scratch')
        return False

    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt['model_state'])
    print(f'loaded resume model: {path}')
    return True


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument('--data', type=str, required=True)
    p.add_argument('--epochs', type=int, default=8)
    p.add_argument('--batch-size', type=int, default=64)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--hidden', type=int, default=64)
    p.add_argument('--resume', type=str, default=None, help='existing model checkpoint to continue from')
    p.add_argument('--out', type=str, default='minesweeper_cnn.pt')
    args = p.parse_args()

    data = np.load(args.data, allow_pickle=True)
    X = data['X'].astype(np.float32)
    Y = data['Y'].astype(np.int64)
    width = int(data['width'])
    height = int(data['height'])
    mines = int(data['mines'])

    n = len(X)
    order = np.random.default_rng(0).permutation(n)
    split = max(1, int(n * 0.85))
    train_idx, val_idx = order[:split], order[split:]

    train_ds = TensorDataset(torch.from_numpy(X[train_idx]), torch.from_numpy(Y[train_idx]))
    val_ds = TensorDataset(torch.from_numpy(X[val_idx]), torch.from_numpy(Y[val_idx]))

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size)

    valid_targets = Y[Y != -100]
    pos = max(1, int((valid_targets == 1).sum()))
    neg = max(1, int((valid_targets == 0).sum()))
    pos_weight_value = neg / pos

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = MinesweeperCNN(in_channels=X.shape[1], hidden=args.hidden).to(device)
    resumed = load_resume_model(model, args.resume, device) if args.resume else False
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    pos_weight = torch.tensor(pos_weight_value, dtype=torch.float32, device=device)

    print(f'device: {device}')
    print(f'samples: {n}, train: {len(train_idx)}, val: {len(val_idx)}')
    print(f'pos_weight: {pos_weight_value:.3f}')
    print(f'resumed: {resumed}')

    best_val = float('inf')

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_losses = []

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad(set_to_none=True)
            logits = model(x)
            loss = masked_bce_loss(logits, y, pos_weight)
            loss.backward()
            opt.step()
            train_losses.append(float(loss.item()))

        val_loss = evaluate_loss(model, val_loader, device, pos_weight) if len(val_idx) else 0.0
        train_loss = float(np.mean(train_losses)) if train_losses else 0.0
        print(f'epoch {epoch:02d} | train_loss={train_loss:.4f} | val_loss={val_loss:.4f}')

        if val_loss < best_val or epoch == args.epochs:
            best_val = val_loss
            torch.save(
                {
                    'model_state': model.state_dict(),
                    'width': width,
                    'height': height,
                    'mines': mines,
                    'in_channels': int(X.shape[1]),
                    'hidden': args.hidden,
                    'resumed': resumed,
                    'samples': n,
                    'best_val_loss': best_val,
                },
                args.out,
            )

    print(f'saved model: {Path(args.out).resolve()}')


if __name__ == '__main__':
    main()
