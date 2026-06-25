from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

import numpy as np

from minesweeper_ai.dataset import collect_samples


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument('--existing', type=str, default='data_online.npz')
    p.add_argument('--out', type=str, default='data_online.npz')
    p.add_argument('--width', type=int, default=30)
    p.add_argument('--height', type=int, default=16)
    p.add_argument('--mines', type=int, default=99)
    p.add_argument('--games', type=int, default=200)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--max-component-cells', type=int, default=20)
    p.add_argument('--record-all', action='store_true')
    args = p.parse_args()

    X_new, Y_new, stats_new = collect_samples(
        games=args.games,
        width=args.width,
        height=args.height,
        mines=args.mines,
        seed=args.seed,
        max_component_cells=args.max_component_cells,
        record_only_guesses=not args.record_all,
    )

    existing = Path(args.existing)
    if existing.exists():
        old = np.load(existing, allow_pickle=True)
        X_old = old['X'].astype(np.float32)
        Y_old = old['Y'].astype(np.int64)
        width_old = int(old['width'])
        height_old = int(old['height'])
        mines_old = int(old['mines'])

        if (width_old, height_old, mines_old) != (args.width, args.height, args.mines):
            raise ValueError(
                f'existing dataset board differs: '
                f'{width_old}x{height_old}/{mines_old} vs {args.width}x{args.height}/{args.mines}'
            )

        X = np.concatenate([X_old, X_new], axis=0)
        Y = np.concatenate([Y_old, Y_new], axis=0)
        old_samples = len(X_old)
    else:
        X = X_new
        Y = Y_new
        old_samples = 0

    merged_stats = {
        'old_samples': old_samples,
        'new_samples': len(X_new),
        'total_samples': len(X),
        'new_collection_stats': stats_new,
    }

    out = Path(args.out)
    np.savez_compressed(
        out,
        X=X,
        Y=Y,
        width=args.width,
        height=args.height,
        mines=args.mines,
        stats=np.array([merged_stats], dtype=object),
    )

    print(f'saved merged dataset: {out.resolve()}')
    print(merged_stats)


if __name__ == '__main__':
    main()
