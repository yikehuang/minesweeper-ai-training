from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

import numpy as np

from minesweeper_ai.dataset import collect_samples


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument('--width', type=int, default=30)
    p.add_argument('--height', type=int, default=16)
    p.add_argument('--mines', type=int, default=99)
    p.add_argument('--games', type=int, default=500)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--max-component-cells', type=int, default=20)
    p.add_argument('--record-all', action='store_true', help='record deterministic states too')
    p.add_argument('--out', type=str, default='minesweeper_dataset.npz')
    args = p.parse_args()

    X, Y, stats = collect_samples(
        games=args.games,
        width=args.width,
        height=args.height,
        mines=args.mines,
        seed=args.seed,
        max_component_cells=args.max_component_cells,
        record_only_guesses=not args.record_all,
    )

    out = Path(args.out)
    np.savez_compressed(
        out,
        X=X,
        Y=Y,
        width=args.width,
        height=args.height,
        mines=args.mines,
        stats=np.array([stats], dtype=object),
    )

    print(f'saved: {out.resolve()}')
    print(f'X shape: {X.shape}, Y shape: {Y.shape}')
    print(stats)


if __name__ == '__main__':
    main()
