from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Optional, Tuple
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

import numpy as np

from minesweeper_ai.dataset import encode_board
from minesweeper_ai.env import UNKNOWN, MinesweeperEnv
from minesweeper_ai.solver import solve_step, unknown_cells

try:
    import torch
    from minesweeper_ai.model import MinesweeperCNN
except Exception:
    torch = None
    MinesweeperCNN = None


def load_model(path: Optional[str], device):
    if not path:
        return None, None
    if torch is None:
        raise RuntimeError('PyTorch is not installed, cannot load model')

    ckpt = torch.load(path, map_location=device)
    model = MinesweeperCNN(
        in_channels=int(ckpt.get('in_channels', 12)),
        hidden=int(ckpt.get('hidden', 64)),
    ).to(device)
    model.load_state_dict(ckpt['model_state'])
    model.eval()
    return model, ckpt


def model_risk_map(model, board, remaining_mines, total_cells, device):
    if torch is None:
        raise RuntimeError('PyTorch is required for model inference')
    with torch.no_grad():
        x = encode_board(board, remaining_mines, total_cells)[None]
        xt = torch.from_numpy(x).to(device)
        logits = model(xt)[0]
        return torch.sigmoid(logits).detach().cpu().numpy()


def choose_model_guess(env: MinesweeperEnv, model, probs: Dict[Tuple[int, int], float], device):
    board = env.observe()
    risks = model_risk_map(model, board, env.remaining_mines_estimate(), env.width * env.height, device)

    best_cell = None
    best_risk = float('inf')

    for r, c in unknown_cells(board):
        model_risk = float(risks[r, c])
        if (r, c) in probs:
            risk = 0.65 * float(probs[(r, c)]) + 0.35 * model_risk
        else:
            risk = model_risk

        if risk < best_risk:
            best_risk = risk
            best_cell = (r, c)

    return best_cell


def play_one_game(width, height, mines, seed, model=None, device=None, max_component_cells=20):
    env = MinesweeperEnv(width, height, mines, seed=seed)
    env.open_cell(height // 2, width // 2)

    steps = 0
    guesses = 0

    while env.alive and not env.won and steps < width * height * 4:
        board = env.observe()
        move = solve_step(board, max_component_cells=max_component_cells, force_guess=(model is None))

        if move.flag_cells:
            for r, c in move.flag_cells:
                if env.observe()[r, c] == UNKNOWN:
                    env.toggle_flag(r, c)

        elif move.open_cells and not move.guessed:
            r, c = move.open_cells[0]
            env.open_cell(r, c)

        else:
            guesses += 1
            if model is not None:
                cell = choose_model_guess(env, model, move.probabilities, device)
            else:
                cell = move.open_cells[0] if move.open_cells else None

            if cell is None:
                break
            env.open_cell(*cell)

        steps += 1

    return env.won, steps, guesses


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument('--width', type=int, default=30)
    p.add_argument('--height', type=int, default=16)
    p.add_argument('--mines', type=int, default=99)
    p.add_argument('--games', type=int, default=200)
    p.add_argument('--seed', type=int, default=123)
    p.add_argument('--model', type=str, default=None)
    p.add_argument('--max-component-cells', type=int, default=20)
    args = p.parse_args()

    device = None
    model = None

    if args.model:
        if torch is None:
            raise RuntimeError('PyTorch is required for --model')
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model, _ = load_model(args.model, device)
        print(f'loaded model: {args.model} on {device}')

    rng = np.random.default_rng(args.seed)
    wins = 0
    steps_total = 0
    guesses_total = 0

    for _ in range(args.games):
        seed = int(rng.integers(0, 2**31 - 1))
        won, steps, guesses = play_one_game(
            args.width,
            args.height,
            args.mines,
            seed,
            model=model,
            device=device,
            max_component_cells=args.max_component_cells,
        )
        wins += int(won)
        steps_total += steps
        guesses_total += guesses

    print({
        'games': args.games,
        'wins': wins,
        'win_rate': wins / args.games,
        'avg_steps': steps_total / args.games,
        'avg_guesses': guesses_total / args.games,
        'model': args.model or 'none',
    })


if __name__ == '__main__':
    main()
