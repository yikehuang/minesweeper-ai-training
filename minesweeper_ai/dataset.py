from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np

from .env import FLAG, UNKNOWN, MinesweeperEnv
from .solver import solve_step, unknown_cells

CHANNELS = 12


def encode_board(board: np.ndarray, remaining_mines: int, total_cells: int) -> np.ndarray:
    """Encode visible board as 12-channel tensor.

    Channels:
    0: unknown
    1: flag
    2-10: opened number 0-8
    11: remaining mine density, repeated on all cells
    """
    h, w = board.shape
    x = np.zeros((CHANNELS, h, w), dtype=np.float32)

    x[0] = (board == UNKNOWN).astype(np.float32)
    x[1] = (board == FLAG).astype(np.float32)

    for n in range(9):
        x[2 + n] = (board == n).astype(np.float32)

    density = remaining_mines / max(1, total_cells)
    x[11].fill(float(density))
    return x


def target_from_hidden(env: MinesweeperEnv, board: np.ndarray) -> np.ndarray:
    """Return flat target.

    Unknown cells: 1 if mine else 0.
    Non-unknown cells: -100, ignored during loss calculation.
    """
    h, w = board.shape
    y = np.full((h, w), -100, dtype=np.int64)
    hidden = env.mines
    for r, c in unknown_cells(board):
        y[r, c] = 1 if hidden[r, c] else 0
    return y.reshape(-1)


def apply_solver_move(env: MinesweeperEnv, move) -> None:
    for r, c in move.flag_cells:
        obs = env.observe()
        if obs[r, c] == UNKNOWN:
            env.toggle_flag(r, c)

    if move.open_cells:
        r, c = move.open_cells[0]
        env.open_cell(r, c)


def collect_samples(
    games: int,
    width: int,
    height: int,
    mines: int,
    seed: int = 0,
    max_steps: int = 5000,
    max_component_cells: int = 20,
    record_only_guesses: bool = True,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, int]]:
    rng = np.random.default_rng(seed)
    X: List[np.ndarray] = []
    Y: List[np.ndarray] = []
    stats = {
        'games': games,
        'wins': 0,
        'losses': 0,
        'recorded_states': 0,
        'guess_states': 0,
    }

    for _ in range(games):
        env = MinesweeperEnv(width, height, mines, seed=int(rng.integers(0, 2**31 - 1)))
        env.open_cell(height // 2, width // 2)

        for _step in range(max_steps):
            if env.won:
                stats['wins'] += 1
                break
            if not env.alive:
                stats['losses'] += 1
                break

            board = env.observe()
            move = solve_step(board, max_component_cells=max_component_cells, force_guess=True)

            should_record = (not record_only_guesses) or move.guessed
            if should_record:
                X.append(encode_board(board, env.remaining_mines_estimate(), width * height))
                Y.append(target_from_hidden(env, board))
                stats['recorded_states'] += 1
                if move.guessed:
                    stats['guess_states'] += 1

            apply_solver_move(env, move)
        else:
            if env.won:
                stats['wins'] += 1
            elif not env.alive:
                stats['losses'] += 1

    if not X:
        raise RuntimeError('no samples collected; try increasing --games or use --record-all')

    return np.stack(X), np.stack(Y), stats
