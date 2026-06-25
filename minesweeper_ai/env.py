from __future__ import annotations

import random
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

UNKNOWN = -1
FLAG = -2


@dataclass
class StepResult:
    alive: bool
    won: bool
    opened_count: int
    hit_mine: bool = False


class MinesweeperEnv:
    """Local Minesweeper environment.

    The minefield is generated after the first click. The first clicked cell
    and its 8-neighbourhood are safe by default, which is close to many modern
    Minesweeper implementations and makes training data less noisy.
    """

    def __init__(
        self,
        width: int = 30,
        height: int = 16,
        mines: int = 99,
        seed: Optional[int] = None,
        safe_first_radius: bool = True,
    ) -> None:
        if width < 5 or height < 5:
            raise ValueError('width and height must be at least 5')
        if mines < 1 or mines >= width * height:
            raise ValueError('invalid mine count')

        self.width = int(width)
        self.height = int(height)
        self.mine_count = int(mines)
        self.safe_first_radius = bool(safe_first_radius)
        self.rng = random.Random(seed)
        self.reset()

    def reset(self) -> np.ndarray:
        self._mines = np.zeros((self.height, self.width), dtype=bool)
        self._opened = np.zeros((self.height, self.width), dtype=bool)
        self._flags = np.zeros((self.height, self.width), dtype=bool)
        self._generated = False
        self.alive = True
        self.won = False
        self.opened_safe_count = 0
        return self.observe()

    @property
    def mines(self) -> np.ndarray:
        return self._mines.copy()

    @property
    def flags(self) -> np.ndarray:
        return self._flags.copy()

    @property
    def opened(self) -> np.ndarray:
        return self._opened.copy()

    def in_bounds(self, r: int, c: int) -> bool:
        return 0 <= r < self.height and 0 <= c < self.width

    def neighbors(self, r: int, c: int) -> List[Tuple[int, int]]:
        out = []
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0:
                    continue
                nr, nc = r + dr, c + dc
                if self.in_bounds(nr, nc):
                    out.append((nr, nc))
        return out

    def _place_mines(self, first_r: int, first_c: int) -> None:
        banned = {(first_r, first_c)}
        if self.safe_first_radius:
            banned.update(self.neighbors(first_r, first_c))

        cells = [
            (r, c)
            for r in range(self.height)
            for c in range(self.width)
            if (r, c) not in banned
        ]
        if len(cells) < self.mine_count:
            raise ValueError('too many mines for safe-first setting')

        self.rng.shuffle(cells)
        for r, c in cells[: self.mine_count]:
            self._mines[r, c] = True
        self._generated = True

    def adjacent_mines(self, r: int, c: int) -> int:
        return sum(1 for nr, nc in self.neighbors(r, c) if self._mines[nr, nc])

    def observe(self) -> np.ndarray:
        board = np.full((self.height, self.width), UNKNOWN, dtype=np.int8)
        board[self._flags] = FLAG

        for r in range(self.height):
            for c in range(self.width):
                if self._opened[r, c]:
                    board[r, c] = self.adjacent_mines(r, c)

        return board

    def remaining_mines_estimate(self) -> int:
        return self.mine_count - int(self._flags.sum())

    def toggle_flag(self, r: int, c: int) -> None:
        if not self.alive or self.won:
            return
        if not self.in_bounds(r, c):
            return
        if self._opened[r, c]:
            return
        self._flags[r, c] = not self._flags[r, c]

    def open_cell(self, r: int, c: int) -> StepResult:
        if not self.alive or self.won:
            return StepResult(self.alive, self.won, self.opened_safe_count)
        if not self.in_bounds(r, c):
            return StepResult(self.alive, self.won, self.opened_safe_count)
        if self._flags[r, c] or self._opened[r, c]:
            return StepResult(self.alive, self.won, self.opened_safe_count)

        if not self._generated:
            self._place_mines(r, c)

        if self._mines[r, c]:
            self.alive = False
            return StepResult(False, False, self.opened_safe_count, hit_mine=True)

        self._flood_open(r, c)
        self._check_win()
        return StepResult(self.alive, self.won, self.opened_safe_count)

    def _flood_open(self, r: int, c: int) -> None:
        stack = [(r, c)]
        while stack:
            cr, cc = stack.pop()
            if self._opened[cr, cc] or self._flags[cr, cc]:
                continue
            if self._mines[cr, cc]:
                continue

            self._opened[cr, cc] = True
            self.opened_safe_count += 1

            if self.adjacent_mines(cr, cc) == 0:
                for nr, nc in self.neighbors(cr, cc):
                    if not self._opened[nr, nc] and not self._flags[nr, nc]:
                        stack.append((nr, nc))

    def chord(self, r: int, c: int) -> None:
        if not self._opened[r, c]:
            return
        number = self.adjacent_mines(r, c)
        flag_count = sum(1 for nr, nc in self.neighbors(r, c) if self._flags[nr, nc])
        if flag_count != number:
            return
        for nr, nc in self.neighbors(r, c):
            if not self._opened[nr, nc] and not self._flags[nr, nc]:
                self.open_cell(nr, nc)

    def _check_win(self) -> None:
        safe_total = self.width * self.height - self.mine_count
        if self.opened_safe_count >= safe_total:
            self.won = True
            self.alive = False

    def hidden_is_mine(self, r: int, c: int) -> bool:
        return bool(self._mines[r, c])
