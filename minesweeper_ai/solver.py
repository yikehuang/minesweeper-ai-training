from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
from itertools import product
from typing import Dict, Iterable, List, Optional, Set, Tuple

import numpy as np

UNKNOWN = -1
FLAG = -2
Cell = Tuple[int, int]


@dataclass
class SolverMove:
    open_cells: List[Cell]
    flag_cells: List[Cell]
    reason: str
    probabilities: Dict[Cell, float]
    guessed: bool = False


def neighbors(r: int, c: int, h: int, w: int) -> List[Cell]:
    out = []
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w:
                out.append((nr, nc))
    return out


def numbered_cells(board: np.ndarray) -> Iterable[Cell]:
    h, w = board.shape
    for r in range(h):
        for c in range(w):
            if 0 <= int(board[r, c]) <= 8:
                yield r, c


def unknown_cells(board: np.ndarray) -> Iterable[Cell]:
    h, w = board.shape
    for r in range(h):
        for c in range(w):
            if int(board[r, c]) == UNKNOWN:
                yield r, c


def build_constraints(board: np.ndarray) -> List[Tuple[Set[Cell], int]]:
    h, w = board.shape
    constraints = []
    for r, c in numbered_cells(board):
        number = int(board[r, c])
        unknowns: Set[Cell] = set()
        flags = 0

        for nr, nc in neighbors(r, c, h, w):
            v = int(board[nr, nc])
            if v == UNKNOWN:
                unknowns.add((nr, nc))
            elif v == FLAG:
                flags += 1

        remaining = number - flags
        if unknowns and 0 <= remaining <= len(unknowns):
            constraints.append((unknowns, remaining))

    return constraints


def basic_deduction(board: np.ndarray) -> Tuple[Set[Cell], Set[Cell]]:
    safe: Set[Cell] = set()
    mines: Set[Cell] = set()

    for cells, mine_count in build_constraints(board):
        if mine_count == 0:
            safe.update(cells)
        elif mine_count == len(cells):
            mines.update(cells)

    return safe, mines


def subset_deduction(board: np.ndarray) -> Tuple[Set[Cell], Set[Cell]]:
    safe: Set[Cell] = set()
    mines: Set[Cell] = set()
    constraints = build_constraints(board)

    for a_cells, a_mines in constraints:
        for b_cells, b_mines in constraints:
            if a_cells == b_cells:
                continue

            if a_cells.issubset(b_cells):
                diff = b_cells - a_cells
                diff_mines = b_mines - a_mines
                if not diff:
                    continue
                if diff_mines == 0:
                    safe.update(diff)
                elif diff_mines == len(diff):
                    mines.update(diff)

    return safe, mines


def split_components(
    constraints: List[Tuple[Set[Cell], int]]
) -> List[Tuple[Set[Cell], List[Tuple[Set[Cell], int]]]]:
    graph: Dict[Cell, Set[Cell]] = defaultdict(set)

    for cells, _ in constraints:
        cells_list = list(cells)
        for cell in cells_list:
            graph[cell]
        for i in range(len(cells_list)):
            for j in range(i + 1, len(cells_list)):
                a, b = cells_list[i], cells_list[j]
                graph[a].add(b)
                graph[b].add(a)

    visited: Set[Cell] = set()
    components = []

    for start in graph:
        if start in visited:
            continue
        q = deque([start])
        visited.add(start)
        comp_cells: Set[Cell] = set()

        while q:
            cell = q.popleft()
            comp_cells.add(cell)
            for nxt in graph[cell]:
                if nxt not in visited:
                    visited.add(nxt)
                    q.append(nxt)

        comp_constraints = []
        for cells, mine_count in constraints:
            inter = cells & comp_cells
            if inter:
                comp_constraints.append((inter, mine_count))

        components.append((comp_cells, comp_constraints))

    return components


def enumerate_component(
    cells: Set[Cell],
    constraints: List[Tuple[Set[Cell], int]],
    max_cells: int = 20,
) -> Optional[Dict[Cell, float]]:
    cells_list = list(cells)
    n = len(cells_list)
    if n == 0 or n > max_cells:
        return None

    valid_count = 0
    mine_counts = {cell: 0 for cell in cells_list}

    cons_idx = [
        ([i for i, cell in enumerate(cells_list) if cell in con_cells], mine_count)
        for con_cells, mine_count in constraints
    ]

    for bits in product((0, 1), repeat=n):
        ok = True
        for idxs, mine_count in cons_idx:
            if sum(bits[i] for i in idxs) != mine_count:
                ok = False
                break

        if not ok:
            continue

        valid_count += 1
        for i, b in enumerate(bits):
            mine_counts[cells_list[i]] += b

    if valid_count == 0:
        return None

    return {cell: mine_counts[cell] / valid_count for cell in cells_list}


def probability_deduction(
    board: np.ndarray,
    max_component_cells: int = 20,
) -> Tuple[Set[Cell], Set[Cell], Dict[Cell, float]]:
    constraints = build_constraints(board)
    components = split_components(constraints)

    probs: Dict[Cell, float] = {}
    safe: Set[Cell] = set()
    mines: Set[Cell] = set()

    for cells, comp_constraints in components:
        result = enumerate_component(cells, comp_constraints, max_cells=max_component_cells)
        if result is None:
            continue

        probs.update(result)
        for cell, p in result.items():
            if p == 0:
                safe.add(cell)
            elif p == 1:
                mines.add(cell)

    return safe, mines, probs


def choose_lowest_risk_guess(board: np.ndarray, probs: Dict[Cell, float]) -> Optional[Cell]:
    unknowns = list(unknown_cells(board))
    if not unknowns:
        return None

    candidates = [(cell, probs[cell]) for cell in unknowns if cell in probs]
    if candidates:
        return min(candidates, key=lambda x: x[1])[0]

    h, w = board.shape
    center = (h / 2, w / 2)
    return min(unknowns, key=lambda cell: abs(cell[0] - center[0]) + abs(cell[1] - center[1]))


def solve_step(
    board: np.ndarray,
    max_component_cells: int = 20,
    force_guess: bool = True,
) -> SolverMove:
    safe, mines = basic_deduction(board)
    if safe or mines:
        return SolverMove(sorted(safe), sorted(mines), 'basic', {}, guessed=False)

    safe, mines = subset_deduction(board)
    if safe or mines:
        return SolverMove(sorted(safe), sorted(mines), 'subset', {}, guessed=False)

    safe, mines, probs = probability_deduction(board, max_component_cells=max_component_cells)
    if safe or mines:
        return SolverMove(sorted(safe), sorted(mines), 'enumeration', probs, guessed=False)

    if force_guess:
        guess = choose_lowest_risk_guess(board, probs)
        if guess is not None:
            return SolverMove([guess], [], 'lowest_risk_guess', probs, guessed=True)

    return SolverMove([], [], 'no_move', probs, guessed=False)
