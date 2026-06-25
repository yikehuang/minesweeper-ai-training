from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

from minesweeper_ai.env import MinesweeperEnv
from minesweeper_ai.solver import solve_step


def main():
    env = MinesweeperEnv(9, 9, 10, seed=1)
    env.open_cell(4, 4)

    for _ in range(30):
        move = solve_step(env.observe(), max_component_cells=16)
        for r, c in move.flag_cells:
            if env.observe()[r, c] == -1:
                env.toggle_flag(r, c)
        if move.open_cells:
            env.open_cell(*move.open_cells[0])
        if env.won or not env.alive:
            break

    print({'alive': env.alive, 'won': env.won, 'opened': env.opened_safe_count})


if __name__ == '__main__':
    main()
