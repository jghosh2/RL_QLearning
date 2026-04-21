import numpy as np
from attrs import define, field

_ACTIONS: dict[int, tuple[int, int]] = {
    0: (-1,  0),
    1: ( 1,  0),
    2: ( 0, -1),
    3: ( 0,  1),
}


@define
class Maze:
    """Grid environment; '0'=open, '1'=wall, 'S'=start, 'G'=goal."""

    grid:  np.ndarray = field(converter=np.array)
    rows:  int        = field(init=False)
    cols:  int        = field(init=False)
    start: tuple      = field(init=False)
    goal:  tuple      = field(init=False)
    state: tuple      = field(init=False)

    def __attrs_post_init__(self) -> None:
        """Derive shape, start, and goal from the grid after construction."""
        self.rows, self.cols = self.grid.shape
        self.start = self._find("S")
        self.goal  = self._find("G")
        self.state = self.start

    def _find(self, marker: str) -> tuple[int, int]:
        """Return (row, col) of the first occurrence of *marker*."""
        positions = list(zip(*np.where(self.grid == marker)))
        if not positions:
            raise ValueError(f"Marker '{marker}' not found in grid.")
        return positions[0]

    def reset(self) -> tuple[int, int]:
        """Place the agent back at start and return the start state."""
        self.state = self.start
        return self.state

    def step(self, action: int) -> tuple[tuple[int, int], float, bool]:
        """Apply *action*; return (next_state, reward, done)."""
        dr, dc     = _ACTIONS[action]
        nr, nc     = self.state[0] + dr, self.state[1] + dc
        next_state = self.state  # stay put on wall hit

        if self._in_bounds(nr, nc) and self.grid[nr, nc] != "1":
            next_state = (nr, nc)

        self.state = next_state
        done       = (self.state == self.goal)
        reward     = 1.0 if done else -0.01

        return self.state, reward, done

    def _in_bounds(self, r: int, c: int) -> bool:
        """Return True if (r, c) is inside the grid."""
        return 0 <= r < self.rows and 0 <= c < self.cols

    def render(self, path: list[tuple] | None = None) -> None:
        """Print the maze to stdout, marking *path* cells with ·."""
        path_set = set(path) if path else set()
        for r in range(self.rows):
            row_str = ""
            for c in range(self.cols):
                cell = self.grid[r, c]
                if   (r, c) == self.goal:   row_str += " G "
                elif (r, c) == self.start:  row_str += " S "
                elif (r, c) in path_set:    row_str += " · "
                elif cell == "1":           row_str += "███"
                else:                       row_str += "   "
            print(row_str)
