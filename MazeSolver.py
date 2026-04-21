from attrs import define, field

from Maze import Maze
from QLearning import QLearningAgent
from Trainer import Trainer


@define
class MazeSolver:
    """Top-level orchestrator: builds maze and agent, trains, and displays the solution."""

    grid:         list
    agent_kwargs: dict          = field(factory=dict)
    maze:         Maze          = field(init=False)
    agent:        QLearningAgent = field(init=False)
    trainer:      Trainer       = field(init=False)

    def __attrs_post_init__(self) -> None:
        """Wire up maze, agent, and trainer after field validation."""
        self.maze    = Maze(self.grid)
        self.agent   = QLearningAgent(**self.agent_kwargs)
        self.trainer = Trainer(self.maze, self.agent)

    def train(self, n_episodes: int = 1_000, log_every: int = 200) -> None:
        """Delegate training to the internal Trainer."""
        self.trainer.train(n_episodes, log_every)

    def solve(self) -> list[tuple]:
        """Follow the greedy policy to extract the solution path."""
        state     = self.maze.reset()
        path      = [state]
        visited   = {state}
        max_steps = self.maze.rows * self.maze.cols * 2

        for _ in range(max_steps):
            action         = self.agent.best_action(state)
            state, _, done = self.maze.step(action)
            if state in visited:
                print("⚠ Loop detected — agent may need more training.")
                break
            path.append(state)
            visited.add(state)
            if done:
                break

        return path

    def display(self) -> None:
        """Solve, render the path, and print the step count."""
        path         = self.solve()
        goal_reached = path[-1] == self.maze.goal
        print(f"\nSolution path ({'✓ goal reached' if goal_reached else '✗ did not reach goal'}):\n")
        self.maze.render(path)
        print(f"\nPath length: {len(path)} steps")
