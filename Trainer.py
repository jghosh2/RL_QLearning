import numpy as np
from attrs import define, field

from Maze import Maze
from QLearning import QLearningAgent


@define
class Trainer:
    """Runs training episodes and collects per-episode metrics."""

    maze:      Maze
    agent:     QLearningAgent
    max_steps: int = 500
    rewards_history: list[float] = field(init=False, factory=list)
    steps_history:   list[int]   = field(init=False, factory=list)

    def run_episode(self) -> tuple[float, int]:
        """Run one full episode; return (total_reward, steps_taken)."""
        state        = self.maze.reset()
        total_reward = 0.0

        for step in range(1, self.max_steps + 1):
            action                   = self.agent.choose_action(state)
            next_state, reward, done = self.maze.step(action)
            self.agent.update(state, action, reward, next_state, done)
            state        = next_state
            total_reward += reward
            if done:
                break

        self.agent.decay_epsilon()
        self.rewards_history.append(total_reward)
        self.steps_history.append(step)
        return total_reward, step

    def train(self, n_episodes: int, log_every: int = 100) -> None:
        """Run *n_episodes* training episodes, logging every *log_every*."""
        print(f"Training for {n_episodes} episodes …\n")
        for ep in range(1, n_episodes + 1):
            reward, steps = self.run_episode()
            if ep % log_every == 0:
                avg_r = np.mean(self.rewards_history[-log_every:])
                print(
                    f"  Episode {ep:>5} | "
                    f"avg reward: {avg_r:+.3f} | "
                    f"steps: {steps:>4} | "
                    f"ε: {self.agent.epsilon:.3f}"
                )
        print("\nTraining complete.")
