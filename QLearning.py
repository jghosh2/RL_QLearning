import random
from collections import defaultdict

import numpy as np
from attrs import define, field


@define
class QLearningAgent:
    """Tabular Q-learning: Q(s,a) ← Q(s,a) + α[r + γ·max Q(s',·) − Q(s,a)]."""

    n_actions: int   = 4
    alpha:     float = 0.1    # learning rate
    gamma:     float = 0.99   # discount factor
    epsilon:   float = 1.0    # initial exploration rate
    eps_min:   float = 0.05   # floor for exploration
    eps_decay: float = 0.995  # multiplicative decay per episode
    q_table: defaultdict = field(init=False)

    def __attrs_post_init__(self) -> None:
        """Initialise the Q-table keyed by state with zero action-value vectors."""
        self.q_table = defaultdict(lambda: np.zeros(self.n_actions))

    def choose_action(self, state: tuple) -> int:
        """Return an action via ε-greedy policy."""
        if random.random() < self.epsilon:
            return random.randrange(self.n_actions)
        return int(np.argmax(self.q_table[state]))

    def update(
        self,
        state:      tuple,
        action:     int,
        reward:     float,
        next_state: tuple,
        done:       bool,
    ) -> None:
        """Apply one Bellman TD update to Q(state, action)."""
        best_next = 0.0 if done else float(np.max(self.q_table[next_state]))
        td_target = reward + self.gamma * best_next
        td_error  = td_target - self.q_table[state][action]
        self.q_table[state][action] += self.alpha * td_error

    def decay_epsilon(self) -> None:
        """Multiply epsilon by eps_decay, clamped at eps_min."""
        self.epsilon = max(self.eps_min, self.epsilon * self.eps_decay)

    def best_action(self, state: tuple) -> int:
        """Return the greedy action for *state* (no exploration)."""
        return int(np.argmax(self.q_table[state]))
