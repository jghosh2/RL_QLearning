# Q-Learning Maze Solver

A self-contained implementation of tabular Q-learning applied to grid-maze navigation. This document explains both the reinforcement learning theory and how the code maps onto it, from the ground up.

---

## Table of Contents

1. [What is Reinforcement Learning?](#1-what-is-reinforcement-learning)
2. [The Markov Decision Process](#2-the-markov-decision-process)
3. [Value Functions and the Bellman Equation](#3-value-functions-and-the-bellman-equation)
4. [Q-Learning](#4-q-learning)
5. [Exploration vs. Exploitation](#5-exploration-vs-exploitation)
6. [The Maze Environment](#6-the-maze-environment)
7. [Code Architecture](#7-code-architecture)
8. [File-by-File Walkthrough](#8-file-by-file-walkthrough)
9. [End-to-End Training Workflow](#9-end-to-end-training-workflow)
10. [Running the Code](#10-running-the-code)
11. [Tuning the Hyperparameters](#11-tuning-the-hyperparameters)

---

## 1. What is Reinforcement Learning?

Most machine learning is about pattern recognition: you hand a model thousands of labelled examples, and it learns to generalise. **Reinforcement Learning (RL)** is fundamentally different. There are no labels. Instead, an **agent** learns by *doing* — it takes actions in an **environment**, observes what happens, and adjusts its behaviour based on the **rewards** it receives.

The analogy to human learning is intentional. A child learning to ride a bike doesn't read a textbook on balance; they fall, adjust, and gradually improve. RL formalises exactly this trial-and-error loop.

```
        ┌─────────────┐
        │    Agent    │
        └──────┬──────┘
      action   │   ▲
               ▼   │ state + reward
        ┌─────────────┐
        │ Environment │
        └─────────────┘
```

Three core ideas:

| Concept | Plain English |
|---|---|
| **State** | A snapshot of the situation the agent finds itself in |
| **Action** | A choice the agent can make |
| **Reward** | A scalar signal — positive for good outcomes, negative for bad |

The agent's goal is to discover a **policy** (a mapping from states to actions) that maximises its *total accumulated reward* over time.

---

## 2. The Markov Decision Process

RL problems are formalised as a **Markov Decision Process (MDP)**, defined by the tuple $(\mathcal{S},\, \mathcal{A},\, P,\, R,\, \gamma)$.

| Symbol | Name | Meaning |
|---|---|---|
| $\mathcal{S}$ | State space | All possible situations |
| $\mathcal{A}$ | Action space | All possible moves |
| $P(s^{\prime} \mid s, a)$ | Transition probability | How likely the environment moves to $s^{\prime}$ after action $a$ in state $s$ |
| $R(s, a, s^{\prime})$ | Reward function | The immediate signal received |
| $\gamma$ | Discount factor | How much future rewards are worth relative to immediate ones |

### The Markov Property

The "Markov" in MDP means: **the future depends only on the present state, not on the history of how you got there.** Knowing where you are is enough — you don't need to remember every step you took to arrive.

In our maze, this holds perfectly. The agent's next position depends only on its current cell and chosen direction, not on the path it took to reach that cell.

### The Discount Factor $\gamma$

Receiving a reward now is better than receiving the same reward later (like money in a bank). The discount factor $\gamma \in [0, 1)$ captures this:

- $\gamma = 0$ → agent is completely short-sighted; only cares about immediate reward
- $\gamma \to 1$ → agent weights all future rewards equally (can be unstable in practice)
- $\gamma = 0.99$ → agent strongly values future rewards but slightly prefers sooner ones

The **discounted return** from time step $t$ is:

$$G_t = r_t + \gamma r_{t+1} + \gamma^{2} r_{t+2} + \gamma^{3} r_{t+3} + \cdots = \sum_{k=0}^{\infty} \gamma^{k} \, r_{t+k}$$

This is the quantity the agent ultimately tries to maximise.

---

## 3. Value Functions and the Bellman Equation

The agent needs a way to judge how good a situation is. **Value functions** provide this.

### State Value $V(s)$

The **state value** $V^{\pi}(s)$ is the expected discounted return starting from state $s$ and following policy $\pi$ thereafter:

$$V^{\pi}(s) = \mathbb{E}_\pi \!\left[ G_t \mid S_t = s \right] = \mathbb{E}_\pi \!\left[ r_t + \gamma \, V^{\pi}(s_{t+1}) \mid S_t = s \right]$$

The second form is the **Bellman equation**: the value of a state equals the immediate reward plus the discounted value of the next state. This recursive relationship is the key insight that makes RL algorithms tractable.

### Action Value $Q(s, a)$

The **action-value function** $Q^{\pi}(s, a)$ answers a more specific question: *how good is it to take action $a$ in state $s$, and then follow policy $\pi$?*

$$Q^{\pi}(s, a) = \mathbb{E}_\pi \!\left[ r + \gamma \max_{a^{\prime}} Q^{\pi}(s^{\prime}, a^{\prime}) \right]$$

$Q$-values are what our algorithm actually learns, because choosing the best action from any state is trivial once you have them: just pick $\arg\max_a Q(s, a)$.

### Optimal $Q$-values

The **optimal** $Q$-function $Q^{*}(s, a)$ satisfies the **Bellman optimality equation**:

$$
Q^{*}(s, a) = \mathbb{E}[r + \gamma \max_{a'} Q^{*}(s', a')]
$$

This says: the optimal value of taking action $a$ in state $s$ equals the immediate reward plus the discounted value of acting optimally from the next state onward. The Q-learning algorithm is essentially an iterative method to solve this equation.

---

## 4. Q-Learning

Q-learning (Watkins, 1989) learns $Q^{*}$ directly, without needing a model of the environment's transition probabilities. It is an **off-policy**, **model-free**, **temporal-difference** algorithm.

### The Update Rule

Each time the agent takes an action, it receives a training signal. First, a **TD target** is constructed from the observed reward and the current best estimate of the future:

$$\text{TD target} = r + \gamma \max_{a^{\prime}} Q(s^{\prime}, a^{\prime})$$

The current $Q$-value is then nudged toward this target by a fraction $\alpha$ (the learning rate):

$$Q(s, a) \;\leftarrow\; Q(s, a) + \alpha \underbrace{\Big[ \underbrace{r + \gamma \max_{a^{\prime}} Q(s^{\prime}, a^{\prime})}_{\text{TD target}} \;-\; Q(s, a) \Big]}_{\text{TD error}}$$

Breaking this down intuitively:

- $Q(s, a)$ — our current estimate of how good this (state, action) pair is
- $r + \gamma \max_{a^{\prime}} Q(s^{\prime}, a^{\prime})$ — a *better* estimate, using the actual reward just observed plus our best guess about the future
- **TD error** — the gap between the two; if positive, we underestimated; if negative, we overestimated
- $\alpha$ — how aggressively to close that gap (small $\alpha$ = slow but stable; large $\alpha$ = fast but noisy)

After enough updates across enough episodes, $Q(s, a)$ converges to $Q^{*}(s, a)$ — provided every $(s, a)$ pair is visited sufficiently often.

### Tabular Q-Learning

In the simplest form — used here — $Q$ is stored as a **dictionary (table)** mapping every $(s, a)$ pair to a scalar. This works when the state space is small enough to enumerate, as in our $5 \times 5$ maze ($25 \text{ cells} \times 4 \text{ actions} = 100$ entries).

For larger problems (images, continuous spaces), the table is replaced with a neural network — this is **Deep Q-Learning (DQN)** — but the update rule stays the same.

---

## 5. Exploration vs. Exploitation

A fundamental tension in RL: should the agent try new actions to discover better strategies (**explore**), or stick with what it already knows works (**exploit**)?

- Pure exploitation → gets stuck in suboptimal habits early in training
- Pure exploration → never benefits from what it has learned

### $\varepsilon$-Greedy Strategy

The simplest solution is **$\varepsilon$-greedy**: with probability $\varepsilon$, pick a random action; otherwise pick the action with the highest $Q$-value.

$$\pi(s) = \begin{cases} \text{random action} & \text{with probability } \varepsilon \\ \arg\max_a\, Q(s, a) & \text{with probability } 1 - \varepsilon \end{cases}$$

### Epsilon Decay

Early in training, the agent knows nothing, so it should explore heavily ($\varepsilon \approx 1$). As it accumulates experience, it should exploit more ($\varepsilon \to \varepsilon_{\min}$). We achieve this by multiplying $\varepsilon$ by a decay factor after every episode:

$$\varepsilon \;\leftarrow\; \max\!\left(\varepsilon_{\min},\; \varepsilon \cdot \varepsilon_{\text{decay}}\right)$$

In our implementation: $\varepsilon$ starts at $1.0$, decays by $\times 0.995$ each episode, and floors at $0.05$ — ensuring the agent never completely stops exploring.

---

## 6. The Maze Environment

The maze is a grid where each cell is one of four types:

```
S   start position
G   goal position
0   open cell (passable)
1   wall (impassable)
```

Example $5 \times 5$ maze:

```
 S  · ███      
    · ███   ███
    ·  ·  · ███
███   ███ ·  · 
         ███ G
```

### Mapping to MDP Concepts

| MDP Concept | Maze Implementation |
|---|---|
| **State** $s$ | Current cell $(\text{row},\, \text{col})$ |
| **Actions** $\mathcal{A}$ | Up, Down, Left, Right (4 actions) |
| **Transition** $P$ | Deterministic: move succeeds unless a wall is hit, in which case the agent stays put |
| **Reward** $R$ | $+1.0$ for reaching the goal; $-0.01$ for every other step |
| **Terminal state** | The goal cell G |

The small **step penalty of $-0.01$** is important: it incentivises the agent to find the *shortest* path rather than wandering. Without it, all paths to the goal yield equal total reward and the agent may meander.

---

## 7. Code Architecture

```
RL/
├── Maze.py          # Environment — state transitions, rewards, rendering
├── QLearning.py     # Agent — Q-table, ε-greedy policy, Bellman update
├── Trainer.py       # Training loop — runs episodes, logs progress
├── MazeSolver.py    # Orchestrator — wires everything together
└── RL_script.ipynb  # Notebook — demo entry point
```

The four modules have a clean layered dependency:

```
MazeSolver
 ├── Maze           (environment)
 ├── QLearningAgent (agent)
 └── Trainer
      ├── Maze
      └── QLearningAgent
```

All classes use `@define` from the [attrs](https://www.attrs.org) library, which auto-generates `__init__`, `__repr__`, and `__eq__` from field declarations — removing boilerplate while making the data model explicit.

---

## 8. File-by-File Walkthrough

### `Maze.py` — The Environment

```python
@define
class Maze:
    grid:  np.ndarray = field(converter=np.array)
    rows:  int        = field(init=False)
    ...
```

`Maze` owns all environment logic. Its three key methods implement the standard **gym-style interface**:

| Method | Role |
|---|---|
| `reset()` | Start a new episode; place agent at S |
| `step(action)` | Apply action; return `(next_state, reward, done)` |
| `render(path)` | Print the maze with an optional solution path overlaid |

The `converter=np.array` on `grid` means you can pass a plain Python list and it is automatically converted to a NumPy array on construction — no manual cast needed.

Because the transition is **deterministic**, $P(s^{\prime} \mid s, a)$ is always $0$ or $1$. This makes the maze a simple MDP, appropriate for tabular Q-learning.

---

### `QLearning.py` — The Agent

```python
@define
class QLearningAgent:
    alpha:     float = 0.1    # learning rate
    gamma:     float = 0.99   # discount factor
    epsilon:   float = 1.0    # initial exploration rate
    eps_min:   float = 0.05
    eps_decay: float = 0.995
    q_table: defaultdict = field(init=False)
```

The $Q$-table is a `defaultdict` — any unseen state automatically gets a zero-initialised vector of length `n_actions`. States are added lazily as the agent encounters them.

**`update()`** is the heart of the algorithm:

```python
best_next = 0.0 if done else float(np.max(self.q_table[next_state]))
td_target = reward + self.gamma * best_next
td_error  = td_target - self.q_table[state][action]
self.q_table[state][action] += self.alpha * td_error
```

Line by line:
1. If the episode ended (`done=True`) there is no future — $\max_{a^{\prime}} Q(s^{\prime}, a^{\prime}) = 0$. Otherwise take the max $Q$-value of the next state.
2. Construct the TD target: $r + \gamma \max_{a^{\prime}} Q(s^{\prime}, a^{\prime})$.
3. Compute the TD error: how wrong our current estimate was.
4. Nudge $Q$ toward the target by $\alpha \times \text{TD error}$.

---

### `Trainer.py` — The Training Loop

```python
@define
class Trainer:
    maze:      Maze
    agent:     QLearningAgent
    max_steps: int = 500
    rewards_history: list[float] = field(init=False, factory=list)
    steps_history:   list[int]   = field(init=False, factory=list)
```

`Trainer` owns the episode loop. One episode looks like:

```
reset maze
while not done and steps < max_steps:
    choose action  (ε-greedy)
    step environment
    update Q-table (Bellman)
    move to next state
decay epsilon
```

`rewards_history` and `steps_history` accumulate per-episode metrics so you can observe convergence: total reward should rise and steps-to-goal should fall as training progresses.

The `factory=list` default is an attrs idiom for mutable defaults — equivalent to `field(default_factory=list)` in dataclasses. Without this, every `Trainer` instance would share the same list object, a classic Python gotcha.

---

### `MazeSolver.py` — The Orchestrator

```python
@define
class MazeSolver:
    grid:         list
    agent_kwargs: dict           = field(factory=dict)
    maze:         Maze           = field(init=False)
    agent:        QLearningAgent = field(init=False)
    trainer:      Trainer        = field(init=False)

    def __attrs_post_init__(self) -> None:
        self.maze    = Maze(self.grid)
        self.agent   = QLearningAgent(**self.agent_kwargs)
        self.trainer = Trainer(self.maze, self.agent)
```

`__attrs_post_init__` is called automatically by attrs after the fields are set. It wires the three components together — the user only needs to provide a grid and (optionally) hyperparameters.

`solve()` extracts the **greedy policy** after training: it always picks $\arg\max_a Q(s, a)$ with no exploration. A loop-detection guard prevents an infinite loop if training was insufficient.

---

## 9. End-to-End Training Workflow

Here is what happens when you call `solver.train(n_episodes=2000)`:

```
Episode 1
  reset → s = (0,0)
  step 1: ε=1.0 → random action (explore)
          → new state, reward=-0.01, update Q
  step 2: ε=1.0 → random action
          ...
  (likely never reaches G in early episodes)
  ε ← ε × 0.995 = 0.995

Episode ~200
  ε ≈ 0.37 — agent starts exploiting partial knowledge
  Occasionally reaches G; Q-values around the path begin rising

Episode ~1000
  ε ≈ 0.07 — mostly exploiting
  Agent reliably finds the goal in near-optimal steps

Episode 2000
  ε = 0.05 (floor)
  Q-table has converged; greedy policy traces the shortest path
```

**Why does it work?** Every time the agent reaches the goal, the $+1.0$ reward flows *backwards* through time via the Bellman update. On the step just before the goal, $Q(s_{\text{prev}}, a_{\text{goal}})$ gets boosted. Next episode, the step before *that* gets boosted — because now $Q(s_{\text{prev}}, \cdot)$ is higher — and so on. This **credit propagation** eventually reaches the start state.

---

## 10. Running the Code

### Setup

```bash
python -m venv .rlvenv
source .rlvenv/bin/activate      # Windows: .rlvenv\Scripts\activate
pip install numpy attrs
```

### Run from the notebook

Open `RL_script.ipynb` in Jupyter and run all cells. The notebook defines a $5 \times 5$ maze, trains for 2000 episodes, and renders the solved path.

### Run from a script

```python
from MazeSolver import MazeSolver

GRID = [
    ['S', '0', '1', '0', '0'],
    ['0', '0', '1', '0', '1'],
    ['0', '0', '0', '0', '1'],
    ['1', '0', '1', '0', '0'],
    ['0', '0', '0', '1', 'G'],
]

solver = MazeSolver(
    grid=GRID,
    agent_kwargs=dict(alpha=0.1, gamma=0.99, epsilon=1.0, eps_min=0.05, eps_decay=0.995),
)
solver.train(n_episodes=2_000, log_every=400)
solver.display()
```

Expected output:

```
Training for 2000 episodes …

  Episode   400 | avg reward: -0.212 | steps:  487 | ε: 0.134
  Episode   800 | avg reward: +0.623 | steps:   28 | ε: 0.018
  ...

Training complete.

Solution path (✓ goal reached):

 S  · ███      
    · ███   ███
    ·  ·  · ███
███   ███ ·  · 
         ███ G 

Path length: 9 steps
```

---

## 11. Tuning the Hyperparameters

| Parameter | Default | Effect of increasing | Effect of decreasing |
|---|---|---|---|
| $\alpha$ | $0.1$ | Learns faster but noisier | Slower, more stable convergence |
| $\gamma$ | $0.99$ | Values long-term rewards more; needs more episodes | Shortsighted; may not find distant goals |
| $\varepsilon$ | $1.0$ | N/A (initial value) | Starting low means less exploration early on |
| $\varepsilon_{\min}$ | $0.05$ | More random behaviour at convergence | Fully greedy at convergence (risky) |
| $\varepsilon_{\text{decay}}$ | $0.995$ | Slower decay; more exploration overall | Faster decay; less exploration |
| `max_steps` | $500$ | Longer episodes; more updates per episode | Agent may time out before reaching goal |
| `n_episodes` | $2000$ | More training; better convergence | May not converge on harder mazes |

**General advice:**
- If the agent rarely reaches the goal → increase `n_episodes` or slow $\varepsilon_{\text{decay}}$ (more exploration time)
- If learning is noisy and $Q$-values fluctuate → decrease $\alpha$
- If the agent finds the goal but takes a long route → increase $\gamma$ to incentivise shorter paths
- If the maze is larger → scale up `n_episodes` proportionally (more states need more visits)
