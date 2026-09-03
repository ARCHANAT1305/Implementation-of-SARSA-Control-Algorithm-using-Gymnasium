# Implementation-of-SARSA-Control-Algorithm-using-Gymnasium

## Aim

To implement the **SARSA control algorithm** using the Gymnasium `FrozenLake-v1` environment and learn an action-value function that helps the agent select better actions for reaching the goal state while avoiding holes.

---

## Problem Statement

Implement the SARSA (State–Action–Reward–State–Action) reinforcement learning algorithm using the Gymnasium FrozenLake environment. The objective is to train an agent to learn an optimal path from the starting state to the goal while avoiding holes. FrozenLake provides a 4×4 grid with 16 states and 4 possible actions: Left, Down, Right, and Up

## Software Requirements
Python 3.x  
Gymnasium – for creating and interacting with the FrozenLake environment  
NumPy – for Q-table and numerical calculations  
Matplotlib – for plotting the learning curve  
IDE/Jupyter Notebook/VS Code – for writing and executing the program  


## Environment Description


The FrozenLake-v1 environment is a 4×4 grid where the agent starts from the starting point and must reach the goal while avoiding holes, with four possible actions: **Left, Down, Right, and Up**.



## Theory

SARSA stands for:

$$
S_t, A_t, R_{t+1}, S_{t+1}, A_{t+1}
$$

It updates the Q-value using the action actually selected in the next state.

The SARSA update rule is:

$$
Q(S_t,A_t) \leftarrow Q(S_t,A_t) + \alpha
\left[
R_{t+1} + \gamma Q(S_{t+1},A_{t+1}) - Q(S_t,A_t)
\right]
$$

Where:

| Symbol | Meaning |
|---|---|
| $S_t$ | Current state |
| $A_t$ | Current action |
| $R_{t+1}$ | Reward received after taking action $A_t$ |
| $S_{t+1}$ | Next state |
| $A_{t+1}$ | Next action selected using the current policy |
| $\alpha$ | Learning rate |
| $\gamma$ | Discount factor |
| $Q(s,a)$ | Action-value function |

---

## Epsilon-Greedy Policy

SARSA uses an epsilon-greedy policy for action selection.

With probability $\epsilon$, the agent explores by selecting a random action.

With probability $1-\epsilon$, the agent exploits by selecting the action with the highest Q-value.

$$
a =
\begin{cases}
\text{random action}, & \text{with probability } \epsilon \\
\arg\max_a Q(s,a), & \text{with probability } 1-\epsilon
\end{cases}
$$

---


## Algorithm
1. Initialize the Q-table
2. Create a Q-table with all values initialized to 0 for every state and action.
3. Start an episode
4. Reset the FrozenLake environment and get the initial state. Select an action using the ε-greedy policy.
5. Take the action
6. Execute the selected action in the environment and observe the next state and reward.
7. Select the next action
8. If the episode is not finished, select the next action from the next state using the ε-greedy policy.

9. Update the Q-value using SARSA


10. Then move to the next state and action.

11. Repeat and learn
12. Repeat the process for many episodes, gradually decrease ε to reduce exploration, and finally obtain the learned policy using the action with the highest Q-value.

## Python Program

```python

import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt

# -------------------------------------------------
# Create FrozenLake Environment
# -------------------------------------------------
env = gym.make("FrozenLake-v1", is_slippery=False)

# -------------------------------------------------
# Hyperparameters
# -------------------------------------------------
num_episodes = 10000
max_steps_per_episode = 100
alpha = 0.1
gamma = 0.99
epsilon = 1.0
epsilon_min = 0.05
epsilon_decay = 0.9995

# -------------------------------------------------
# Initialize Q-table
# -------------------------------------------------
Q = np.zeros((env.observation_space.n, env.action_space.n))
episode_rewards = []

# -------------------------------------------------
# Epsilon-Greedy Action Selection
# -------------------------------------------------
def epsilon_greedy_action(state, epsilon):
    if np.random.random() < epsilon:
        return env.action_space.sample()
    else:
        return np.argmax(Q[state])

# -------------------------------------------------
# SARSA Training
# -------------------------------------------------
for episode in range(num_episodes):

    state, info = env.reset()
    action = epsilon_greedy_action(state, epsilon)
    total_reward = 0

    for step in range(max_steps_per_episode):

        next_state, reward, terminated, truncated, info = env.step(action)

        if terminated or truncated:
            Q[state, action] = Q[state, action] + alpha * (
                reward - Q[state, action]
            )
            total_reward += reward
            break

        next_action = epsilon_greedy_action(next_state, epsilon)

        Q[state, action] = Q[state, action] + alpha * (
            reward + gamma * Q[next_state, next_action]
            - Q[state, action]
        )

        state = next_state
        action = next_action
        total_reward += reward

    episode_rewards.append(total_reward)

    epsilon = max(epsilon_min, epsilon * epsilon_decay)

# -------------------------------------------------
# Display Functions
# -------------------------------------------------
def print_value_function(values):
    print("\nEstimated State-Value Function:")
    print(np.round(values.reshape(4, 4), 3))

def print_policy(policy):
    action_symbols = {
        0: "L",
        1: "D",
        2: "R",
        3: "U"
    }

    policy_grid = np.array(
        [action_symbols[action] for action in policy]
    ).reshape(4, 4)

    print("\nLearned Policy:")
    print(policy_grid)

# -------------------------------------------------
# Output
# -------------------------------------------------
state_values = np.max(Q, axis=1)
learned_policy = np.argmax(Q, axis=1)

print("\nFinal Q-table:")
print(np.round(Q, 3))

print_value_function(state_values)
print_policy(learned_policy)

average_reward = np.mean(episode_rewards[-1000:])
print("\nAverage reward over last 1000 episodes:", average_reward)

# -------------------------------------------------
# Plot Learning Curve
# -------------------------------------------------
window = 500

moving_average = np.convolve(
    episode_rewards,
    np.ones(window) / window,
    mode="valid"
)

plt.figure(figsize=(8, 5))
plt.plot(moving_average)
plt.xlabel("Episode")
plt.ylabel("Average Reward")
plt.title("SARSA Learning Curve - FrozenLake")
plt.grid(True)
plt.show()

env.close()
```

## Output

#### Final Q-table:




<img width="272" height="261" alt="image" src="https://github.com/user-attachments/assets/c11465a7-6a0f-4759-87a0-ed14f79f0290" />

#### Estimated State-Value Function:


<img width="266" height="100" alt="image" src="https://github.com/user-attachments/assets/e4150737-f64b-4d21-ab55-50c7ecdfe376" />



#### Learned Policy:

<img width="175" height="86" alt="image" src="https://github.com/user-attachments/assets/6a8d4920-7bfa-4474-8aae-741d9f30521b" />



#### Average reward over last 1000 episodes: 
<img width="396" height="31" alt="image" src="https://github.com/user-attachments/assets/3e54560a-2a9c-4ce1-a104-8fcc2904e65d" />
<img width="688" height="392" alt="image" src="https://github.com/user-attachments/assets/b37fed34-59f1-485c-b6cf-0bfaf1a3dcc5" />



## Result


After training the agent for 10,000 episodes, the SARSA algorithm updates the Q-table based on the actions actually selected by the agent. SARSA is an on-policy TD control algorithm, using the next selected action in its Q-value update


## Inference

The SARSA agent initially explores the FrozenLake environment using random actions. Through repeated interaction and Q-value updates, it learns which actions are better for reaching the goal while avoiding holes. After 10,000 episodes, the learned Q-table and policy demonstrate the agent's ability to navigate the environment successfully.
