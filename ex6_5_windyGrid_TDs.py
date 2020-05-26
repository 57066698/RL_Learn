"""
    windy grid world
    有风方格, 目标点附近有不定向的风
    action的 0123 代表 上右下左
"""

import numpy as np
import gym
import gym_windy_gridworlds

env = gym.make('WindyGridWorld-v0')
obs_space = env.observation_space
print(obs_space)
print(env.reset())
print(env.action_space)
print(env.step(0))

epsilon = 0.1
alpha = 0.9
lr = 0.5

#  ---------------------  SARSA


Q = np.zeros((7, 10, 4))  # [agent_y, agent_x, next action]
actions = [0, 1, 2, 3]

sarsa_steps = []
episodes = 0

while len(sarsa_steps) < 10000:
    # new episode
    agent_y, agent_x = env.reset()
    s = (agent_y, agent_x)
    done = False
    if np.random.rand() > epsilon:
        action = np.argmax(Q[s])
    else:
        action = np.random.choice(actions)

    while not done:
        (agent_y, agent_x), reward, done, _ = env.step(action)
        s_ = (agent_y, agent_x)

        if np.random.rand() > epsilon:
            action_ = np.argmax(Q[s])
        else:
            action_ = np.random.choice(actions)

        s_a = (*s, action)
        s_a_ = (*s_, action_)

        Q[s_a] = Q[s_a] + lr * (reward + alpha * Q[s_a_] - Q[s_a])
        s, action = s_, action_

        if done:
            episodes = episodes + 1

        sarsa_steps.append(episodes)

# ------------------- Q-Learn


Q = np.zeros((7, 10, 4))  # [agent_y, agent_x, next action]
actions = [0, 1, 2, 3]

Q_steps = []
episodes = 0

while len(Q_steps) < 10000:
    # new episode
    agent_y, agent_x = env.reset()
    s = (agent_y, agent_x)
    done = False

    while not done:

        if np.random.rand() > epsilon:
            action = np.argmax(Q[s])
        else:
            action = np.random.choice(actions)

        (agent_y, agent_x), reward, done, _ = env.step(action)
        s_ = (agent_y, agent_x)

        s_a = (*s, action)

        Q[s_a] = Q[s_a] + lr * (reward + alpha * np.max(Q[s_]) - Q[s_a])
        s = s_

        if done:
            episodes = episodes + 1

        Q_steps.append(episodes)

#  -----------------------  Expected Sarsa


Q = np.zeros((7, 10, 4))  # [agent_y, agent_x, next action]
PI = np.ones((7, 10, 4)) / 4.0
actions = [0, 1, 2, 3]

expected_sarsa_steps = []
episodes = 0


def one_hot_with_eps(n, i, eps):
    arr = np.ones(n) * eps / n
    arr[i] = 1 - eps + eps / n
    return arr


while len(expected_sarsa_steps) < 10000:
    # new episode
    agent_y, agent_x = env.reset()
    s = (agent_y, agent_x)
    done = False

    while not done:

        action = np.argmax(PI[s])
        (agent_y, agent_x), reward, done, _ = env.step(action)
        s_ = (agent_y, agent_x)
        s_a = (*s, action)

        Q[s_a] = Q[s_a] + lr * (reward + alpha * np.sum(PI[s_] * Q[s_]) - Q[s_a])
        PI[s] = one_hot_with_eps(4, np.argmax(Q[s]), epsilon)
        s = s_

        if done:
            episodes = episodes + 1

        expected_sarsa_steps.append(episodes)

import matplotlib.pyplot as plt

plt.plot(sarsa_steps, color='blue')
plt.plot(Q_steps, color='g')
plt.plot(expected_sarsa_steps, color="r")
plt.xlabel("num steps")
plt.ylabel("done episodes")
plt.show()
