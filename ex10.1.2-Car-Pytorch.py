import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import numpy as np
import random
from collections import deque

import gym

env = gym.make('MountainCar-v0')

episodes = 10000000
memory_size = 20000
batch_size = 50

discount_factor = 0.98

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.dense1 = nn.Linear(env.observation_space.shape[0], 64*4)
        self.dense2 = nn.Linear(64*4, 64*4)
        self.dense3 = nn.Linear(64*4, env.action_space.n)

    def forward(self, x):
        x = F.tanh(self.dense1(x))
        x = F.tanh(self.dense2(x))
        x = F.tanh(self.dense3(x))
        return x

net = Net()
print(net)

experiences = deque([], maxlen=memory_size)
experience_weights = deque([], maxlen=memory_size)
weight_sum = 0

ep_lens = []
steps = 1
e = 1
epsilon_decay = 0.995

criterion = nn.MSELoss()
optimizer = optim.SGD(net.parameters(), lr=0.1, momentum=0.9)

def train_on_batch(batch, batch_size=batch_size):
    # batch: [N, [s, a, r, s_next, done]]
    states = torch.tensor([b[0] for b in batch], dtype=torch.float32)
    next_states = np.array([b[3] for b in batch])
    outs = net(states)
    targets = np.copy(outs.data.numpy())
    next_qs = net(torch.tensor(next_states, dtype=torch.float32))

    for i in range(batch_size):
        max_q = max(next_qs[i])
        targets[i][batch[i][1]] = (batch[i][2] + (discount_factor * max_q if not batch[i][4] else 0))

    loss = criterion(outs, torch.tensor(targets, dtype=torch.float32))
    loss.backward()
    optimizer.step()

def batch_weighted_selection(items, weights, weight_sum, num_selections):
    selection_numbers = sorted([random.randint(0, weight_sum-1) for i in range(num_selections)])
    selections = []
    running_weight_sum = 0
    for i in range(len(items)):
        running_weight_sum += weights[i]
        while selection_numbers[0] <= running_weight_sum:
            selections.append(items[i])
            selection_numbers = selection_numbers[1:]
            if not selection_numbers:
                return selections

episode_reward = -200
train = False
saved = False

for n in range(episodes):
    e *= epsilon_decay
    episode_reward = 0
    episode_survival = 0

    observation = env.reset()
    start_point = observation[0]
    episode_experiences = []

    while True:
        episode_survival += 1

        action = None
        outputs = net(torch.tensor([observation], dtype=torch.float32))[0]
        outputs = outputs.data.numpy()

        if random.uniform(0, 1) < e:
            action = env.action_space.sample()
        else:
            action = np.argmax(outputs)
        new_observation, reward, done, info = env.step(action)
        episode_experiences.append(
            [observation, action, reward, new_observation, done]
        )
        reward = 1 if done else reward
        observation = new_observation
        episode_reward += reward
        steps += 1
        if len(experiences) < 2*batch_size or n < 5:
            pass
        else:
            selections = [random.choice(experiences) for i in range(batch_size)]
            train_on_batch(selections)
        if done:
            break

    experiences += episode_experiences

    ep_lens.append(episode_survival)
    print("%s, %s, %s, %s" % (n, episode_reward, episode_survival, e))
