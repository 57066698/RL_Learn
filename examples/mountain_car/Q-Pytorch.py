import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.init import xavier_uniform
from torch.utils.tensorboard import SummaryWriter

import numpy as np
import random
from collections import deque

import gym

env = gym.make('MountainCar-v0')

episodes = 10000000
memory_size = 20000
batch_size = 50

discount_factor = 0.98

d = 128

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.dense1 = nn.Linear(env.observation_space.shape[0], d)
        self.dense2 = nn.Linear(d, env.action_space.n)

    def forward(self, x):
        x = torch.tanh(self.dense1(x))
        x = self.dense2(x)
        return x

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
net = Net()
net.to(device)
print(net)
writer = SummaryWriter()

experiences = []
weight_sum = 0

ep_lens = []
steps = 1
e = 1.
epsilon_decay = 0.995

criterion = nn.MSELoss()
optimizer = optim.Adam(net.parameters(), lr=0.001)

batch_num = 0

def train_on_batch(batch, batch_size=batch_size):
    global batch_num
    # batch: [N, [s, a, r, s_next, done]]
    s = torch.tensor([b[0] for b in batch], dtype=torch.float32).to(device)
    q = net(s)

    s_ = torch.tensor([b[3] for b in batch], dtype=torch.float32).to(device)
    q_ = net(s_)

    targets = np.copy(q.detach().cpu().numpy())
    next_qs = q_.detach().cpu().numpy()

    for i in range(batch_size):
        max_q = max(next_qs[i])
        targets[i][batch[i][1]] = (batch[i][2]
                                   + (discount_factor * max_q
                                   if not batch[i][4] else 0))

    y = torch.tensor(targets, dtype=torch.float32).to(device)
    optimizer.zero_grad()
    loss = criterion(q, y)
    loss.backward()
    optimizer.step()
    writer.add_scalar("Loss/train", loss, batch_num)
    batch_num += 1

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
        outputs = net(torch.tensor([observation], dtype=torch.float32).to(device))
        outputs = outputs.detach().cpu().numpy()[0]

        if random.uniform(0, 1) < e:
            action = env.action_space.sample()
        else:
            action = np.argmax(outputs)
        new_observation, reward, done, info = env.step(action)
        reward = 1 if (done and new_observation[0] > 0.5) else reward
        episode_experiences.append(
            [observation,
             action,
             reward,
             new_observation,
             done]
        )

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
    writer.flush()

writer.close()