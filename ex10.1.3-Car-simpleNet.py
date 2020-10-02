import simpleNet
from simpleNet import Moduel
from simpleNet.layers import Dense, Tanh
from simpleNet.optims import Adam
from simpleNet.losses import MeanSquaredError
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

class Net(Moduel):
    def __init__(self):
        super().__init__()
        self.dense1 = Dense(env.observation_space.shape[0], d, bias=True)
        self.tanh = Tanh()
        self.dense2 = Dense(d, env.action_space.n, bias=True)

    def forwards(self, x):
        x = self.dense1(x)
        x = self.tanh(x)
        x = self.dense2(x)
        return x

net = Net()
print(net)
writer = SummaryWriter()

experiences = deque([], maxlen=memory_size)
experience_weights = deque([], maxlen=memory_size)
weight_sum = 0

ep_lens = []
steps = 1
e = 1.
epsilon_decay = 0.995

criterion = MeanSquaredError()
optimizer = Adam(net, lr=0.001)

num_train = 0

def train_on_batch(batch, batch_size=batch_size):
    global num_train
    # batch: [N, [s, a, r, s_next, done]]
    states = np.array([b[0] for b in batch])
    next_states = np.array([b[3] for b in batch])

    outs = net(states)
    targets = np.copy(outs)
    next_qs = net(next_states)

    for i in range(batch_size):
        # now, set the target output for the action taken to be the
        # updated Q val
        max_q = max(next_qs[i])
        targets[i][batch[i][1]] = (batch[i][2]
                                   + (discount_factor * max_q
                                   if not batch[i][4] else 0))

    loss = criterion(outs, targets)
    da = criterion.backwards()
    net.backwards(da)
    optimizer.step()
    writer.add_scalar("Loss/train", loss, num_train)
    num_train += 1

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
        outputs = net([observation])

        if random.uniform(0, 1) < e:
            action = env.action_space.sample()
        else:
            action = np.argmax(outputs)
        new_observation, reward, done, info = env.step(action)
        episode_experiences.append(
            [observation,
             action,
             reward,
             new_observation,
             done]
        )
        reward = 1 if (done and new_observation[0] > 0.5) else reward
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