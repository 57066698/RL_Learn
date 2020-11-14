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

weight_sum = 0

steps = 1
e = 1.
epsilon_decay = 0.995
n_steps = 1

criterion = MeanSquaredError()
optimizer = Adam(net, lr=0.001)

num_train = 0

def train_on_batch(batch, num_step):
    global num_train

    # batch: [N, [s, a, [r, r, r, ...], s_n, done]]
    states = np.array([b[0] for b in batch])
    next_states = np.array([b[3] for b in batch])

    outs = net(states)
    targets = np.copy(outs)
    next_qs = net(next_states)

    r_arr = np.power(discount_factor, np.arange(num_step))

    for i in range(len(batch)):
        max_q = max(next_qs[i])
        rs = r_arr * batch[i][2]
        targets[i][batch[i][1]] = (sum(rs) + (discount_factor * r_arr[-1] * max_q if not batch[i][4] else 0))

    loss = criterion(outs, targets)
    da = criterion.backwards()
    net.backwards(da)
    optimizer.step()
    writer.add_scalar("Loss/train", loss, num_train)
    num_train += 1

def make_batchs(n_step, episode_experiences):
    # 从episode_experiences中提取符合n_step的所有情况
    # s, a, r, done
    result_experiences = []
    for i in range(len(episode_experiences)-n_step-1):
        result_experiences.append([episode_experiences[i][0],
                                  episode_experiences[i][1],
                                  [episode_experiences[j][2] for j in range(i, i+n_step)],
                                  episode_experiences[i+n_step][0],
                                  episode_experiences[i+n_step][3]])
    return result_experiences

episode_reward = -200
train = False
saved = False

def choose_action(Q):
    if random.uniform(0, 1) < e:
        action = env.action_space.sample()
    else:
        action = np.argmax(Q)
    return action

for n in range(episodes):
    e *= epsilon_decay

    episode_reward = 0
    episode_survival = 0

    observation = env.reset()
    start_point = observation[0]
    episode_experiences = []

    while True:
        episode_survival += 1

        outputs = net([observation])
        action = choose_action(outputs)
        new_observation, reward, done, info = env.step(action)
        episode_experiences.append(
            [observation,
             action,
             reward,
             done]
        )
        reward = 1 if (done and new_observation[0] > 0.5) else reward
        observation = new_observation
        episode_reward += reward
        steps += 1
        # if len(experiences) < 2*batch_size or n < 5:
        #     pass
        # else:
        #     selections = [random.choice(experiences) for i in range(batch_size)]
        #     train_on_batch(selections)
        if done:
            break

    # todo: train
    batch = make_batchs(n_steps, episode_experiences)

    epochs = 1
    n_batch = int(np.ceil(len(batch)/batch_size))

    for ee in range(epochs):
        for i in range(n_batch):
            left = i * batch_size
            right = min((i+1) * batch_size, len(batch))
            train_on_batch(batch[left:right], n_steps)

    print("%s, %s, %s, %s" % (n, episode_reward, episode_survival, e))
    writer.flush()

writer.close()