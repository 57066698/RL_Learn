import simpleNet
from simpleNet import Moduel
from simpleNet.layers import Dense, Tanh
from simpleNet.optims import Adam
from simpleNet.losses import MeanSquaredError
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import gym
import random

env = gym.make('MountainCar-v0')


# 定义神经网络
class QNet(Moduel):
    def __init__(self, in_dim, out_dim, hidden_dim):
        self.dense1 = Dense(in_dim, hidden_dim)
        self.tanh1 = Tanh()
        self.dense2 = Dense(hidden_dim, out_dim)

    def forwards(self, x):
        x = self.dense1(x)
        x = self.tanh1(x)
        x = self.dense2(x)
        return x


qNet = QNet(env.observation_space.shape[0], env.action_space.n, 128)
criterion = MeanSquaredError()
optimizer = Adam(qNet, lr=0.001)


# 定义训练函数

def train_batch(qNet, off_policy_batch, discount_factor):
    # off_policy_batch [N, [s, a, r, s_next, done]]
    s = off_policy_batch[:, 0]
    q = qNet(s)
    s_ = off_policy_batch[:, 3]
    q_ = qNet(s_)
    max_q_ = np.max(q_, axis=1, keepdims=False)
    done = off_policy_batch[:, 4]
    r = off_policy_batch[:, 2]

    targets = np.copy(q)
    a = off_policy_batch[:, 1]
    a = a[:, None]
    targets[a] = r + (discount_factor * max_q_ * (1 - done))

    # train
    optimizer.zero_grad()
    loss = criterion(q, targets)
    criterion.backwards()
    optimizer.step()
    return loss


# 定义探索步骤

episodes = 10000
memory_size = 20000
batch_size = 50
e = 1.0
discount_factor = 0.98
episode_decay = 0.995

writer = SummaryWriter()
experiences = []
weight_sum = 0
steps = 0
ep_lens = []
episode_reward = 0

for n in range(episodes):

    e = e * episode_decay
    episode_reward = 0
    episode_survival = 0

    observation = env.reset()
    episode_experiences = []

    while True:
        episode_survival += 1
        action = None
        q_value = qNet(observation)[0, :]

        if random.uniform(0, 1) < e:
            action = env.action_space.sample()
        else:
            action = np.argmax(q_value)
