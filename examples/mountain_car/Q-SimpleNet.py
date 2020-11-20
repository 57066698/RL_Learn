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
        super().__init__()
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
    s = np.array([batch[0] for batch in off_policy_batch])
    a = np.array([batch[1] for batch in off_policy_batch])
    r = np.array([batch[2] for batch in off_policy_batch])
    s_ = np.array([batch[3] for batch in off_policy_batch])
    done = np.array([batch[4] for batch in off_policy_batch], dtype=float)

    q = qNet(s)
    q_ = qNet(s_)
    max_q_ = np.argmax(q_, axis=1)
    targets = np.copy(q)

    for i in range(len(max_q_)):
        targets[i, a[i]] = r[i] + (discount_factor * max_q_[i] * (1 - done[i]))

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
        q_value = qNet(observation)

        if random.uniform(0, 1) < e:
            action = env.action_space.sample()
        else:
            action = np.argmax(q_value)

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
            train_batch(qNet, selections, discount_factor)
        if done:
            break

    experiences += episode_experiences

    ep_lens.append(episode_survival)
    print("%s, %s, %s, %s" % (n, episode_reward, episode_survival, e))
    writer.flush()

writer.close()