import gym
import numpy as np
import torch
import torch.nn as nn

#todo: 未成功
"""
    Cartpole Monte Carlo
    
    -----------------------
    gym小车问题
"""

env = gym.make('CartPole-v1')
print(env.observation_space)
print(env.reset())
print(env.action_space)
print(env.step(0))


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# 规范s到[0, 1]
def normalize_state(s):
    [position, velocity, angle, pole_v] = s
    pos = (position - 4.8) / 9.6
    voc = sigmoid(velocity)
    ang = (angle - 24) / 48
    p_v = sigmoid(pole_v)
    return pos, voc, ang, p_v


# model
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.dense1 = nn.Linear(5, 16)
        self.dense2 = nn.Linear(16, 1)

    def forward(self, x):
        x = self.dense1(x)
        y = self.dense2(x)
        return y


class StateData:
    def __init__(self):
        self.X = None
        self.Y = None
        self.batch_size = 1024

    def record(self, s, action, reward):
        X = self.X
        Y = self.Y

        pos, voc, ang, p_v = normalize_state(s)
        assert action == 0.0 or action == 1.0
        r = sigmoid(reward)
        if X is None:
            X = np.array([[pos, voc, ang, p_v, action]])  # [N, 5]
            Y = np.array([[r]])  # [N, 1]
        else:
            X = np.concatenate((X, torch.Tensor([[pos, voc, ang, p_v, action]])), axis=0)
            Y = np.concatenate((Y, torch.Tensor([[r]])), axis=0)
        self.X = X
        self.Y = Y

    def gen_batch(self, i):
        if self.X is None:
            return None

        left = self.batch_size * i
        right = min(self.batch_size * (i+1), self.X.shape[0])

        return self.X[left:right], self.Y[left:right]

    def empty(self):
        self.X = None
        self.Y = None

    def __len__(self):
        if self.X is None:
            return 0
        else:
            return int(np.ceil(self.X.shape[0] / self.batch_size))


class QValueModel:
    def __init__(self):
        self.net = Net()
        self.critrion = nn.MSELoss()
        self.optim = torch.optim.SGD(self.net.parameters(), lr=0.1)
        self.batch_size = 1024
        self.actions = torch.Tensor([0.0, 1.0])

    def train_model(self, x_batch, y_batch):
        x = torch.Tensor(x_batch)
        y_true = torch.Tensor(y_batch)
        y_pred = self.net(x)
        loss = self.critrion(y_pred, y_true)
        loss.backward()
        self.optim.step()
        return loss

    def state_value(self, s):
        """
            返回1-D数组 Q[s, a]
        """
        position, velocity, angle, pole_v = normalize_state(s)
        s = torch.Tensor([[position, velocity, angle, pole_v]])
        x = torch.zeros((len(self.actions), 5))
        x[:, :4] = s
        x[:, 4] = self.actions
        y = self.net(x)  # [N, 1]
        return y.cpu().data.numpy().reshape(-1)  # [N]


stateData = StateData()
qValueModel = QValueModel()
num_episodes = 1000
episode_rewards = []

for i in range(num_episodes):
    # reset environment
    s = env.reset()
    done = None
    S = [s]
    A = []
    R = [0]
    # loop until end
    while (not done) and len(S) < 1000:
        # choose action
        a = np.argmax(qValueModel.state_value(S[-1]))
        # play environment
        s_, r_, done, _ = env.step(a)
        # record action on t, state and reward on t_next
        S.append(s_)
        A.append(a)
        R.append(r_)
        # when game end, record episode rewards
        if done:
            episode_rewards.append(sum(R))

    # do monte carlo backward for values
    num = len(S)
    G = 0
    for j in range(len(S)-2, -1, -1):  # len(S) = len(R) = T, len(A) = T - 1
        G = R[j+1] + 0.9 * G
        G = min(100, G)
        stateData.record(S[j], A[j], G)

    # train Q-value model
    for k in range(len(stateData)):
        x_batch, y_batch = stateData.gen_batch(k)
        qValueModel.train_model(x_batch, y_batch)

# plot episode_rewards
import matplotlib.pyplot as plt
plt.plot(episode_rewards)
plt.show()