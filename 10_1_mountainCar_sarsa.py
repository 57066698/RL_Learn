import numpy as np
import gym
import torch
import torch.nn as nn

env = gym.make('MountainCar-v0')
print(env.observation_space)
print(env.reset())
print(env.action_space)
print(env.step(0))


def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_back(y):
    y = np.clip(y, 1e-12, 1-1e-12)
    y = 1/y - 1
    y = np.clip(y, 1e-12, 1-1e-12)
    return -np.log(1/y - 1)

# 规范s到[0, 1]
def normalize_state(s):
    [position, velocity] = s
    pos = (position + 1.2) / 1.8
    voc = (velocity + 0.07) / 0.14
    return pos, voc


# model
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.dense1 = nn.Linear(5, 32)
        self.dense2 = nn.Linear(32, 1)

    def forward(self, x):
        x = self.dense1(x)
        x = self.dense2(x)
        x = torch.sigmoid(x)
        return x


def one_Hot(n, i):
    arr = np.zeros(n)
    arr[i] = 1.0
    return arr


class QValueModel:
    def __init__(self):
        self.net = Net()
        self.critrion = nn.L1Loss()
        self.optim = torch.optim.SGD(self.net.parameters(), lr=0.01)
        self.action_num = 3

    def get_x_tensor(self, s, a):
        pos, voc = normalize_state(s)
        x = torch.zeros((1, 5))
        x[0, :2] = torch.Tensor([pos, voc])
        x[0, 2:] = torch.Tensor(one_Hot(self.action_num, a))
        return x

    def get_y_tensor(self, g):
        g = sigmoid(g)
        y = torch.zeros((1, 1))
        y[0, 0] = g
        return y

    def Qsa_value(self, s, a):
        x_tensor = self.get_x_tensor(s, a)
        y_tensor = self.net(x_tensor)
        value = y_tensor.cpu().data.numpy()
        return sigmoid_back(value[0, 0])

    def train_model_sarsa(self, s, a, r, s_next = None, a_next = None):
        Q_sa = self.Qsa_value(s, a)

        if s_next is not None and a_next is not None:
            Q_sa_next = self.Qsa_value(s_next, a_next)
            y = self.get_y_tensor(r+Q_sa_next-Q_sa)
        else:
            y = self.get_y_tensor(r-Q_sa)

        x = self.get_x_tensor(s, a)
        y_pred = self.net(x)
        loss = self.critrion(y_pred, y)
        loss.backward()
        self.optim.step()
        return loss

    def Qs(self, s):
        """
            返回1-D数组 Q[s, a]
        """
        position, velocity = normalize_state(s)
        s = torch.Tensor([position, velocity])
        x = torch.zeros((3, 5))
        x[:, :2] = s
        x[0, 2:] = torch.Tensor(one_Hot(3, 0))
        x[1, 2:] = torch.Tensor(one_Hot(3, 1))
        x[2, 2:] = torch.Tensor(one_Hot(3, 2))
        y = self.net(x)  # [3, 1]
        return y.cpu().data.numpy().reshape(-1)  # [N]

qValueModel = QValueModel()
num_episodes = 1000
episode_steps = []
epsilon = 0.1


def choose_action(s):
    if np.random.rand() > epsilon:
        Qs = qValueModel.Qs(s)
        best_action = np.argmax(Qs)
        return best_action
    else:
        return np.random.randint(0, 3)


for i in range(num_episodes):
    # reset environment
    s = env.reset()
    step_count = 0
    done = None
    # loop until end
    while not done:
        # choose action
        a = choose_action(s)
        # play environment
        s_, r, done, _ = env.step(a)
        env.render()
        step_count += 1
        # when game end, record episode rewards
        if done:
            qValueModel.train_model_sarsa(s, a, r)
            episode_steps.append(step_count)
        else:
            a_ = choose_action(s_)
            qValueModel.train_model_sarsa(s, a, r, s_, a_)

            s = s_
            a = a_

# plot episode_rewards
import matplotlib.pyplot as plt

plt.plot(episode_steps)
plt.show()
