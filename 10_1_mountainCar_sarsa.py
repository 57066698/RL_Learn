"""
    mountainCar 使用 sarsa
"""

import numpy as np
import gym
import torch
import torch.nn as nn
import math

"""
    --------------   tools   --------------------
"""

# 规范s到[0, 1]
def normalize_state(s):
    [position, velocity] = s
    pos = (position + 1.2) / 1.8
    voc = (velocity + 0.07) / 0.14
    return (pos, voc)

def one_Hot(n, i):
    arr = np.zeros(n)
    arr[i] = 1.0
    return arr

"""
    --------------  Approximation Torch model  --------------------
    knowing s, a
    predict q(s, a)
"""


class LinearNet(nn.Module):
    def __init__(self):
        super(LinearNet, self).__init__()
        self.dense1 = nn.Linear(5, 52, bias=False)
        self.dense2 = nn.Linear(52, 1, bias=False)

    def forward(self, x):
        x = self.dense1(x)
        x = self.dense2(x)
        return x

class WarpModel():
    def __init__(self):
        self.net = LinearNet()
        self.critrion = nn.MSELoss()
        self.optimizer = torch.optim.SGD(self.net.parameters(), lr=0.001)
        # self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=1, gamma=0.9)

    def train(self, x_np, y_np):
        x_tensor = torch.Tensor(x_np).float()
        y_tensor = torch.Tensor(y_np).float()
        y_pred = self.net(x_tensor)
        self.net.zero_grad()
        loss = self.critrion(y_pred, y_tensor)
        # if math.isnan(loss):
        #     raise ValueError("nan")
        loss.backward()
        self.optimizer.step()
        return loss

    def predict(self, x_np):
        x_tensor = torch.Tensor(x_np).float()
        y_pred = self.net(x_tensor)
        y_pred_np = y_pred.cpu().data.numpy().reshape(-1)
        # if math.isnan(y_pred_np[0]):
        #     raise ValueError("nan")
        return y_pred_np

"""
    -----------------  Sarsa model  ---------------------
    q(s, a) = q(s_next, a_next)
"""
class SarsaModel:
    def __init__(self):
        self.warpModel = WarpModel()
        self.action_num = 3

    def get_x(self, s, a):
        pos, voc = s
        x = np.zeros((1, 5))
        x[0, :2] = [pos, voc]
        x[0, 2:] = one_Hot(self.action_num, a)
        return x

    def get_y(self, g):
        y = np.zeros((1, 1))
        y[0, 0] = g
        return y

    def train_model_sarsa(self, s, a, r, s_next=None, a_next=None):
        Q_sa = self.warpModel.predict(self.get_x(s, a))

        if s_next is not None and a_next is not None:
            Q_sa_next = self.warpModel.predict(self.get_x(s_next, a_next))
            y = self.get_y(r + 0.9 * Q_sa_next - Q_sa)
        else:
            y = self.get_y(r - Q_sa)

        x = self.get_x(s, a)
        loss = self.warpModel.train(x, y)
        loss = loss.cpu().data.numpy()
        return loss

    def best_action(self, s):
        """
            返回1-D数组 Q[s, a]
        """
        x0 = self.get_x(s, 0)
        x1 = self.get_x(s, 1)
        x2 = self.get_x(s, 2)
        x_batch = np.concatenate((x0, x1, x2), axis=0)
        y_batch = self.warpModel.predict(x_batch)  # [3, 1]
        best_action = np.argmax(y_batch)
        return best_action

"""
    ----------  main loop ----------------------
"""


if __name__ == "__main__":

    env = gym.make('MountainCar-v0')
    print(env.observation_space)
    print(env.reset())
    print(env.action_space)
    print(env.step(0))

    sarsaModel = SarsaModel()
    num_episodes = 1000
    end_pos = []
    epsilon = 0.3

    for i in range(num_episodes):
        # reset environment
        s = env.reset()
        s = normalize_state(s)
        step_count = 0
        done = None
        # loop until end
        while not done:
            # choose action
            if np.random.rand() > epsilon:
                a = sarsaModel.best_action(s)
            else:
                a = np.random.randint(0, 3)
            # play environment
            s_, r, done, _ = env.step(a)
            s_ = normalize_state(s_)
            # env.render()
            # when game end, record episode rewards
            if done:
                loss = sarsaModel.train_model_sarsa(s, a, r)
                end_pos.append(s[0])
                print("epsilon %d" % i, loss, s[0], r==1)
            else:
                if np.random.rand() > epsilon:
                    a_ = sarsaModel.best_action(s)
                else:
                    a_ = np.random.randint(0, 3)
                loss = sarsaModel.train_model_sarsa(s, a, r, s_, a_)

                s = s_
                a = a_

    # plot episode_rewards
    import matplotlib.pyplot as plt

    plt.plot(end_pos)
    plt.show()
