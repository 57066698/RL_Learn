import gym
import numpy as np
import torch
import torch.nn as nn

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
        x = self.dense2(x)
        y = torch.sigmoid(x)
        return y


class Model:
    def __init__(self):
        self.net = Net()
        self.critrion = nn.MSELoss()
        self.optim = torch.optim.SGD(self.net.parameters(), lr=0.01)
        self.X_tensor = None
        self.Y_tensor = None
        self.batch_size = 1024
        self.actions = [0.0, 1.0]

    # train Q model
    def train_model(self):
        num = self.X_tensor.shape[0]
        batch_num = int(np.ceil(num / self.batch_size))
        for i in range(batch_num):
            batch_x = self.X_tensor[i * 1024: (i + 1) * 1024]
            batch_v = self.Y_tensor[i * 1024: (i + 1) * 1024]
            y = self.net(batch_x)
            loss = self.critrion(y, batch_v)
            loss.backward()
            self.optim.step()
        return loss

    def record(self, s, action, reward):
        pos, voc, ang, p_v = normalize_state(s)
        assert action == 0.0 or action == 1.0
        r = sigmoid(reward)
        if self.X_tensor is None:
            self.X_tensor = torch.Tensor([[pos, voc, ang, p_v, action]])  # [N, 5]
            self.Y_tensor = torch.Tensor([[r]])  # [N, 1]
        else:
            self.X_tensor = torch.cat((self.X_tensor, torch.Tensor([[pos, voc, ang, p_v, action]])), dim=1)
            self.Y_tensor = torch.cat((self.Y_tensor, torch.Tensor([[r]])), dim=1)

    def PI(self, s, model):
        """
        从模型PI中选出对策
        """
        position, velocity, angle, pole_v = normalize_state(s)
        s = torch.Tensor([position, velocity, angle, pole_v])
        x = torch.zeros((len(self.actions), 5))
        x[:, :4] = s
        x[:, 5] = self.actions
        y = model(x)
        best_action = torch.argmax(y.view(-1)).data
        return best_action

    def value(self, s, a, model):
        """
        Q(s, a) 对应的值
        """

