import numpy as np
import gym
import torch
import torch.nn as nn
from torch.autograd import Variable

class Policy(nn.Module):
    def __init__(self, env):
        super(Policy, self).__init__()
        self.state_space = env.observation_space.shape[0]
        self.action_space = env.action_space.n
        self.hidden = 200
        self.l1 = nn.Linear(self.state_space, self.hidden, bias=False)
        self.l2 = nn.Linear(self.hidden, self.action_space, bias=False)

    def forward(self, x):
        model = torch.nn.Sequential(
            self.l1,
            self.l2,
        )
        return model(x)

if __name__ == "__main__":
    # 参数和env
    env = gym.make('MountainCar-v0')
    env._max_episode_steps = 2000

    num_episodes = 100
    epsilon = 0.3
    steps = 2000

    # 组件
    policy = Policy(env)
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.SGD(policy.parameters(), lr=0.001)

    # 迭代
    for i in range(num_episodes):
        # 本迭代初始化
        s = env.reset()
        l = 0
        r = 0
        success = False
        N_steps = 0

        # 一步
        for j in range(steps):
            Q = policy(Variable(torch.from_numpy(s).type(torch.FloatTensor)))
            # choose action
            if np.random.rand() > epsilon:
                _, a = torch.max(Q, -1)
                a = a.item()
            else:
                a = np.random.randint(0, 3)

            s_, r, done, _ = env.step(a)
            if num_episodes <= 20:
                r = s_ * 0.1
            Q_ = policy(Variable(torch.from_numpy(s_).type(torch.FloatTensor)))
            max_Q_, _ = torch.max(Q_, -1)
            Q_target = Q.clone()
            Q_target = Variable(Q_target.data)
            Q_target[a] = Q_target[a] + 0.99 * ( r + max_Q_.detach() - Q_target[a] )

            loss = loss_fn(Q, Q_target)
            policy.zero_grad()
            loss.backward()
            optimizer.step()

            l = loss.cpu().data.numpy()
            s = s_
            N_steps += 1

            if done:
                # epsilon *= .99
                if s[0] >= 0.5:
                    success = True
                break

        epsilon = epsilon * 0.9

        print("epsilon %d, finally loss: %02f, end position: %02f, success: %s, N_steps: %d" % (i, l, s[0], str(success), N_steps))

    s = env.reset()

    for j in range(1000):
        env.render()
        Q = policy(Variable(torch.from_numpy(s).float()))
        _, action = torch.max(Q, -1)
        action = action.item()
        s_, r, done, _ = env.step(action)
        if done:
            break
        s = s_

    input("aaa")