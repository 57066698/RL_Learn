# failed


import numpy as np
import gym
from net.DenseNet import DenseNet

env = gym.make('MountainCar-v0')
print(env.observation_space)
print(env.reset())
print(env.action_space)
print(env.step(0))

num_episodes = 30
max_step = 4000

env._max_episode_steps = max_step
epsilon = .7
alpha = 0.99
lr = 0.0001
dim = 512

net = DenseNet(2, 3, dim)

def choose_action(Q):
    if np.random.rand() > epsilon:
        action = np.argmax(Q)
    else:
        action = np.random.randint(0, 3)
    return action

for n in range(num_episodes):
    s = env.reset()
    success = False

    for i in range(max_step):
        # choose action
        Q = net.forward(s)
        a = choose_action(Q)
        s_, r, done, _ = env.step(a)
        Q_ = net.forward(s_)

        if done == True:
            Q_tar = np.array(Q)
            Q_tar[a] = r

        else:
            Q_ = net.forward(s_)
            a_ = choose_action(Q_)
            Q_tar = np.array(Q)
            Q_tar[a] = alpha * Q_[a_] + r - Q[a]

        net.backward(Q - Q_tar)
        net.step(lr)

        s = s_

        if done == True:
            if s[0] >= 0.5:
                epsilon = epsilon * 0.9
                success = True

            break

    print("epsilon %d, end position: %02f, success: %s, n_steps: %s" % (
    n, s[0], str(success), i))

s = env.reset()

for j in range(1000):
    env.render()
    Q = net.forward(s)
    action = np.argmax(Q)
    s_, r, done, _ = env.step(action)
    if done:
        break
    s = s_

input("aaa")