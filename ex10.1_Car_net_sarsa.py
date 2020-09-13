import numpy as np
import gym
from net.DenseNet import DenseNet

env = gym.make('MountainCar-v0')
print(env.observation_space)
print(env.reset())
print(env.action_space)
print(env.step(0))

net = DenseNet(2, 3, 512)

num_episodes = 10
max_step = 2000

env._max_episode_steps = max_step

for n in range(num_episodes):
    s = env.reset()

    for i in range(max_step):
        # choose action
        Q = net.forward(s)
        if np.random.rand() > 0.3:
            action = np.argmax(Q)
        else:
            action = np.random.randint(0, 3)
        s_, reward, done, _ = env.step(action)
        Q_tar = np.array(Q)
        # Q_tar[action] =