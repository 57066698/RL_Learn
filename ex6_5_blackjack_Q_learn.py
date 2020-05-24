import gym
import numpy as np
import matplotlib.pylab as plt
from mpl_toolkits.mplot3d import Axes3D

"""
    Q-learn
    每一步更新Q, Q[s, a] += lr*(R + alpha * max(Q(s_next, a)) - Q(s,a))
    ----------------------------
    
    21点游戏
    env:
    起手一人拿2张, JQK算10点，2-10算2-10点，1算1点或者10点(没爆就算10点)
    玩家先，拿牌继续拿一张，超过21点输；揭牌庄家拿牌到超过17点. 接近21点赢
    玩家全程只能看到庄家第一张牌
    obs:
    (玩家点数和, 庄家第一张牌, 有无ace:可以变为10点的1)
    action_space: 
    (0: 揭牌，1: 拿牌)
"""

env = gym.make('Blackjack-v0')
obs_space = env.observation_space
print(obs_space)
print(env.reset())
print(env.action_space)
print(env.step(0))

epsilon = 0.1  # 随机action概率
alpha = 0.3
lr = 0.1
S_shape = (obs_space[0].n, obs_space[1].n, obs_space[2].n)
Q = np.zeros((*S_shape, 2), dtype=np.float)  # 行动回报

actions = np.array([0, 1])


def one_hot(n, i):
    arr = np.zeros(n)
    arr[i] = 1
    return arr


for i in range(100000):
    Q_episode = []
    my_hand, dealer, ace = env.reset()
    s = (my_hand, dealer, int(ace))
    done = False
    G = 0
    W = np.ones((*S_shape, 2), dtype=np.float)  # 更新权重
    while not done:
        done = False
        if np.random.rand() > epsilon:
            action = np.argmax(Q[s])
        else:
            action = np.random.choice(actions)
        Q_episode.append((s, action))
        obs, reward, done, _ = env.step(action)
        my_hand, dealer, ace = obs
        s_a = (*s, action)
        s_next = (my_hand, dealer, int(ace))
        Q[s_a] = Q[s_a] + lr * (reward + alpha * np.argmax(Q[s_next]) - Q[s_a])

        s = (my_hand, dealer, int(ace))
        s_a = (*s, action)

R = 0
cards = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 10, 10, 10])  # JQK is 10
n = 1000

for i in range(n):
    my_hand, dealer, ace = env.reset()
    done = False
    while not done:

        s = (my_hand, dealer, int(ace))
        action = np.argmax(Q[s])
        obs, reward, done, _ = env.step(action)
        my_hand, dealer, ace = obs

        if done:
            R += reward

print("Q-Learn，预期收入为：", R / n)
