import gym
import numpy as np
import matplotlib.pylab as plt
from mpl_toolkits.mplot3d import Axes3D

"""
    Monte Carlo Exploring-Starts
    将PI(s)设置成 argmax(Q(s, A))
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

S_shape = (obs_space[0].n, obs_space[1].n, obs_space[2].n)
PI = np.ones((*S_shape, 2), dtype=np.float) / 2  # 行动概率
Q = np.zeros((*S_shape, 2), dtype=np.float)  # 行动回报
count_Q = np.zeros(Q.shape)  # 统计采取这个行动(s, a)的次数

actions = np.array([0, 1])


def one_hot(n, i):
    arr = np.zeros(n)
    arr[i] = 1
    return arr


for i in range(100000):
    Q_episode = []
    my_hand, dealer, ace = env.reset()
    done = False
    while not done:
        done = False
        s = (my_hand, dealer, int(ace))
        action = np.random.choice(actions, p=PI[s])
        obs, reward, done, _ = env.step(action)
        if not done:
            Q_episode.append((s, action))
            my_hand, reward, done = obs
        else:
            G = reward
            for i in range(len(Q_episode) - 1, -1, -1):
                G = 1.0 * G
                if Q_episode[i] not in Q_episode[i + 1:]:
                    s, a = Q_episode[i]
                    s_a = (*s, a)
                    average = (Q[s_a]*count_Q[s_a] + G) / (count_Q[s_a] + 1)
                    count_Q[s_a] += 1
                    Q[s_a] = average
                    PI[s] = one_hot(2, np.argmax(Q[s]))

X = np.arange(obs_space[1].n)
Y = np.arange(obs_space[0].n)
X, Y = np.meshgrid(X, Y)
Z = Q[:, :, 0, 0]

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=plt.get_cmap('rainbow'))
plt.show()

env.close()
