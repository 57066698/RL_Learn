import gym
import numpy as np
import matplotlib.pylab as plt
from mpl_toolkits.mplot3d import Axes3D

"""
    Frist-Visit prediction
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

R = 0

for _ in range(1000):
    my_hand, dealer, ace = env.reset()
    done = False
    while not done:
        done = False
        if np.random.rand(1) > 0.5:
            obs, reward, done, _ = env.step(1)
            R += reward
        else:
            obs, reward, done, _ = env.step(0)
            if done:
                R += reward


print("随机选择, 预期收益为:", R/1000.0)

V = np.zeros((obs_space[0].n, obs_space[1].n, obs_space[2].n), dtype=np.float)
count_V = np.zeros(V.shape)  # 统计进入这个状态的次数


for i in range(100000):
    S = []
    my_hand, dealer, ace = env.reset()
    done = False
    while not done:
        done = False
        if np.random.rand(1) > 0.5:
            obs, reward, done, _ = env.step(1)
            my_hand, dealer, ace = obs
            S.append((my_hand, dealer, int(ace)))
        else:
            obs, reward, done, _ = env.step(0)

        if done:
            G = reward
            for i in range(len(S)-1, -1, -1):
                G = 1.0 * G
                s = S[i]
                if s not in S[i+1:]:
                    average = (count_V[s] * V[s] + G) / (count_V[s] + 1)
                    count_V[s] += 1
                    V[s] = average

X = np.arange(obs_space[1].n)
Y = np.arange(obs_space[0].n)
X, Y = np.meshgrid(X, Y)
Z = V[:, :, 0]

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=plt.get_cmap('rainbow'))
plt.show()

# test 在12以上时，将拿牌的概率收益相加与自身对比，考虑拿牌还是揭牌
# sum(prop(s_) * V(s_)) vs V(s)

R = 0
cards = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 10, 10, 10]) # JQK is 10
n = 1000

for i in range(n):
    my_hand, dealer, ace = env.reset()
    done = False
    while not done:
        if my_hand <= 11:
            obs, reward, done, _ = env.step(1)
        else:
            draw_prob = np.ones(len(cards)) / len(cards)
            draw_hand = my_hand + cards
            draw_value = V[draw_hand.tolist(), dealer, int(ace)]
            draw_value[0] = V[draw_hand[0], dealer, 1]  # 抽到1要把ace置为1
            draw_V = np.sum(draw_prob * draw_value)

            if draw_V > V[my_hand, dealer, int(ace)]:
                obs, reward, done, _ = env.step(1)
            else:
                obs, reward, done, _ = env.step(0)

            if done:
                R += reward


print("Monte Carlo First-Visit 选择，预期收入为：", R/n)

env.close()
