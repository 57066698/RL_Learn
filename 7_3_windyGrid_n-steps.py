import gym
import numpy as np
import gym_windy_gridworlds

'''
    N-Steps
    四种 n steps 实现
    
    ---------------------
    
    Windy Grid World
    全空方格矩阵，只有一个目标，目标附近有不定向的若干风
'''

env = gym.make('WindyGridWorld-v0')
print(env.observation_space)
print(env.reset())
print(env.action_space)
print(env.step(0))

epsilon = 0.1
lr = 0.5
alpha = 0.9
actions = [0, 1, 2, 3]

# 由于是对V更新，不好选择a，这里不使用
def n_step_TD_V(train_spisode, n_step):
    spisode_T = []
    alphas = [pow(alpha, i) for i in range(n_step + 1)]
    V = np.zeros((7, 10))

    for i in range(train_spisode):
        (agent_y, agent_x) = env.reset()
        T = float('inf')  # end ind
        t = 0
        S = [(agent_y, agent_x)]
        R = [0]
        stop_cond = False
        while not stop_cond:
            if t < T:  # 游戏没结束，继续模拟
                a = np.random.choice(actions)
                s_, r_, done, _ = env.step(a)
                S.append(s_)
                R.append(r_)
                if done:
                    T = t + 1

            left_ind = t - n_step + 1  # 之前第n step的ind
            if left_ind >= 0:  # 至少走了n_step，开始计算V了
                right_ind = min(left_ind + n_step, T)
                G = sum(alpha[:right_ind - left_ind] * R[left_ind:right_ind])
                left_s, right_s = S[left_ind], S[right_ind]
                right_V = alphas[right_ind - left_ind + 1] * V[right_s]  # 如果是终点，那么终点必然为0
                V[left_s] = V[left_s] + lr * [G + right_V]

            t = t + 1
            if left_ind == T - 1:
                stop_cond = True

            spisode_T.append(T)
    return spisode_T

def e_greedy(Q, s, actions):
    if np.random.rand() <= epsilon:
        a = np.random.choice(actions)
    else:
        a = np.argmax(Q[s])
    return a


def n_step_sarsa(train_spisode, n_step):
    spisode_T = []
    train_T = []
    alphas = [pow(alpha, i) for i in range(n_step + 1)]
    Q = np.zeros((7, 10, 4))

    for i in range(train_spisode):
        (agent_y, agent_x) = env.reset()
        T = float('inf')  # end ind
        t = 0
        S = [(agent_y, agent_x)]
        A = [e_greedy(Q, S[-1], actions)]
        R = []
        stop_cond = False
        while not stop_cond:
            if t < T:  # 游戏没结束，继续模拟
                # epsilon-greedy
                s_, r_, done, _ = env.step(A[-1])
                S.append(s_)
                R.append(r_)
                if done:
                    T = t + 1
                    train_T.append(t)
                else:
                    A.append(e_greedy(Q, S[-1], actions))

            left_ind = t - n_step + 1  # 之前第n step的ind
            if left_ind >= 0:  # 至少走了n_step，开始计算left的Q了
                right_ind = min(left_ind + n_step, T-1)  # 考虑末尾边界
                G = sum(np.multiply(alphas[:right_ind - left_ind], R[left_ind:right_ind]))  # 一路的alpha * R
                left_s, right_s = S[left_ind], S[right_ind]
                left_s_a, right_s_a = (*left_s, A[left_ind]), (*right_s, A[right_ind])
                if left_ind + n_step < T:
                    G = G + alphas[right_ind-left_ind] * Q[right_s_a]
                Q[left_s_a] = Q[left_s_a] + lr * (G - Q[left_s_a])

            t = t + 1
            if left_ind == T - 1:
                stop_cond = True

        (agent_y, agent_x) = env.reset()
        done = False
        t = 0
        while (not done) and t < 100:
            a = e_greedy(Q, (agent_y, agent_x), actions)
            (agent_y, agent_x), r, done, _ = env.step(a)
            t = t + 1
        spisode_T.append(t)
    return spisode_T, np.sum(Q==0)

sarsa_spisodes, train_T = n_step_sarsa(100, 5)

import matplotlib.pylab as plt

plt.plot(sarsa_spisodes)
# plt.plot(np.clip(train_T, 0, 100))
plt.show()
print(train_T)