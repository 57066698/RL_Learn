import keras
from keras.models import Sequential
from keras.layers import Dense, Activation

import numpy as np
import random
from collections import deque

import gym
from gym import wrappers

env = gym.make('MountainCar-v0')
# env = wrappers.Monitor(env, 'experiment', force=True)

episodes = 1000000  # number of training 'episodes' to run
memory_size = 20000  # number of frames to store in memory
batch_size = 50  # number of training frames to grab from memory and train on

discount_factor = 0.98  # the q learning future discount parameter

inputs = env.observation_space.shape[0]

# Setup our NN and training environment in keras
model = Sequential()
# model.add(Dense(units=50, input_dim=env.observation_space.shape[0],
#           kernel_initializer='zeros'))
model.add(Dense(units=64*4, input_dim=env.observation_space.shape[0]))
model.add(Activation('tanh'))
model.add(Dense(units=64*4))
model.add(Activation('tanh'))
model.add(Dense(units=env.action_space.n))
model.add(Activation('linear'))

model.compile(loss=keras.losses.mean_squared_error,
              optimizer=keras.optimizers.RMSprop(lr=0.001))

model.save('models/0.h5')
# store experiences. list of (s, a, r, s', done)
experiences = deque([], maxlen=memory_size)
experience_weights = deque([], maxlen=memory_size)
weight_sum = 0

ep_lens = []  # store the length of episodes in a list
steps = 1  # store the total number of frames in a list
e = 1
epsilon_decay = 0.995


def train_on_batch(model, batch, batch_size=batch_size):
    states = np.array([b[0] for b in batch])
    next_states = np.array([b[3] for b in batch])
    # feed our samples through the network to get our
    # predictions, which once updated with the target q vals will be
    # our targets
    outs = model.predict_on_batch(states)
    targets = np.copy(outs)
    # now, using the Q-value formula, come up with our list of
    # 'better' outputs
    next_qs = model.predict_on_batch(next_states)
    for i in range(batch_size):
        # now, set the target output for the action taken to be the
        # updated Q val
        max_q = max(next_qs[i])
        targets[i][batch[i][1]] = (batch[i][2]
                                   + (discount_factor * max_q
                                   if not batch[i][4] else 0))
    model.train_on_batch(states, targets)


def batch_weighted_selection(items, weights, weight_sum, num_selections):
    selection_numbers = sorted([random.randint(0, weight_sum-1) for i in range(num_selections)])
    selections = []
    running_weight_sum = 0
    for i in range(len(items)):
        running_weight_sum += weights[i]
        while selection_numbers[0] <= running_weight_sum:
            selections.append(items[i])
            selection_numbers = selection_numbers[1:]
            if not selection_numbers:
                return selections


episode_reward = -200
train = False
saved = False
for n in range(episodes):
    # e = -episode_reward/200
    # e = 1/((episode_reward+201.))
    e *= epsilon_decay
    # store the total reward and survival time for this episode
    episode_reward = 0
    episode_survival = 0

    # first observation
    observation = env.reset()
    start_point = observation[0]
    episode_experiences = []

    while True:
        episode_survival += 1

        action = None
        outputs = model.predict_on_batch(np.array([observation]))[0]
        # print(outputs)
        if random.uniform(0, 1) < e:
            action = env.action_space.sample()
        else:
            action = np.argmax(outputs)
        new_observation, reward, done, info = env.step(action)
        episode_experiences.append(
                [observation, action, reward, new_observation, done])
        # reward = (1 if done and episode_survival < 200 else
        #           -1+abs(observation[0]-start_point)+abs(observation[1]))
        reward = 1 if done else reward
        observation = new_observation
        episode_reward += reward
        steps += 1
        if len(experiences) < 2*batch_size or n < 5:
            pass
        else:
            # selections = batch_weighted_selection(experiences, experience_weights, weight_sum, batch_size)
            selections = [random.choice(experiences) for i in range(batch_size)]
            train_on_batch(model, selections)
        if done:
            break

    # ep_weight = 201-episode_survival
    # for i in range(episode_survival):
    #         if len(experience_weights) >= memory_size:
    #             weight_sum -= experience_weights.popleft()
    #         experience_weights.append(ep_weight)
    # weight_sum += episode_survival*(201-episode_survival)
    experiences += episode_experiences

    ep_lens.append(episode_survival)
    print("%s,%s,%s,%s" % (n, episode_reward, episode_survival, e))
    if episode_survival < 100 and not saved:
        saved = True
        model.save('models/a.h5')