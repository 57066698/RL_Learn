from examples.mountain_car.Q_SimpleNet import train_batch, qNet
import numpy as np

batch = [[[0.1, 0.2], 1, 0, [1, 2], 0]]


for i in range(10000):
    train_batch(qNet, batch, 1)

q_tar = qNet(np.array([[1., 2.]]))
q_now = qNet(np.array([[0.1, 0.2]]))

print(q_tar)
print(q_now)

