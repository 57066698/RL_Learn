import numpy as np


class DenseNet:
    def __init__(self, input=2, output=3, dim=512):

        self.input=input
        self.output=output

        self.W1 = np.random.rand(input, dim)
        self.W2 = np.random.rand(dim, output)

        self.W1_grad = None
        self.W2_grad = None

        self.cached_x = None
        self.cached_h = None

    def forward(self, x):
        # x: [1]
        x = np.reshape(x, (1, self.input))
        h = np.matmul(x, self.W1)
        z = np.matmul(h, self.W2)

        self.cached_x = x
        self.cached_h = h  # [1, 512]
        return np.squeeze(z, axis=0)

    def backward(self, dz):
        # dz: [3]
        dz = np.reshape(dz, (1, self.output))
        self.W2_grad = np.matmul(self.cached_h.T, dz)
        dh = np.matmul(dz, self.W2.T)
        self.W1_grad = np.matmul(self.cached_x.T, dh)

    def step(self, lr=0.001):
        self.W1 -= lr * self.W1_grad
        self.W2 -= lr * self.W2_grad


if __name__ == "__main__":

    net = DenseNet(2, 3, 512)
    x = [1, -1]
    y = [0, 3, -3]

    for i in range(100):
        z = net.forward(x)
        dz = z - y
        net.backward(dz)
        net.step()

    print(net.forward(x))
