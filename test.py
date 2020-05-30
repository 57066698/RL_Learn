import numpy as np
import torch

x = 10
y = 1/(1+np.exp(-x))

print(-np.log(1/y - 1))

