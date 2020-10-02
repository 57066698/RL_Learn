import torch
import torch.nn as nn
import numpy as np

print(torch.cuda.is_available())

a = torch.tensor(np.zeros((3, 3)))
a = a.float().cuda()
# a = a.flatten()

b = torch.tensor([[0, 2], [1, 0]], dtype=torch.long).cuda()
a[b] = 1.0

print(a.detach().cpu().numpy())