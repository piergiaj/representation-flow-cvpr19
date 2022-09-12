'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np 

a_list = [[1,1,1,1,1],[2,2,2,2,2]]
a_np = np.array(a_list)

print(a_np[:,:])
print(a_np[:,:-1])

value = torch.FloatTensor([[[[-0.5,0,0.5]]]])

# torch.Size([1, 1, 1, 3])
print(value.shape)
print(value)


value1 = value.transpose(3,2)

print(value1.shape)
print(value1)

value_repeat = value.repeat(8,8,1,1)

# torch.Size([8, 8, 1, 3])
print(value_repeat.shape)
'''

import torch

a = [2,3,1,2,31,3,1,2,3,2]

a_tensor = torch.FloatTensor(a)
v1 = torch.zeros_like(a_tensor.data)
a_bool = (a_tensor>3)

v1[a_bool] = a_tensor[a_bool]


print(a_bool)
print(a_bool==False)
print(v1)