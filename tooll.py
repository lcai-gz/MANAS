import numpy as np
import math
import torch
import os
import torch.nn as nn
from operations import *
from genotypes import *
from Models import *

os.environ['CUDA_VISIBLE_DEVICES'] = '2'
model = MixedSearchSpace(steps=2)
input = torch.randn(4, 3, 64, 64)
model.cuda()
input = input.cuda()
out = model(input)




