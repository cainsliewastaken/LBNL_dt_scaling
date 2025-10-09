import torch.nn.functional as F
from utilities3 import *
from timeit import default_timer

import numpy as np
import torch
print(torch.__version__)

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
#from torchinfo import summary
import sys
import netCDF4 as nc

def denormalize (m1,s1,autoreg_pred):

    autoreg_pred_denorm1 = autoreg_pred*s1+m1
    return autoreg_pred_denorm1

