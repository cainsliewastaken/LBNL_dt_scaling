import torch.nn.functional as F
from utilities3 import *
from timeit import default_timer

import numpy as np
import torch
# print(torch.__version__)

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
#from torchinfo import summary
import sys
import netCDF4 as nc
import scipy.io as sio


def load_train_data (ind1,Ndecomp):

    Nx=256
    Ny=256
    input_data = np.zeros([Ndecomp,1,Nx,Ny])
    label_data = np.zeros([Ndecomp,1,Nx,Ny])

    count=0
    for k in range(ind1,ind1+Ndecomp):
    #  print(k)
     F1='/global/homes/c/cainslie/LBL_J_Spectral_Stability/training_data_scratch/data_lowres/' +str(k)+'.mat'
    #  F2='/global/homes/c/cainslie/LBL_J_Spectral_Stability/training_data_scratch/data_lowres/' +str(k+1)+'.mat'
     f1= sio.loadmat(F1)
    # #  f2= sio.loadmat(F2)
     omega_input = f1['Omega']
    #  omega_output= f2['Omega'] 
     input_data[count,0,:,:]=omega_input
    #  label_data[count,0,:,:]=omega_output
     count=count+1

    return torch.from_numpy(input_data).float()#, torch.from_numpy(label_data).float()



def load_test_data(Ntrain,N_test):

    Nx=256
    Ny=256
    input_data = np.zeros([N_test,1,Nx,Ny])
    label_data = np.zeros([N_test,1,Nx,Ny])

    count=0
    for k in range(Ntrain,Ntrain+N_test):

     F1='/global/homes/c/cainslie/LBL_J_Spectral_Stability/training_data_scratch/data_lowres/' +str(k)+'.mat'
    #  F2='/global/homes/c/cainslie/LBL_J_Spectral_Stability/training_data_scratch/data_lowres/' +str(k+1)+'.mat'
     f1= sio.loadmat(F1)
    # #  f2= sio.loadmat(F2)
     omega_input = f1['Omega']
    #  omega_output= f2['Omega'] 
     input_data[count,0,:,:]=omega_input
    #  label_data[count,0,:,:]=omega_output
     count=count+1

    return torch.from_numpy(input_data).float()#, torch.from_numpy(label_data).float()


def load_train_data_v2(ind1, Ndecomp):
    Nx=256
    Ny=256
    input_data = np.zeros([Ndecomp,1,Nx,Ny])
    label_data = np.zeros([Ndecomp,1,Nx,Ny])

    count=0
    for k in range(ind1, ind1+Ndecomp):
    #  print(k)
     F1='/global/homes/c/cainslie/LBNL_dt_scaling/training_data/py2d_turb_sims/results/Re1000_fkx4fky4_r0.1_b20/NoSGS/NX256/dt0.0001_IC1/data/' +str(ind1+count)+'.mat'
    #  F2='/global/homes/c/cainslie/LBL_J_Spectral_Stability/training_data_scratch/py2d_turb_sims/results/Re1000_fkx4fky4_r0.1_b20/NoSGS/NX256/dt0.0001_IC1/data/' +str(ind1+lead*(1+count))+'.mat'
     f1= sio.loadmat(F1)
    #  f2= sio.loadmat(F2)
     omega_input = f1['Omega']
    #  omega_output= f2['Omega'] 
     input_data[count,0,:,:]=omega_input
    #  label_data[count,0,:,:]=omega_output
     count=count+1
    return torch.from_numpy(input_data).float()#, torch.from_numpy(label_data).float()



def load_test_data_v2(Ntrain, N_test, lead):
    Nx=256
    Ny=256
    input_data = np.zeros([N_test,1,Nx,Ny])
    label_data = np.zeros([N_test,1,Nx,Ny])

    count=0
    for k in range(Ntrain,Ntrain+N_test):
     F1='/global/homes/c/cainslie/LBNL_dt_scaling/training_data/py2d_turb_sims/results/Re1000_fkx4fky4_r0.1_b20/NoSGS/NX256/dt0.0001_IC1/data/' +str(Ntrain+lead*count)+'.mat'
     F2='/global/homes/c/cainslie/LBNL_dt_scaling/training_data/py2d_turb_sims/results/Re1000_fkx4fky4_r0.1_b20/NoSGS/NX256/dt0.0001_IC1/data/' +str(Ntrain+lead*(1+count))+'.mat'
     f1= sio.loadmat(F1)
     f2= sio.loadmat(F2)
     omega_input = f1['Omega']
     omega_output= f2['Omega'] 
     input_data[count,0,:,:]=omega_input
     label_data[count,0,:,:]=omega_output
     count=count+1

    return torch.from_numpy(input_data).float(), torch.from_numpy(label_data).float()


def load_data_with_lead(start_ind, sequence_len, lead):
    Nx=256
    Ny=256
    input_data = np.zeros([sequence_len, 1, Nx,Ny])
    # label_data = np.zeros([sequence_len, 1, Nx,Ny])
    count=0
    for k in range(start_ind, start_ind+sequence_len):
     F1='/global/homes/c/cainslie/LBNL_dt_scaling/training_data/py2d_turb_sims/results/Re1000_fkx4fky4_r0.1_b20/NoSGS/NX256/dt0.0001_IC1/data/' +str(start_ind+lead*count)+'.mat'
     f1= sio.loadmat(F1)
     omega_input = f1['Omega']
     input_data[count,0,:,:]=omega_input
     count=count+1

    return torch.from_numpy(input_data).float()

