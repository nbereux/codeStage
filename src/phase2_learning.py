from RBM import GBRBM
from genGaussianData import genGaussianData
import torch
import h5py
import argparse
import gzip
import pickle
import sys
import os
import numpy as np
import pathlib
path = str(pathlib.Path(__file__).parent.absolute())+'/'
# sys.path.insert(1, '/home/nicolas/Stage/code/stage/src')
# sys.path.insert(1, '/home/nicolas/Stage/code/stage/data')

device = torch.device(torch.device(
    'cuda' if torch.cuda.is_available() else 'cpu'))
#device = torch.device("cpu")
dtype = torch.float

fname = path+"../model/AllParametersGBRBM_NGibbs_CLEAN_MNIST_50_Nh500_Ns10000_Nmb100_var_set_reg_100_x1_lr_0.01.h5"
f = h5py.File(fname, 'r')
data_t = "CLEAN_MNIST"
ep_max = 1000
Nv = torch.tensor(f['W_10'], device=device).shape[1]
Nh = torch.tensor(f['W_10'], device=device).shape[0]
lr_W1 = 0.01
lr_W2 = 0.0001
NGibbs = 50
n_mb = 100
n_pcd = n_mb
var_set = True
myRBM = GBRBM(num_visible=Nv,
              num_hidden=Nh,
              device=device,
              lr_W1=lr_W1,
              lr_W2=lr_W2,
              gibbs_steps=NGibbs,
              UpdCentered=False,
              mb_s=n_mb,
              num_pcd=n_pcd,
              var_set=var_set)
fq_msr_RBM = 1000

dataFile = path+"../data/cleanMNIST10000.h5"
dataf = h5py.File(dataFile, 'r')
X = torch.tensor(dataf['clean'], device=device)
X = X/torch.std(X, 1).reshape(X.shape[0], 1)


alltimes = []
for t in f['alltime'][:]:
    if 'W_1'+str(t) in f:
        alltimes.append(t)

t = alltimes[-1]
myRBM.W_1 = torch.tensor(f['W_1'+str(t)], device=myRBM.device)
myRBM.W_2 = torch.tensor(f['W_2'+str(t)], device=myRBM.device)
#myRBM.sigV = torch.tensor(f['sigV'], device = myRBM.device)
myRBM.vbias = torch.tensor(
    f['vbias'+str(t)], device=myRBM.device)
myRBM.hbias = torch.tensor(
    f['hbias'+str(t)], device=myRBM.device)

myRBM.ResetPermChainBatch = True  # Put False for PCD, False give Rdm
stamp = "_phase_2_GBRBM_NGibbs_CLEAN_MNIST_50_Nh500_Ns10000_Nmb100_var_set_reg_100_x1_lr_0.01"
myRBM.file_stamp = stamp
base = 1.7
v = np.array([0, 1], dtype=int)
allm = np.append(np.array(0), base**np.array(list(range(30))))
for k in range(30):
    for m in allm:
        v = np.append(v,int(base**k)+int(m)) 
v = np.array(list(set(v)))
v = np.sort(v)
myRBM.list_save_time = v

v = np.array(list(set(v)))
v = np.sort(v)
myRBM.list_save_time = v
myRBM.list_save_rbm = np.arange(1, ep_max, fq_msr_RBM)
fq_msr_RBM = 1000
myRBM.list_save_rbm = np.arange(1,ep_max,fq_msr_RBM)
myRBM.fit(X, ep_max = ep_max)
print("model updates saved at "+ path + "../model/AllParameters"+stamp+".h5")
print("model saved at "+ path +"../model/RBM"+stamp+".h5")

