import numpy as np
import matplotlib.pyplot as plt
import torch
import sys

from RBM import RBM
from scipy.integrate import simps
from scipy import stats


device = torch.device("cuda")
dtype = torch.float
torch.set_num_threads(4)

data = np.genfromtxt('../data/data_1d2c_bal_seed14.dat')
data = torch.tensor((data+1)/2, device=device, dtype=dtype)


lr = 0.01
NGibbs = 100
annSteps = 0
mb_s = 600
num_pcd = 100
Nh = 20
Nv = data.shape[1]
ep_max = 100
w_hat = torch.linspace(0, 1, steps=100)
_, _, V = torch.svd(data)
V = V[:, 0]
N = 20000
it_mean = 50

myRBM = RBM(num_visible=Nv,
            num_hidden=Nh,
            device=device,
            lr=lr,
            gibbs_steps=NGibbs,
            UpdCentered=False,
            mb_s=mb_s,
            num_pcd=num_pcd,
            w_hat=w_hat,
            N=N,
            it_mean=it_mean,
            V=V,
            TMCLearning=True
            )

myRBM.fit(data.T, ep_max)
