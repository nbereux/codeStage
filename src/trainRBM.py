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

data = np.genfromtxt('../dataset/data_2d.dat')
data = torch.tensor((data+1)/2, device=device, dtype=dtype)


lr = 0.01
NGibbs = 50
annSteps = 0
mb_s = 600
num_pcd = 600
Nh = 100
Nv = data.shape[1]
ep_max = 1000
w_hat = torch.linspace(0, 1, steps=100)
_, _, V = torch.svd(data)
V = V[:, 0]
N = 20000
it_mean = 50
fq_msr_RBM = 1000

myRBM = RBM(num_visible=Nv,
            num_hidden=Nh,
            device=device,
            lr=lr,
            gibbs_steps=NGibbs,
            UpdCentered=True,
            mb_s=mb_s,
            num_pcd=num_pcd,
            )

stamp = 'RBM_NGibbs_'+str(NGibbs)+'_Nh'+str(Nh)+'_Ns' + \
    str(Nv)+'_Nmb'+str(mb_s)+'_Nepoch'+str(ep_max) + \
    '_lr_'+str(lr)+"_TMCTEST2D_updCentered_TRUE"
myRBM.file_stamp = stamp
base = 1.7
v = np.array([0, 1], dtype=int)
allm = np.append(np.array(0), base**np.array(list(range(30))))
for k in range(30):
    for m in allm:
        v = np.append(v, int(base**k)+int(m))
v = np.array(list(set(v)))
v = np.sort(v)
myRBM.list_save_time = v

v = np.array(list(set(v)))
v = np.sort(v)
myRBM.list_save_time = v
myRBM.list_save_rbm = np.arange(1, ep_max, fq_msr_RBM)

fq_msr_RBM = 1000
myRBM.list_save_rbm = np.arange(1, ep_max, fq_msr_RBM)
myRBM.SetVisBias(data.T)
myRBM.fit(data.T, ep_max)
print("model updates saved at " + "../model/AllParameters"+stamp+".h5")
print("model saved at " + "../model/RBM"+stamp+".h5")
