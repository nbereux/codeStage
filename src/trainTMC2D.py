import numpy as np
import matplotlib.pyplot as plt
import torch
from TMCRBM2D import TMCRBM2D


device = torch.device("cuda")
dtype = torch.float
torch.set_num_threads(4)

#data = torch.tensor(np.genfromtxt('../dataset/data_1d2c_bal_seed14.dat'), device=device, dtype=dtype)
data = torch.tensor(np.genfromtxt('../dataset/data_2d.dat'), device = device, dtype = dtype)
data = (data+1)/2

Nv = data.shape[1]
Nh = 100

verbose = 0
save_fig = True

lr = 0.01
NGibbs = 30
it_mean = 20

mb_s = 200
num_pcd = 200
ep_max = 100
N = 20000
nb_chain = 10
nb_point_dim = torch.tensor([50,50])

stamp = 'TMC2DRBM_NGibbs_'+str(NGibbs)+'_Nh'+str(Nh)+'_Nv' + str(Nv)+'_Nmb'+str(mb_s)+'_Nepoch'+str(ep_max)+'_lr_'+str(lr) + '_N' + str(N) + '_Npoint' + str(nb_point_dim.prod().item()) + '_Nchain' + str(nb_chain)
myRBM = TMCRBM2D(num_visible=Nv,
            num_hidden=Nh,
            device=device,
            lr=lr,
            gibbs_steps=NGibbs,
            UpdCentered=True,
            mb_s=mb_s,
            num_pcd=num_pcd,
            ResetPermChainBatch=False,
            CDLearning=True,
            N = N,
            nb_chain = nb_chain,
            nb_point_dim=nb_point_dim,
            verbose=verbose,
            save_fig = save_fig
            )

fq_msr_RBM = 10
myRBM.file_stamp = stamp
base = 1.7
v = np.array([0,1],dtype=int)
allm = np.append(np.array(0),base**np.array(list(range(30))))
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

myRBM.list_save_rbm = np.arange(1,ep_max,fq_msr_RBM)
myRBM.SetVisBias(data.T)
myRBM.fit(data.T, ep_max)

