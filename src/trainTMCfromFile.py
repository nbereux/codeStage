import numpy as np
import matplotlib.pyplot as plt
import torch
import h5py
from TMCRBM import TMCRBM

device = torch.device("cuda")
dtype = torch.float
torch.set_num_threads(4)

#data = np.genfromtxt('../dataset/data_1d3c.dat')
data = np.genfromtxt('../dataset/data_5.dat')
data = torch.tensor((data+1)/2, device=device, dtype=dtype)

fmodel = h5py.File("../model/AllParametersTMCRBM_NGibbs_50_Nh40_Nv1000_Nmb200_Nepoch1000_lr_0.1_N20000_Npoint1000_Nchain15_phase3.h5", "r")

Nv = data.shape[1]
Nh = 40

verbose = 0
save_fig = True

lr = 0.01
NGibbs = 50
it_mean = 20

mb_s = 200
num_pcd = 200
ep_max = 1000
N = 20000
nb_chain = 15
nb_point = 1000
stamp = 'TMCRBM_NGibbs_'+str(NGibbs)+'_Nh'+str(Nh)+'_Nv' + str(Nv)+'_Nmb'+str(mb_s)+'_Nepoch'+str(ep_max)+'_lr_'+str(lr) + '_N' + str(N) + '_Npoint' + str(nb_point) + '_Nchain' + str(nb_chain) + '_phase3'

myRBM = TMCRBM(num_visible=Nv,
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
            nb_point=nb_point,
            verbose=verbose,
            save_fig = save_fig
            )

fq_msr_RBM = 1000
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

alltimes = []
for t in fmodel['alltime'][:]:
    if 'W'+str(t) in fmodel:
        alltimes.append(t)
t = alltimes[-1]
myRBM.W = torch.tensor(fmodel['W'+str(t)], device = myRBM.device)
myRBM.Nh = myRBM.W.shape[0]
myRBM.hbias = torch.tensor(fmodel['hbias'+str(t)], device = myRBM.device)
myRBM.vbias = torch.tensor(fmodel['vbias'+str(t)], device = myRBM.device)
fmodel.close()
myRBM.fit(data.T, ep_max)
