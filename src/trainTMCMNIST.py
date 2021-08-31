import numpy as np
import matplotlib.pyplot as plt
import torch
from TMCRBM import TMCRBM
import gzip
import pickle


device = torch.device("cuda")
dtype = torch.float
torch.set_num_threads(4)

f = gzip.open('../dataset/mnist.pkl.gz', 'rb')
u = pickle._Unpickler(f)
u.encoding = 'latin1'
p = u.load()
train_set, _, _ = p

data = torch.as_tensor(train_set[0][:10000,:].T, device = device, dtype = dtype)

Nv = data.shape[0]
Nh = 100

verbose = 0
save_fig = False

lr = 0.01
NGibbs = 50
it_mean = 20

mb_s = 200
num_pcd = 200
ep_max = 1000
N = 20000
nb_chain = 15
nb_point = 1000

stamp = 'TMCRBM_MNIST_NGibbs_'+str(NGibbs)+'_Nh'+str(Nh)+'_Nv' + str(Nv)+'_Nmb'+str(mb_s)+'_Nepoch'+str(ep_max)+'_lr_'+str(lr) + '_N' + str(N) + '_Npoint' + str(nb_point) + '_Nchain' + str(nb_chain)
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
myRBM.SetVisBias(data)
myRBM.fit(data, ep_max)

