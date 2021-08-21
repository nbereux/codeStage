from RBM import GBRBM
import torch
import h5py
import numpy as np
import pathlib
path = str(pathlib.Path(__file__).parent.absolute())+'/'
#path = str(pathlib.Path(__file__).absolute())+'/'
import sys
sys.path.insert(1, '/home/nberieux/codeStage/src')
sys.path.insert(1, '/home/nberieux/codeStage/data')

device = torch.device("cuda")
dtype = torch.float
torch.set_num_threads(4)

data = torch.tensor(torch.load(path+"../data/yeast.pt"), device = device, dtype = dtype)

Nh = 100  # number of hidden nodes
lr_W1 = 0.01
lr_W2 = 0.0001
NGibbs = 100
n_mb = 50
n_pcd = n_mb
ep_max = 1000

fq_msr_RBM = 1000

Nv = data.shape[0]  # numbder of visible nodes
var_set = False
var_fold = "var_est"
if var_set:
    var_fold = "var_set"

myRBM = GBRBM(num_visible=Nv,
                   num_hidden=Nh,
                   device=device,
                   lr_W1=lr_W1,
                   lr_W2=lr_W2,
                   gibbs_steps=NGibbs,
                   UpdCentered=False,
                   mb_s=n_mb,
                   num_pcd=n_pcd,
                   var_set = var_set)

myRBM.ResetPermChainBatch = True  # Put False for PCD, False give Rdm
stamp = 'GBRBM_YEAST_NGibbs'+str(NGibbs)+'_Nh'+str(Nh)+'_Nmb'+str(n_mb)+'_Nepoch'+str(ep_max)+'_'+var_fold+'_lrW1'+str(lr_W1)+'_lrW2'+str(lr_W2)
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
myRBM.fit(data, ep_max = ep_max)

print("model updates saved at "+ path + "../model/AllParameters"+stamp+".h5")
print("model saved at "+ path +"../model/RBM"+stamp+".h5")