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

parser = argparse.ArgumentParser()
parser.add_argument(
    "mode", help="0 to train, 1 to generate data", type=int, default=0)
parser.add_argument(
    "var_set", help="1 if the variance is set, 0 else", type=int, default=1)
parser.add_argument(
    "reg", help="10,100 or 1000 to divide the lr for W_2", type=int, default=100)
parser.add_argument(
    "gen_mult", help="1 to multiply ngibbs for gen by 10, 0 else", type=int, default=0)
parser.add_argument("--lr", help="learning rate for W_1",
                    type=float, default=0.01)
parser.add_argument("--Ngibbs", help="Ngibbs for training",
                    type=int, default=10)
parser.add_argument("--Nmb", help="size of minibatches", type=int, default=50)
parser.add_argument("--ep_max", help="number of epochs", type=int, default=200)
parser.add_argument("--Nh", help="number of hidden nodes",
                    type=int, default=50)
parser.add_argument(
    "--data_gen", help="1 to use generated data, 0 to use MNIST, 2 for YEAST", type=int, default=1)
parser.add_argument("--var", help="the var list ", type=float, nargs='+', default=1)
args = parser.parse_args()

device = torch.device(torch.device(
    'cuda' if torch.cuda.is_available() else 'cpu'))
dtype = torch.float

dim = 100
Nsample = 10000
l_cube = 5
Nh = args.Nh  # number of hidden nodes
lr = args.lr  # learning rate
NGibbs = args.Ngibbs  # number of gibbs steps
n_mb = args.Nmb  # size of minibatches
n_pcd = n_mb  # size of the negative chain
ep_max = args.ep_max
fq_msr_RBM = 1000
var = torch.tensor([0.5, 1, 1.5])
var_str = ''
for i in range(len(var)):
    var_str += str(var[i].item())+'_'

n_centers = var.shape[0]

print("------------------------------------------------------------------")
if args.var_set == 1:
    varfold = "var_set"
    print(varfold)
    var_set = True
else:
    varfold = "var_est"
    print(varfold)
    var_set = False
if args.reg == 10:
    divfold = "reg_10"
    print(divfold)
    eps = 10
elif args.reg == 100:
    divfold = "reg_100"
    print(divfold)
    eps = 100
else:
    divfold = "reg_1000"
    print(divfold)
    eps = 1000

if args.gen_mult == 0:
    nb_genfold = "x1"
    print(nb_genfold)
    Ngibbs_gen = NGibbs
else:
    nb_genfold = "x10"
    print(nb_genfold)
    Ngibbs_gen = NGibbs*10


if args.data_gen == 1:
    fname = 'data_Nclusters'+str(n_centers)+'_dim' + \
        str(dim)+'_Lcube'+str(l_cube)+'_var'+var_str+'.h5'
    data_t = "data_gen"
    if args.mode == 0:
        if os.path.isfile(fname):
            f = h5py.File(path+'../data/'+fname, 'r')
            centers = torch.tensor(f['centers'], device=device)
            var = torch.tensor(f['var'], device=device)
            X = torch.tensor(f['data'], device=device)
        else:
            genGaussianData(dim, Nsample, l_cube, torch.tensor(var))
            f = h5py.File(path+'../data/'+fname, 'r')
            centers = torch.tensor(f['centers'], device=device)
            var = torch.tensor(f['var'], device=device)
            X = torch.tensor(f['data'], device=device)
        f.close()

    elif args.mode == 1:
        f = h5py.File(path+'../data/'+fname, 'r')
        centers = torch.tensor(f['centers'], device = device)
        var = torch.tensor(f['var'], device = device)
        X = torch.tensor(f['data'], device = device)

elif args.data_gen == 0:
    data_t = "MNIST"
    f = gzip.open(path+'../data/mnist.pkl.gz', 'rb')
    u = pickle._Unpickler(f)
    u.encoding = 'latin1'
    p = u.load()
    train_set, _, _ = p
    X = torch.as_tensor(train_set[0][:10000,:].T, device = device, dtype = dtype)
    centers = 0
elif args.data_gen == 2:
    data_t = "YEAST"
#    X = torch.tensor(torch.load(path+"../data/yeast.pt"), device = device, dtype = dtype)
    centers = 0
else :
    data_t = "CLEAN MNIST"
    dataFile = path+"../data/cleanMNIST10000.h5"
    dataf = h5py.File(dataFile, 'r')
    X = torch.tensor(dataf['clean'], device = device)
    print(X.shape)
    X = X/torch.std(X,1).reshape(X.shape[0],1)
    centers=0
Nv = X.shape[0]  # numbder of visible nodes




myRBM = GBRBM(num_visible=Nv,
                   num_hidden=Nh,
                   device=device,
                   lr_W1=lr,
                   lr_W2=lr/eps,
                   gibbs_steps=NGibbs,
                   UpdCentered=False,
                   mb_s=n_mb,
                   num_pcd=n_pcd,
                   var_set = var_set)

if args.mode == 0:
    print("TRAIN")
    myRBM.SetVisBias(X)  # initialize the visible biases
    myRBM.ResetPermChainBatch = True  # Put False for PCD, False give Rdm
    stamp = 'GBRBM_NGibbs_'+data_t+'_'+str(NGibbs)+'_Nh'+str(Nh)+'_Ns' + \
        str(Nsample)+'_Nmb'+str(n_mb)+'_'+varfold+'_'+divfold+'_'+nb_genfold+'_lr_'+str(lr)
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
    myRBM.centers = centers
    fq_msr_RBM = 1000
    myRBM.list_save_rbm = np.arange(1,ep_max,fq_msr_RBM)
    myRBM.fit(X, ep_max = ep_max)
    print("model updates saved at "+ path + "../model/AllParameters"+stamp+".h5")
    print("model saved at "+ path +"../model/RBM"+stamp+".h5")
elif args.mode == 1:
    print("GENERATE DATA")
    frname = path+'../model/AllParametersGBRBM_NGibbs_'+data_t+'_'+str(NGibbs)+'_Nh'+str(Nh)+'_Ns' + \
        str(Nsample)+'_Nmb'+str(n_mb)+'_'+varfold+'_'+divfold+'_'+nb_genfold+'_lr_'+str(lr)+'.h5'
    fr = h5py.File(frname,'r')
    myRBM.centers = torch.load(path + "../data/centers_GBRBM.pt")
    Nh = fr['W_10'].shape[0] # get the visible shape
    Nv = fr['W_10'].shape[1] # get the hidden shape
    # usefull if you want to use the RBm
    alltimes = []
    for t in fr['alltime'][:]:
        if 'W_1'+str(t) in fr:
            alltimes.append(t)

    fwname = '../data/'+data_t+'_'+'_GBRBM_NGibbs'+str(NGibbs)+'_Nh'+str(Nh)+'_Ns' + \
        str(Nsample)+'_Nmb'+str(n_mb)+'_'+varfold+'_'+divfold+'_'+nb_genfold+'_lr_'+str(lr)+'.h5'
    fw = h5py.File(path+fwname, 'w')
    fw.create_dataset('alltime', data=alltimes)
    fw.close()
    for t in alltimes:
        myRBM.W_1 = torch.tensor(fr['W_1'+str(t)], device=myRBM.device)
        myRBM.W_2 = torch.tensor(fr['W_2'+str(t)], device=myRBM.device)
        myRBM.sigV = torch.tensor(fr['sigV'], device = myRBM.device)
        myRBM.vbias = torch.tensor(
            fr['vbias'+str(t)], device=myRBM.device)
        myRBM.hbias = torch.tensor(
            fr['hbias'+str(t)], device=myRBM.device)
        vinit = torch.bernoulli(torch.rand(
            (myRBM.Nv, 10000), device=myRBM.device, dtype=myRBM.dtype))

        si, _, _, _ = myRBM.Sampling(vinit, it_mcmc=Ngibbs_gen)

        var = [[] for i in range(myRBM.centers.shape[0])]
        c = [[] for i in range(myRBM.centers.shape[0])]
        for i in range(si.shape[1]):
            dist = []
            for k in range(myRBM.centers.shape[0]):
                dist.append(torch.linalg.norm(si.T[i]-myRBM.centers[k].cuda()))
                idx = np.argmin(dist)
                c[idx].append(si.T[i])
        for i in range(myRBM.centers.shape[0]):
            if len(c[i]) != 0:
                c[i] = torch.stack(c[i])
            else:
                c[i] = torch.zeros(si.T[0].shape)
            var[i].append(torch.mean(torch.std(c[i], 0)).cpu())
        var = np.array(var)
        fw = h5py.File(path+fwname, 'a')
        print('Saving nb_upd='+str(t))
        fw.create_dataset('dataIT'+str(t),
                          data=si.cpu())
        fw.create_dataset('varIT'+str(t), data=var)
        fw.close()


