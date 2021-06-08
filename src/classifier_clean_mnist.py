from torch.optim.lr_scheduler import StepLR
from ClassifierMetric import ComparisonDataset
from ClassifierMetric import BinaryClassifier, train, test
import torch.optim as optim
import torch
import functions
import numpy as np
import gzip
import pickle
import sys
import argparse
sys.path.insert(1, '/home/nicolas/Stage/code/stage/src')
sys.path.insert(1, '/home/nicolas/Stage/code/stage/data')
sys.path.insert(1, '/home/nicolas/Stage/code/stage/model')
import pathlib
import h5py
import RBM
path = str(pathlib.Path(__file__).parent.absolute())+'/'

device = torch.device("cuda")
dtype = torch.float

parser = argparse.ArgumentParser()
parser.add_argument("filename", help="name of the machine")
parser.add_argument("ngibbs_gen", help="Ngibbs for data generation", type = int, default = 10000)
parser.add_argument("time", help="Time at which we evaluate the machine", type = int, default = 1)
args = parser.parse_args()


f = gzip.open(path+'../data/mnist.pkl.gz', 'rb')
u = pickle._Unpickler(f)
u.encoding = 'latin1'
p = u.load()
train_set, _, _ = p
X = train_set[0]
X = (X > 0.5)*1.0
print(X.shape)
data_mnist = torch.as_tensor(
    X[:10000, :].T, device=device, dtype=dtype)
labels_mnist = torch.tensor(np.zeros(len(data_mnist[0])), device=device)

train_kwargs = {'batch_size': 128}
test_kwargs = {'batch_size': 1000}
cuda_kwargs = {'num_workers': 0,
               'pin_memory': False,
               'shuffle': False}
train_kwargs.update(cuda_kwargs)
test_kwargs.update(cuda_kwargs)

ffname = path+"../data/cleanMNIST10000.h5"
ff = h5py.File(ffname, 'r')
var_rm = ff['id_col']
data_mnist = torch.tensor(ff['original'], device = device)





success_rate = []
fname = path + "../model/" + args.filename
f = h5py.File(fname, 'r')
Nh = f['W_10'].shape[0]
Nv = f['W_10'].shape[1]

lr = 0.01
NGibbs = 100
mb_s = 500
num_pcd = 500
eps = 100
alltimes = []
for t in f['alltime'][:]:
    if 'W_1'+str(t) in f:
        alltimes.append(t)
myRBM = RBM.GBRBM(num_visible=Nv,
                   num_hidden=Nh,
                   device=device,
                   lr_W1=lr,
                   lr_W2=lr/eps,
                   gibbs_steps=NGibbs,
                   UpdCentered=False,
                   mb_s=mb_s,
                   num_pcd=num_pcd,
                   var_set = True)


times = np.array(alltimes)[range(len(alltimes))]
alls = []
base = 1.7
v = np.array([0,1],dtype=int)
for k in range(30):
	v = np.append(v,int(base**k))
for t2 in v:
    at = np.abs(times-t2)
    idx = np.argmin(at)
    t = times[idx]

    myRBM.W_1 = torch.tensor(f['W_1'+str(t)], device=myRBM.device)
    myRBM.W_2 = torch.tensor(f['W_2'+str(t)], device=myRBM.device)
    myRBM.vbias = torch.tensor(
        f['vbias'+str(t)], device=myRBM.device)
    myRBM.hbias = torch.tensor(
        f['hbias'+str(t)], device=myRBM.device)
    vinit = torch.bernoulli(torch.rand(
        (myRBM.Nv, 10000), device=myRBM.device, dtype=myRBM.dtype))

    tmp, _, _, _ = myRBM.Sampling(vinit, it_mcmc=myRBM.gibbs_steps)
    rebuiltMNIST = torch.zeros(data_mnist.shape[0], tmp.shape[1], device = myRBM.device)
    passed = 0
    for i in range(data_mnist.shape[0]):
        if np.isin(i, var_rm):
            rebuiltMNIST[i, :] = torch.zeros(rebuiltMNIST[i, :].shape)
            passed +=1
        else :
            rebuiltMNIST[i, :] = tmp[i-passed, :]
    si = rebuiltMNIST
    data_gen = torch.tensor(si, device=device)
    labels_gen = torch.tensor(np.ones(len(data_gen[0])), device=device)
    data = torch.cat((data_gen, data_mnist), dim=1)
    labels = torch.cat((labels_gen, labels_mnist), dim=0)
    data = data.T
    shuffle_index = torch.randperm(data.shape[0])
    newdata = torch.empty(data.shape)
    newlabel = torch.empty(labels.shape)
    for i in range(shuffle_index.shape[0]):
        newdata[i] = data[shuffle_index[i]]
        newlabel[i] = labels[shuffle_index[i]]

    net = BinaryClassifier()
    net.cuda()
    train_set = ComparisonDataset(
        device, data=newdata, labels=newlabel, train=True)
    test_set = ComparisonDataset(
        device, data=newdata, labels=newlabel, train=False)
    train_loader = torch.utils.data.DataLoader(
        train_set, **train_kwargs)
    test_loader = torch.utils.data.DataLoader(test_set, **test_kwargs)
    optimizer = optim.Adadelta(net.parameters(), lr=0.1)

    scheduler = StepLR(optimizer, step_size=1, gamma=0.7)
    tmp = []
    for epoch in range(1, 100 + 1):
        train(net, device, train_loader, optimizer, epoch)
        test(net, device, test_loader)
        scheduler.step()
        tmp.append(torch.linalg.norm(net(test_set.data[:2500].cuda()).round().view(
            2500)-test_set.label[:2500].cuda(), 1).item()*100/2500)
    print(tmp[-1])
    success_rate.append(tmp[-1])
    torch.save(success_rate, path + "../data/classification_error_" +
               args.filename+".pt")
