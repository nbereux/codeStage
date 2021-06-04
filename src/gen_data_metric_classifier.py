from torch.optim.lr_scheduler import StepLR
from ComparisonDataset import ComparisonDataset
from BinaryClassifier import BinaryClassifier, train, test
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

device = torch.device("cuda")
dtype = torch.float

parser = argparse.ArgumentParser()
parser.add_argument("filename", help="name of the machine")
args = parser.parse_args()


f = gzip.open('./data/mnist.pkl.gz', 'rb')
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


success_rate = []
fname = "model/" + args.filename
myRBM, f, alltimes = functions.retrieveRBM(device, fname)

times = np.array(alltimes)[range(len(alltimes))]

for t in [times[-1]]:
    myRBM.W = torch.tensor(f['paramW'+str(t)], device=myRBM.device)
    myRBM.vbias = torch.tensor(
        f['paramVB'+str(t)], device=myRBM.device)
    myRBM.hbias = torch.tensor(
        f['paramHB'+str(t)], device=myRBM.device)
    vinit = torch.bernoulli(torch.rand(
        (myRBM.Nv, 10000), device=myRBM.device, dtype=myRBM.dtype))

    si, _, _, _ = myRBM.Sampling(vinit, it_mcmc=10000)
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
    success_rate.append(tmp[-1])
    torch.save(success_rate, "data/classification_error_" +
               args.filename+".pt")
