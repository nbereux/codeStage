from scipy.integrate import simps
from RBM import RBM
import numpy as np
import matplotlib.pyplot as plt
import torch
import sys
import time
sys.path.insert(1, '/home/nicolas/code/src')
sys.path.insert(1, '/home/nicolas/code/data')


device = torch.device("cuda")
dtype = torch.float
torch.set_num_threads(4)


def TMCSample(v, w_hat, N, V, it_mcmc=100, ß=1):
    vtab = torch.zeros((v.shape[0], it_mcmc+1))
    vtab[:, 0] = v[:, 0]
    v_curr = v.cpu()
    V = V.cpu()
    norm = 1/(v_curr.shape[0]**0.5)
    w_curr = (torch.dot(v_curr[:, 0], V)*norm).item()
    h_curr, _ = myRBM.SampleHiddens01(v_curr.cuda())

    # print(w_curr)
    for t in range(it_mcmc):
        start = time.time()
        h_curr, _ = myRBM.SampleHiddens01(v_curr.cuda())
        h_i = (torch.mm(myRBM.W.T, h_curr) +
               myRBM.vbias.reshape(v.shape[0], 1)).cpu()
        w_next = w_curr

        v_next = torch.clone(v_curr)
        #index = torch.randperm(v_curr.shape[0])

        for idx in range(v_curr.shape[0]):
            #i = index[idx]
            i = idx
            # start_2 = time.time()
            v_next[i] = 1-v_curr[i]
            # print(h_i.shape)
            # print(w_hat)
            #print("bef ",w_next)
            w_next += ((2*v_next[i, 0]-1)*V[i]*norm).item()
            #print("aft ",w_next)
            # print(w_curr-w_next)
            #w_next = torch.dot(v_next[:,0], V)/(v.shape[0]**0.5)
            # On calcul -DeltaE
            ΔE = ß*((2*v_next[i]-1)*h_i[i])-(N/2) * \
                ((w_hat-w_next)**2-(w_hat-w_curr)**2)
            # tmp = torch.exp(ß*((2*v_next[i]-1)*h_i[i])-N*((w_hat-w_next)**2-(w_hat-w_curr)**2))
            # print(tmp.shape)
            # prob = min(1, tmp)
            if ΔE >= 0:
                # print("t1")
                v_curr[i] = v_next[i]
                w_curr = w_next
            elif (torch.rand(1, 1, device=torch.device("cpu")) < torch.exp(ΔE)):
                # print("t1")
                v_curr[i] = v_next[i]
                w_curr = w_next
            else:
                w_next = w_curr
                v_next[i] = 1-v_curr[i]

            #print("upd 1 var exec time = ", start_2-time.time())
        vtab[:, t+1] = v_curr[:, 0]
        #print("it_mcmc exec time = ", time.time()-start)
    return v_curr, h_curr, vtab


W = np.genfromtxt('../data/C1d5c/rbm_W.dat').T
vbias = np.genfromtxt('../data/C1d5c/rbm_vis.dat')
hbias = -np.genfromtxt('../data/C1d5c/rbm_hid.dat')
data = np.genfromtxt('../data/C1d5c/data_5.dat')
data = (data+1)/2


lr = 0.01
l2 = 0
NGibbs = 10
annSteps = 0
mb_s = 500
num_pcd = 500
Nh = W.shape[0]
Nv = W.shape[1]

ep_max = 100

myRBM = RBM(num_visible=Nv,
            num_hidden=Nh,
            device=device,
            lr=lr,
            # regL2=l2,
            gibbs_steps=NGibbs,
            # anneal_steps=annSteps,
            UpdCentered=True,
            # mb_self=mb_s,
            num_pcd=num_pcd)

myRBM.W = torch.tensor(4*W).float().cuda()
myRBM.vbias = torch.tensor(2*vbias - 2*W.sum(0)).float().to(device)
myRBM.hbias = torch.tensor(2*hbias - 2*W.sum(1)).float().to(device)
U, S, V = torch.svd(torch.tensor(4*W).float().cuda())

start_points = torch.bernoulli(torch.rand(myRBM.Nv, 1000, device=device))
arrival, _, _, _ = myRBM.Sampling(start_points, it_mcmc=1000)
proj_gen = torch.mm(arrival.T, V).cpu()/myRBM.Nv**0.5

it_mean = [90, 350, 1500]
it_mc = [100, 400, 1600]
for it in range(len(it_mc)):
    start = torch.bernoulli(torch.rand(myRBM.Nv, 1, device=device))
    V0 = V[:, 0]
    # w_hat = torch.dot(start.T, V)[0:,]
    w_hat = torch.linspace(0, 1, steps=100)
    y = []
    N = 10000
    for i in range(len(w_hat)):
        print(i)
        tmpv, tmph, vtab = TMCSample(start, w_hat[i], N, V0, it_mcmc=it_mc[i])
        y.append(torch.mean(torch.dot(vtab[:, -it_mean[i]].T.cpu(), V0.cpu())))
    y = np.array(y)/myRBM.Nv**0.5
    res = np.zeros(len(w_hat)-1)
    print(simps(y-w_hat.numpy(), w_hat.numpy()))
    for i in range(1, len(w_hat)):
        res[i-1] = simps(y[:i]-w_hat[:i].numpy(), w_hat[:i].numpy())
    const = 1/res[-1]
    p_m = np.exp(N*res)*const
    potential = res + (1/N)*np.log(res[-1])
    #plt.plot(w_hat[1:], potential)
    plt.plot(w_hat[1:], res)

    proj_data = torch.mm(torch.tensor(data, device=device,
                                      dtype=dtype), V).cpu()/myRBM.Nv**0.5
    plt.figure(dpi=200)
    plt.plot(w_hat[1:], potential, color="green", label="potential")
    plt.plot(w_hat, y-w_hat.numpy(), color='red', label="grad potential")
    plt.hlines(0, 0, 1, color='black')
    plt.scatter(proj_gen[:, 0], proj_gen[:, 1], alpha=0.2, label='data_gen')
    rdm_y = torch.randn(proj_data[:, 0].shape)/20
    # plt.plot(proj_data[:,0],rdm_y,'o',markersize=1,alpha=0.3)
    # plt.hist(proj_data[:,0].numpy(), label = 'data', density=True, bins=100)
    plt.xlabel("w_hat")
    plt.legend()
    plt.savefig('../fig/TMC_IT_'+str(it))
    plt.close()
