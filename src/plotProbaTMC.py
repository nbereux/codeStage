import h5py
from scipy import stats
from scipy.integrate import simps
from RBM import RBM
import numpy as np
import matplotlib.pyplot as plt
import torch
import sys
sys.path.insert(1, '/home/nicolas/code/src')
sys.path.insert(1, '/home/nicolas/code/data')


device = torch.device("cuda")
dtype = torch.float
torch.set_num_threads(4)

data = np.genfromtxt('../data/data_1d2c_bal_seed14.dat')
data = torch.tensor((data+1)/2, device=device, dtype=dtype)


def TMCSample(v, w_hat, N, V, it_mcmc=100, it_mean=50, ß=1):
    # print("Initialisation")
    #s = time.time()
    vtab = torch.zeros(v.shape, device=device)
    v_curr = v
    V = V
    norm = 1/(v_curr.shape[0]**0.5)
    w_curr = (torch.mv(v_curr.T, V)*norm)

    index = torch.randperm(v_curr.shape[0])
    # print(time.time()-s)
    #print("IT MCMC")
    #s = time.time()

    for t in range(it_mcmc):
        #print('init it')
        # print(t)
        h_curr, _ = myRBM.SampleHiddens01(v_curr)
        h_i = (torch.mm(myRBM.W.T, h_curr) +
               myRBM.vbias.reshape(v.shape[0], 1))  #  Nv x Ns
        w_next = w_curr.clone()

        v_next = torch.clone(v_curr)
        index = torch.randperm(v_curr.shape[0])
        for idx in range(v_curr.shape[0]):
            #print('upd comp')
            #s = time.time()
            i = idx
            v_next[i, :] = 1-v_curr[i, :]
            w_next += ((2*v_next[i, :]-1)*V[i]*norm)

            # On calcul -DeltaE
            ΔE = ß*((2*v_next[i, :]-1)*h_i[i, :])-(N/2) * \
                ((w_hat-w_next)**2-(w_hat-w_curr)**2)

            tir = torch.rand(v_curr.shape[1], 1,
                             device=torch.device("cuda")).squeeze()
            prob = torch.exp(ΔE).squeeze()
            v_curr[i, :] = torch.where(tir < prob, v_next[i, :], v_curr[i, :])
            v_next[i, :] = torch.where(
                tir < prob, v_next[i, :], 1-v_next[i, :])
            w_curr = torch.where(tir < prob, w_next, w_curr)
            w_next = torch.where(tir < prob, w_next, w_curr)
            # print(time.time()-s)
        if (t >= (it_mcmc-it_mean)):
            vtab += v_curr
    # print(time.time()-s)

    vtab = vtab*(1/it_mean)
    return v_curr, h_curr, vtab


lr = 0.01
NGibbs = 100
annSteps = 0
mb_s = 600
num_pcd = 100
Nh = 20
Nv = data.shape[1]
ep_max = 10
w_hat = torch.linspace(0, 1, steps=100)
_, S_d, V = torch.linalg.svd(data)
V0 = V[:, 0]
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
            V=V0,
            TMCLearning=True
            )

f = h5py.File(
    "../model/AllParametersTMC_NGibbs_100_Nh20_Ns1000_Nmb600_Nepoch100_lr_0.05.h5", 'r')

# Get all registered times
alltimes = []
for t in f['alltime'][:]:
    if 'W'+str(t) in f:
        alltimes.append(t)
print(alltimes)

nb_chain = 15  # Nb de chaines pour chaque w_hat
it_mcmc = 100  # Nb it_mcmc pour chaque chaine
it_mean = 70  # Nb it considérée pour la moyenne temporelle de chaque chaine
N = 20000  # Contrainte
nb_point = 10000  # Nb de points de discrétisation pour w_hat
xmin = -0.0
xmax = 1.0


for t in alltimes:
    print(t)
    myRBM.W = torch.tensor(f['W'+str(t)], device=myRBM.device)
    myRBM.hbias = torch.tensor(f['hbias'+str(t)], device=myRBM.device)
    myRBM.vbias = torch.tensor(f['vbias'+str(t)], device=myRBM.device)
    _, _, V_g = torch.linalg.svd(myRBM.W)
    if torch.mean(V_g[:, 0]) < 0:
        V_g = -V_g
    start = torch.bernoulli(torch.rand(
        myRBM.Nv, nb_chain*nb_point, device=device))
    #V0 = V[:, 0]
    # w_hat = torch.dot(start.T, V)[0:,]
    w_hat_b = torch.linspace(xmin, xmax, steps=nb_point, device=device)
    w_hat = torch.zeros(nb_chain*nb_point, device=device)
    for i in range(nb_point):
        for j in range(nb_chain):
            w_hat[i*nb_chain+j] = w_hat_b[i]
    tmpv, tmph, vtab = TMCSample(
        start, w_hat, N, V_g[:, 0], it_mcmc=it_mcmc, it_mean=it_mean)
    y = np.array(torch.mm(vtab.T, V_g).cpu().squeeze())/myRBM.Nv**0.5
    newy = np.array([np.mean(y[i*nb_chain:i*nb_chain+nb_chain])
                     for i in range(nb_point)])
    w_hat = w_hat.cpu().numpy()
    w_hat_b = w_hat_b.cpu().numpy()
    res = np.zeros(len(w_hat_b)-1)
    for i in range(1, len(w_hat_b)):
        res[i-1] = simps(newy[:i]-w_hat_b[:i], w_hat_b[:i])

    const = simps(np.exp(N*res-np.max(N*res)), w_hat_b[:-1])
    p_m = np.exp(N*res-np.max(N*res))/const
    proj_data = torch.mm(torch.tensor(data, device=device,
                                      dtype=dtype), V_g).cpu()/myRBM.Nv**0.5
    potential = res + (1/N)*np.log(const)
    fig, ax1 = plt.subplots(dpi=200)

    color = 'tab:red'
    ax1.set_xlabel("w_hat")
    ax1.plot(w_hat_b, newy-w_hat_b, color='red', label="grad potential")
    ax1.plot(w_hat_b[1:], potential, label='potential')
    ax1.hlines(0, 0, 1, color='black')
    #ax1.scatter(proj_gen[:,0], proj_gen[:,1],alpha=0.2, label = 'data_gen')
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

    color = 'tab:blue'
    rdm_y = torch.randn(proj_data[:, 0].shape)/20
    ax2.hist(proj_data[:, 0].numpy(), label='data', density=True, bins=200)
    ax2.plot(w_hat_b[1:], p_m, color="green", label="prob")

    ax2.tick_params(axis='y', labelcolor=color)

    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    ax1.legend()
    ax2.legend()
    plt.savefig('../fig/evolprob/upd'+str(t)+'.png')
    plt.close()
    del V_g
    del proj_data
    del rdm_y
    del w_hat
    del w_hat_b
    del tmpv
    del tmph
    del vtab

    torch.cuda.empty_cache()
