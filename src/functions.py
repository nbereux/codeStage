import h5py
import RBM
import torch
import math
import pathlib
path = str(pathlib.Path(__file__).parent.absolute())+'/'
from scipy.interpolate import UnivariateSpline
from scipy.integrate import simps

import numpy as np
dtype = torch.float


def retrieveRBM(device, fname):
    f = h5py.File(fname, 'r')

    Nh = f['paramW1'].shape[0]  # get the visible shape
    Nv = f['paramW1'].shape[1]  # get the hidden shape
    # usefull if you want to use the RBN
    lr = 0.01
    NGibbs = 100
    mb_s = 500
    num_pcd = 500

    alltimes = []
    for t in f['alltime'][:]:
        if 'paramW'+str(t) in f:
            alltimes.append(t)

    # creating the RBM
    myRBM = RBM.RBM(num_visible=Nv,
                    num_hidden=Nh,
                    device=device,
                    lr=lr,
                    # regL2=l2,
                    gibbs_steps=NGibbs,
                    # anneal_steps=annSteps,
                    UpdCentered=True,
                    mb_s=mb_s,
                    num_pcd=num_pcd)
    return myRBM, f, alltimes


def compute_H(W, b, dev):
    mat2 = 1+torch.exp(b)
    Nv = W.shape[1]
    H = torch.zeros((Nv, Nv), device=dev)
    for i in range(Nv):
        for j in range(i, Nv):
            num = torch.log((1+torch.exp(b+W[:, i]+W[:, j])))+torch.log(mat2)
            denom = torch.log(
                (1+torch.exp(b+W[:, i])))+torch.log((1+torch.exp(b+W[:, j])))
            H[i, j] = torch.sum(num-denom)
            assert(not(math.isnan(H[i, j])))
    H = H.T + H - torch.eye(Nv, device=dev)*torch.diag(H)
    H = 1/8*H
    return H


def gradE(z, y, mu, gamma):
    return (z.T*mu - torch.matmul(y.T, gamma.T)).T


def Sinkhorn2D(mu, nu, C, eps, dev):
    precision = torch.tensor([1e-6, 1e-6], device=dev)
    gamma_barre = torch.exp(-C/eps)
    a_curr = torch.ones(len(mu), device=dev)
    b_curr = torch.ones(len(nu), device=dev)
    a_next = torch.zeros(len(mu), device=dev)
    b_next = torch.zeros(len(nu), device=dev)
    k = 0
    diff = torch.tensor([torch.linalg.norm(a_next-a_curr),
                         torch.linalg.norm(b_next-b_curr)], device=dev)
    while(True):
        k += 1
        a_next = mu / torch.mv(gamma_barre, b_curr)
        b_next = nu / torch.mv(gamma_barre.T, a_next)
        diff = torch.tensor([torch.linalg.norm(
            a_next-a_curr), torch.linalg.norm(b_next-b_curr)], device=dev)
        if(torch.less(diff, precision).all()):
            break
        a_curr = torch.clone(a_next)
        b_curr = torch.clone(b_next)
    gamma_barre = torch.matmul(torch.unsqueeze(
        a_next, 1), torch.unsqueeze(b_next, 1).T)*gamma_barre
    u = eps*torch.log(a_next)
    v = eps*torch.log(b_next)
    return u, v, gamma_barre


def Wasserstein_distance(data_1, data_2, eps, dev):

    n = data_1.shape[0]
    m = data_2.shape[0]
    z0 = data_1
    mu_tab = torch.ones(n, device=dev)/n
    nu_tab = torch.ones(m, device=dev)/m
    y = data_2
    nu = (1/m)*torch.sum(y)
    C = torch.cdist(z0, y)
    _, _, gamma = Sinkhorn2D(mu_tab, nu_tab, C, 1, dev)
    part_1 = torch.sum(C*gamma)
    tmp = gamma*(torch.log(gamma/torch.matmul(mu_tab, nu_tab.T))-1)
    part_2 = torch.sum(tmp)
    return part_1 + eps*part_2


def gradWas(data_1, data_2, eps, iter_max, filename, dev, prt=False, tau=0.9):

    l_grad = []

    n = data_1.shape[0]
    m = data_2.shape[0]
    z0 = data_1
    mu_tab = torch.ones(n, device=dev)/n
    nu_tab = torch.ones(m, device=dev)/m
    y = data_2

    # Construction de la matrice de contrainte
    C = torch.cdist(z0, y)
    # Determine le plan optimal
    _, _, gamma = Sinkhorn2D(mu_tab, nu_tab, C, eps, dev)
    # gradient
    grad = gradE(z0, y, mu_tab, gamma)
    l_grad.append(grad)
    cpt = 0
    while (torch.linalg.norm(grad) > 1e-6):
        z0 -= tau * grad
        torch.cdist(z0, y)
        _, _, gamma = Sinkhorn2D(mu_tab, nu_tab, C, eps, dev)
        grad = gradE(z0, y, mu_tab, gamma)
        l_grad.append(grad)
        cpt += 1
        print(cpt)
        print(torch.linalg.norm(grad))
        if (cpt > iter_max):
            break
    return z0.T, l_grad

# w_hat : nDim x nb_point
# V : matrice de projection
def TMCSampleND(myRBM, v, w_hat, N, V, device, nb_point, nb_chain, it_mcmc=100, it_mean=50, ß=1):
    #print("Initialisation")
    vtab = torch.zeros(v.shape, device=device)
    v_curr = v
    norm = 1/(v_curr.shape[0]**0.5)
    w_curr = (torch.mm(v_curr.T, V)*norm)[:,:w_hat.shape[0]]
    index = torch.randperm(v_curr.shape[0])
    for t in range(it_mcmc):
        #print('init it')
        print(t)
        h_curr, _ = myRBM.SampleHiddens01(v_curr)
        h_i = (torch.mm(myRBM.W.T, h_curr)+myRBM.vbias.reshape(v.shape[0],1)) # Nv x Ns
        w_next = w_curr.clone()
        
        v_next = torch.clone(v_curr)
        index = torch.randperm(v_curr.shape[0])
        for idx in range(v_curr.shape[0]):
            i = idx
            v_next[i,:] = 1-v_curr[i,:]
            for j in range(w_next.shape[1]):
                w_next[:,j] += ((2*v_next[i,:]-1)*V[i,j]*norm)
                
            # On calcul -DeltaE
            ΔE = ß*((2*v_next[i,:]-1)*h_i[i,:])-(N/2)*(torch.sum((w_hat.T-w_next)**2, dim=1)-torch.sum((w_hat.T-w_curr)**2, dim=1))

            tir = torch.rand(v_curr.shape[1],1, device = torch.device("cuda")).squeeze()
            prob = torch.exp(ΔE).squeeze()
            v_curr[i,:] = torch.where(tir<prob, v_next[i,:], v_curr[i,:])
            v_next[i,:] = torch.where(tir<prob, v_next[i,:], 1-v_next[i,:])
            neg_index = torch.ones(w_curr.shape[0], dtype = bool)
            index = torch.where(tir<prob)[0]
            neg_index[index] = False
            w_curr[index,:]=  w_next[index, :]
            w_next[neg_index,:] =  w_curr[neg_index,:]
        if (t>= (it_mcmc-it_mean)):
            vtab += v_curr
    vtab = vtab*(1/it_mean)
    vtab = vtab.reshape(myRBM.Nv, nb_point, nb_chain)
    v_curr = v_curr.reshape(myRBM.Nv, nb_point, nb_chain)
    h_curr = h_curr.reshape(myRBM.Nh, nb_point, nb_chain)
    return v_curr, h_curr, vtab

def singValTrain_1(fname, alltimes, device, nval=-1):
    f = h5py.File(fname, "r")
    if nval == -1 :
        nval = torch.svd(torch.tensor(f["paramW0"], device = device), compute_uv = False)[1].shape[0]
    ret = torch.zeros(alltimes.shape[0], nval)
    for i in range(alltimes.shape[0]):
        t = alltimes[i]
        ret[i,:] = torch.svd(torch.tensor(f["paramW"+str(t)],device = device), compute_uv = False)[1][:nval]
    f.close()
    return ret

def singValTrain_2(fname, alltimes, device, nval=-1):
    f = h5py.File(fname, "r")
    if nval == -1 :
        nval = torch.svd(torch.tensor(f["W0"], device = device), compute_uv = False)[1].shape[0]
    ret = torch.zeros(alltimes.shape[0], nval)
    for i in range(alltimes.shape[0]):
        t = alltimes[i]
        ret[i,:] = torch.svd(torch.tensor(f["W"+str(t)],device = device), compute_uv = False)[1][:nval]
    f.close()
    return ret

def singValTrain_3(fname, alltimes, device, nval=-1):
    f = h5py.File(fname, "r")
    if nval == -1 :
        nval = torch.svd(f["W_10"], compute_uv = False).shape[0]
    ret = torch.zeros(alltimes.shape[0], nval)
    for t in alltimes:
        ret[i,:] = torch.svd(f["W_1"+str(t)], compute_uv = False)[:nval]
    f.close()
    return ret

def ComputeProbabilityTMC1D(myRBM, data, nb_chain, it_mcmc, it_mean, N, nb_point, border_length, device):
    start = torch.bernoulli(torch.rand(myRBM.Nv, nb_chain*nb_point, device=device))
    _, _, V_g = torch.svd(myRBM.W)
    V_g = V_g[:, 0]
    if torch.mean(V_g) < 0:
        V_g = -V_g
    proj_data = torch.mv(data, V_g)/myRBM.Nv**0.5
    xmin = torch.min(proj_data) - border_length
    xmax = torch.max(proj_data) + border_length
    w_hat_b = torch.linspace(xmin, xmax, steps=nb_point, device=device)
    w_hat = torch.zeros(nb_chain*nb_point, device=device)
    for i in range(nb_point):
        for j in range(nb_chain):
            w_hat[i*nb_chain+j] = w_hat_b[i]
    tmpv, tmph, vtab = myRBM.TMCSample(start, w_hat, N, V_g, it_mcmc=it_mcmc, it_mean=it_mean)
    y = np.array(torch.mm(vtab.T, V_g.unsqueeze(1)).cpu().squeeze())/myRBM.Nv**0.5
    newy = np.array([np.mean(y[i*nb_chain:i*nb_chain+nb_chain]) for i in range(nb_point)])
    w_hat = w_hat.cpu().numpy()
    w_hat_b_np = w_hat_b.cpu().numpy()
    res = np.zeros(len(w_hat_b)-1)
    for i in range(4, len(w_hat_b)):
    #    res[i-1] = simps(newy[:i]-w_hat_b_np[:i], w_hat_b_np[:i])
        TMPF = UnivariateSpline( w_hat_b_np[:i], newy[:i]-w_hat_b_np[:i])
        res[i-1] = TMPF.integral(w_hat_b_np[0], w_hat_b_np[-1])

    #const = simps(np.exp(N*res-np.max(N*res)), w_hat_b_np[:-1])
    TMPF2 = UnivariateSpline( w_hat_b_np[:-1], np.exp(N*res-np.max(N*res)))
    const = TMPF2.integral(w_hat_b_np[0], w_hat_b_np[-1])
    p_m = torch.tensor(np.exp(N*res-np.max(N*res))/const, device=myRBM.device)
    grad_pot = newy-w_hat_b_np
    potential = res + (1/N)*np.log(const)
    return p_m, grad_pot, potential, w_hat_b

def ComputeProbabilityTMC2D(myRBM, data, nb_chain, it_mcmc, it_mean, N, nb_point_dim, border_length, device, nDim = 2):
    _, _, V_g = torch.svd(myRBM.W)
    if torch.mean(V_g[:,0])<0:
        V_g = -V_g
    proj_data = torch.mm(data, V_g).cpu()/myRBM.Nv**.5
    limits = torch.zeros((2, nDim))
    for i in range(nDim):
        limits[0, i] = proj_data[:,i].min()-border_length
        limits[1, i] = proj_data[:,i].max()+border_length
    nb_point = nb_point_dim.prod()
    x_grid = np.linspace(limits[0,0], limits[1,0], nb_point_dim[0])
    x_grid = np.array([x_grid for i in range(nb_point_dim[1])])
    x_grid = x_grid.reshape(nb_point)
    y_grid = []
    y_d = np.linspace(limits[0,1], limits[1,1], nb_point_dim[1])
    for i in range(nb_point_dim[1]):
        for j in range(nb_point_dim[0]):
            y_grid.append(y_d[i])
    grid = torch.tensor([x_grid, y_grid], device = device, dtype = dtype)
    start = torch.bernoulli(torch.rand(myRBM.Nv, nb_chain*nb_point, device = device))
    # w_hat = torch.dot(start.T, V)[0:,]
    w_hat_b = grid
    w_hat = torch.zeros((2, nb_chain*nb_point), device = device)
    for i in range(nb_point):
        for j in range(nb_chain):
            w_hat[:,i*nb_chain+j] = w_hat_b[:,i]
    tmpv, tmph, vtab = TMCSampleND(myRBM, start, w_hat, N, V_g, device, nb_point, nb_chain, it_mcmc = it_mcmc, it_mean=it_mean)
    newy = torch.mm(torch.mean(vtab, dim = 2).T, V_g)[:,:nDim]/myRBM.Nv**0.5
    grad_pot = newy.T-w_hat_b
    square = torch.zeros(2, nb_point_dim[0], nb_point_dim[1])
    w_hat_tmp = np.zeros((2, nb_point_dim[0], nb_point_dim[1]))
    for i in range(0,grad_pot.shape[1], nb_point_dim[0]):
            w_hat_tmp[:,:,int(i/nb_point_dim[0])] = w_hat_b[:, i:(i+nb_point_dim[0])].cpu().numpy()
            square[:,:, int(i/nb_point_dim[0])] = grad_pot[:,i:(i+nb_point_dim[0])]
    w_hat_dim = []
    for i in range(nDim):
        w_hat_dim.append(np.linspace(limits[0,i], limits[1,i], nb_point_dim[i]))

    res_x = np.zeros(nb_point_dim[0])
    for i in range(nb_point_dim[0]):
        res_x[i] = simps(square[0][:(i+1),0].cpu().numpy(), w_hat_tmp[0,:(i+1),0])
    res_y = np.zeros((nb_point_dim[0], nb_point_dim[1]))
    for i in range(nb_point_dim[0]):
        for j in range(nb_point_dim[1]):
            res_y[i,j] = simps(square[1][i,:(j+1)].cpu().numpy(), w_hat_tmp[1,i,:(j+1)])

    pot = np.expand_dims(res_x, 1).repeat(nb_point_dim[1],1) + res_y    
    res = np.exp(N*(pot-np.max(pot)))
    const = np.zeros(res.shape[0])
    for i in range(res.shape[0]):
        const[i-1] = simps(res[:,i], w_hat_tmp[1, i, :])
    const = simps(const, w_hat_tmp[0,:,0])
    p_m = res/const
    return square, p_m, w_hat_tmp

def SampleTMC1D(p_m, w_hat_b, n_sample):
    cdf = np.zeros(len(p_m)-1)
    for i in range(1,len(p_m)):
        cdf[i-1] = simps(p_m[:i], w_hat_b[:i])

    sample = torch.rand(n_sample)
    sample = sample.sort()[0]
    i = 0
    for k in range(len(cdf)-1):
        while(cdf[k+1]>sample[i]):
            sample[i] = w_hat_b[k]
            i+=1
            if i==n_sample:
                return sample
    return sample