import h5py
import RBM
import torch
import math
import pathlib
path = str(pathlib.Path(__file__).parent.absolute())+'/'


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
