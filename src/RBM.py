import torch
import h5py
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import simps
import time


class GBRBM:
    # NEEDED VAR:
    # * num_visible
    # * num_hidden
    def __init__(self, num_visible,  #  number of visible nodes
                 num_hidden,  # number of hidden nodes
                 device,  #  CPU or GPU ?
                 gibbs_steps=10,  # number of MCMC steps for computing the neg term
                 var_init=1e-4,  # variance of the init weights
                 dtype=torch.float,
                 num_pcd=100,  # number of permanent chains
                 lr_W1=0.01,  # learning rate for W_1
                 lr_W2=1e-4,  # learning rate for W_2
                 ep_max=100,  # number of epochs
                 mb_s=50,  # size of the minibatch
                 UpdCentered=False,  # Update using centered gradients
                 CDLearning=False,
                 var_set=True
                 ):
        self.Nv = num_visible
        self.Nh = num_hidden
        self.device = device
        self.var_set = var_set
        self.var = torch.ones(self.Nv, device=self.device)
        self.gibbs_steps = gibbs_steps
        self.dtype = dtype
        # weight of the RBM
        self.W_1 = torch.randn(size=(self.Nh, self.Nv),
                               device=self.device, dtype=self.dtype)*var_init
        self.W_2 = torch.randn(size=(self.Nh, self.Nv),
                               device=self.device, dtype=self.dtype)*var_init
        # self.W_2 = torch.zeros(size=(self.Nh, self.Nv), device = self.device, dtype=self.dtype)
        self.var_init = var_init

        # visible and hidden biases
        self.vbias = torch.zeros(self.Nv, device=self.device, dtype=self.dtype)
        self.hbias = torch.zeros(self.Nh, device=self.device, dtype=self.dtype)

        # permanent chain
        self.X_pc = torch.bernoulli(torch.rand(
            (self.Nv, num_pcd), device=self.device, dtype=self.dtype))
        self.lr_W1 = lr_W1
        self.lr_W2 = lr_W2
        self.ep_max = ep_max
        self.mb_s = mb_s
        self.num_pcd = num_pcd

        self.ep_tot = 0
        self.up_tot = 0
        self.list_save_time = []
        self.list_save_rbm = []
        self.file_stamp = ''
        self.VisDataAv = 0
        self.HidDataAv = 0
        self.UpdCentered = UpdCentered
        self.ResetPermChainBatch = False
        self.CDLearning = CDLearning

    def SetVisBias(self, X):
        self.vbias = torch.mean(X, 1)

    # Sampling and getting the mean value using Sigmoid
    # using CurrentState

    def SampleHiddens01(self, V, β=1):
        t = 1/self.var
        tmp = self.W_2*t
        mh = torch.sigmoid(
            β*(torch.unsqueeze(self.hbias, 1) +
               torch.mm(self.W_1, V)
               + torch.mm(tmp, V*V)))
        # mh = torch.where(mh < 0, torch.zeros(mh.shape, device = self.device), mh)
        # mh = torch.where(mh > 1, torch.ones(mh.shape, device = self.device), mh)
        # if torch.sum(torch.isnan(mh))!=0:
        # print(t)
        # print('W1 ',torch.isnan(self.W_1).sum())
        # print('W2 ',torch.isnan(self.W_2).sum())

        # print(test)
        h = torch.bernoulli(mh)
        return h, mh

    # H is Nh X M
    # W is Nh x Nv
    # Return Visible sample and average value for gaussian variable
    def SampleVisiblesGaus(self, H, eps=1e-3):
        tmp = 1/(1-2*torch.mm(self.W_2.T, H))
        # assert torch.sum(torch.isnan(tmp)) == 0
        mu = (torch.mm(self.W_1.t(), H) + self.vbias.reshape(self.Nv, 1))*tmp

        t = 1/self.var
        t = torch.unsqueeze(t, 1)
        t = t.repeat(1, mu.shape[1])
        var = torch.div(t, 1-2*torch.matmul(self.W_2.T, H))

        var = torch.where(var > 0, var, torch.ones(
            var.shape, device=self.device)*eps).to(self.device)
        # var = torch.abs(var)
        sample = torch.normal(mean=mu, std=var)
        return sample, mu

    def GetAv(self, it_mcmc=0):
        if it_mcmc == 0:
            it_mcmc = self.gibbs_steps

        v = self.X_pc
        mh = 0
        β = 1
        h, mh = self.SampleHiddens01(v, β=β)
        v, mu = self.SampleVisiblesGaus(h)

        for t in range(1, it_mcmc):
            h, mh = self.SampleHiddens01(v, β=β)
            v, mu = self.SampleVisiblesGaus(h)
        return v, mu, h, mh

    def Sampling(self, X, it_mcmc=0):
        if it_mcmc == 0:
            it_mcmc = self.gibbs_steps
        if self.var_set:
            self.sigV = torch.ones(self.Nv, device=self.device)
        v = X
        β = 1

        h, mh = self.SampleHiddens01(v, β=β)
        v, mu = self.SampleVisiblesGaus(h)

        for t in range(it_mcmc-1):
            h, mh = self.SampleHiddens01(v, β=β)
            v, mu = self.SampleVisiblesGaus(h)

        return v, mu, h, mh

    def updateWeights(self, v_pos, h_pos, v_neg, h_neg_v, h_neg_m):

        lr_p_W1 = self.lr_W1/self.mb_s
        lr_n_W1 = self.lr_W1/self.num_pcd
        lr_p_W2 = self.lr_W2/self.mb_s
        lr_n_W2 = self.lr_W2/self.num_pcd
        t = 1  # /self.var
        v_pos_W1 = v_pos.T*t
        v_pos_W2 = (v_pos*v_pos).T*t
        v_neg_W2 = ((v_neg*v_neg).T*t).T
        v_neg_W1 = (v_neg.T*t).T

        self.vbias += torch.sum(v_pos_W1.T, 1)*lr_p_W1 - \
            torch.sum(v_neg_W1, 1)*lr_n_W1
        self.hbias += torch.sum(h_pos, 1)*lr_p_W1 - \
            torch.sum(h_neg_m, 1)*lr_n_W1

        NegTerm_ia = h_neg_v.mm(v_neg_W1.t())

        self.W_1 += h_pos.mm(v_pos_W1) * lr_p_W1 - NegTerm_ia*lr_n_W1
        self.W_2 += h_pos.mm(v_pos_W2) * lr_p_W2 - \
            h_neg_v.mm(v_neg_W2.T)*lr_n_W2

    def fit_batch(self, X):
        h_pos_v, h_pos_m = self.SampleHiddens01(X)
        if self.CDLearning:
            self.X_pc = X
            self.X_pc, _, h_neg_v, h_neg_m = self.GetAv()
        else:
            self.X_pc, _, h_neg_v, h_neg_m = self.GetAv()

        self.updateWeights(X, h_pos_m, self.X_pc, h_neg_v, h_neg_m)

    def getMiniBatches(self, X, m):
        return X[:, m*self.mb_s:(m+1)*self.mb_s]

    def getImages(self):
        ffname = "../data/cleanMNIST10000.h5"
        ff = h5py.File(ffname, 'r')
        var_rm = ff['id_col']
        data_mnist = ff['original']
        vinit = torch.bernoulli(torch.rand(
            (self.Nv, 10), device=self.device, dtype=self.dtype))
        tmp, _, _, _ = self.Sampling(vinit, it_mcmc=self.gibbs_steps)
        rebuiltMNIST = torch.zeros(
            data_mnist.shape[0], tmp.shape[1], device=self.device)
        passed = 0
        for i in range(data_mnist.shape[0]):
            if np.isin(i, var_rm):
                rebuiltMNIST[i, :] = torch.zeros(rebuiltMNIST[i, :].shape)
                passed += 1
            else:
                rebuiltMNIST[i, :] = tmp[i-passed, :]
        fig, ax = plt.subplots(1, tmp.shape[1])
        for i in range(tmp.shape[1]):
            ax[i].imshow(rebuiltMNIST[:, i].view(28, 28).cpu())
        plt.savefig("../tmp/ep"+str(self.ep_tot)+".png")
        plt.close()

    def fit(self, X, ep_max=0):
        if ep_max == 0:
            ep_max = self.ep_max

        NB = int(X.shape[1]/self.mb_s)

        if self.ep_tot == 0:
            self.VisDataAv = torch.mean(X, 1)

        if (len(self.list_save_time) > 0) & (self.up_tot == 0):
            f = h5py.File('../model/AllParameters'+self.file_stamp+'.h5', 'w')
            f.create_dataset('alltime', data=self.list_save_time)
            f.close()

        if (len(self.list_save_rbm) > 0) & (self.ep_tot == 0):
            f = h5py.File('../model/RBM'+self.file_stamp+'.h5', 'w')
            f.create_dataset('lr_W1', data=self.lr_W1)
            f.create_dataset('lr_W2', data=self.lr_W2)
            f.create_dataset('NGibbs', data=self.gibbs_steps)
            f.create_dataset('UpdByEpoch', data=NB)
            f.create_dataset('miniBatchSize', data=self.mb_s)
            f.create_dataset('numPCD', data=self.num_pcd)
            f.create_dataset('alltime', data=self.list_save_rbm)
            f.close()
        if not(self.var_set):
            self.sigV = torch.std(X, dim=1)
        for t in range(ep_max):

            print("IT ", self.ep_tot)
            self.ep_tot += 1
            Xp = X[:, torch.randperm(X.size()[1])]
            for m in range(NB):
                if self.ResetPermChainBatch:
                    self.X_pc = torch.normal(
                        torch.zeros(self.Nv, self.X_pc.shape[1],
                                    device=self.device), std=1)  # torch.unsqueeze(torch.sqrt(self.var), 1).repeat(1, self.X_pc.shape[1]))
                    self.X_pc = self.X_pc.to(self.device)
                Xb = self.getMiniBatches(Xp, m)
                self.fit_batch(Xb)

                if self.up_tot in self.list_save_time:
                    # self.getImages()
                    f = h5py.File('../model/AllParameters' +
                                  self.file_stamp+'.h5', 'a')
                    print('Saving nb_upd='+str(self.up_tot))
                    f.create_dataset('W_1'+str(self.up_tot),
                                     data=self.W_1.cpu())
                    f.create_dataset('W_2'+str(self.up_tot),
                                     data=self.W_2.cpu())
                    f.create_dataset('vbias'+str(self.up_tot),
                                     data=self.vbias.cpu())
                    f.create_dataset('hbias'+str(self.up_tot),
                                     data=self.hbias.cpu())
                    f.close()

                self.up_tot += 1

            if self.ep_tot in self.list_save_rbm:
                f = h5py.File('../model/RBM'+self.file_stamp+'.h5', 'a')
                f.create_dataset('W_1'+str(self.up_tot),
                                 data=self.W_1.cpu())
                f.create_dataset('W_2'+str(self.up_tot),
                                 data=self.W_2.cpu())
                f.create_dataset('vbias'+str(self.ep_tot),
                                 data=self.vbias.cpu())
                f.create_dataset('hbias'+str(self.ep_tot),
                                 data=self.hbias.cpu())
                f.close()


class RBM:
    # NEEDED VAR:
    # * num_visible
    # * num_hidden
    def __init__(self, num_visible,  #  number of visible nodes
                 num_hidden,  # number of hidden nodes
                 device,  #  CPU or GPU ?
                 gibbs_steps=10,  # number of MCMC steps for computing the neg term
                 var_init=1e-4,  # variance of the init weights
                 dtype=torch.float,
                 num_pcd=100,  #  number of permanent chains
                 lr=0.01,  # learning rate
                 ep_max=100,  #  number of epochs
                 mb_s=50,  # size of the minibatch
                 w_hat=torch.linspace(0, 1, steps=100),
                 N=20000,
                 V=0,
                 it_mean=5,
                 UpdCentered=False,  # Update using centered gradients
                 CDLearning=False,
                 TMCLearning=False
                 ):
        self.Nv = num_visible
        self.Nh = num_hidden
        self.gibbs_steps = gibbs_steps
        self.device = device
        self.dtype = dtype
        # weight of the RBM
        self.W = torch.randn(size=(self.Nh, self.Nv),
                             device=self.device, dtype=self.dtype)*var_init
        self.var_init = var_init
        # visible and hidden biases
        self.vbias = torch.zeros(self.Nv, device=self.device, dtype=self.dtype)
        self.hbias = torch.zeros(self.Nh, device=self.device, dtype=self.dtype)
        # permanent chain
        self.X_pc = torch.bernoulli(torch.rand(
            (self.Nv, num_pcd), device=self.device, dtype=self.dtype))
        self.lr = lr
        self.ep_max = ep_max
        self.mb_s = mb_s
        self.num_pcd = num_pcd
        # TMC Sampling
        self.w_hat = w_hat
        self.N = N
        self.V = V
        self.it_mean = it_mean

        self.ep_tot = 0
        self.up_tot = 0
        self.list_save_time = []
        self.list_save_rbm = []
        self.file_stamp = ''
        self.VisDataAv = 0
        self.HidDataAv = 0
        self.UpdCentered = UpdCentered
        self.ResetPermChainBatch = False
        self.CDLearning = CDLearning
        self.TMCLearning = TMCLearning

    # save RBM's parameters
    def saveRBM(self, fname):
        f = h5py.File(fname, 'w')
        f.create_dataset('W', data=self.W.cpu())
        f.create_dataset('hbias', data=self.hbias.cpu())
        f.create_dataset('vbias', data=self.vbias.cpu())
        f.create_dataset('X_pc', data=self.X_pc.cpu())
        f.close()

    # load a RBM from file
    def loadRBM(self, fname, stamp='', PCD=False):
        f = h5py.File(fname, 'r')
        self.W = torch.tensor(f['W'+stamp])
        self.vbias = torch.tensor(f['vbias'+stamp])
        self.hbias = torch.tensor(f['hbias'+stamp])
        self.Nv = self.W.shape[1]
        self.Nh = self.W.shape[0]
        if PCD:
            self.x_pc = torch.tensor(f['X_pc']+stamp)
        if self.device.type != "cpu":
            self.W = self.W.cuda()
            self.vbias = self.vbias.cuda()
            self.hbias = self.hbias.cuda()
            if PCD:
                self.X_pc = self.X_pc.cuda()

    # tile a set of 2d arrays
    def ImConcat(self, X, ncol=10, nrow=5, sx=28, sy=28, ch=1):
        tile_X = []
        for c in range(nrow):
            L = torch.cat((tuple(X[i, :].reshape(sx, sy, ch)
                                 for i in np.arange(c*ncol, (c+1)*ncol))))
            tile_X.append(L)
        return torch.cat(tile_X, 1)

    # X is Nv x Ns
    # return the free energy of all samples -log(p(x))

    def FreeEnergy(self, X):
        vb = torch.sum(X.t() * self.vbias, 1)  # vb: Ns
        fe_exp = 1 + torch.exp(self.W.mm(X).t() +
                               self.hbias)  # fe_exp: Ns x Nh
        Wx_b_log = torch.sum(torch.log(fe_exp), 1)  # Wx_b_log: Ns
        result = - vb - Wx_b_log  # result: Ns
        return result

    # Compute the energy of the RBM
    # V is Nv x Ns x NT
    # H is Nh x Ns x NT
    # E is Ns x NT
    def computeE(self, V, H):
        INT = torch.sum((H * torch.tensordot(self.W, V, dims=1)), 0)
        FIELDS = torch.tensordot(self.hbias, H, dims=1) + \
            torch.tensordot(self.vbias, V, dims=1)
        return -(INT + FIELDS)

    # Compute ratio of log(Z) using AIS
    def ComputeFE(self, nβ=1000, NS=1000):
        FE_RATIO_AIS = self.ComputeFreeEnergyAIS(nβ, NS)
        FE_PRIOR = self.ComputeFreeEnergyPrior()

        return FE_PRIOR, FE_RATIO_AIS

    def ComputeFreeEnergyAIS(self, nβ, nI):

        βlist = torch.arange(0, 1.000001, 1.0/nβ)
        x = torch.zeros(self.Nv+self.Nh2, nI, device=self.device)
        H = torch.zeros(self.Nh, nI, device=self.device)
        E = torch.zeros(nI, device=self.device)

        # initialize xref
        x = torch.bernoulli(torch.sigmoid(self.vbias_prior).repeat(nI, 1).t())
        H = torch.bernoulli(torch.rand((self.Nh, nI), device=self.device))
        E = self.computeE(x, H).double().to(
            self.device) - self.computeE_prior(x)
        self.V_STATE = x
        self.H_STATE = H
        for idβ in range(1, nβ+1):
            H, _ = self.SampleHiddens01(x, β=βlist[idβ])
            x, _ = self.SampleVisibles01(H, β=βlist[idβ])
            E += self.computeE(x, H)

        Δ = 0
        # E = self.computeE(x,H) - self.computeE_prior(x)
        Δβ = 1.0/nβ
        Δ = -Δβ*E  # torch.sum(E,1)
        # for idβ in range(nβ):
        #    Δβ = 1/nβ
        #    Δ += -Δβ*E[:,idβ]
        Δ = Δ.double()
        Δ0 = torch.mean(Δ)

        AIS = torch.log(torch.mean(torch.exp(Δ-Δ0).double()))+Δ0
        # AIS = torch.log(torch.mean(torch.exp(Δ)))
        return AIS

    # Compute both AATS scores
    def ComputeAATS(self, X, fake_X, s_X):
        CONCAT = torch.cat((X[:, :s_X], fake_X[:, :s_X]), 1)
        dAB = torch.cdist(CONCAT.t(), CONCAT.t())
        torch.diagonal(dAB).fill_(float('inf'))
        dAB = dAB.cpu().numpy()

        # the next line is use to tranform the matrix into
        #  d_TT d_TF   INTO d_TF- d_TT-  where the minus indicate a reverse order of the columns
        #  d_FT d_FF        d_FT  d_FF
        dAB[:int(dAB.shape[0]/2), :] = dAB[:int(dAB.shape[0]/2), ::-1]
        closest = dAB.argmin(axis=1)
        n = int(closest.shape[0]/2)

        ninv = 1/n
        # np.concatenate([(closest[:n] < n), (closest[n:] >= n)])
        correctly_classified = closest >= n
        # for a true sample, proba that the closest is in the set of true samples
        AAtruth = (closest[:n] >= n).sum()*ninv
        # for a fake sample, proba that the closest is in the set of fake samples
        AAsyn = (closest[n:] >= n).sum()*ninv

        return AAtruth, AAsyn

    # init the visible bias using the empirical frequency of the training dataset
    def SetVisBias(self, X):
        NS = X.shape[1]
        prob1 = torch.sum(X, 1)/NS
        prob1 = torch.clamp(prob1, min=1e-5)
        prob1 = torch.clamp(prob1, max=1-1e-5)
        self.vbias = -torch.log(1.0/prob1 - 1.0)

    # define an initial value fo the permanent chain
    def InitXpc(self, V):
        self.X_pc = V

    # Sampling and getting the mean value using Sigmoid
    # using CurrentState
    def SampleHiddens01(self, V, β=1):
        mh = torch.sigmoid(β*(self.W.mm(V).t() + self.hbias).t())
        h = torch.bernoulli(mh)

        return h, mh

    # H is Nh X M
    # W is Nh x Nv
    # Return Visible sample and average value for binary variable
    def SampleVisibles01(self, H, β=1):
        mv = torch.sigmoid(β*(self.W.t().mm(H).t() + self.vbias).t())
        v = torch.bernoulli(mv)
        return v, mv

    # Compute the negative term for the gradient
    # IF it_mcmc=0 : use the class variable self.gibbs_steps for the number of MCMC steps
    # IF self.anneal_steps>= : perform anealing for the corresponding number of steps
    # FOR ANNEALING: only if the max eigenvalues is above self.ann_threshold
    # βs : effective temperure. Used only if =! -1
    def GetAv(self, it_mcmc=0, βs=-1):
        if it_mcmc == 0:
            it_mcmc = self.gibbs_steps

        v = self.X_pc
        mh = 0

        β = 1
        h, mh = self.SampleHiddens01(v, β=β)
        v, mv = self.SampleVisibles01(h, β=β)

        for t in range(1, it_mcmc):
            h, mh = self.SampleHiddens01(v, β=β)
            v, mv = self.SampleVisibles01(h, β=β)

        return v, mv, h, mh

    # Return samples and averaged values
    # IF it_mcmc=0 : use the class variable self.gibbs_steps for the number of MCMC steps
    # IF self.anneal_steps>= : perform anealing for the corresponding number of steps
    # FOR ANNEALING: only if the max eigenvalues is above self.ann_threshold
    # βs : effective temperure. Used only if =! -1
    def Sampling(self, X, it_mcmc=0):
        if it_mcmc == 0:
            it_mcmc = self.gibbs_steps

        v = X
        β = 1

        h, mh = self.SampleHiddens01(v, β=β)
        v, mv = self.SampleVisibles01(h, β=β)

        h, mh = self.SampleHiddens01(v, β=β)
        v, mv = self.SampleVisibles01(h, β=β)

        for t in range(it_mcmc-1):
            h, mh = self.SampleHiddens01(v, β=β)
            v, mv = self.SampleVisibles01(h, β=β)

        return v, mv, h, mh

    def TMCSample(self, v, w_hat, N, V, it_mcmc=0, it_mean=0, ß=1):
        if it_mcmc == 0:
            it_mcmc = self.gibbs_steps
        if it_mean == 0:
            it_mean = self.it_mean
        vtab = torch.zeros(v.shape, device=self.device)
        v_curr = v
        # V = V
        norm = 1/(v_curr.shape[0]**0.5)
        w_curr = (torch.mv(v_curr.T, V)*norm)
        # index = torch.randperm(v_curr.shape[0])
        for t in range(it_mcmc):
            # print(t)
            h_curr, _ = self.SampleHiddens01(v_curr)
            h_i = (torch.mm(self.W.T, h_curr) +
                   self.vbias.reshape(v.shape[0], 1))  # Nv x Ns
            w_next = w_curr.clone()
            v_next = torch.clone(v_curr)
            # index = torch.randperm(v_curr.shape[0])

            for idx in range(v_curr.shape[0]):
                i = idx
                v_next[i, :] = 1-v_curr[i, :]
                w_next += ((2*v_next[i, :]-1)*V[i]*norm)

                # On calcul -DeltaE
                ΔE = ß*((2*v_next[i, :]-1)*h_i[i, :])-(N/2) * \
                    ((w_hat-w_next)**2-(w_hat-w_curr)**2)

                tir = torch.rand(
                    v_curr.shape[1], 1, device=self.device).squeeze()
                prob = torch.exp(ΔE).squeeze()
                v_curr[i, :] = torch.where(
                    tir < prob, v_next[i, :], v_curr[i, :])
                v_next[i, :] = torch.where(
                    tir < prob, v_next[i, :], 1-v_next[i, :])
                w_curr = torch.where(tir < prob, w_next, w_curr)
                w_next = torch.where(tir < prob, w_next, w_curr)
            if (t >= (it_mcmc-it_mean)):
                vtab += v_curr
        vtab = vtab*(1/it_mean)
        return v_curr, h_curr, vtab

    def updateWeightsTMC(self, v_pos, h_pos, negTermV, negTermH, negTermW):
        lr_p = self.lr/self.mb_s
        lr_n = self.lr
        self.W += h_pos.mm(v_pos.t())*lr_p - negTermW*lr_n
        self.vbias += torch.sum(v_pos, 1)*lr_p - negTermV*lr_n
        self.hbias += torch.sum(h_pos, 1)*lr_p - negTermH*lr_n

        fname = '../data/valGradTMC.h5'
        f = h5py.File(fname, 'a')
        f.create_dataset('negTermW'+str(self.up_tot), data=negTermW.cpu())
        f.create_dataset('negTermH'+str(self.up_tot), data=negTermH.cpu())
        f.create_dataset('negTermV'+str(self.up_tot), data=negTermV.cpu())
        f.close()

    # Update weights and biases

    def updateWeights(self, v_pos, h_pos, v_neg, h_neg_v, h_neg_m):

        lr_p = self.lr/self.mb_s
        lr_n = self.lr/self.num_pcd
        # lr_reg = self.lr*self.regL2

        NegTerm_ia = h_neg_v.mm(v_neg.t())

        self.W += h_pos.mm(v_pos.t())*lr_p - NegTerm_ia*lr_n
        self.vbias += torch.sum(v_pos, 1)*lr_p - torch.sum(v_neg, 1)*lr_n
        self.hbias += torch.sum(h_pos, 1)*lr_p - \
            torch.sum(h_neg_m, 1)*lr_n

        fname = '../data/valGradNorm.h5'
        f = h5py.File(fname, 'a')
        f.create_dataset('negTermW'+str(self.up_tot), data=NegTerm_ia.cpu())
        f.create_dataset('negTermH'+str(self.up_tot),
                         data=torch.sum(h_neg_m, 1).cpu())
        f.create_dataset('negTermV'+str(self.up_tot),
                         data=torch.sum(v_neg, 1).cpu())
        f.close()

    # Update weights and biases
    def updateWeightsCentered(self, v_pos, h_pos_v, h_pos_m, v_neg, h_neg_v, h_neg_m, ν=0.2, ε=0.01):

        # self.HidDataAv = (1-ν)*self.HidDataAv + ν*torch.mean(h_pos_m,1)
        self.VisDataAv = torch.mean(v_pos, 1)
        self.HidDataAv = torch.mean(h_pos_m, 1)
        Xc_pos = (v_pos.t() - self.VisDataAv).t()
        Hc_pos = (h_pos_m.t() - self.HidDataAv).t()

        Xc_neg = (v_neg.t() - self.VisDataAv).t()
        Hc_neg = (h_neg_m.t() - self.HidDataAv).t()

        NormPos = 1.0/self.mb_s
        NormNeg = 1.0/self.num_pcd
        # NormL2 = self.regL2

        siτa_neg = Hc_neg.mm(Xc_neg.t())*NormNeg
        si_neg = torch.sum(v_neg, 1)*NormNeg
        τa_neg = torch.sum(h_neg_m, 1)*NormNeg

        ΔW = Hc_pos.mm(Xc_pos.t())*NormPos - siτa_neg

        self.W += ΔW*self.lr

        ΔVB = torch.sum(v_pos, 1)*NormPos - si_neg - \
            torch.mv(ΔW.t(), self.HidDataAv)
        self.vbias += self.lr*ΔVB

        ΔHB = torch.sum(h_pos_m, 1)*NormPos - τa_neg - \
            torch.mv(ΔW, self.VisDataAv)
        self.hbias += self.lr*ΔHB

    # Compute positive and negative term

    def fit_batch(self, X):
        h_pos_v, h_pos_m = self.SampleHiddens01(X)
        if self.CDLearning:
            self.X_pc = X
            self.X_pc, _, h_neg_v, h_neg_m = self.GetAv()
        elif self.TMCLearning:
            # time_start = time.time()
            nb_chain = 15  # Nb de chaines pour chaque w_hat
            it_mcmc = 25  # Nb it_mcmc pour chaque chaine
            it_mean = 10  # Nb it considérée pour la moyenne temporelle de chaque chaine
            N = 20000  # Contrainte
            nb_point = 1000  # Nb de points de discrétisation pour w_hat
            start = torch.bernoulli(torch.rand(
                self.Nv, nb_chain*nb_point, device=self.device))
            # SVD des poids
            _, _, V0 = torch.svd(self.W)
            V0 = V0[:, 0]
            if torch.mean(V0) < 0:
                V0 = -V0
            
            # pour adapter la taille de l'intervalle discrétisé à chaque itération
            # proj_data = torch.mv(X.T, V0)
            # xmin = torch.min(proj_data) - 0.2
            # xmax = torch.max(proj_data) + 0.2

            xmin = -1.5
            xmax = 1.5
            w_hat_b = torch.linspace(
                xmin, xmax, steps=nb_point, device=self.device)
            w_hat = torch.zeros(nb_chain*nb_point, device=self.device)
            for i in range(nb_point):
                for j in range(nb_chain):
                    w_hat[i*nb_chain+j] = w_hat_b[i]
            tmpv, tmph, vtab = self.TMCSample(
                start, w_hat, N, V0, it_mcmc=it_mcmc, it_mean=it_mean)
            
            y = np.array(torch.mm(vtab.T, V0.unsqueeze(1)
                                  ).cpu().squeeze())/self.Nv**0.5
            newy = np.array([np.mean(y[i*nb_chain:i*nb_chain+nb_chain])
                             for i in range(nb_point)])
            #w_hat_np = w_hat.cpu().numpy()
            w_hat_b_np = w_hat_b.cpu().numpy()
            res = np.zeros(len(w_hat_b)-1)
            for i in range(1, len(w_hat_b)):
                res[i-1] = simps(newy[:i]-w_hat_b_np[:i], w_hat_b_np[:i])
            const = simps(np.exp(N*res-np.max(N*res)), w_hat_b_np[:-1])
            p_m = torch.tensor(np.exp(N*res-np.max(N*res)) /
                               const, device=self.device)
            s_i = torch.stack([torch.mean(
                tmpv[:, i*nb_chain:i*nb_chain+nb_chain], dim=1) for i in range(nb_point)], 1)
            tau_a = torch.stack([torch.mean(
                tmph[:, i*nb_chain:i*nb_chain+nb_chain], dim=1) for i in range(nb_point)], 1)
            s_i = torch.trapz(s_i[:, 1:]*p_m, w_hat_b[1:], dim=1)
            tau_a = torch.trapz(tau_a[:, 1:]*p_m, w_hat_b[1:], dim=1)
            fname = "../data/saveDistrib.h5"
            f = h5py.File(fname, 'w')
            f.create_dataset('p_m'+str(self.up_tot), data=p_m.cpu())
            f.close()
            prod = torch.zeros(
                (self.Nv, self.Nh, nb_point*nb_chain), device=self.device)
            for i in range(tmpv.shape[1]):
                prod[:, :, i] = torch.outer(tmpv[:, i], tmph[:, i])
            prod = torch.stack([torch.mean(
                prod[:, :, i*nb_chain:i*nb_chain+nb_chain], dim=2) for i in range(nb_point)], 2)
            prod = torch.trapz(prod[:, :, 1:]*p_m, w_hat_b[1:], dim=2)

            # print(time.time() - time_start)

        else:
            self.X_pc, _, h_neg_v, h_neg_m = self.GetAv()

        if self.UpdCentered:
            self.updateWeightsCentered(
                X, h_pos_v, h_pos_m, self.X_pc, h_neg_v, h_neg_m)
        elif self.TMCLearning:
            self.updateWeightsTMC(X, h_pos_m, s_i, tau_a, prod.T)
        else:
            self.updateWeights(X, h_pos_m, self.X_pc, h_neg_v, h_neg_m)

    def getMiniBatches(self, X, m):
        return X[:, m*self.mb_s:(m+1)*self.mb_s]

    def fit(self, X, ep_max=0):
        if ep_max == 0:
            ep_max = self.ep_max

        NB = int(X.shape[1]/self.mb_s)

        if self.ep_tot == 0:
            self.VisDataAv = torch.mean(X, 1)

        # _,h_av = self.SampleHiddens01(X)
        # self.HidDataAv = torch.mean(h_av,1)

        if (len(self.list_save_time) > 0) & (self.up_tot == 0):
            f = h5py.File('../model/AllParameters'+self.file_stamp+'.h5', 'w')
            f.create_dataset('alltime', data=self.list_save_time)
            f.close()

        if (len(self.list_save_rbm) > 0) & (self.ep_tot == 0):
            f = h5py.File('../model/RBM'+self.file_stamp+'.h5', 'w')
            f.create_dataset('lr', data=self.lr)
            f.create_dataset('NGibbs', data=self.gibbs_steps)
            f.create_dataset('UpdByEpoch', data=NB)
            f.create_dataset('miniBatchSize', data=self.mb_s)
            f.create_dataset('numPCD', data=self.num_pcd)
            f.create_dataset('alltime', data=self.list_save_rbm)
            f.close()

        _, S_d, _ = torch.svd(X/np.sqrt(X.shape[1]))
        for t in range(ep_max):
            print("IT ", self.ep_tot)
            self.ep_tot += 1

            Xp = X[:, torch.randperm(X.size()[1])]
            for m in range(NB):
                # print(m)
                if self.ResetPermChainBatch:
                    self.X_pc = torch.bernoulli(torch.rand(
                        (self.Nv, self.num_pcd), device=self.device, dtype=self.dtype))

                Xb = self.getMiniBatches(Xp, m)
                self.fit_batch(Xb)

                if self.up_tot in self.list_save_time:
                    f = h5py.File('../model/AllParameters' +
                                  self.file_stamp+'.h5', 'a')
                    print('Saving nb_upd='+str(self.up_tot))
                    f.create_dataset('W'+str(self.up_tot), data=self.W.cpu())
                    f.create_dataset('vbias'+str(self.up_tot),
                                     data=self.vbias.cpu())
                    f.create_dataset('hbias'+str(self.up_tot),
                                     data=self.hbias.cpu())
                    f.close()
                    _, S, _ = torch.svd(self.W)
                    plt.plot(S.cpu(), label="W")
                    plt.plot(S_d.cpu()[:len(S)], label="data")
                    plt.semilogy()
                    plt.legend()
                    plt.savefig("../tmp/TMCeig"+str(self.up_tot)+".png")
                    plt.close()

                self.up_tot += 1

            if self.ep_tot in self.list_save_rbm:
                f = h5py.File('../model/RBM'+self.file_stamp+'.h5', 'a')
                f.create_dataset('W'+str(self.ep_tot), data=self.W.cpu())
                f.create_dataset('vbias'+str(self.ep_tot),
                                 data=self.vbias.cpu())
                f.create_dataset('hbias'+str(self.ep_tot),
                                 data=self.hbias.cpu())
                f.close()
