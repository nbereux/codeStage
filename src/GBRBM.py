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
        if not(self.var_set):
            tmp = 1/(1-2*torch.mm(self.W_2.T, H))
            # assert torch.sum(torch.isnan(tmp)) == 0
            mu = (torch.mm(self.W_1.t(), H) + self.vbias.reshape(self.Nv, 1))*tmp

            t = 1/self.var
            t = torch.unsqueeze(t, 1)
            t = t.repeat(1, mu.shape[1])
            var = torch.div(t, 1-2*torch.matmul(self.W_2.T, H))

            var = torch.where(var > 0, var, torch.ones(
                var.shape, device=self.device)*eps).to(self.device)
        else :
            mu = torch.matmul(self.W_1.T, H)+torch.unsqueeze(self.vbias, 1)
            var = self.sigV
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
            self.sigV =1# torch.ones(self.Nv, device=self.device)
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
        else :
            self.sigV = 1
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
