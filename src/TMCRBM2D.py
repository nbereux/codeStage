import torch
import numpy as np
import h5py
from scipy.integrate import simps
import matplotlib.pyplot as plt
import time

class TMCRBM2D:
    # NEEDED VAR:
    # * num_visible
    # * num_hidden
    def __init__(self, num_visible, # number of visible nodes
                 num_hidden, # number of hidden nodes
                 device, # CPU or GPU ?
                 gibbs_steps=10, # number of MCMC steps for computing the neg term
                 var_init=1e-4, # variance of the init weights
                 dtype=torch.float,
                 num_pcd = 100, # number of permanent chains
                 lr = 0.01, # learning rate
                 ep_max = 100, # number of epochs
                 mb_s = 50, # size of the minibatch
                 UpdCentered = True, # Update using centered gradients
                 CDLearning = False,
                 ResetPermChainBatch = False,
                 it_mean = 8, # Nb de chaines pour chaque w_hat
                 nb_chain = 15, # Nb it considérée pour la moyenne temporelle de chaque chaine
                 N = 20000, # Contrainte
                 nb_point_dim = torch.tensor([100,100]), # Nb de points de discrétisation pour w_hat
                 verbose = 0,
                 save_fig = False
                 ): 
        self.Nv = num_visible        
        self.Nh = num_hidden
        self.gibbs_steps = gibbs_steps
        self.device = device
        self.dtype = dtype
        # weight of the RBM
        self.W = torch.randn(size=(self.Nh,self.Nv), device=self.device, dtype=self.dtype)*var_init
        self.var_init = var_init
        # visible and hidden biases
        self.vbias = torch.zeros(self.Nv, device=self.device, dtype=self.dtype)
        self.hbias = torch.zeros(self.Nh, device=self.device, dtype=self.dtype)
        # permanent chain
        self.X_pc = torch.bernoulli(torch.rand((self.Nv,num_pcd), device=self.device, dtype=self.dtype))
        self.lr = lr
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
        self.ResetPermChainBatch = ResetPermChainBatch
        self.CDLearning = CDLearning

        # TMC param
        self.nb_chain = nb_chain  
        self.it_mean = it_mean  
        self.N = N  
        self.nb_point_dim = nb_point_dim.to(device)
        self.nb_point = self.nb_point_dim.prod()

        self.verbose = verbose
        self.save_fig = save_fig

        self.p_m = torch.zeros(self.nb_point-1)
        self.w_hat_b = torch.zeros(self.nb_point)
        _, _, self.V0 = torch.svd(self.W)
        self.nDim = 2

    # save RBM's parameters
    def saveRBM(self,fname):
        f = h5py.File(fname,'w')
        f.create_dataset('W',data=self.W.cpu())
        f.create_dataset('hbias',data=self.hbias.cpu())
        f.create_dataset('vbias',data=self.vbias.cpu())
        f.create_dataset('X_pc',data=self.X_pc.cpu())
        f.close()

    # load a RBM from file
    def loadRBM(self,fname,stamp='',PCD=False):
        f = h5py.File(fname,'r')
        self.W = torch.tensor(f['W'+stamp]); 
        self.vbias = torch.tensor(f['vbias'+stamp]); 
        self.hbias = torch.tensor(f['hbias'+stamp]); 
        self.Nv = self.W.shape[1]
        self.Nh = self.W.shape[0]
        if PCD:
            self.x_pc = torch.tensor(f['X_pc']+stamp); 
        if self.device.type != "cpu":
            self.W = self.W.cuda()
            self.vbias = self.vbias.cuda()
            self.hbias = self.hbias.cuda()
            if PCD:
                self.X_pc = self.X_pc.cuda()

    def SetVisBias(self,X):
        NS = X.shape[1]
        prob1 = torch.sum(X,1)/NS
        prob1 = torch.clamp(prob1,min=1e-5)
        prob1 = torch.clamp(prob1,max=1-1e-5)
        self.vbias = -torch.log(1.0/prob1 - 1.0)
    
    # define an initial value fo the permanent chain
    def InitXpc(self,V):
        self.X_pc = V

    # Sampling and getting the mean value using Sigmoid
    # using CurrentState
    def SampleHiddens01(self,V,β=1):             
        mh = torch.sigmoid(β*(self.W.mm(V).t() + self.hbias).t())
        h = torch.bernoulli(mh)

        return h,mh

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
        norm = 1/(v_curr.shape[0]**0.5)
        w_curr = (torch.mm(v_curr.T, V)*norm)[:,:w_hat.shape[0]]
        for t in range(it_mcmc):
            h_curr, _ = self.SampleHiddens01(v_curr)
            h_i = (torch.mm(self.W.T, h_curr)+self.vbias.reshape(v.shape[0],1)) # Nv x Ns
            w_next = w_curr.clone()
            
            v_next = torch.clone(v_curr)
            for i in range(v_curr.shape[0]):
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
        vtab = vtab.reshape(self.Nv, self.nb_point, self.nb_chain)
        v_curr = v_curr.reshape(self.Nv, self.nb_point, self.nb_chain)
        h_curr = h_curr.reshape(self.Nh, self.nb_point, self.nb_chain)
        return v_curr, h_curr, vtab

    def updateWeights(self, v_pos, h_pos, negTermV, negTermH, negTermW):
        lr_p = self.lr/self.mb_s
        lr_n = self.lr
        self.W += h_pos.mm(v_pos.t())*lr_p - negTermW*lr_n
        self.vbias += torch.sum(v_pos, 1)*lr_p - negTermV*lr_n
        self.hbias += torch.sum(h_pos, 1)*lr_p - negTermH*lr_n

    def updateWeightsCentered(self, v_pos, h_pos_v, h_pos_m, v_neg, h_neg_v, h_neg_m, ν=0.2, ε=0.01):
        self.VisDataAv = torch.mean(v_pos, 1).float()
        self.HidDataAv = torch.mean(h_pos_m, 1).float()    
        
        NormPos = 1.0/self.mb_s
        #NormNeg = 1.0/self.num_pcd
        
        
        Xc_pos = (v_pos.t() - self.VisDataAv).t()
        Hc_pos = (h_pos_m.t() - self.HidDataAv).t()
        
        si_neg = v_neg
        τa_neg = h_neg_v
        ΔW_neg = h_neg_m - torch.outer(h_neg_v.float(), self.VisDataAv) - torch.outer(self.HidDataAv, v_neg) + torch.outer(self.HidDataAv, self.VisDataAv)
        ΔW =Hc_pos.mm(Xc_pos.t())*NormPos - ΔW_neg

        self.W += ΔW*self.lr

        ΔVB = torch.sum(v_pos, 1)*NormPos - si_neg - \
            torch.mv(ΔW.t().float(), self.HidDataAv)
        self.vbias += self.lr*ΔVB

        ΔHB = torch.sum(h_pos_m, 1)*NormPos - τa_neg - \
            torch.mv(ΔW.float(), self.VisDataAv)
        self.hbias += self.lr*ΔHB

    def fit_batch(self,X):
        h_pos_v, h_pos_m = self.SampleHiddens01(X)
        
        s = time.time()
        
        # SVD des poids
        _, _, self.V0 = torch.svd(self.W)
        if torch.mean(self.V0[:,0]) < 0:
            self.V0 = -self.V0
        
        # pour adapter la taille de l'intervalle discrétisé à chaque itération
        proj_data = torch.mm(X.T, self.V0)
        width_plus = 0.2
        limits = torch.zeros((2, self.nDim))
        for i in range(self.nDim):
            limits[0, i] = proj_data[:,i].min()-width_plus
            limits[1, i] = proj_data[:,i].max()+width_plus
        x_grid = np.linspace(limits[0,0], limits[1,0], self.nb_point_dim[0])
        x_grid = np.array([x_grid for i in range(self.nb_point_dim[1])])
        x_grid = x_grid.reshape(self.nb_point)
        y_grid = []
        y_d = np.linspace(limits[0,1], limits[1,1], self.nb_point_dim[1])
        for i in range(self.nb_point_dim[1]):
            for j in range(self.nb_point_dim[0]):
                y_grid.append(y_d[i])
        self.w_hat_b = torch.tensor([x_grid, y_grid], device = self.device, dtype = self.dtype)
        
        start = torch.bernoulli(torch.rand(self.Nv, self.nb_chain*self.nb_point, device = self.device))
        w_hat = torch.zeros((self.nDim, self.nb_chain*self.nb_point), device = self.device)
        for i in range(self.nb_point):
            for j in range(self.nb_chain):
                w_hat[:,i*self.nb_chain+j] = self.w_hat_b[:,i]
        
        print("Initialisation time : ", time.time()-s)
        
        s=time.time()
        
        tmpv, tmph, vtab = self.TMCSample(start, w_hat, self.N, self.V0, it_mcmc = self.gibbs_steps, it_mean=self.it_mean)
        
        print("Sampling time : ", time.time()-s)
        
        newy = torch.mm(torch.mean(vtab, dim = 2).T, self.V0)[:,:self.nDim]/self.Nv**0.5
        grad_pot = newy.T-self.w_hat_b
        square = torch.zeros(2, self.nb_point_dim[0], self.nb_point_dim[1])
        w_hat_tmp = np.zeros((2, self.nb_point_dim[0], self.nb_point_dim[1]))
        for i in range(0,grad_pot.shape[1], self.nb_point_dim[0]):
                w_hat_tmp[:,:,int(i/self.nb_point_dim[0])] = self.w_hat_b[:, i:(i+self.nb_point_dim[0])].cpu().numpy()
                square[:,:, int(i/self.nb_point_dim[0])] = grad_pot[:,i:(i+self.nb_point_dim[0])]
        
        w_hat_dim = []
        for i in range(self.nDim):
            w_hat_dim.append(np.linspace(limits[0,i], limits[1,i], self.nb_point_dim[i]))

        res_x = np.zeros(self.nb_point_dim[0])
        for i in range(self.nb_point_dim[0]):
            res_x[i] = simps(square[0][:(i+1),0].cpu().numpy(), w_hat_tmp[0,:(i+1),0])
        res_y = np.zeros((self.nb_point_dim[0], self.nb_point_dim[1]))
        for i in range(self.nb_point_dim[0]):
            for j in range(self.nb_point_dim[1]):
                res_y[i,j] = simps(square[1][i,:(j+1)].cpu().numpy(), w_hat_tmp[1,i,:(j+1)])

        pot = np.expand_dims(res_x, 1).repeat(self.nb_point_dim[1].cpu(),1) + res_y    
        res = np.exp(self.N*(pot-np.max(pot)))
        
        const = np.zeros(res.shape[0])
        for i in range(res.shape[0]):
            const[i-1] = simps(res[:,i], w_hat_tmp[1, i, :])
        const = simps(const, w_hat_tmp[0,:,0])
        self.p_m = torch.tensor(res/const, device=self.device, dtype=self.dtype)
        
        s_i = torch.mean(tmpv, dim = 2)
        tau_a = torch.mean(tmph, dim = 2)
        s_i_square = torch.zeros([s_i.shape[0], self.nb_point_dim[0], self.nb_point_dim[1]])
        tau_a_square = torch.zeros([tau_a.shape[0], self.nb_point_dim[0], self.nb_point_dim[1]])

        for i in range(0,grad_pot.shape[1], self.nb_point_dim[0]):
            s_i_square[:,:,int(i/self.nb_point_dim[0])] = s_i[:, i:(i+self.nb_point_dim[0])]
            tau_a_square[:,:,int(i/self.nb_point_dim[0])] = tau_a[:, i:(i+self.nb_point_dim[0])]
    
        prod = torch.zeros(
            (self.Nv, self.Nh, self.nb_point), device=self.device)
        tmpcompute = torch.zeros(self.Nv, self.Nh, self.nb_chain)
        s = time.time()
        for i in range(self.nb_point):
            for k in range(self.nb_chain):
                tmpcompute[:,:,k] = torch.outer(tmpv[:, i, k], tmph[:, i, k])
            prod[:, :, i] = torch.mean(tmpcompute, dim = 2)
        print("si tau_a prod : ", time.time()-s)
        
        s_i_square = torch.zeros([s_i.shape[0], self.nb_point_dim[0], self.nb_point_dim[1]], device=self.device, dtype=self.dtype)
        tau_a_square = torch.zeros([tau_a.shape[0], self.nb_point_dim[0], self.nb_point_dim[1]], device=self.device, dtype=self.dtype)
        prod_square = torch.zeros((prod.shape[0], prod.shape[1], self.nb_point_dim[0], self.nb_point_dim[1]), device=self.device, dtype=self.dtype)
        for i in range(0,grad_pot.shape[1], self.nb_point_dim[0]):
            s_i_square[:,:,int(i/self.nb_point_dim[0])] = s_i[:, i:(i+self.nb_point_dim[0])]
            tau_a_square[:,:,int(i/self.nb_point_dim[0])] = tau_a[:, i:(i+self.nb_point_dim[0])]
            prod_square[:,:,:,int(i/self.nb_point_dim[0])] = prod[:, :, i:(i+self.nb_point_dim[0])]

        tmpres_s_i = torch.zeros(self.Nv, self.nb_point_dim[0], device=self.device, dtype=self.dtype)
        tmpres_tau_a = torch.zeros(self.Nh, self.nb_point_dim[0], device=self.device, dtype=self.dtype)
        tmpres_prod = torch.zeros((prod_square.shape[0], prod_square.shape[1], prod_square.shape[2]), device=self.device, dtype=self.dtype)
        s_i_square = self.p_m*s_i_square #Ca fait bien ce qu'on veut
        tau_a_square = self.p_m*tau_a_square
        prod_square = self.p_m*prod_square
        s_i_fin = torch.zeros(self.Nv, device=self.device, dtype=self.dtype)
        tau_a_fin = torch.zeros(self.Nh, device=self.device, dtype=self.dtype)
        prod_fin = torch.zeros(prod_square.shape[0], prod_square.shape[1], device=self.device, dtype=self.dtype)
        for i in range(self.nb_point_dim[0]):
            tmpres_s_i[:,i] = torch.trapz(s_i_square[:,i,:], torch.tensor(w_hat_dim[1], device=self.device, dtype=self.dtype))
            tmpres_tau_a[:,i] = torch.trapz(tau_a_square[:,i,:], torch.tensor(w_hat_dim[1], device=self.device, dtype=self.dtype))
            tmpres_prod[:,:,i] = torch.trapz(prod_square[:,:,i,:], torch.tensor(w_hat_dim[1], device=self.device, dtype=self.dtype))
        tau_a_fin = torch.trapz(tmpres_tau_a, torch.tensor(w_hat_dim[0], device=self.device, dtype=self.dtype))
        s_i_fin = torch.trapz(tmpres_s_i, torch.tensor(w_hat_dim[0], device=self.device, dtype=self.dtype))
        prod_fin = torch.trapz(tmpres_prod, torch.tensor(w_hat_dim[0], device=self.device, dtype=self.dtype))


        if self.UpdCentered:
            self.updateWeightsCentered(X, h_pos_v, h_pos_m, s_i_fin, tau_a_fin, prod_fin.T)
        else:
            self.updateWeights(X, h_pos_m, s_i_fin, tau_a_fin, prod_fin.T)

   
    def getMiniBatches(self,X,m):
        return X[:,m*self.mb_s:(m+1)*self.mb_s]

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
            f.create_dataset('lr', data=self.lr)
            f.create_dataset('NGibbs', data=self.gibbs_steps)
            f.create_dataset('UpdByEpoch', data=NB)
            f.create_dataset('miniBatchSize', data=self.mb_s)
            f.create_dataset('numPCD', data=self.num_pcd)
            f.create_dataset('alltime', data=self.list_save_rbm)
            f.close()

        for t in range(ep_max):
            print("IT ", self.ep_tot)
            self.ep_tot += 1

            Xp = X[:, torch.randperm(X.size()[1])]

            for m in range(NB):
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
                    f.create_dataset('p_m'+str(self.up_tot), data=self.p_m.cpu())
                    f.close()
                    if self.verbose == 1 :
                        _, S, _ = torch.svd(self.W)
                        print(S[:2])
                self.up_tot += 1

            if self.ep_tot in self.list_save_rbm:
                f = h5py.File('../model/RBM'+self.file_stamp+'.h5', 'a')
                f.create_dataset('W'+str(self.ep_tot), data=self.W.cpu())
                f.create_dataset('vbias'+str(self.ep_tot),
                                 data=self.vbias.cpu())
                f.create_dataset('hbias'+str(self.ep_tot),
                                 data=self.hbias.cpu())
                f.create_dataset('p_m'+str(self.ep_tot), data=self.p_m.cpu())

                f.close()
            
            if self.save_fig:
                vinit = torch.bernoulli(torch.rand((self.Nv, 1000), device=self.device, dtype=self.dtype))
                si, _, _, _ = self.Sampling(vinit, it_mcmc=self.gibbs_steps)
                proj_gen = torch.mv(si.T, self.V0)/self.Nv**0.5
                proj_data = torch.mv(X.T, self.V0)/self.Nv**0.5
                plt.figure(dpi = 200)
                plt.xlim(torch.min(proj_data).item()-0.2, torch.max(proj_data).item()+0.2)

                plt.hist(proj_data.cpu().numpy(), bins = 100, density = True);
                plt.hist(proj_gen.cpu().numpy(), bins = 100, density = True);
                plt.plot(self.w_hat_b.cpu().numpy()[1:],self.p_m.cpu().numpy(), '-')
                plt.savefig("../fig/TMC/distrib_ep_"+str(self.ep_tot)+".png")
                plt.close()



                
        print("model updates saved at " + "../model/AllParameters"+self.file_stamp+".h5")
        print("model saved at " +"../model/RBM"+self.file_stamp+".h5")
