import torch
import h5py
import pathlib
path = str(pathlib.Path(__file__).parent.absolute())+'/'

def genGaussianData(dim, Nsample, l_cube, var):
    var_str = ''
    for i in range(len(var)):
        var_str += str(var[i].item())+'_' 

    n_centers = var.shape[0]
    centers = torch.rand(size = (n_centers,dim))*l_cube

    fname = 'data_Nclusters'+str(n_centers)+'_dim'+str(dim)+'_Lcube'+str(l_cube)+'_var'+var_str+'.h5'

    print('generate sample')
    
    sample = []
    for i in range(Nsample):
        k = torch.randint(0, n_centers, (1, 1)).item()
        sample.append(torch.normal(mean=centers[k], std=torch.sqrt(var[k])))
    X = torch.stack(sample).T
    
    f = h5py.File(path+'../data/'+fname, 'w')
    f.create_dataset('centers', data=centers.cpu())
    f.create_dataset('var', data=var.cpu())
    f.create_dataset('data', data = X.cpu())
    f.close()

    print('saved at 'path+'data/' + fname)
