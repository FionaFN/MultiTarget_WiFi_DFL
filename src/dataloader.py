import numpy as np
import torch
from torch.utils.data import Dataset

class dataset(Dataset):
    def __init__(self, dRSS, GT, prior):
        self.dRSS = torch.from_numpy(dRSS).float()
        self.GT   = torch.from_numpy(GT).float()
        self.prior= torch.from_numpy(prior).float()
            
    def __len__(self):
        return len(self.dRSS)
        
    def __getitem__(self, index):    
        dRSS = torch.unsqueeze(self.dRSS[index], 0)
        GT   = torch.unsqueeze(self.GT[index], 0) 
        prior= torch.unsqueeze(self.prior[index], 0)
        return dRSS, GT, prior
    
def get_dataset(dataset_dir,split):    
    vars()[split+'_dRSS'] = np.load(dataset_dir+split+'_dRSS_M40.npy', allow_pickle=True)
    vars()[split+'_GT']  = np.load(dataset_dir+split+'_GT_probmap.npy', allow_pickle=True)  
    vars()[split+'_prior'] = np.load(dataset_dir+split+'_prior.npy', allow_pickle=True) 
    vars()[split+'_set'] = dataset(vars()[split+'_dRSS'], vars()[split+'_GT'], vars()[split+'_prior'])
    return vars()[split+'_set']