import torch
from torch.utils.data import Dataset
import numpy as np
import os

class HCPOrigDataset(Dataset):
    def __init__(self, data_root, task_id,mode='train', transform=None):
        super(HCPOrigDataset, self).__init__()
        self.data_root = data_root
        self.mode = mode
        self.transform = transform
        self.task_id = task_id
        
        data_dir = os.path.join(data_root, mode)

        X0_name = '0.npy'
        X1_name = '1.npy'
        X2_name = '2.npy'

        self.X0 = np.load(os.path.join(data_dir, X0_name))
        self.X1 = np.load(os.path.join(data_dir, X1_name))
        self.X2 = np.load(os.path.join(data_dir, X2_name))

        self.y=np.load(os.path.join(data_dir, 'y_task_{}.npy'.format(task_id)))

    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        X0 = self.X0[idx]
        X1 = self.X1[idx]
        X2 = self.X2[idx]
        y = self.y[idx]
        
        sample = {'X0': X0, 'X1': X1, 'X2': X2, 'y': y}
        
        if self.transform:
            sample = self.transform(sample)
        
        return sample
    
class HCPGraphDataset(Dataset):
    def __init__(self, data_root, task_id,mode='train', transform=None):
        super(HCPGraphDataset, self).__init__()
        self.data_root = data_root
        self.mode = mode
        self.transform = transform
        self.task_id = task_id
        
        data_dir = os.path.join(data_root, mode)

        X0_name = '0.npy'
        X1_name = '1.npy'
        X2_name = '2.npy'

        self.X0 = np.load(os.path.join(data_dir, X0_name))
        self.X1 = np.load(os.path.join(data_dir, X1_name))
        self.X2 = np.load(os.path.join(data_dir, X2_name))

        self.y=np.load(os.path.join(data_dir, 'y_task_{}.npy'.format(task_id)))

    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        X0 = self.X0[idx]
        X1 = self.X1[idx]
        X2 = self.X2[idx]
        y = self.y[idx]

        # X: half of the adjacency matrix, shape: (79800, )
        X0_adjmat = np.zeros((400,400))
        X1_adjmat = np.zeros((400,400))
        X2_adjmat = np.zeros((400,400))

        X0_adjmat[np.triu_indices(400,1)] = X0
        X0_adjmat = X0_adjmat + X0_adjmat.T + np.eye(400)

        X1_adjmat[np.triu_indices(400,1)] = X1
        X1_adjmat = X1_adjmat + X1_adjmat.T + np.eye(400)

        X2_adjmat[np.triu_indices(400,1)] = X2
        X2_adjmat = X2_adjmat + X2_adjmat.T + np.eye(400)

        
        sample = {'X0': X0_adjmat, 'X1': X1_adjmat, 'X2': X2_adjmat, 'y': y}
        
        if self.transform:
            sample = self.transform(sample)
        
        return sample



class HCPDataset(Dataset):
    def __init__(self, data_root, task_id, dim_reduce='isomap',mode='train', transform=None):
        super(HCPDataset, self).__init__()
        self.data_root = data_root
        self.mode = mode
        self.transform = transform
        self.task_id = task_id
        
        data_dir = os.path.join(data_root, dim_reduce, mode)

        X0_name = '0_task_{}.npy'.format(task_id)
        X1_name = '1_task_{}.npy'.format(task_id)
        X2_name = '2_task_{}.npy'.format(task_id)

        self.X0 = np.load(os.path.join(data_dir, X0_name))
        self.X1 = np.load(os.path.join(data_dir, X1_name))
        self.X2 = np.load(os.path.join(data_dir, X2_name))

        self.y=np.load(os.path.join(data_dir, 'y_task_{}.npy'.format(task_id)))

    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        X0 = self.X0[idx]
        X1 = self.X1[idx]
        X2 = self.X2[idx]
        y = self.y[idx]
        
        sample = {'X0': X0, 'X1': X1, 'X2': X2, 'y': y}
        
        if self.transform:
            sample = self.transform(sample)
        
        return sample
    
class ToTensor(object):
    def __call__(self, sample):
        X0, X1, X2, y = sample['X0'], sample['X1'], sample['X2'], sample['y']
        
        return {'X0': torch.from_numpy(X0).float(),
                'X1': torch.from_numpy(X1).float(),
                'X2': torch.from_numpy(X2).float(),
                'y': torch.from_numpy(y).float()}
    
class Normalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std
        
    def __call__(self, sample):
        X0, X1, X2, y = sample['X0'], sample['X1'], sample['X2'], sample['y']
        
        X0 = (X0 - self.mean[0]) / self.std[0]
        X1 = (X1 - self.mean[1]) / self.std[1]
        X2 = (X2 - self.mean[2]) / self.std[2]
        
        return {'X0': X0, 'X1': X1, 'X2': X2, 'y': y}
    

if __name__ == "__main__":
    data_root = "/Datasets/recogbio/"
    task_id = 0
    dataset = HCPGraphDataset(data_root, task_id, mode='train')
    print(len(dataset))
    print(dataset[0]['X0'].shape)
    print(dataset[0]['X1'].shape)
    print(dataset[0]['X2'].shape)
    print(dataset[0]['y'].shape)
    
