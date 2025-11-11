# %% create a daatset with transforms
import torch
import torchvision
import numpy as np
from torch.utils.data import DataLoader, Dataset

# %% create a dataset with transforms
class WineDataset(Dataset):
    def __init__(self, transform=None):
        xy = np.loadtxt('./wine.csv', delimiter=',', dtype=np.float32, skiprows=1)
        self.n_samples = xy.shape[0]
        self.x = xy[:, 1:]
        self.y = xy[:, [0]]
        self.transform = transform
        
    def __getitem__(self, index):
        sample = self.x[index], self.y[index]
        
        if self.transform:
            sample = self.transform(sample)
            
        return sample
    
    def __len__(self):
        return self.n_samples
# %%
class ToTensor:
    def __call__(self, sample):
        inputs, targets = sample
        return torch.from_numpy(inputs), torch.from_numpy(targets)
# %%
class MulTransform:
    def __init__(self, factor):
        self.factor = factor
        
    def __call__(self, sample):
        inputs, targets = sample
        inputs *= self.factor
        return inputs, targets  
# %%
dataset = WineDataset(transform=ToTensor())
first_data = dataset[0]
print(first_data)
print(type(first_data[0]), type(first_data[1]))     
composed = torchvision.transforms.Compose([ToTensor(), MulTransform(4)])
dataset = WineDataset(transform=composed)
first_data = dataset[0]
print(first_data)
print(type(first_data[0]), type(first_data[1])) 
# %%
