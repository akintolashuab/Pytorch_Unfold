# %%import of PAckages
import numpy as np
import torch
import pandas as pd
import torchvision
from torch.utils.data import DataLoader, Dataset
import math
# %% lets create a custom dataset class
class WineDataset(Dataset):
    def __init__(self):
        #data loading
        xy = np.loadtxt('./wine.csv', delimiter=',', dtype=np.float32, skiprows=1)
        self.n_samples = xy.shape[0]
        self.x = torch.from_numpy(xy[:, 1:])  #features
        self.y = torch.from_numpy(xy[:, [0]])  #labels
        
    def __getitem__(self, index):  
        return self.x[index], self.y[index]
    
    def __len__(self):
        return self.n_samples

# %%
dataset = WineDataset()
#first_data = dataset[0]
features, labels = dataset[103:108] #first_data
print(features, labels)
# %%
print(len(dataset))
print(dataset[0:5])
# %% loading the data with dataloader
dataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=0)
dataiter = iter(dataloader)
#data = next(dataiter)
features, labels = next(dataiter)
print(features, labels)

# %%
num_epochs = 2
total_samples = len(dataset)
n_iterations = math.ceil(total_samples/4)
print(total_samples, n_iterations)

# %% lets loop through the dataset multiple times
for epoch in range(num_epochs):
    for i, (inputs, labels) in enumerate(dataloader):
        #forward backward, update
        if (i+1) % 5 == 0:
            print(f'epoch {epoch+1}/{num_epochs}, step {i+1}/{n_iterations}, inputs {inputs.shape}')

# %%
