# %%# %% imports of necessary packages
import torch
import torch.nn as nn
import lightning.pytorch as pl
from lightning.pytorch import Trainer
#import pytorch_lightning as pl
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F
import sys
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
# from pytorch_lightning import Trainer

# %% Hyperparameters
input_size = 784 #28x28
hidden_size1 = 500
hidden_size2 = 200
num_classes = 10
num_epochs = 5
learning_rate = 0.001
# %% Building Feed Forward Neural Network using Lightning
# %% feed forward neural network model
class FeedForwardNetLightning(pl.LightningModule):
    def __init__(self, input_size, hidden_size1, hidden_size2, num_classes):
        super(FeedForwardNetLightning, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size1) 
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size1, hidden_size2) 
        self.fc3 = nn.Linear(hidden_size2, num_classes) 
    
    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.fc3(out)
        return out
    
    def training_step(self, batch, batch_idx):
        images, labels = batch
        images = images.reshape(-1, 28*28)
        outputs = self.forward(images)
        loss = F.cross_entropy(outputs, labels)
        train_tensorboard = {'train_loss': loss}
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return {'loss': loss, 'log': train_tensorboard}
    
    def train_dataloader(self):
        train_dataset = torchvision.datasets.MNIST(root='./data', 
                    train=True, transform=transforms.ToTensor(), download=True)
        train_loader = DataLoader(dataset=train_dataset, 
                    batch_size=100, num_workers=0, shuffle=True)
        return train_loader
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        return optimizer
    
    def validation_step(self, batch, batch_idx):
        images, labels = batch
        images = images.reshape(-1, 28*28)
        outputs = self.forward(images)
        loss = F.cross_entropy(outputs, labels)
        self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return {"val_loss": loss}
    
    def val_dataloader(self):
        val_dataset = torchvision.datasets.MNIST(root='./data', 
                    train=False, transform=transforms.ToTensor())
        val_loader = DataLoader(dataset=val_dataset, 
                    batch_size=100, num_workers=0, shuffle=False)
        return val_loader
    
    def validation_epoch_end(self,output):
        avg_loss = torch.stack([x for x in output]).mean()
        self.log('val_loss', avg_loss)
        val_tensorboard = {'val_loss': avg_loss}
        self.log('val_loss', avg_loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return {"val_loss": avg_loss, 'val_log': val_tensorboard}



# %%
trainer = Trainer(fast_dev_run=False, max_epochs=num_epochs, log_every_n_steps=1)
model = FeedForwardNetLightning(input_size, hidden_size1, hidden_size2, num_classes)
trainer.fit(model)
# %%
print(pl.__version__)
# %%
