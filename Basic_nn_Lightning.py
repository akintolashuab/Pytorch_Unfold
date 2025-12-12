# %% import of libraries
import lightning as L
import torch
from torch import nn as nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import SGD
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

#%%create Basic neural network class
class BasicLightning(L.LightningModule):
    def __init__(self):
        super(BasicLightning, self).__init__()
        self.w00 = nn.Parameter(torch.tensor(1.70), requires_grad=False)
        self.w10 = nn.Parameter(torch.tensor(12.6), requires_grad=False)
        self.w01 = nn.Parameter(torch.tensor(-40.8), requires_grad=False)
        self.w11 = nn.Parameter(torch.tensor(2.7), requires_grad=False)
        self.b00 = nn.Parameter(torch.tensor(-0.85), requires_grad=False)
        self.b10 = nn.Parameter(torch.tensor(0.00), requires_grad=False)
        self.bfinal = nn.Parameter(torch.tensor(-16.0), requires_grad=False)

    def forward(self, input):
        top_relu = F.relu((self.w00*input) + self.b00)
        bottom_relu = F.relu((self.w10*input) + self.b10)
        top_relu1 = top_relu * self.w01
        bottom_relu1 = bottom_relu * self.w11
        output = F.relu(top_relu1 + bottom_relu1 + self.bfinal)
        return output

    
# %%
doses = torch.linspace(0, 1, steps=11) # Create input tensor
model = BasicLightning()
outputs = model(doses)
print(doses)
print(outputs)
sns.set_style("whitegrid")
plt.figure(figsize=(8, 5))
plt.plot(doses.detach().numpy(), outputs.detach().numpy())
plt.title('Basic Neural Network Dose-Response Curve', fontsize=16)
plt.xlabel('Dose')
plt.ylabel('Response')
plt.show()


## NEWRAL NETWORK WITH LIGHTNING FRAMEWORK.
# %%
inputs = torch.tensor([0.0, 0.5, 1.0])
labels = torch.tensor([0.0, 1.0, 0.0])
dataset = TensorDataset(inputs, labels)
dataloader = DataLoader(dataset)

# %%
class BasicLightningTrain(L.LightningModule):
    def __init__(self):
        super(BasicLightningTrain, self).__init__()
        self.w00 = nn.Parameter(torch.tensor(1.70), requires_grad=False)
        self.w10 = nn.Parameter(torch.tensor(12.6), requires_grad=False)
        self.w01 = nn.Parameter(torch.tensor(-40.8), requires_grad=False)
        self.w11 = nn.Parameter(torch.tensor(2.7), requires_grad=False)
        self.b00 = nn.Parameter(torch.tensor(-0.85), requires_grad=False)
        self.b10 = nn.Parameter(torch.tensor(0.00), requires_grad=False)
        self.bfinal = nn.Parameter(torch.tensor(0.0), requires_grad=True)
        self.learning_rate = 0.01


    def forward(self, input):
        top_relu = F.relu((self.w00*input) + self.b00)
        bottom_relu = F.relu((self.w10*input) + self.b10)
        top_relu1 = top_relu * self.w01
        bottom_relu1 = bottom_relu * self.w11
        output = F.relu(top_relu1 + bottom_relu1 + self.bfinal)
        return output
    
model = BasicLightningTrain()
optimizer = SGD(model.parameters(), lr=0.1)
print("Before training, bfinal =" + str(model.bfinal.item()) + "\n")
for epoch in range(100):
    total_loss = 0
    for i in range(len(inputs)):
        input= inputs[i]
        label= labels[i]
        output = model(input)
        loss = (output-label)**2
        loss.backward()
        total_loss += float(loss.item())
    optimizer.step()
    optimizer.zero_grad()

    if total_loss < 0.0001:
        print("num_steps to converge:", str(epoch))
        break
    print("Step:" + str(epoch), "Loss:" + str(round(total_loss, 3)), "bfinal:" + str(model.bfinal.item()))
print("After training, bfinal =" + str(model.bfinal.item()) + "\n")


# %%
model.b00
model.bfinal
# %%

model = BasicLightningTrain()
with torch.no_grad():
    doses = torch.linspace(0, 1, steps=11)
    outputs = model(doses)
    print(doses)
    print(outputs)
    sns.set_style("whitegrid")
    plt.figure(figsize=(8, 5))
    plt.plot(doses.detach().numpy(), outputs.detach().numpy())
    plt.title('Basic Neural Network Dose-Response Curve', fontsize=16)
    plt.xlabel('Dose')
    plt.ylabel('Response')
    plt.show()


# %% Using Lightning Trainer
class BasicLightningTrainnew(L.LightningModule):
    def __init__(self):
        super(BasicLightningTrainnew, self).__init__()
        self.w00 = nn.Parameter(torch.tensor(1.70), requires_grad=False)
        self.w10 = nn.Parameter(torch.tensor(12.6), requires_grad=False)
        self.w01 = nn.Parameter(torch.tensor(-40.8), requires_grad=False)
        self.w11 = nn.Parameter(torch.tensor(2.7), requires_grad=False)
        self.b00 = nn.Parameter(torch.tensor(-0.85), requires_grad=False)
        self.b10 = nn.Parameter(torch.tensor(0.00), requires_grad=False)
        self.bfinal = nn.Parameter(torch.tensor(0.0), requires_grad=True)
        self.learning_rate = 0.01


    def forward(self, input):
        top_relu = F.relu((self.w00*input) + self.b00)
        bottom_relu = F.relu((self.w10*input) + self.b10)
        top_relu1 = top_relu * self.w01
        bottom_relu1 = bottom_relu * self.w11
        output = F.relu(top_relu1 + bottom_relu1 + self.bfinal)
        return output

    def configure_optimizers(self):
        optimizer = SGD(self.parameters(), lr=self.learning_rate)
        return optimizer
      
    def training_step(self, batch, batch_idx):
        input, label = batch
        output = self(input)
        loss = (output-label)**2
        return loss
# %%   
from lightning.pytorch.tuner import Tuner
model = BasicLightningTrainnew()
trainer = L.Trainer(max_epochs=2000)
tuner = Tuner(trainer)
lr_find_result = tuner.lr_find(model, 
                                       train_dataloaders=dataloader,
                                       min_lr=0.001,
                                       max_lr=1.0,
                                       early_stop_threshold=None)

new_lr = lr_find_result.suggestion()
print(f"Suggested learning rate is: , {new_lr:.4f}")

# %%
model.learning_rate = new_lr
trainer.fit(model, train_dataloaders = dataloader)
print(model.bfinal.data)
# %%
doses = torch.linspace(0, 1, steps=11)
outputs = model(doses)
sns.set_style("whitegrid")
plt.figure(figsize=(8, 5))
sns.lineplot(x= doses.detach().numpy(), y = outputs.detach().numpy(), 
             color = 'green', linewidth=2.5)
           
plt.title('Basic Neural Network Dose-Response Curve', fontsize=16)
plt.xlabel('Dose')
plt.ylabel('Response')
plt.show()
# %%
