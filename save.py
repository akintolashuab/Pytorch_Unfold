# %%
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

# %%
class Model(nn.Module):
    def __init__(self, n_input_features):
        super(Model, self).__init__()
        self.linear = nn.Linear(n_input_features, 1)

    def forward(self, x):
        y_pred = self.linear(x)
        y_pred = F.Sigmoid(y_pred)
        return y_pred
    
# %%
model = Model(6)
for param in model.parameters():
    print(param)
# %%
for i in range(19):
    if (i%2 == 0) & (i <= 8):
        print(i)
# %% Lazy way to save the model
## save the model only
FILE = 'model.pth'
torch.save(model, FILE)

## Load the model
loaded_model = torch.load('model.pth')
loaded_model.eval()
for param in loaded_model.parameters():
    print(param)
# %% Prefered way to save the model
## save only the state_dict
FILE = 'model.pth1'
torch.save(model.state_dict(), FILE)

## Load the model
loaded_model = Model(6)
loaded_model.load_state_dict(torch.load('model.pth1'))
loaded_model.eval()
for param in loaded_model.parameters():
    print(param)
# %%
model.state_dict()
loaded_model.state_dict()

# %%
# saving a checkpoint
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
optimizer.state_dict()
checkpoint = {
    'epoch': 90,    
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
}
torch.save(checkpoint, 'my_checkpoint.pth')
# %%
loaded_checkpoint = torch.load('my_checkpoint.pth')
model = Model(6)
model.load_state_dict(loaded_checkpoint['model_state_dict'])
optimizer = torch.optim.SGD(model.parameters(), lr=0)
optimizer.load_state_dict(loaded_checkpoint['optimizer_state_dict'])
epoch = loaded_checkpoint['epoch']
print(epoch)
print(model.state_dict())
print(optimizer.state_dict())
# %%
import torch
print(torch.cuda.is_available())
print(torch.cuda.device_count())
#print(torch.cuda.get_device_name(0))
# %%
# 1) Save on GPU, Load on CPU
device = torch.device("cuda")
model.to(device)
torch.save(model.state_dict(), PATH)

device = torch.device('cpu')
model = Model(*args, **kwargs)
model.load_state_dict(torch.load(PATH, map_location=device))

# 2) Save on GPU, Load on GPU
device = torch.device("cuda")
model.to(device)
torch.save(model.state_dict(), PATH)

model = Model(*args, **kwargs)
model.load_state_dict(torch.load(PATH))
model.to(device)

# Note: Be sure to use the .to(torch.device('cuda')) function 
# on all model inputs, too!

# 3) Save on CPU, Load on GPU
torch.save(model.state_dict(), PATH)

device = torch.device("cuda")
model = Model(*args, **kwargs)
model.load_state_dict(torch.load(PATH, map_location="cuda:0"))  # Choose whatever GPU device number you want
model.to(device)

# This loads the model to a given GPU device. 
# Next, be sure to call model.to(torch.device('cuda')) to convert the model’s parameter tensors to CUDA tensors
