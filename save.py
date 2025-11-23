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
new_model = loaded_model.load_state_dict(6)
print(new_model)
# %%
# saving a checkpoint
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
optimizer.state_dict()
# %%
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
