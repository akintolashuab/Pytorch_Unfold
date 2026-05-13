#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import seaborn as sns
from torch.utils.data import DataLoader, TensorDataset, dataset, Dataset

# %%
num_points = 360*4
X = np.arange(num_points)
print(X.shape)
y = [np.cos(X[i]*np.pi/180) * (1+i/num_points) + np.random.normal(0, 0.05) for i in range(num_points)]
print(len(y))
print(y)
sns.lineplot(x=X, y=y)

# %%
x_restruct = []
y_restruct = []

for i in range(num_points-10):
    list_1 = []
    for j in range(i, i+10):
        list_1.append(y[j])
    x_restruct.append(list_1)
    y_restruct.append(y[j+1])

x_restruct = np.array(x_restruct)
y_restruct = np.array(y_restruct)
# %%
print(x_restruct[0])
print(y_restruct[0])
# %%
print(x_restruct.shape)
print(y_restruct.shape)
# %%
print(x_restruct[0].shape)
print(y_restruct[0].shape)

# %%
print(x_restruct)
# %%
train_test_clipping = 360*3
x_train = x_restruct[:train_test_clipping]
y_train = y_restruct[:train_test_clipping]
x_test = x_restruct[train_test_clipping:]
y_test = y_restruct[train_test_clipping:]
print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)
# %%
class TrigonometricDataset(Dataset):
    def __init__(self, x, y):
        self.x = torch.tensor(x, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]
    
#dateloaders
train_loader = DataLoader(TrigonometricDataset(x_train, y_train), batch_size=32, shuffle=True)
test_loader = DataLoader(TrigonometricDataset(x_test, y_test), batch_size=32, shuffle=False)
sns.lineplot(x=range(len(y_train)), y=y_train, label = "Training Data")

# %%
hidden_size = 5
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size) -> None:
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=1,  batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        out = self.fc(lstm_out[:, -1, :])  # Use the last output of the LSTM
        return out
    
# %%
model = LSTMModel(input_size=1, hidden_size=5, output_size=1)
loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
num_epochs = 200

# %%
for epoch in range(num_epochs):
    for j, (x, y) in enumerate(train_loader):

        optimizer.zero_grad()
        y_pred = model(x.view(-1, 10, 1))
        loss = loss_fn(y_pred, y.unsqueeze(1))

        loss.backward()
        optimizer.step()
        
    if (epoch+1) % 2 == 0:
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.4f}")
       



# %% Testing the model
x_test, y_test = next(iter(test_loader))
with torch.no_grad():
    y_pred = model(torch.unsqueeze(x_test, 2)).detach().squeeze().numpy()
y_acct = y_test.numpy()
x_acct = range(y_acct.shape[0])
sns.lineplot(x=x_acct, y=y_acct, label="Actual", color="blue")
sns.lineplot(x=x_acct, y=y_pred, label="Predicted", color="red")
plt.legend()
plt.show()


# %%
model.eval()

predictions = []
actuals = []

with torch.no_grad():

    for i, (x_test, y_test) in enumerate(test_loader):

        x_test = x_test.unsqueeze(2)

        y_pred = model(x_test)

        predictions.extend(y_pred.squeeze().numpy())
        actuals.extend(y_test.numpy())

sns.lineplot(x=range(len(actuals)), y=actuals, label="Actual", color="blue")
sns.lineplot(x=range(len(predictions)), y=predictions, label="Predicted", color="red")

plt.legend()
plt.show()

