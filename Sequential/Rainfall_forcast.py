#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
import torch.nn.functional as F
from tqdm.notebook import trange, tqdm

# %%
df_data = pd.read_csv("Weather.csv")
df_data.head()

#%%
from dataset import WeatherDataset
# %%
#Define the date to split the dataset into training and testing sets
split_date = pd.to_datetime('2015-01-01')

# Number of days in the input sequence
day_range = 15

# Number of days the MLP will take as input
days_in = 14

# Ensure that the total number of days in the input sequence is larger than the MLP input size
assert day_range > days_in, "The total day range must be larger than the input days for the MLP"

# Define the hyperparameters for training the model
learning_rate = 1e-4  # Learning rate for the optimizer
nepochs = 500  # Number of training epochs
batch_size = 32  # Batch size for training

#%%
# Create training dataset
# This will load the weather data, consider sequences of length day_range,
# and split the data such that data before split_date is used for training
dataset_train = WeatherDataset(df_data,
         day_range=day_range, split_date=split_date, train_test="train")
 # %%
dataset_train
# %%
x, y = dataset_train[1]
print(x.shape)
print(y)

#%%
dataset_test = WeatherDataset(df_data, day_range=day_range, split_date=split_date, train_test="test", 
                              mean=dataset_train.mean, std=dataset_train.std)
# %%
x, y = dataset_test[1]
print(x.shape)
print(y)
# %%
df_data.weather.value_counts()
# %%
x, y = dataset_train[0]
x
# %%
y
# %%
x, y = dataset_train[1]
x
# %%
y
# %%
df_data.head(15)
# %%
print(len(dataset_train))
print(len(dataset_test))
print(len(df_data))
# %%
print(f'Number of training examples: {len(dataset_train)}')
print(f'Number of testing examples: {len(dataset_test)}')
data_loader_train = DataLoader(dataset=dataset_train, batch_size=batch_size, shuffle=False, drop_last=True)
data_loader_test = DataLoader(dataset=dataset_test, batch_size=batch_size, shuffle=False, drop_last=True)

#%%
for batch, (x, y) in enumerate(data_loader_train):
    print(f"Batch {batch}:")
    print(f"Input shape: {x.shape}")
    print(f"Target shape: {y.shape}")
    print(batch, (x, y))
    break
#%%
fig = plt.figure(figsize=(10, 5))
plt.title("Melbourne Max Daily Temperature (C)")

plt.plot(dataset_train.data[:,1])
plt.plot(dataset_test.data[:,1])

plt.legend(["Train","Test"])
plt.show()
# %%
# Note:see here how we can just directly access the data from the dataset class
#Create RNN Network
class ResBlockMLP(nn.Module):
    def __init__(self, input_size, output_size):
        super(ResBlockMLP, self).__init__()
        # Layer normalization for the input
        self.norm1 = nn.LayerNorm(input_size)
        # First fully connected layer that reduces the dimensionality by half
        self.fc1 = nn.Linear(input_size, input_size // 2)
        
        # Layer normalization after the first fully connected layer
        self.norm2 = nn.LayerNorm(input_size // 2)
        # Second fully connected layer that outputs the desired output size
        self.fc2 = nn.Linear(input_size // 2, output_size)
        
        # Skip connection layer to match the output size
        self.fc3 = nn.Linear(input_size, output_size)

        # Activation function
        self.act = nn.ELU()

    def forward(self, x):
        # Apply normalization and activation function to the input
        x = self.act(self.norm1(x))
        # Compute the skip connection output
        skip = self.fc3(x)
        
        # Apply the first fully connected layer, normalization, and activation function
        x = self.act(self.norm2(self.fc1(x)))
        # Apply the second fully connected layer
        x = self.fc2(x)
        
        # Add the skip connection to the output
        return x + skip

#%%
class RNN(nn.Module):
    def __init__(self, seq_len, num_features, output_size=1, num_blocks=1, buffer_size=128):
        super(RNN, self).__init__()

        self.seq_len = seq_len
        self.num_features = num_features

        # Total flattened input size
        seq_data_len = seq_len * num_features   # ✅ FIXED

        # Input MLP
        self.input_mlp = nn.Sequential(
            nn.Linear(seq_data_len, 4 * seq_data_len),
            nn.ELU(),
            nn.Linear(4 * seq_data_len, 128),
            nn.ELU()
        )

        # "RNN" layer
        self.rnn = nn.Linear(128 + buffer_size, 128)

        # Residual blocks
        self.res_blocks = nn.Sequential(
            *[ResBlockMLP(128, 128) for _ in range(num_blocks)]
        )

        # Output layer (predict temp_max)
        self.fc_out = nn.Linear(128, output_size)

        # Buffer update
        self.fc_buffer = nn.Linear(128, buffer_size)

        self.act = nn.ELU()

    def forward(self, input_seq, buffer_in):
        # Flatten: (B, seq_len, features) → (B, seq_len*features)
        #x= input_seq.view(input_seq.size(0), -1)  # (B, seq_len*num_features)
        x = input_seq.reshape(input_seq.shape[0], -1)

        # Encode input
        input_vec = self.input_mlp(x)   # (B,128)

        # Combine with memory
        x_cat = torch.cat((buffer_in, input_vec), dim=1)  # (B, 128+128)

        # RNN step
        x = self.rnn(x_cat)

        # Residual learning
        x = self.act(self.res_blocks(x))

        # Output + new memory
        output = self.fc_out(x)
        new_buffer = torch.tanh(self.fc_buffer(x))

        return output, new_buffer                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           


#%%
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

num_features = 5   # ✅ your dataset
buffer_size = 128

weather_rnn = RNN(
    seq_len=days_in,
    num_features=num_features,
    output_size=1,   # predict temp_max
    buffer_size=buffer_size
).to(device)

optimizer = optim.Adam(weather_rnn.parameters(), lr=learning_rate)
loss_fn = nn.MSELoss()

#%%
training_loss_logger = []

for epoch in trange(nepochs, desc="Epochs"):
    weather_rnn.train()

    for i, (x_batch, y_batch) in enumerate(data_loader_train):

        x_batch = x_batch.to(device)   # (B, 14, 5)
        y_batch = y_batch.to(device).unsqueeze(1)  # (B,1)

        # Initialize memory
        buffer = torch.zeros(x_batch.size(0), buffer_size, device=device)

        # Forward pass
        y_pred, buffer = weather_rnn(x_batch, buffer)

        # Compute loss
        loss = loss_fn(y_pred, y_batch)

        # Backprop
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        training_loss_logger.append(loss.item())
# %%
plt.figure(figsize=(10,5))
plt.plot(training_loss_logger)
plt.title("Training Loss")
plt.show()


# %%
weather_rnn.eval()

data_tensor = torch.FloatTensor(dataset_test.data).to(device)

predictions = []

with torch.no_grad():

    for i in range(len(data_tensor) - days_in):

        # Use REAL data window
        seq_block = data_tensor[i:i+days_in].unsqueeze(0)  # (1,14,5)

        buffer = torch.zeros(1, buffer_size, device=device)

        pred, buffer = weather_rnn(seq_block, buffer)

        predictions.append(pred.cpu())


# %%
log_predictions = []

weather_rnn.eval()

with torch.no_grad():

    buffer = torch.zeros(1, buffer_size, device=device)

    # Start with real data
    seq_block = data_tensor[:days_in].unsqueeze(0).to(device)

    for j in range(data_tensor.shape[0] - days_in):

        # Predict next temp_max
        data_pred, buffer = weather_rnn(seq_block, buffer)

        log_predictions.append(data_pred.cpu())

        # ---- FIX STARTS HERE ----
        # Take last row as template
        new_step = seq_block[:, -1:, :].clone()   # (1,1,5)

        # Replace ONLY temp_max (column index 1)
        new_step[:, :, 1] = data_pred

        # Slide window
        seq_block = torch.cat(
            (seq_block[:, 1:, :], new_step),
            dim=1
        )
predictions_cat = torch.cat(log_predictions)


# %%
# Unnormalize
un_norm_predictions = (predictions_cat * dataset_test.std[1]) + dataset_test.mean[1]
un_norm_data = (data_tensor * dataset_test.std) + dataset_test.mean
un_norm_data = un_norm_data[days_in:, 1]

# MSE
test_mse = (un_norm_data - un_norm_predictions.squeeze()).pow(2).mean().item()
print(f"Test MSE value {test_mse:.2f}")

# Plot
plt.figure(figsize=(10,5))
plt.plot(un_norm_data, label="Ground Truth")
plt.plot(un_norm_predictions.squeeze(), label="Prediction")
plt.title("Max Daily Temperature (C)")
plt.legend()
plt.show()
# %%
print(un_norm_predictions[:10])
print(un_norm_data[:10])
# %%
