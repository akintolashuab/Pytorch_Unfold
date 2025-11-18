# %% imports of necessary packages
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F
import sys
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
#writer = SummaryWriter('runs/mnist_experiment_2')
writer = SummaryWriter('runs_fresh/mnist_experiment_1')
# %% device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

# %% import of MNIST dataset
train_dataset = torchvision.datasets.MNIST(root='./data', train=True, transform=transforms.ToTensor(), download=True)
test_dataset = torchvision.datasets.MNIST(root='./data', train=False, transform=transforms.ToTensor())  
train_loader = DataLoader(dataset=train_dataset, batch_size=100, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=100, shuffle=False)

image_, label = train_dataset[0]
print(image_.shape, label)  # torch.Size([1, 28, 28
image_iter = iter(train_loader)
images, labels = next(image_iter)
print(images.shape, labels.shape)  # torch.Size([100, 1, 28, 28]) torch.Size([100])
print(len(train_dataset), len(test_dataset))  # 60000 10000

# %% code to display some sample images
example = iter(train_loader)
sample_images, sample_labels = next(example)
for i in range(6):
    plt.subplot(2, 3, i+1)
    plt.imshow(sample_images[i][0], cmap='gray')
    plt.title(f'Label: {sample_labels[i]}')
img_grid = torchvision.utils.make_grid(sample_images)
writer.add_image('mnist_images', img_grid)
#sys.exit()
#writer.close()
# %% hyperparameters
input_size = 784 #28x28
hidden_size1 = 500
hidden_size2 = 200
num_classes = 10
num_epochs = 3
learning_rate = 0.001
# %% feed forward neural network model
class FeedForwardNet(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, num_classes):
        super(FeedForwardNet, self).__init__()
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
# %%
model = FeedForwardNet(input_size, hidden_size1, hidden_size2, num_classes).to(device)
# %% loss and optimizer functions
criterion = nn.CrossEntropyLoss()   
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
writer.add_graph(model, images.reshape(-1, 28*28).to(device))
writer.close()
# %% training loop
total_step = len(train_loader)
print(total_step)
# %% training loop

for epoch in range(num_epochs):
    running_loss = 0.0
    running_correct = 0
    for i, (images, labels) in enumerate(train_loader):
        #reshaping the image from (100, 1, 28, 28) to (100, 784)
        images = images.reshape(-1, 28*28).to(device)
        labels = labels.to(device)
        
        #forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        #backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        running_correct += (predicted == labels).sum().item()

        if (i+1) % 100 == 0:
            print (f'epoch {epoch+1}/{num_epochs}, step {i+1}/{total_step}, loss: {loss.item():.4f}')
            writer.add_scalar('training loss', running_loss / 100, epoch * total_step + i)
            writer.add_scalar('accuracy', running_correct / (100 * 100), epoch * total_step + i)
            running_loss = 0.0
            running_correct = 0
        

# %%
# testing loop
preds = []
labels= []
with torch.no_grad():
    n_correct = 0
    n_samples = 0
    for images, labels_1 in test_loader:
        images = images.reshape(-1, 28*28).to(device)
        labels_1 = labels_1.to(device)
        outputs = model(images)
        
        #max returns (value ,index)
        _, predicted = torch.max(outputs, 1)
        n_samples += labels_1.size(0)
        n_correct += (predicted == labels_1).sum().item()

        labels.append(predicted)
        class_prediction = [F.softmax(output, dim=0) for output in outputs]
        preds.append(class_prediction)

    pred = torch.cat([torch.stack(batch) for batch in preds])
    label = torch.cat(labels)
    
    acc = 100.0 * n_correct / n_samples
    print(f'Accuracy of the network on the 10000 test images: {acc} %')

    classes = range(10)
    for i in classes:
        labels_i = label==i
        pred_i = pred[:,i]
        writer.add_pr_curve(str(i), labels_i, pred_i, global_step=0)
        writer.close()
# %%
