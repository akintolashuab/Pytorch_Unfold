# %% importing libraries
import torch
import torch.nn as nn   
import torch.nn.functional as F
import torchvision
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report, accuracy_score

#%% Hyperparameters
num_epochs = 2
batch_size = 50
learning_rate = 0.01
num_classes = 10

.,#%% loading the dataset
transform = transforms.Compose([transforms.ToTensor(), 
                                transforms.Normalize((.5, .5, .5), 
                                                     (.5,.5, .5))])
train_dataset = datasets.CIFAR10(root='./data', train=True, transform=transform, download=True)
test_dataset = datasets.CIFAR10(root='./data', train=False, transform=transform, download=True)
trainloader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
testloader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')  
# %%
imgiter = iter(trainloader)
images, label = next(imgiter)
images[0].shape
label[0]

imgiter = iter(trainloader)
images, labels = next(imgiter)

# Pick the first image and its label
image = images[9]      # shape: [3, H, W]
label = labels[9]

# Undo normalization (assuming mean=0.5, std=0.5 for all channels)
image = image * 0.5 + 0.5  

# Convert from tensor (C, H, W) -> (H, W, C)
npimg = image.numpy().transpose((1, 2, 0))

# Show the image
plt.imshow(npimg)
plt.title(f"Label: {label.item()}")
plt.axis("off")
plt.show()

# %%model
class ConvNet(nn.Module):
    def __init__(self, num_classes=10):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 3) #30x30
        self.pool = nn.MaxPool2d(2, 2) #15x15
        self.conv2 = nn.Conv2d(16, 32, 3) #13x13
        # After conv2 and pool, the image size will be 6x6
        self.fc1 = nn.Linear(32 * 6 * 6, 120) 
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # [batch_size, 16, 16, 16]
        x = self.pool(F.relu(self.conv2(x)))  # [batch_size, 32, 6, 6]
        x = x.view(-1, 32 * 6 * 6)            # flattening
        x = F.relu(self.fc1(x))               # [batch_size, 120]
        x = F.relu(self.fc2(x))               # [batch_size, 84]
        x = self.fc3(x)                       # [batch_size, num_classes]
        return x

# %% creating training loop
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')   
print(device)
model = ConvNet(num_classes).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
total_step = len(trainloader)
print(total_step)

# %%
len(train_dataset)
# %%
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(trainloader):
        images = images.to(device)
        labels = labels.to(device)
        
        #forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        #backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if (i+1) % 100 == 0:
            print (f'epoch {epoch+1}/{num_epochs}, step {i+1}/{total_step}, loss: {loss.item():.4f}')
# %%
with torch.no_grad():
    n_correct = 0
    n_samples = 0
    for images, labels in testloader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        n_samples += labels.size(0)
        n_correct += (predicted == labels).sum().item()
    
    acc = 100.0 * n_correct / n_samples
    print(f'Accuracy of the network on the 10000 test images: {acc} %')
# %%
