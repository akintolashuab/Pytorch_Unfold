# %% Import of packages
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn import datasets
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay

# %% Dataset generation and data preprocessing
bc = datasets.load_breast_cancer()
X, y = bc.data, bc.target   
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
sc = StandardScaler()
X_train = sc.fit_transform(X_train) 
X_test = sc.transform(X_test)
X_train = torch.tensor(X_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
y_test = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)  
X_train.shape, X_test.shape, y_train.shape, y_test.shape

# %% Design Model (input layer, hidden layer, output layer and forward pass)
class LogisticRegressionModel(nn.Module):
    def __init__(self, input_dim, hidden1, hidden2, output_dim):
        super(LogisticRegressionModel, self).__init__()
        self.lin1 = nn.Linear(input_dim, hidden1)
        self.lin2 = nn.Linear(hidden1, hidden2)
        self.lin3 = nn.Linear(hidden2, output_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.lin1(x)
        x = F.relu(x)
        x = self.lin2(x)
        x = F.relu(x)
        x = self.lin3(x)
        return x
      
 
# %% creating model, define loss function and optimizer
model = LogisticRegressionModel(input_dim=30, hidden1=16, hidden2=8, output_dim=1)
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

# %% Training the model
num_epochs = 1000
for epoch in range(num_epochs):
    #set the gradients to zero
    optimizer.zero_grad()

    #forward pass
    y_pred = model(X_train)
    
    loss = criterion(y_pred, y_train)
    loss.backward()

    #update gradient
    optimizer.step()
    
    if (epoch+1) % 100 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# %% Testing the model
with torch.no_grad():
    y_pred = torch.sigmoid(model(X_test))  # apply sigmoid here
    y_pred_cls = y_pred.round()
    acc = accuracy_score(y_test, y_pred_cls)
    print(f'Accuracy: {acc:.4f}')
    
    cm = confusion_matrix(y_test, y_pred_cls)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0, 1])
    disp.plot(cmap=plt.cm.Blues)
    plt.show()

# %% Alternative heatmap visualization
sns.heatmap(confusion_matrix(y_test, y_pred_cls), annot=True, fmt='d', cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.show()
# %%
result = pd.DataFrame({'True': y_test.flatten(), 'Predicted': y_pred_cls.flatten()})
print(result)
# %%
result.to_csv('logistic_regression_results.csv', index=False)
# %%
