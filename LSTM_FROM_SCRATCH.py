# %% import of libraries
import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning.pytorch as pl
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset, TensorDataset
import torchvision
from lightning.pytorch import Trainer

# %%
class LSTMFromScratch(pl.LightningModule):
    def __init__(self): # __init__() is the class constructor
        super(LSTMFromScratch, self).__init__()
        pl.seed_everything(seed=42)
        
        ###################
        ##
        ## Initialize the tensors for the LSTM
        ##
        ###################
        
        ## NOTE: nn.LSTM() uses random values from a uniform distribution to initialize the tensors
        ## Here we can do it 2 different ways 1) Normal Distribution and 2) Uniform Distribution
        ## We'll start with the Normal Distribtion...
        mean = torch.tensor(0.0)
        std = torch.tensor(1.0)   

        self.wlr1 = nn.Parameter(torch.normal(mean=mean, std=std), requires_grad=True)
        self.wlr2 = nn.Parameter(torch.normal(mean=mean, std=std), requires_grad=True)
        self.blr1 = nn.Parameter(torch.tensor(0.), requires_grad=True)

        ## These are the Weights and Biases in the second stage, which determins the new
        ## potential long-term memory and what percentage will be remembered.
        self.wpr1 = nn.Parameter(torch.normal(mean=mean, std=std), requires_grad=True)
        self.wpr2 = nn.Parameter(torch.normal(mean=mean, std=std), requires_grad=True)
        self.bpr1 = nn.Parameter(torch.tensor(0.), requires_grad=True)

        self.wp1 = nn.Parameter(torch.normal(mean=mean, std=std), requires_grad=True)
        self.wp2 = nn.Parameter(torch.normal(mean=mean, std=std), requires_grad=True)
        self.bp1 = nn.Parameter(torch.tensor(0.), requires_grad=True)
        
        ## These are the Weights and Biases in the third stage, which determines the
        ## new short-term memory and what percentage will be sent to the output.
        self.wo1 = nn.Parameter(torch.normal(mean=mean, std=std), requires_grad=True)
        self.wo2 = nn.Parameter(torch.normal(mean=mean, std=std), requires_grad=True)
        self.bo1 = nn.Parameter(torch.tensor(0.), requires_grad=True)


    def lstm_unit(self, input_value, long_memory, short_memory):
        ## lstm_unit does the math for a single LSTM unit.
        
        ## NOTES:
        ## long term memory is also called "cell state"
        ## short term memory is also called "hidden state"
        
        ## 1) The first stage determines what percent of the current long-term memory
        ##    should be remembered
        long_remember_percent = torch.sigmoid((short_memory * self.wlr1) + 
                                              (input_value * self.wlr2) + 
                                              self.blr1)
        
        ## 2) The second stage creates a new, potential long-term memory and determines what
        ##    percentage of that to add to the current long-term memory
        potential_remember_percent = torch.sigmoid((short_memory * self.wpr1) + 
                                                   (input_value * self.wpr2) + 
                                                   self.bpr1)
        potential_memory = torch.tanh((short_memory * self.wp1) + 
                                      (input_value * self.wp2) + 
                                      self.bp1)
        
        ## Once we have gone through the first two stages, we can update the long-term memory
        updated_long_memory = ((long_memory * long_remember_percent) + 
                       (potential_remember_percent * potential_memory))
        
        ## 3) The third stage creates a new, potential short-term memory and determines what
        ##    percentage of that should be remembered and used as output.
        output_percent = torch.sigmoid((short_memory * self.wo1) + 
                                       (input_value * self.wo2) + 
                                       self.bo1)         
        updated_short_memory = torch.tanh(updated_long_memory) * output_percent
        
        ## Finally, we return the updated long and short-term memories
        return([updated_long_memory, updated_short_memory])
        
    
    def forward(self, input): 
        ## forward() unrolls the LSTM for the training data by calling lstm_unit() for each day of training data 
        ## that we have. forward() also keeps track of the long and short-term memories after each day and returns
        ## the final short-term memory, which is the 'output' of the LSTM.
        
        long_memory = 0 # long term memory is also called "cell state" and indexed with c0, c1, ..., cN
        short_memory = 0 # short term memory is also called "hidden state" and indexed with h0, h1, ..., cN
        day1 = input[0]
        day2 = input[1]
        day3 = input[2]
        day4 = input[3]
        
        ## Day 1
        long_memory, short_memory = self.lstm_unit(day1, long_memory, short_memory)
        
        ## Day 2
        long_memory, short_memory = self.lstm_unit(day2, long_memory, short_memory)
        
        ## Day 3
        long_memory, short_memory = self.lstm_unit(day3, long_memory, short_memory)
        
        ## Day 4
        long_memory, short_memory = self.lstm_unit(day4, long_memory, short_memory)
        
        ##### Now return short_memory, which is the 'output' of the LSTM.
        return short_memory
        
        
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters())


    def training_step(self, batch, batch_idx): # take a step during gradient descent.
        input_i, label_i = batch # collect input
        output_i = self.forward(input_i[0]) # run input through the neural network
        loss = (output_i - label_i)**2 ## loss = squared residual
        
    ##
        ## Logging the loss and the predicted values so we can evaluate the training
        ##
        ###################
        self.log("train_loss", loss)
        ## NOTE: Our dataset consists of two sequences of values representing Company A and Company B
        ## For Company A, the goal is to predict that the value on Day 5 = 0, and for Company B,
        ## the goal is to predict that the value on Day 5 = 1. We use label_i, the value we want to
        ## predict, to keep track of which company we just made a prediction for and 
        ## log that output value in a company specific file
        if (label_i == 0):
            self.log("out_0", output_i)
        else:
            self.log("out_1", output_i)
            
        return loss
    
# %%
model = LSTMFromScratch() # First, make model from the class

print("Before optimization, the parameters are...")
for name, param in model.named_parameters():
    print(name, ":", round(param.item(), 4))

# %%
print("\nNow let's compare the observed and predicted values without training the model...")
## NOTE: To make predictions, we pass in the first 4 days worth of stock values 
## in an array for each company. In this case, the only difference between the
## input values for Company A and B occurs on the first day. Company A has 0 and
## Company B has 1.
print("Company A: Observed = 0, Predicted =", 
      model(torch.tensor([0., 0.5, 0.25, 1.])).detach())
print("Company B: Observed = 1, Predicted =", 
      model(torch.tensor([1., 0.5, 0.25, 1.])).detach())

# %%
inputs = torch.tensor([[0., 0.5, 0.25, 1.], [1., 0.5, 0.25, 1.]])
labels = torch.tensor([0., 1.])
dataset = TensorDataset(inputs, labels) 
dataloader = DataLoader(dataset)
trainer = pl.Trainer(max_epochs=2000) # with default learning rate, 0.001 (this tiny learning rate makes learning slow)
trainer.fit(model, train_dataloaders=dataloader)


print("\nNow let's compare the observed and predicted values...")
print("Company A: Observed = 0, Predicted =", model(torch.tensor([0., 0.5, 0.25, 1.])).detach())
print("Company B: Observed = 1, Predicted =", model(torch.tensor([1., 0.5, 0.25, 1.])).detach())

# %% import of libraries
path_to_checkpoint = trainer.checkpoint_callback.best_model_path ## By default, "best" = "most recent"
print("The new trainer will start where the last left off, "
"and the check point data is here: " + 
      path_to_checkpoint + "\n")

## Then create a new Lightning Trainer
trainer = pl.Trainer(max_epochs=3000) # Before, max_epochs=2000, so, by setting it to 3000, we're adding 1000 more.
## And then call fit() using the path to the most recent checkpoint files
## so that we can pick up where we left off.
trainer.fit(model, train_dataloaders=dataloader, ckpt_path=path_to_checkpoint)
print("\nNow let's compare the observed and predicted values...")
print("Company A: Observed = 0, Predicted =", model(torch.tensor([0., 0.5, 0.25, 1.])).detach())
print("Company B: Observed = 1, Predicted =", model(torch.tensor([1., 0.5, 0.25, 1.])).detach())

# %% import of libraries
path_to_checkpoint = trainer.checkpoint_callback.best_model_path ## By default, "best" = "most recent"
print("The new trainer will start where the last left off, and the check point data is here: " + 
      path_to_checkpoint + "\n")

## Then create a new Lightning Trainer
trainer = pl.Trainer(max_epochs=5000) # Before, max_epochs=3000, so, by setting it to 5000, we're adding 2000 more.
## And then call fit() using the path to the most recent checkpoint files
## so that we can pick up where we left off.
trainer.fit(model, train_dataloaders=dataloader, ckpt_path=path_to_checkpoint)
# %%
print("\nNow let's compare the observed and predicted values...")
print("Company A: Observed = 0, Predicted =", model(torch.tensor([0., 0.5, 0.25, 1.])).detach())
print("Company B: Observed = 1, Predicted =", model(torch.tensor([1., 0.5, 0.25, 1.])).detach())

print("After optimization, the parameters are...")
for name, param in model.named_parameters():
    print(name, ":", param.data)

# %%
## %% LSTM FROM NN.LSTM() IMPLEMENTATION USING LIGHTNING
## Instead of coding an LSTM by hand, let's see what we can do with PyTorch's nn.LSTM()
class LightningLSTM(pl.LightningModule):

    def __init__(self): # __init__() is the class constructor function, and we use it to initialize the Weights and Biases.
        
        super().__init__() # initialize an instance of the parent class, LightningModule.

        pl.seed_everything(seed=42)
        self.train_losses = []
        
        ## input_size = number of features (or variables) in the data. In our example
        ##              we only have a single feature (value)
        ## hidden_size = this determines the dimension of the output
        ##               in other words, if we set hidden_size=1, then we have 1 output node
        ##               if we set hiddeen_size=50, then we hve 50 output nodes (that can then be 50 input
        ##               nodes to a subsequent fully connected neural network.
        self.lstm = nn.LSTM(input_size=1, hidden_size=1) 
         

    def forward(self, input):
        ## transpose the input vector
        input_trans = input.view(len(input), 1)
        
        lstm_out, temp = self.lstm(input_trans)
        
        ## lstm_out has the short-term memories for all inputs. We make our prediction with the last one
        prediction = lstm_out[-1] 
        return prediction
        
        
    def configure_optimizers(self): # this configures the optimizer we want to use for backpropagation.
        return Adam(self.parameters(), lr=0.1) ## we'll just go ahead and set the learning rate to 0.1

    
    def training_step(self, batch, batch_idx): # take a step during gradient descent.
        input_i, label_i = batch # collect input
        output_i = self.forward(input_i[0]) # run input through the neural network
        loss = (output_i - label_i)**2 ## loss = squared residual
        

        # Convert tensor loss to scalar number
        loss_value = loss.item()

        # Save loss for plotting
        self.train_losses.append(loss_value)

        ###################
        ##
        ## Logging the loss and the predicted values so we can evaluate the training
        ##
        ###################
        self.log("train_loss", loss)
        
        if (label_i == 0):
            self.log("out_0", output_i)
        else:
            self.log("out_1", output_i)

        return loss
    
model = LightningLSTM() # First, make model from the class

## print out the name and value for each parameter
print("Before optimization, the parameters are...")
for name, param in model.named_parameters():
    print(name, param.data)
    
print("\nNow let's compare the observed and predicted values...")
print("Company A: Observed = 0, Predicted =", model(torch.tensor([0., 0.5, 0.25, 1.])).detach())
print("Company B: Observed = 1, Predicted =", model(torch.tensor([1., 0.5, 0.25, 1.])).detach())

# %%
trainer = pl.Trainer(max_epochs=300, log_every_n_steps=2)

trainer.fit(model, train_dataloaders=dataloader)

print("After optimization, the parameters are...")
for name, param in model.named_parameters():
    print(name, param.data)

print("\nNow let's compare the observed and predicted values...")
print("Company A: Observed = 0, Predicted =", model(torch.tensor([0., 0.5, 0.25, 1.])).detach())
print("Company B: Observed = 1, Predicted =", model(torch.tensor([1., 0.5, 0.25, 1.])).detach())
# %%
plt.figure(figsize=(8,5))

plt.plot(model.train_losses)

plt.xlabel("Training Step")
plt.ylabel("Loss")
plt.title("Training Loss Curve")

plt.grid(True)

plt.show()