import torch
import torch.optim as optim
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F 



class NeuralNetwork1(nn.Module):
    """
    Define a Neural Network with 1 Hidden Layer of size 1 and ReLU activator
    
    "h1, h2" represent parameters for size of first and second Hidden Layer
    In this case h2 is never used
   
    "m" is a multiplier applied at output where we have Sigmoid or Tanh Activators or Negative Output
    In this case is never used
    """
    
    def __init__(self, h1, h2, m):
        super(NeuralNetwork1, self).__init__()
        
        #PARAMETERS THAT WILL CALL IN TRAINING FUNCTION FOR PLOT
        self.size_of_hidden_layer1 = h1 #The size of first Hidden Layer
        self.size_of_hidden_layer2 = h2 #The size of first Hidden Layer
        self.multiplier = m #output multiplier
        
        self.in_h1 = 0 # Values from input to First Hidden Layer
        self.h1_act = 0 # Values with Activator applied
        self.act_out = 0 # Ouptut of Neural Network
        
        self.linear1 = nn.Linear(1, self.size_of_hidden_layer1)
        self.linear2 = nn.Linear(self.size_of_hidden_layer1, 1)
    
    def forward(self, x):
        output = self.linear1(x)
        self.in_h1 = output.detach()
        
        output = F.relu(output)
        self.h1_act = output.detach()
        
        output = self.linear2(output)
        self.act_out = output.detach()
        
        return output


class NeuralNetwork2(nn.Module):
    """
    Define a Neural Network with 1 Hidden Layer of size 1 and Tanh activator
   
    h1, h2 represent parameters for size of first and second Hidden Layer
    In this case h2 is never used
   
    "m" is a multiplier applied at output where we have Sigmoid or Tanh Activators or Negative Output
    In this case is never used
    """
    
    def __init__(self, h1, h2, m):
        super(NeuralNetwork2, self).__init__()
        
        #PARAMETERS THAT WILL CALL IN TRAINING FUNCTION FOR PLOT
        self.size_of_hidden_layer1 = h1 #The size of first Hidden Layer
        self.size_of_hidden_layer2 = h2 #The size of first Hidden Layer
        self.multiplier = m #output multiplier
        
        self.in_h1 = 0 # Values from input to First Hidden Layer
        self.h1_act = 0 # Values with Activator applied
        self.act_out = 0 # Ouptut of Neural Network
        
        self.linear1 = nn.Linear(1, self.size_of_hidden_layer1)
        self.linear2 = nn.Linear(self.size_of_hidden_layer1, 1)
    
    def forward(self, x):
        output = self.linear1(x)
        self.in_h1 = output.detach()
        
        output = torch.tanh_(output)
        self.h1_act = output.detach()
        
        output = self.linear2(output)
        self.act_out = output.detach()
        
        return output
    

class NeuralNetwork3(nn.Module):
    """
    Define a Neural Network with 1 Hidden Layer of size 1 and Sigmoid activator
    
    h1, h2 represent parameters for size of first and second Hidden Layer
    In this case h2 is never used
   
    "m" is a multiplier applied at output where we have Sigmoid or Tanh Activators or Negative Output
    In this case is never used
    """
    
    def __init__(self, h1, h2, m):
        super(NeuralNetwork3, self).__init__()
        
        #PARAMETERS THAT WILL CALL IN TRAINING FUNCTION FOR PLOT
        self.size_of_hidden_layer1 = h1 #The size of first Hidden Layer
        self.size_of_hidden_layer2 = h2 #The size of first Hidden Layer
        self.multiplier = m #output multiplier
        
        self.in_h1 = 0 # Values from input to First Hidden Layer
        self.h1_act = 0 # Values with Activator applied
        self.act_out = 0 # Ouptut of Neural Network
        
        self.linear1 = nn.Linear(1, self.size_of_hidden_layer1)
        self.linear2 = nn.Linear(self.size_of_hidden_layer1, 1)
    
    def forward(self, x):
        output = self.linear1(x)
        self.in_h1 = output.detach()
        
        output = torch.sigmoid(output)
        self.h1_act = output.detach()
        
        output = self.linear2(output)
        self.act_out = output.detach()
        
        return output
    
class NeuralNetwork4(nn.Module):
    """
    Define a Neural Network with 1 Hidden Layer of size 1 and ReLU activator
    
    h1, h2 represent parameters for size of first and second Hidden Layer
    
    "m" is a multiplier applied at output where we have Sigmoid or Tanh Activators or Negative Output
    In this case is never used
    """
    
    def __init__(self, h1, h2, m):
        super(NeuralNetwork4, self).__init__()
        
        #PARAMETERS THAT WILL CALL IN TRAINING FUNCTION FOR PLOT
        self.size_of_hidden_layer1 = h1 #The size of first Hidden Layer
        self.size_of_hidden_layer2 = h2 #The size of second Hidden Layer
        self.multiplier = m #output multiplier
        
        self.in_h1 = 0 # Values from input to First Hidden Layer
        self.h1_act1 = 0 # Values with Activator applied
        self.act1_h2 = 0 # Values from First Hidden Layer to Second Hidden Layer
        self.h2_act2 = 0 # Values with Activator applied
        self.act2_out = 0 # Ouptut of Neural Network
        
        self.linear1 = nn.Linear(1, self.size_of_hidden_layer1)
        self.linear2 = nn.Linear(self.size_of_hidden_layer1, self.size_of_hidden_layer2)
        self.linear3 = nn.Linear(self.size_of_hidden_layer2, 1)
    
    def forward(self, x):
        output = self.linear1(x)
        self.in_h1 = output.detach()
        
        output = F.relu(output)
        self.h1_act1 = output.detach()
        
        output = self.linear2(output)
        self.act1_h2 = output.detach()
        
        output = F.relu(output)
        self.h2_act2 = output.detach()
        
        output = self.linear3(output)
        self.act2_out = output.detach()
        
        return output


class NeuralNetwork5(nn.Module):
    """
    Define a Neural Network with 1 Hidden Layer of size 1 and Tanh activator
    
    h1, h2 represent parameters for size of first and second Hidden Layer
    
    "m" is a multiplier applied at output where we have Sigmoid or Tanh Activators or Negative Output
    In this case is never used
    """
    
    def __init__(self, h1, h2, m):
        super(NeuralNetwork5, self).__init__()
        
        #PARAMETERS THAT WILL CALL IN TRAINING FUNCTION FOR PLOT
        self.size_of_hidden_layer1 = h1 #The size of first Hidden Layer
        self.size_of_hidden_layer2 = h2 #The size of second Hidden Layer
        self.multiplier = m #output multiplier
        
        self.in_h1 = 0 # Values from input to First Hidden Layer
        self.h1_act1 = 0 # Values with Activator applied
        self.act1_h2 = 0 # Values from First Hidden Layer to Second Hidden Layer
        self.h2_act2 = 0 # Values with Activator applied
        self.act2_out = 0 # Ouptut of Neural Network
        
        self.linear1 = nn.Linear(1, self.size_of_hidden_layer1)
        self.linear2 = nn.Linear(self.size_of_hidden_layer1, self.size_of_hidden_layer2)
        self.linear3 = nn.Linear(self.size_of_hidden_layer2, 1)
    
    def forward(self, x):
        output = self.linear1(x)
        self.in_h1 = output.detach()
        
        output = torch.tanh_(output)
        self.h1_act1 = output.detach()
        
        output = self.linear2(output)
        self.act1_h2 = output.detach()
        
        output = torch.tanh_(output)
        self.h2_act2 = output.detach()
        
        output = self.linear3(output)
        self.act2_out = output.detach()
        
        return output

class NeuralNetwork6(nn.Module):
    """
    Define a Neural Network with 1 Hidden Layer of size 1 and Sigmoid activator
    
    h1, h2 represent parameters for size of first and second Hidden Layer
   
    "m" is a multiplier applied at output where we have Sigmoid or Tanh Activators or Negative Output
    In this case is never used
    """
    
    def __init__(self, h1, h2, m):
        super(NeuralNetwork6, self).__init__()
        
        #PARAMETERS THAT WILL CALL IN TRAINING FUNCTION FOR PLOT
        self.size_of_hidden_layer1 = h1 #The size of first Hidden Layer
        self.size_of_hidden_layer2 = h2 #The size of second Hidden Layer
        self.multiplier = m #output multiplier
        
        self.in_h1 = 0 # Values from input to First Hidden Layer
        self.h1_act1 = 0 # Values with Activator applied
        self.act1_h2 = 0 # Values from First Hidden Layer to Second Hidden Layer
        self.h2_act2 = 0 # Values with Activator applied
        self.act2_out = 0 # Ouptut of Neural Network
        
        self.linear1 = nn.Linear(1, self.size_of_hidden_layer1)
        self.linear2 = nn.Linear(self.size_of_hidden_layer1, self.size_of_hidden_layer2)
        self.linear3 = nn.Linear(self.size_of_hidden_layer2, 1)
    
    def forward(self, x):
        output = self.linear1(x)
        self.in_h1 = output.detach()
        
        output = torch.sigmoid(output)
        self.h1_act1 = output.detach()
        
        output = self.linear2(output)
        self.act1_h2 = output.detach()
        
        output = torch.sigmoid(output)
        self.h2_act2 = output.detach()
        
        output = self.linear3(output)
        self.act2_out = output.detach()
        
        return output

class NeuralNetwork7(nn.Module):
    """
    Define a Neural Network with 1 Hidden Layer of size 1 and ReLU activator
    
    h1, h2 represent parameters for size of first and second Hidden Layer
    In this case h2 is never used
   
    "m" is a multiplier applied at output where we have Sigmoid or Tanh Activators or Negative Output
    """
    
    def __init__(self, h1 , h2, m):
        super(NeuralNetwork7, self).__init__()
        
        #PARAMETERS THAT WILL CALL IN TRAINING FUNCTION FOR PLOT
        self.size_of_hidden_layer1 = h1 #The size of first Hidden Layer
        self.size_of_hidden_layer2 = h2 #The size of first Hidden Layer
        self.multiplier = m #output multiplier
        
        self.in_h1 = 0 # Values from input to First Hidden Layer
        self.h1_act = 0 # Values with Activator applied
        self.act_out = 0 # Ouptut of Neural Network
        
        self.linear1 = nn.Linear(1, self.size_of_hidden_layer1)
        self.linear2 = nn.Linear(self.size_of_hidden_layer1, 1)
    
    def forward(self, x):
        output = self.linear1(x)
        self.in_h1 = output.detach()
        
        output = F.relu(output)
        self.h1_act = output.detach()
        
        output = self.linear2(output)
        self.act_out = output.detach()
        
        output = F.relu(output)
        return output * self.multiplier


class NeuralNetwork8(nn.Module):
    """
    Define a Neural Network with 1 Hidden Layer of size 1 and Tanh activator
    
    h1, h2 represent parameters for size of first and second Hidden Layer
    In this case h2 is never used
   
    "m" is a multiplier applied at output where we have Sigmoid or Tanh Activators or Negative Outpu
    """
    
    def __init__(self, h1, h2, m):
        super(NeuralNetwork8, self).__init__()
        
        #PARAMETERS THAT WILL CALL IN TRAINING FUNCTION FOR PLOT
        self.size_of_hidden_layer1 = h1 #The size of first Hidden Layer
        self.size_of_hidden_layer2 = h2 #The size of first Hidden Layer
        self.multiplier = m #output multiplier
        
        self.in_h1 = 0 # Values from input to First Hidden Layer
        self.h1_act = 0 # Values with Activator applied
        self.act_out = 0 # Ouptut of Neural Network
        
        self.linear1 = nn.Linear(1, self.size_of_hidden_layer1)
        self.linear2 = nn.Linear(self.size_of_hidden_layer1, 1)
    
    def forward(self, x):
        output = self.linear1(x)
        self.in_h1 = output.detach()
        
        output = torch.tanh_(output)
        self.h1_act = output.detach()
        
        output = self.linear2(output)
        self.act_out = output.detach()
        
        output = torch.tanh_(output)
        return output * self.multiplier
    

class NeuralNetwork9(nn.Module):
    """
    Define a Neural Network with 1 Hidden Layer of size 1 and Sigmoid activator
    
    h1, h2 represent parameters for size of first and second Hidden Layer
    In this case h2 is never used
    
    "m" is a multiplier applied at output where we have Sigmoid or Tanh Activators or Negative Outpu
    """
    
    def __init__(self, h1, h2, m):
        super(NeuralNetwork9, self).__init__()
        
        #PARAMETERS THAT WILL CALL IN TRAINING FUNCTION FOR PLOT
        self.size_of_hidden_layer1 = h1 #The size of first Hidden Layer
        self.size_of_hidden_layer2 = h2 #The size of first Hidden Layer
        self.multiplier = m #output multiplier
        
        self.in_h1 = 0 # Values from input to First Hidden Layer
        self.h1_act = 0 # Values with Activator applied
        self.act_out = 0 # Ouptut of Neural Network
        
        self.linear1 = nn.Linear(1, self.size_of_hidden_layer1)
        self.linear2 = nn.Linear(self.size_of_hidden_layer1, 1)
    
    def forward(self, x):
        output = self.linear1(x)
        self.in_h1 = output.detach()
        
        output = torch.sigmoid(output)
        self.h1_act = output.detach()
        
        output = self.linear2(output)
        self.act_out = output.detach()
        
        output = torch.sigmoid(output)
        return output * self.multiplier 


    
    
    
    