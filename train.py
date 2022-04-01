import torch.optim as optim
import matplotlib.pyplot as plt
import torch.nn as nn
from networks import *
import time
import numpy

start_time = time.time()

def network_selector(index, h1_size, h2_size, multiplier):
    """
    Here we select the type of network we want to use
    h1_size and h2_size represent parameter for size of first and second Hidden Layer 
    """
    sel = {
        1: NeuralNetwork1(h1_size, h2_size, multiplier),
        2: NeuralNetwork2(h1_size, h2_size, multiplier),
        3: NeuralNetwork3(h1_size, h2_size, multiplier),
        4: NeuralNetwork4(h1_size, h2_size, multiplier),
        5: NeuralNetwork5(h1_size, h2_size, multiplier),
        6: NeuralNetwork6(h1_size, h2_size, multiplier),
        7: NeuralNetwork7(h1_size, h2_size, multiplier),
        8: NeuralNetwork8(h1_size, h2_size, multiplier),
        9: NeuralNetwork9(h1_size, h2_size, multiplier)
        }
    return sel[index]

def mesages (index):
    msg = {
        1: "Training Neural Network with 1 Hidden Layer and ReLU Activator",
        2: "Training Neural Network with 1 Hidden Layer and Tanh Activator",
        3: "Training Neural Network with 1 Hidden Layer and Sigmoid Activator",
        4: "Training Neural Network with 2 Hidden Layers and ReLU Activator",
        5: "Training Neural Network with 2 Hidden Layers and Tanh Activator",
        6: "Training Neural Network with 2 Hidden Layers and Sigmoid Activator",
        7: "Training Neural Network with 1 Hidden Layer and ReLU Activator",
        8: "Training Neural Network with 1 Hidden Layer and Tanh Activator",
        9: "Training Neural Network with 1 Hidden Layer and Sigmoid Activator",
        }
    return msg[index]


def train(num_episodes, learning_rate, h1_size, h2_size, index, start, end ,step, a ,b, multiplier):
    
    # Exit function if index is out of range
    if (index <1) or (index>9):
        print ("!!! INDEX OUT OF RANGE. Index can be: 1, 2, 3, 4, 5, 6, 7, 8, 9 ")
        return
    
        
    x = torch.arange(start, end + step, step, dtype=torch.float32) #Generate 1-D tensor of size [(end-start)/step]
                                                                   #with values in interval [start, end)
                                                                   
    x = x.unsqueeze(1) # Returns a new tensor with a dimension of size one inserted at the specified position
                       # x = tensor([1,2,3]) -> x = tensor([[1],   
                       #                                    [2],
                       #                                    [3]])
    
    y = a * x**2 + b #Target ecuation
    
    
    """ *** START TRAIN NEURAL NETWORK *** """    
    
    mynn = network_selector(index, h1_size, h2_size, multiplier) #Create "mynn" object
    optimizer = optim.Adam(mynn.parameters(), lr=learning_rate)
    loss_function = nn.MSELoss() #A built in PyTorch function that measures the mean squared error
                                 #between each element in the input x and target y
                                 #https://pytorch.org/docs/stable/generated/torch.nn.MSELoss.html
    
    loss_fnc = []
    
    for i_ep in range(num_episodes):
        predicted_value = mynn(x) #The outputs of Neural Network
        loss = loss_function(predicted_value, y) #Compute Loss Function. First parameter is the output of
                                                 #Neural Network and the second parameter "y" is the target (how our Neural Network should look like) 
        
        loss_fnc.append(loss.detach())
        
        optimizer.zero_grad() #Set gradients to zero
        loss.backward()  #minimize the cost function by adjusting network’s weights and biases
        optimizer.step()
    
    """ *** END TRAIN NEURAL NETWORK *** """
    
    ########################################################################################
    
    """         ######    PLOTTING    ######        """
    """  ##############   START   ################  """
    
    if index == 1 or index == 2 or index == 3:
        plt.figure(figsize=(15,15))
        plt.subplot(2,2,1)
        plt.plot(x, mynn.in_h1)
        plt.title("Neurons from Input to first Hidden Layer")
        
        plt.subplot(2,2,2)
        plt.plot(x, mynn.h1_act)
        plt.title("Neurons with Ativator applied")
        
        plt.subplot(2,2,3)
        plt.plot(x, mynn.act_out)
        plt.title("Output of Neural Network")
        
        plt.subplot(2,2,4)
        plt.plot(x, y, label = "Target")
        plt.plot(x, mynn(x).detach(),label = "Predicted")
        plt.legend(loc="upper center")
        plt.title("Target and Predicted function")
        plt.xlabel("Training parameter (x)")
        plt.ylabel("Target parameters (y) ")
    
    elif index == 4 or index == 5 or index == 6:
        plt.figure(figsize=(15,15))
        plt.subplot(2,2,1)
        plt.plot(x, mynn.in_h1)
        plt.title("Neurons from Input to first Hidden Layer")
        
        plt.subplot(2,2,2)
        plt.plot(x, mynn.h1_act1)
        plt.title("Neurons with Activator applied")
        
        plt.subplot(2,2,3)
        plt.plot(x, mynn.act1_h2)
        plt.title("First Hidden Layer to Second Hidden Layer")
        
        plt.subplot(2,2,4)
        plt.plot(x, mynn.h2_act2)
        plt.title("Neurons with Activator applied")
        
        
        plt.figure(figsize=(10,5))
        plt.subplot(1,2,1)
        plt.plot(x, mynn(x).detach())
        plt.title("output of Neural Network")
        
        plt.subplot(1,2,2)
        plt.plot(x, y, label = "Target")
        plt.plot(x, mynn(x).detach(),label = "Predicted")
        plt.legend(loc="upper center")
        plt.title("Target and Predicted function")
        plt.xlabel("Training parameter (x)")
        plt.ylabel("Target parameters (y) ")
        
    elif index == 7 or index == 8 or index == 9:
        
        plt.figure(figsize=(15,15))
        plt.subplot(2,2,1)
        plt.plot(x, mynn.in_h1)
        plt.title("Neurons from Input to first Hidden Layer")
        
        plt.subplot(2,2,2)
        plt.plot(x, mynn.h1_act)
        plt.title("Neurons with Ativator applied")
        
        plt.subplot(2,2,3)
        plt.plot(x,mynn.act_out)
        plt.title("Output of Neural Network")
        
        plt.subplot(2,2,4)
        plt.plot(x, mynn(x).detach())
        plt.title("Output with activator applied")
        
        plt.figure()
        plt.plot(x, y, label = "Target")
        plt.plot(x, mynn(x).detach(),label = "Predicted")
        plt.legend(loc="upper center")
        plt.title("Target and Predicted function")
        plt.xlabel("Training parameter (x)")
        plt.ylabel("Target parameters (y) ")
    
        
    plt.figure()
    plt.plot(loss_fnc)
    plt.title("Loss function")
    plt.xlim([0,150])
    
    """         ######    PLOTTING    ######        """
    """  #############   END    ##################  """
    
    ####################################################################
    
    """     ########   MESSAGES    ########     """
    """ ################ START ################ """
    
    print ("\n")
    print (mesages(index))
    
    if ((index >= 1) and (index <= 3)) or  ((index >= 7) and (index <= 9)):
        print ("The size of Hidden Layer is: ", h1_size)
    
    elif (index >= 4) and (index <= 6):
        print ("The size of First Hidden Layer is: ", h1_size)
        print ("The size of Second Hidden Layer is: ", h2_size) 
        
    
    print ("\nLast value of Loss Function: ",numpy.round_((loss.detach().numpy()),2))
    
    elapsed_time = time.time() - start_time #Time cronometer
    print ("Total training time: ", round(elapsed_time,2), "[s]")
    
    """     ########   MESSAGES    ########   """
    """ ################ END ################ """