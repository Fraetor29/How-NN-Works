from train import train

"""
We will train some Neural Networks to learn an order II ecuation (y = ax^2 +b), 
in order to see how Hidden Layers and Activators Works.

In "networks.py" are defined few types of Neural Network.

In "train.py" is defined "train()" method, that will train our choosen Neural Network.

Our input parameters will be a tensor "x". To define it we will use "torch.arange" in train() method,
with parameters "start" -> where 'x' starts; "end" -> where 'x' end; "step" -> how many numbers are between "start" and "end"

Our target function will be "y", an order II ecuation (y = ax^2 +b) 

"num_episodes" -> number of training episodes
"learning_rate" -> Learning rate is a hyper-parameter that controls how much we are adjusting 
                   the weights of our network with respect the loss gradient  
"h1_size" -> size of First Hidden Layer
"h2_size" -> size of Second Hidden Layer (Just NeuralNetwork4, NeuralNetwork5, NeuralNetwork6 have two Hidden Layers)

"index" -> parameter that select types of Neural Networks we want to use. It takes values from 1 to 9
1 -> Neural Network with 1 Hidden Layer and ReLU Activator
2 -> Neural Network with 1 Hidden Layer and Tanh Activator
3 -> Neural Network with 1 Hidden Layer and Sigmoid Activator
4 -> Neural Network with 2 Hidden Layers and ReLU Activator
5 -> Neural Network with 2 Hidden Layers and Tanh Activator
6 -> Neural Network with 2 Hidden Layers and Sigmoid Activator
7 -> Neural Network with 1 Hidden Layer and ReLU Activator **AND SAME ACTIVATOR APPLIED AT OUTPUT
8 -> Neural Network with 1 Hidden Layer and Tanh Activator **AND SAME ACTIVATOR APPLIED AT OUTPUT
9 -> Neural Network with 1 Hidden Layer and Sigmoid Activator **AND SAME ACTIVATOR APPLIED AT OUTPUT

!!! NeuralNetwork7, NeuralNetwork8, NeuralNetwork9 ARE PARTICULAR NETWORKS BECAUSE HAVE ACTIVATOR AT OUTPUT

For NeuralNetwork7 we have ReLU Activator thus the output can be positive or zero
For NeuralNetwork8 we have Tanh Activator thus the output can be in interval [-1, 1]
For NeuralNetwork9 we have Sigmoid Activator thus the output can be in interval [0, 1]

"multiplier" -> parameter applied at output of those three Networks (NeuralNetwork7, NeuralNetwork8, NeuralNetwork9) to increase the output

For NeuralNetwork8, NeuralNetwork9 "multiplier" can be set:
    multiplier = a*start^2 + b
    The max value of 'y' for 'x' in the interval [start, end] is: [y = a*start^2 + b] or [y = a*end^2 + b]
    The output of Tanh or Sigmoid is [-1, 1] or [0, 1], thus if we multiply the output of this two activators with multiplier, we get the max value of 'y'


The user can play with those parameters below and see on plots how Neural Networks work inside
"""

""" ***** PARAMETERS FOR USER ***** """

# Generate 1-D vector of size [(end-start)/step] with values in interval [start, end)
# Generate "x" -> input for Neural Network,  x = [start, end)
start = -5
end = 5
step = 0.1 

# Parameters for second order ecuation y = a*x^2 +b
a = 0.5
b = 3 

learning_rate = 0.1 # Learning rate is a hyper-parameter that controls how much 
                    # we are adjusting the weights of our network with respect the loss gradient
num_episodes = 1000 # Number of training episodes
h1_size = 2 # Size of First Hidden Layer
h2_size = 2 # Size of Second Hidden Layer

index = 1 # Explained in description above
multiplier = 1 # Explained in description above

# Call training function
train(num_episodes, learning_rate, h1_size, h2_size, index, start, end ,step, a ,b, multiplier) 
