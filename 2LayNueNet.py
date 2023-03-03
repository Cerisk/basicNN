import numpy as np





# Sigmoid function (used to map any value to a value between 0 and 1)
# This is used to convert numbers to probabilities in order to train neural networks
# This is used as a basic kind of function for our "nonlinearity"
def sigFunc (x, deriv = False):
    # Can be used to generate derivative (the rate of change of a 
    # function with respect to a variable). Incredibly efficient!
    if (deriv == True):
        return x * (1 - x)
    
    return 1 / (1 + np.exp(-x))




# Dataset used for input (can be changed to file of input)
# Using a file (csv) makes it easier to manage, must modify code
# in order to accomplish this. This example has 3 input nodes to the
# neural network and 4 training examples.
inpData = np.array ([    [0, 0, 1],
                         [0, 1, 1],
                         [1, 0, 1],
                         [1, 1, 1]   ])




# Dataset used for output (can be changed to file of output)
# This dataset is generated horizontally (1 row and 4 columns for space).
# "T" is the transpose function (move from one side of an equation to another)
outData = np.array ( [[0, 0, 1, 1]] ).T
# After transposing the Y-Matrix has 4 rows and one column (like the input)
# So now our network has 3 inputs and 1 output.



# Used to seed random numbers to make calculations (randomly distributing
# the numbers), but will be randomly distributed in the exact same way
# each time you train it. Making it easier to see how changes affect it.
np.random.seed( 1 )





# Inititiates the weight matrix for the neural network (synapse zero).
# 2 is used since we have only 2 layers (input and out).
# We only need one matrix of weight to connect them. The dimension (3, 1)
# is used because we have 3 inputs and 1 output.
# Example to look at it is l0 size of 3 and l1 size of 1 and we want to
# connect every node of l0 to every node of l1, requireing a matrix
# dimensionality of (3, 1). It's best practice to initiate with a mean
# weight of 0 in weight initialization.
syn0 = 2 * np.random.random( (3, 1) ) - 1
# Note, this is the "neural network" as this matrix as the layers
# are used to form the syn0 matrix.





# Loop used to start training the neural network that iterates
# through the training code multiple times to optimize the network
# to the dataset
for itr in range (10000):


    # Explicitly describes our first layer as our data
    l0 = inpData

    # This is the prediction step, used to let the network "try" to
    # predict the output given an input. This can be studied to see how
    # it performs and djust it to do better after each iteration
    # First step has 1st matrix multiples l0 by syn0 with the 2nd passing
    # our ouput through the sigmoid function.
    l1 = sigFunc(np.dot(l0, syn0))


    # l1 had a "guess" for each input. This is used to compare how well
    # it did by subtracting the true answer (outData) to the guess (l1).
    # All l1_error is just a vecctor of + and - #s that reflect how much
    # the network missed.
    l1_error = outData - l1



    # sigFunc(l1, True) represents the derivative
    # The entire statement is the error weighted derivative (the change
    # in the variable)
    l1_delta = l1_error * sigFunc(l1, True)


    # This line updates our network with our new training data
    syn0 += np.dot(l0.T, l1_delta)


print("Your output after training: ")
print(l1)