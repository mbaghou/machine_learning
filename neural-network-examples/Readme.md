# Neural network

## Sample Neural Network
The beginning of the program just defines libraries and the values of the parameters, and creates a list which contains the values of the weights that will be modified (those are generated randomly).

outputNeuronOperation() :  defines the work of the output neuron. It takes 3 parameters (the 2 values of the neurons and the expected output). “outputP” is the variable corresponding to the output given by the Perceptron

perceptron() : calculate the error, used to modify the weights of every connections to the output neuron.

learn() : We create a loop that makes the neural network repeat every situation several times. This part is the learning phase. The number of iteration is chosen according to the precision we want. 