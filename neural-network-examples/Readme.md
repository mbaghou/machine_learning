# Neural network

## Sample Neural Network
The beginning of the program just defines libraries and the values of the parameters, and creates a list which contains the values of the weights that will be modified (those are generated randomly).

outputNeuronOperation() :  defines the work of the output neuron. It takes 3 parameters (the 2 values of the neurons and the expected output). “outputP” is the variable corresponding to the output given by the Perceptron

perceptron() : calculate the error, used to modify the weights of every connections to the output neuron.

learn() : We create a loop that makes the neural network repeat every situation several times. This part is the learning phase. The number of iteration is chosen according to the precision we want. 

## Simple Image Classification using Convolutional Neural Network — Deep Learning
We will be building a convolutional neural network that will be trained on few thousand images of cats and dogs, and later be able to predict if the given image is of a cat or a dog.

The process of building a Convolutional Neural Network always involves four major steps.

Step - 1 : Convolution

Step - 2 : Pooling

Step - 3 : Flattening

Step - 4 : Full connection

train/test dataset : https://drive.google.com/drive/folders/1XaFM8BJFligrqeQdE-_5Id0V_SubJAZe?usp=sharing
