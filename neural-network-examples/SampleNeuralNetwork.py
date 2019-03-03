import numpy, random


# Sample <In Inclusive> Neural Network
# Example : true and true = true
#           true and false = true
#           false and true = true
#           false and false = false
class SampleNeuralNetwork:
    a = 1  # learning rate
    bias = 1  # value of bias
    weights = [random.random(), random.random(), random.random()]  # weights

    def activationFunc(self, t):
        # if output > 0:
        #     output = 1
        # else:
        #     output = 0
        return 1 / (1 + numpy.exp(-t))

    def outputNeuronOperation(self, input1, input2):
        output = input1 * self.weights[0] + input2 * self.weights[1] + self.bias * self.weights[2]
        return self.activationFunc(output)

    def perceptron(self, input1, input2, output):
        outputP = self.outputNeuronOperation(input1, input2)
        error = output - outputP
        self.weights[0] += error * input1 * self.a
        self.weights[1] += error * input2 * self.a
        self.weights[2] += error * self.bias * self.a

    def learn(self, iteration):
        for i in range(iteration):
            self.perceptron(1, 1, 1)
            self.perceptron(1, 0, 1)
            self.perceptron(0, 1, 1)
            self.perceptron(0, 0, 0)
        print('Used weights :')
        print(*self.weights)

    def test(self, x, y):
        res = self.outputNeuronOperation(x, y)
        print(x, "or", y, "is : ", res)


if __name__ == '__main__':
    sample = SampleNeuralNetwork()
    sample.learn(10000)
    sample.test(1, 1)
    sample.test(0, 1)
    sample.test(1, 0)
    sample.test(0, 0)
