"""
Author: Jennifer Cao
Email: jennifer.cao@wisc.edu
"""
import numpy as np
np.random.seed(0)
import random
random.seed(0)
""" this part includes some calculation tools"""
def sigmoid(x):
    return 1.0 / (1 + np.exp(-x))

def derivative(x):
    return x * (1 - x)

"""an implementation of symple perception
    where all of the attributes are numeric
"""
class Perceptron(object):
    def __init__(self, attrNum, labels,
            learningRate = 0.1, epochNum = 50):
        self.attrNum = attrNum
        self.labels = labels

        # input layer without bias unit
        self.w0 = np.random.uniform(-1, 1, (attrNum, attrNum))
        # hidden layer with bias unit
        self.w1 = np.random.uniform(-1, 1, (attrNum + 1, 1))

        self.eta = self.setLearningRate(learningRate)
        self.epoch = epochNum

    def setLearningRate(self, learningRate):
        """update learning rate manually"""
        return learningRate

    def setEpoch(self, epochNum):
        """update epoch manually"""
        self.epoch = epochNum

    def confidence(self, x):
        x1 = sigmoid(np.dot(x, self.w0))
        ones = np.ones(x1.shape[0])
        x1 = np.c_[ones, x1]
        x = sigmoid(np.dot(x1, self.w1))
        return x

    def predict(self, x):
        """predict result for x
        @ param x: input matrix (for training process, this is the batch input; for testing process, this is test data matrix)
        @ rparam: predict class
        """
        output = self.confidence(x)
        return np.where(output < 0.5, self.labels[0], self.labels[1])

    def fit(self, x, labels):
        """ fit perceptron to input data, using back propogation
        @ param x: input matrix
        @ param labels: input original y (for each instance, it is the original classification)
        @ rparam: void. This function updates weights of the neurals iteratively.
        """
        y = np.where(labels == self.labels[0], 0, 1)
        y = y.reshape(y.shape[0], 1)
        print x.shape, y.shape
        training_data = np.c_[x, y]
        random.shuffle(training_data)
        y = training_data[:,-1]
        x = training_data[:,:-1]
        for _ in range(self.epoch):
            for xi, yi in zip(x, y):
                # forward
                hiddenInput = xi
                hiddenOutput = sigmoid(np.dot(hiddenInput, self.w0))
                # xi = [x1, x2, x3, ..., xn] -> [1, x1, x2, x3, ..., xn]
                finalInput = np.insert(hiddenOutput, 0, 1)
                finalOutput = sigmoid(np.dot(finalInput, self.w1))

                # backward
                finalError = yi - finalOutput
                finalDelta = finalError * derivative(finalOutput)
                hiddenError = finalDelta.dot(self.w1[1:,:].T)
                hiddenDelta = hiddenError * derivative(hiddenOutput)
                self.w1 += self.eta * finalInput.reshape(finalInput.shape[0], 1) * finalDelta
                self.w0 += self.eta * hiddenInput.T.dot(hiddenDelta)

""" unit test """
from data_provider import data_provider, data_parser
def utPredict():
    attrNum, labels, instances = data_provider('../sonar.arff')
    x, y = data_parser(instances[:100])
    perceptron = Perceptron(attrNum, labels)
    print perceptron.predict(x)

def utFit():
    attrNum, labels, instances = data_provider('../sonar.arff')
    trainx, trainy = data_parser(instances[:100])
    testx, testy = data_parser(instances[101:])
    perceptron = Perceptron(attrNum, labels, 0.1, 10)
    perceptron.fit(trainx, trainy)
    predicty = perceptron.predict(testx)
    acc = total = 0
    for i in range(predicty.shape[0]):
        if predicty[i] == testy[i]:
            acc += 1
        total += 1
    print float(acc) / total


if __name__ == '__main__':
#   utPredict()
   utFit()
