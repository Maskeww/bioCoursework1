import numpy as np
import random
import math
from random import uniform
from random import randint

#######learningrule isn't connected, when should all the delta data etc be changed?

class NeuralNetwork:

    def __init__(self): #rename hiddenNodeOutput etc
        #network structure
        self.numInputs = 2
        self.numOutputs = 1
        self.numHidden1 = 5
        self.numHidden2 = 5

        #current weight values
        self.W1 = np.random.randn(self.numInputs,self.numHidden1-1) #layer 1 weights
        self.W2 = np.random.randn(self.numHidden1, self.numHidden2-1)#layer 2 weights
        self.W3 = np.random.randn(self.numHidden2, self.numOutputs)#layer 3 weights
        #bestOutputValues parameters
        self.testSet = []
        self.cost = 0

        self.bestOutputValues = []
        self.actualOutput = 0 #actual output value from running the network for a single network input
        self.expectedOutput= 0 # expected output value for a single network input
        self.averageOutputDifference = 0 #total difference between the input and output values for all of the inputs

    def sphere(self, inputTuple):
        answer=0
        newTestSet = []
        if inputTuple[0] <0:
            input1 = -inputTuple[0]
        else:
            input1 = inputTuple[0]
        if inputTuple[1] <0:
            input2 = -inputTuple[1]
        else:
            input2 = inputTuple[1]
                #print("AFTER", newTestSet2)

            answer += input1**2 + input2**2
        #answer = inputTuple[0] * inputTuple[0]
        return answer

    def rastrigin(self, inputTuple):
        inputs = 10*len(inputTuple)
        for i in range(len(inputTuple)):
            inputs += inputTuple[i]**2 - (10*math.cos(2*math.pi*inputTuple[i]))
        return inputs

    def Schaffers(self, inputTuple):
          if inputTuple[0] <0:
              input1 = -inputTuple[0]
          else:
              input1 = inputTuple[0]
          if inputTuple[1] <0:
              input2 = -inputTuple[1]
          else:
              input2 = inputTuple[1]
          return 418.9829 *2 -(-input1 * math.sin(math.sqrt(input1)))+(input2 * math.sin(math.sqrt(input2)))

    def rosenbrock(self,inputTuple):
        return (100 *(inputTuple[1]-(inputTuple[0])**2)**2)+((inputTuple[0]) - 1)**2

    def differentPowers(self, inputTuple):

        if inputTuple[0] <0:
            input1 = -inputTuple[0]
        else:
            input1 = inputTuple[0]
        if inputTuple[1] <0:
            input2 = -inputTuple[1]
        else:
            input2 = inputTuple[1]
        return ((input1)**2) + ((input2)**6)
    def sigmoid(self, z):
        return 1/(1+np.exp(-z))

    def ReLU(self, x):
        return x * (x > 0)

    def costFunction(self):
        self.cost = self.cost/len(self.testSet) #works out total cost of an iteration
        return

    def propagate(self, X, fun): #make this work for any network structure
        self.hiddenNodeInput1  = np.dot(X, self.W1) #weights *inputs from layer 1
        self.hiddenNodeOutput1  = self.ReLU(self.hiddenNodeInput1) #sigmoid function performed in 2nd layer neurons
        self.hiddenNodeInput2  = np.dot(self.hiddenNodeOutput1, self.W2[0:self.numHidden1-1]) + 1 * self.W2[self.numHidden1-1] #weights *inputs from layer 1
        self.hiddenNodeOutput2  = self.ReLU(self.hiddenNodeInput2) #sigmoid function performed in 2nd layer neurons
        self.outputNodeInput = np.dot(self.hiddenNodeOutput2, self.W3[0:self.numHidden2-1])+ 1 * self.W3[self.numHidden2-1]#weights *inputs from layer 2
        self.actualOutput = self.ReLU(self.outputNodeInput)#self.ReLU(self.outputNodeInput)
        print("Actual output ==", self.actualOutput)
        self.bestOutputValues.append(self.actualOutput)
        return


    def run(self, X, fun): #run an iteration of the program
        #print("Input values ==",(X[0],X[1]))

        self.propagate(X, fun) #feeds the input through the network
        if fun ==1:
            self.expectedOutput = self.sphere(X[0:2]) #finds the expected output from the sphere function
        elif fun ==3:
            self.expectedOutput = self.rastrigin(X[0:2])
        elif fun ==7:
            self.expectedOutput = self.Schaffers(X[0:2])
        elif fun ==8:
            self.expectedOutput = self.rosenbrock(X[0:2])
        elif fun ==14:
            self.expectedOutput = self.differentPowers(X[0:2])

        print("Expected output ==",self.expectedOutput)
        print("\n")
        self.cost+= 0.5*((self.expectedOutput-self.actualOutput)**2) #adds to the total cost of the current iteration
        self.averageOutputDifference += (self.actualOutput- self.expectedOutput) #keeps track of the actual output/expected output difference for working out output delta
        #self.hiddenNodeCopies() #puts all hidden neuron output into an array in order to calculate neuron deltas
        return


    def loop(self, fun): #controls number of iterations of program and updates the program data
        for index in self.testSet:
            tup = (index[0], index[1])
            self.run(tup, fun)


    def start(self, testSet, fun):
        newTestSet = []
        nn = NeuralNetwork()

        #testSet = newTestSet
        nn.testSet = testSet
        nn.loop(fun)

        cost = nn.cost
        topCost = nn.cost
        weights = (nn.W1, nn.W2, nn.W3)
        outputDifference = nn.averageOutputDifference
        bestOutputValues = nn.bestOutputValues
        topOutput = []

        bestCostArray = []
        generations = 100
        for i in range(generations):
            #print("GENERATION", i)
            returned = self.GA(weights, cost, 3, 0.0001, 100, outputDifference, bestOutputValues, topCost, topOutput, testSet, fun)
            weights = returned[0:3]
            cost = returned[3]
            outputDifference = returned[4]
            bestOutputValues = returned[5]
            topCost = returned[6]
            bestCostArray.append(cost)
            topOutput = returned[7]
            counter = 0
            #print("BEST DIFFERENCE ==", outputDifference)
            #print("BEST bestOutputValues ==", bestOutputValues)
            print("TOP COST ==", topCost)

        for index in bestCostArray:
            #print("TOP COST ==", topCost)
            array = []
            for index in topOutput:
                array.append(index[0])
            #print("TOP OUTPUT ==", array)
            #print("---------------------------------"))
        print("Output Values", array)
        return array

    def GA(self, bestWeights, cost, mutationFrequency, learningRate, population, difference, bestOutputValues, tCost, topOutput, testSet, fun):
        #print("IN THE THE GENETIC ALGORITHM METHOD")
        bestW1 = bestWeights[0]
        bestW2 = bestWeights[1]
        bestW3 = bestWeights[2]
        bestCost = 10000
        bestDifference = 0
        bestbestOutputValues = []

        for i in range(0,population):
            nn = NeuralNetwork()
            nn.testSet = testSet

            for k in range(0,2):
                for n in range(0,self.numHidden1-1):
                    mutate = random.randint(1, mutationFrequency)
                    if (mutate == 1):
                        if(difference > 0):
                            nn.W1[k][n] = bestW1[k][n] -(learningRate) * cost
                        else:
                            nn.W1[k][n] = bestW1[k][n] + (learningRate) * cost

            for k in range(0, self.numHidden1):
                for n in range(0,self.numHidden2-1):
                    mutate = random.randint(1, mutationFrequency)
                    if (mutate == 1):
                        if(difference > 0):
                            nn.W2[k][n] = bestW2[k][n] -(learningRate) * cost
                        else:

                            nn.W2[k][n] = bestW2[k][n] + (learningRate) * cost


            for n in range(0,self.numHidden2):
                mutate = random.randint(1, mutationFrequency)
                if (mutate == 1):
                    if(difference > 0):
                        nn.W3[n] = bestW3[n] -(learningRate) * cost
                    else:
                        nn.W3[n] = bestW3[n] + (learningRate) * cost
            nn.loop(fun)

            if(nn.cost < tCost):
                tCost = nn.cost
                topOutput = nn.bestOutputValues

            if(nn.cost<bestCost):
                bestCost = nn.cost
                bestW1 = nn.W1
                bestW2 = nn.W2
                bestW3 = nn.W3
                bestDifference = nn.averageOutputDifference
                bestOutput = nn.actualOutput
                bestbestOutputValues = nn.bestOutputValues
        #print("BEST COST of previous generation ==", bestCost)

        return bestW1, bestW2, bestW3, bestCost, bestDifference, bestbestOutputValues, tCost, topOutput

if __name__ == "__main__": # outputs greater than 1 and then loop inputs
    nn = NeuralNetwork()

    #change the number 1 at the end of the (start) method call to 3 to run the rastrigin function and to 14 to run the different powers function
    #If you keep the number at 1, the sphere function will run

    nn.start([(1,1), (1,2),(2,2), (2,3), (3,3,)], 1)
