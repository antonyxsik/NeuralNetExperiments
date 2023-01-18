"""
babyNN_model
~~~~~~~~~~~~~
My first Neural Network
From scratch, no ML libraries
with lots and lots of help
Antony Sikorski
"""

# Libraries
import numpy as np 
import random



# Sigmoid nonlinearity function
# may add other nonlinearities, but for now using only sigmoid is fine
def sigmoid(z):
    a = 1.0 / (1 + np.exp(-z))
    return a

# Need the derivative of sigmoid for backpropogation  
def sigmoidPrime(z):
    aPrime = sigmoid(z)*(1 - sigmoid(z))
    return aPrime



class Network:

    def __init__(self, layerSizes, learnRate, miniBatchSize, numEpochs): 

        """
        Constructor for the Network class

        - layerSizes takes a list of layer sizes (input, hiddens, output) (int list)
        - learnRate is the learning rate (float)
        - miniBatchSize is the size of the mini batches when chopping up the data for 
            stochastic gradient descent (int)
        - numEpochs is the number of training epochs (int)

        Also randomly generates the weights and biases by sampling from a normal 
        (Gaussian) distribution with a mean 0 and variance 1. 
        """

        self.layerSizes = layerSizes
        self.numLayers = len(layerSizes)
        self.learnRate = learnRate
        self.miniBatchSize = miniBatchSize 
        self.numEpochs = numEpochs
        # generates an array filled with column vectors of biases for 
        # each layer following the input layer (based on number of neurons)
        self.biases = [np.random.randn(i, 1) for i in layerSizes[1:]]
        # generates an array filled with matrices of weights based on 
        # the sizes of each layer (number of neurons)
        self.weights = [np.random.randn(j,i) for i,j in zip(layerSizes[:-1], layerSizes[1:])]

    
     
    def forward(self, a):
        """
        The forward pass function. 
        Takes the image input vector and returns the NN output vector based on 
        the current weights and biases 
        """
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a)+b)
        return a


    
    def forwardStore(self, a):
        """
        Similar to the forward pass function except it stores each layer's activations
        and z vectors. To be used in backpropogation. 
        Takes the image input vector and returns the lists of all z vectors and 
        activations (including input, hidden, and output)
        """
        activations = [a] 
        zList = [] 
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w,a) + b
            zList.append(z)
            a = sigmoid(z) 
            activations.append(a) 

        return (zList, activations)



    # not yet being used, currently okay with just displaying accuracy 
    def cost(self, output, correct, datLength):
        """
        Calculates the cost/loss. Used for evaluating performance. Takes in the 
        output of the NN, the desired output, and the length of the data. 
        Returns a floating point value for the loss/cost. 
        """
        step1 = (output - correct)
        step2 = np.square(step1)
        loss = 0.5 * np.sum(step2)/(datLength)
        return loss



    def costPrime(self, output, correct):
        """
        Derivative of the cost function. Takes in the output of the NN
        and the desired output. Returns the difference vector. Used 
        in backpropogation. 
        """
        return(output-correct)
    


    def backward(self, imageInput, correct):
        """
        The backpropogation function. Most important bit in this whole file. Takes in the 
        actual image input vector (784x1) and the desired output vector (10x1), and 
        calculates the gradient in order to updates the weights and biases for one step. 
        Returns two lists: A list of matrices for the weight gradient, and a list of vectors
        for the bias gradient. 

        Note: the * operation between two vectors below represents the Hadamard product, 
        which is just elementwise multiplication of two vectors of the same shape. It
        results in a vector of the same shape as the original two. 
        """

        # initializes empty arrays of the same shapes so that we can conveniently put the gradient in 
        weightGradient = [np.zeros(w.shape) for w in self.weights]
        biasGradient = [np.zeros(b.shape) for b in self.biases]

        # Forward pass
        # saved lists of z vectors and activations for each layer
        zList, activations = self.forwardStore(imageInput)


        # Backward pass
        # all updates will need this constant
        constant = self.costPrime(activations[-1], correct) * sigmoidPrime(zList[-1])

        for i in range(1, self.numLayers):

            # for i = 1, initializes the partial deriv equations
            if i == 1:
                biasGradient[-i] = constant
                weightGradient[-i] = np.dot(constant, activations[-i-1].transpose())
            
            # for i > 1, you effectively just keep multiplying them by the weight of the layer after and
            # the sigmoid of the current z
            else:
                constant = np.dot(self.weights[-i+1].transpose(), constant) * sigmoidPrime(zList[-i])
                biasGradient[-i] = constant
                weightGradient[-i] = np.dot(constant, activations[-i-1].transpose())

        return (weightGradient, biasGradient)



    def train(self,training_data, test_data = None):
        """
        Here we train our NN using stochastic gradient descent with mini batches. We iterate over 
        multiple epochs (complete training cycles). 

        In each epoch, we cut the data up into mini batches, compute the gradient using 
        backpropogation for each entry in the mini batch, and then calculate the average per
        mini batch and use that and the learning rate to update our weights and biases. 
        We repeat this until we have gone through all of the mini batches, and that is considered 
        a completed epoch. 

        The functions takes in the training_data and testing_data. It first trains the network on 
        the training_data in each epoch as described above, and then applies that set of weights and
        biases to the testing data. The epoch, along with the accuracy/success rate of the network 
        on the corresponding testing data is so that progress can be tracked. 

        The test_data parameter is set to None by default, so that we have the option of training 
        the network and not doing any testing. If test data is provided, accuracy will be shown. 
        """
        training_data = list(training_data)

        #represents one whole cycle of training and testing
        for i in range(self.numEpochs):
            
            #randomly shuffled so that we get different mini-batches in each iteration
            random.shuffle(training_data)
            #cut into mini batches using list comprehension
            miniBatches = [training_data[ind: ind + self.miniBatchSize] for ind in range(0, len(training_data), self.miniBatchSize)]

            #iterating over each mini batch 
            for miniBatch in miniBatches: 
                
                #lists to keep a running total of the gradient
                weightTotalGrad = [np.zeros(w.shape) for w in self.weights]
                biasTotalGrad = [np.zeros(b.shape) for b in self.biases]

                #iterating over each image/classification tuple in the mini batch
                for x,y in miniBatch:

                    #backpropogation  
                    weightGradUpdate, biasGradUpdate = self.backward(x,y)

                    #update the running total
                    weightTotalGrad = [total + update for total, update in zip(weightTotalGrad, weightGradUpdate)]
                    biasTotalGrad = [total + update for total, update in zip(biasTotalGrad, biasGradUpdate)]
                
                # the average gradient is computed, then multiplied by the learning rate
                # this is then used to update our weights and biases 
                self.weights = [weights - (self.learnRate/len(miniBatch)) * update for weights, update in zip(self.weights, weightTotalGrad)]
                self.biases = [bias - (self.learnRate/len(miniBatch)) * update for bias, update in zip(self.biases, biasTotalGrad)]

            if test_data: 
                test_data = list(test_data)

                # epoch and accuracy of the network is printed each time if testing data is provided 
                print("Epoch ", i + 1, " complete")
                results = [(np.argmax(self.forward(x)), y) for (x, y) in test_data]
                numCorrect = sum(int(classified == actual) for classified, actual in results)
                print("Accuracy: ", numCorrect, "/", len(test_data), "\n")


                #In the future maybe also print the loss number to show that it is decreasing
                
            
            #shows which epoch we are on if no test data is provided 
            else: 
                print("Epoch ", i + 1, " complete")



    # not yet being used, currently okay with just displaying accuracy 
    def cost(self, output, correct, datLength):
        """
        Calculates the cost/loss. Used for evaluating performance. Takes in the 
        output of the NN, the desired output, and the length of the data. 
        Returns a floating point value for the loss/cost. 
        """
        step1 = (output - correct)
        step2 = np.square(step1)
        loss = 0.5 * np.sum(step2)/(datLength)
        return loss