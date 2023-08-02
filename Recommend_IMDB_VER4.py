import numpy as np
import random
import os
import pickle
import xlwt
from xlwt import Workbook 
from xlrd import open_workbook
import time

os.system('cls||clear')

# .pkl matrix data writing. I've used it to create our dataset.
def saveData(Data, File):
    with open(File, 'wb') as f:
        pickle.dump(Data, f)

# .pkl matrix data reading
def loadList(File):
    DataList = []
    with open(File, 'rb') as f:
        DataList = pickle.load(f)
    return DataList

__location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))
dataFile = os.path.join(__location__, 'training.pkl')
trainList = loadList(dataFile)
__location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))
dataFileAll = os.path.join(__location__, 'training_all.pkl')
trainListAll = loadList(dataFileAll)
dataTestFile = os.path.join(__location__, 'test.pkl')
testList = loadList(dataTestFile)
dataRecomFile = os.path.join(__location__, 'recomdata.pkl')
recomList = loadList(dataRecomFile)
dataNameFile = os.path.join(__location__, 'name_list.pkl')
film_name_list = loadList(dataNameFile)
dimension = 31
# Lists are taken from our pkl data files.

# Function to save our results to an excel file.
def saveResults(result, filedir):
    wb = Workbook()
    sheet1 = wb.add_sheet('Sheet1')
    sheet1.write(0, 0, 'Learning Rate') 
    for i in range(9):
        sheet1.write(i+1, 0, (i+1)/10)
    sheet1.write(11, 0, 'Learning Rate (Without Momentum)')
    for i in range(9):
        sheet1.write(i+12, 0, (i+1)/10)
    sheet1.write(22, 0, 'Hidden Layer 1 N count (x 15 5)') 
    for i in range(9):
        sheet1.write(i+23, 0, (i+1)*5)
    sheet1.write(34, 0, 'Hidden Layer 2 N count (15 x 5)')
    for i in range(9):
        sheet1.write(i+35, 0, (i+1)*5)  
    sheet1.write(0, 1, 'Education Iteration')
    sheet1.write(0, 2, 'Education Time')
    sheet1.write(0, 3, 'Error Percentage')
    for i in range(9):
        sheet1.write(i+1, 1, result[i, 0]) 
        sheet1.write(i+1, 2, result[i, 1]) 
        sheet1.write(i+1, 3, result[i, 2])   
    for i in range(9):
        sheet1.write(i+12, 1, result[i+9, 0]) 
        sheet1.write(i+12, 2, result[i+9, 1]) 
        sheet1.write(i+12, 3, result[i+9, 2])  
    for i in range(10):
        sheet1.write(i+23, 1, result[i+18, 0]) 
        sheet1.write(i+23, 2, result[i+18, 1]) 
        sheet1.write(i+23, 3, result[i+18, 2])  
    for i in range(10):
        sheet1.write(i+35, 1, result[i+28, 0]) 
        sheet1.write(i+35, 2, result[i+28, 1]) 
        sheet1.write(i+35, 3, result[i+28, 2])  
    wb.save(filedir) 

class Perceptron():
    def __init__(self, dimensions, rate_of_learning = 1e-2, bias = True, activation_type = "tanh"):
        #Initialize
        self.dimensions = dimensions
        self.rate_of_learning = rate_of_learning
        self.bias = bias
        self.activation_type = activation_type

        self.reset()

    def reset(self):
        #Reset perceptron variables
        if self.bias is True:
            self.weights = np.zeros((1, self.dimensions + 1), dtype=float)
            self.weights[0,:] = np.random.rand() - 0.5
            self.prevweights = np.zeros((1, self.dimensions + 1), dtype=float)
        else:
            self.weights = np.zeros((1, self.dimensions), dtype=float)
            self.weights[0,:] = np.random.rand() - 0.5
            self.prevweights = np.zeros((1, self.dimensions), dtype=float)
        self.data = np.zeros(self.dimensions)
        self.lin_comb = 0
        self.activation = 0

    def forward(self, data):
        #Forward pass data through the perceptron
        if self.bias is True:
            self.data = np.concatenate((data, np.array([1],dtype=float).reshape(1,1)), axis = 0)
        else:
            self.data = np.array(data)
        w = self.weights
        self.lin_comb = np.dot(w, self.data)
        self.activation = self.activation_function(self.lin_comb)
        return self.activation

    def activation_function(self, data):
        #Select activation function and use it
        #Sigmoid is used in this project
        if self.activation_type == "tanh":
            y = tanh(data)
            self.my_derivative = y[1]
            return y[0]
        elif self.activation_type == "sigmoid":
            y = sigmoid(data)
            self.my_derivative = y[1]
            return y[0]
        else:
            return data

    def __str__(self):
        return "Dimensions: %d, Bias: %d, Learning Rate: %f, Activation Type: %s"% (self.dimensions, 
                                                                                    self.bias, 
                                                                                    self.rate_of_learning, 
                                                                                    self.activation_type)

    def update(self, grad, moment = True):
        # Weight function including momentum term, if momentum=False its taken zero
        # If not, it's momentum = n + (1 - n)/5 for n=learning rate.
        momentum = (1 - self.rate_of_learning)/5 + self.rate_of_learning
        if  moment == False:
            momentum = 0
        momentumterm = momentum*np.subtract(self.weights, self.prevweights)
        self.prevweights = self.weights
        self.weights = np.add(np.add(self.weights,np.multiply((self.rate_of_learning*grad[0]),self.data).reshape(1, self.weights.size)), momentumterm)
# Perceptron code is written with the help of the Github of Mr. Halil Durmuş (Hunterhal)

# tanh and sigmoid functions with their derivatives.
def tanh(x):
    t=(np.exp(x)-np.exp(-x))/(np.exp(x)+np.exp(-x))
    dt=1-t**2
    return t,dt
def sigmoid(x):
    s=1/(1+np.exp(-x))
    ds=s*(1-s)  
    return s,ds

# Creation of 2 layer perceptron network
def network_2layer_create(data_size, l1_size, l2_size, lo_size, rate_of_learning, bias, act_function):
    List_layer1 = []
    List_layer2 = []
    List_layerout = []
    # Layer sizes are taken as argument. Learning rate is same for all perceptrons. Bias=True. 
    # Result from previous layer is used in the layer.
    for i in range(l1_size):
        List_layer1.append(Perceptron(data_size, rate_of_learning, bias, act_function))
    for i in range(l2_size):
        List_layer2.append(Perceptron(l1_size, rate_of_learning, bias, act_function))
    for i in range(lo_size):
        List_layerout.append(Perceptron(l2_size, rate_of_learning, bias, act_function))
    Layers = np.array([List_layer1, List_layer2, List_layerout])
    return Layers

# Local gradient calculation
def local_grad(list1, list2, nextgrad = None, outputLayer = False):
    ncount = len(list1)
    derive_activation = np.zeros((ncount, 1), dtype=float)
    grad = np.zeros((ncount, 1), dtype=float)
    # Another function for output layer gradients is created. Since it's definition is different.
    if outputLayer == True:
        for i in range(ncount):
            derive_activation[i] = list1[i].my_derivative
        grad = np.multiply(list2, derive_activation) 
    # Local gradients are calculated with backpropagation algorithm.
    else:
        wForRow = (list2[0].weights.size) - 1
        counterout = len(list2)
        wMatrix = np.zeros((wForRow, counterout), dtype=float)
        for i in range(ncount):
            derive_activation[i] = list1[i].my_derivative
        for i in range(counterout):
            wMatrix[:, i] = list2[i].weights[0, 0:wForRow]
        grad = np.multiply(np.dot(wMatrix, nextgrad), derive_activation)
    return grad

# Testing function. Uses forward path algorithm, gives output.
def testdata(testdata, network):
    i1 = 0
    i2 = 0
    io = 0
    y1 = np.zeros((len(network[0]), 1), dtype=float)
    y2 = np.zeros((len(network[1]), 1), dtype=float)
    yout = np.zeros((len(network[2]), 1), dtype=float)
    for perceptron in network[0]:
        y1[i1] = perceptron.forward(testdata)
        i1 += 1
    for perceptron in network[1]:
        y2[i2] = perceptron.forward(y1)
        i2 += 1
    for perceptron in network[2]:
        yout[io] = perceptron.forward(y2)
        io += 1
    return yout

def recommendfilm(datalist, network, dim):
    recomdata = np.zeros((234,5), dtype=float)
    i=0
    for data in datalist:
        prediction = testdata(data, network)
        prediction = prediction.reshape(1,5)
        recomdata[i,:] = prediction[0,:]
        i=i+1
    recomdata = recomdata.reshape(234,5)
    col=np.array([np.sum(recomdata,axis=1)])
    col = col.reshape(234,1)
    for i in range(5):
        recomdata[:, i] = recomdata[:, i]/col[:, 0]
        recomdata[:, i] = recomdata[:, i]*(i/2-1)
    weighted_recom = np.sum(recomdata,axis=1)
    weighted_recom = weighted_recom.reshape(234,1)
    max_index_row = np.where(weighted_recom == np.amax(weighted_recom))
    max_index_int, myZeroVariable = max_index_row
    print("Best recommendation for you is %s ." % (film_name_list[max_index_int[0]]))
    weighted_recom[max_index_int] = -1
    max_index_row = np.where(weighted_recom == np.amax(weighted_recom))
    max_index_int, myZeroVariable = max_index_row
    print("2nd recommendation for you is %s ." % (film_name_list[max_index_int[0]]))
    weighted_recom[max_index_int] = -1
    max_index_row = np.where(weighted_recom == np.amax(weighted_recom))
    max_index_int, myZeroVariable = max_index_row
    print("3rd recommendation for you is %s ." % (film_name_list[max_index_int[0]]))
    weighted_recom[max_index_int] = -1
    max_index_row = np.where(weighted_recom == np.amax(weighted_recom))
    max_index_int, myZeroVariable = max_index_row
    print("4th recommendation for you is %s ." % (film_name_list[max_index_int[0]]))
    weighted_recom[max_index_int] = -1
    max_index_row = np.where(weighted_recom == np.amax(weighted_recom))
    max_index_int, myZeroVariable = max_index_row
    print("5th recommendation for you is %s ." % (film_name_list[max_index_int[0]]))
    weighted_recom[max_index_int] = -1
    max_index_row = np.where(weighted_recom == np.amax(weighted_recom))
    max_index_int, myZeroVariable = max_index_row
    print("Recommendation is done")
    
# Testing phase
def testcomplete(datalist, network, dim):
    Numoferrors = 0
    datatotal = 0
    for data in datalist:
        # Testdata is used for every data, network output is given to pred.
        prediction = testdata(data[:dim], network)
        prediction = prediction/np.amax(prediction, axis = 0)
        for i in range(prediction.size):
            if prediction[i] == float(1):
                indice = i
        if indice == 1:
            predicted = "1"
        elif indice == 2:
            predicted = "2"
        elif indice == 3:
            predicted = "3"
        elif indice == 4:
            predicted = "4"
        elif indice == 5:
            predicted = "5"
        else:
            predicted = "None"
        # Network output is divided to maximum output. Only one output neuron is activated.
        # Indice of this neuron is our output.
        if data[dim + indice] != prediction[indice]:
            Numoferrors += 1
        datatotal += 1
    # Error and total tests are taken out.
    result = np.array([Numoferrors, datatotal])
    return result

# Neuron education
def education(data, network, expection_data, moment = True):
    y1 = np.zeros((len(network[0]), 1), dtype=float)
    y2 = np.zeros((len(network[1]), 1), dtype=float)
    yout = np.zeros((len(network[2]), 1), dtype=float)
    i1 = 0
    i2 = 0
    io = 0
    # Starts from first layer, follows the path until the output.
    for perceptron in network[0]:
        y1[i1] = perceptron.forward(data)
        i1 += 1
    for perceptron in network[1]:
        y2[i2] = perceptron.forward(y1)
        i2 += 1
    for perceptron in network[2]:
        yout[io] = perceptron.forward(y2)
        io += 1
    # Error calculation
    e = np.subtract(expection_data, yout)
    E = np.dot(np.transpose(e), e)/2

    gradient1 = np.zeros((len(network[0]), 1), dtype=float)
    gradient2 = np.zeros((len(network[1]), 1), dtype=float)
    gradientout = np.zeros((len(network[2]), 1), dtype=float)
    
    # Local gradients are calculated
    gradientout = local_grad(network[2], e, outputLayer = True)
    gradient2 = local_grad(network[1], network[2], nextgrad = gradientout)
    gradient1 = local_grad(network[0], network[1], nextgrad = gradient2)

    i1 = 0
    i2 = 0
    io = 0
    # Weights are updated
    for perceptron in network[0]:
        perceptron.update(gradient1[i1], moment)
        i1 += 1
    for perceptron in network[1]:
        perceptron.update(gradient2[i2], moment)
        i2 += 1
    for perceptron in network[2]:
        perceptron.update(gradientout[io], moment)
        io += 1
    return E

# Education is iterated for every neuron
def educationcomplete(datalist, network, epoch, errorthreshold, dim, moment = True):
    Eortprev = 0
    ResetCount = 0
    result = np.zeros(2)
    result[0] = epoch
    # Education continues until epoch is reached or error is low enough
    for i in range(epoch):
        Eort = 0 
        for data in datalist:
            attributes = data[:dimension]
            expection_data = data[dimension:]
            # First 31 numbers are data, last 5 are expected results.
            E = education(attributes, network, expection_data, moment)
            Eort += E
        # Error of every data is found.
        Eort = Eort/len(datalist)
        if abs(Eortprev - Eort) < Eortprev*errorthreshold and Eort > errorthreshold*100:
            ResetCount += 1
            if ResetCount >= 31:
                for perceptron in network[0]:
                    perceptron.reset()
                for perceptron in network[1]:
                    perceptron.reset()
                for perceptron in network[2]:
                    perceptron.reset()
                ResetCount = 0
        # Ends education if error is low enough or system does not improve itself.
        if (abs(Eortprev - Eort) < Eortprev*errorthreshold and Eort < errorthreshold*100) or Eort < errorthreshold:
            result[0] = i + 1
            break
        Eortprev = Eort
    # Education error and total iterations are taken out.
    result[1] = Eort
    return result

# We create 31, 15, 20, 5 network with 0.9 learning rate and sigmoid activation function. 
# Decided for this network after doing a wide iteration and a precise iteration. This gives the best results.
network = network_2layer_create(dimension, 15, 20, 5, 9e-1, True, "sigmoid")
epoch = 500
eth = 1e-3

# Now the education for recommendation
start_time = time.time()
result = educationcomplete(trainListAll, network, epoch, eth, dimension, False)
print("Education time: %f" % (time.time() - start_time))
print("Education is over in %d steps with %.7f error" % (result[0], result[1]))
recommendfilm(recomList, network, dimension)


result = educationcomplete(trainList, network, epoch, eth, dimension, False)
print("Education time: %f" % (time.time() - start_time))
print("Education is over in %d steps with %.7f error" % (result[0], result[1]))

# Errors are taken from function
testresult = testcomplete(testList, network, dimension)
print("Made %d errors out of %d tries" % (testresult[0], testresult[1]))

# Iteratively tried for best results. For which neuron numbers in each layer we get the best result?
# What should be the optimum learning rate, should we use momentum?
resultMomentum = np.zeros((9,3), dtype=float)
for i in range(9):
    education_iteration = 0
    test_accuracy = 0
    education_time = 0
    for j in range(10):
        network = network_2layer_create(dimension, 15, 15, 5, (i+1)*1e-1, True, "sigmoid")
        start_time = time.time()
        result = educationcomplete(trainList, network, epoch, eth, dimension)
        education_time += time.time() - start_time
        print("Education time for resultMomentum, %d iteration: %f" % (i+1, (time.time() - start_time)))
        education_iteration += result[0]
        testresult = testcomplete(testList, network, dimension)
        print("Made %d errors out of %d tries" % (testresult[0], testresult[1]))
        test_accuracy += testresult[0]/testresult[1]
    resultMomentum[i,0] = education_iteration/10
    resultMomentum[i,1] = education_time/10
    resultMomentum[i,2] = test_accuracy/10
print("resultMomentum is done")

resultNoMomentum = np.zeros((9,3), dtype=float)
for i in range(9):
    education_iteration = 0
    test_accuracy = 0
    education_time = 0
    for j in range(10):
        network = network_2layer_create(dimension, 15, 15, 5, (i+1)*1e-1, True, "sigmoid")
        start_time = time.time()
        result = educationcomplete(trainList, network, epoch, eth, dimension, False)
        education_time += time.time() - start_time
        print("Education time for resultNoMomentum, %d iteration: %f" % (i+1, (time.time() - start_time)))
        education_iteration += result[0]
        testresult = testcomplete(testList, network, dimension)
        print("Made %d errors out of %d tries" % (testresult[0], testresult[1]))
        test_accuracy += testresult[0]/testresult[1]
    resultNoMomentum[i,0] = education_iteration/10
    resultNoMomentum[i,1] = education_time/10
    resultNoMomentum[i,2] = test_accuracy/10
print("resultNoMomentum is done")

layer1iteration = np.zeros((10,3), dtype=float)
for i in range(10):
    education_iteration = 0
    test_accuracy = 0
    education_time = 0
    for j in range(10):
        network = network_2layer_create(dimension, (i+1)*5, 15, 5, 5e-1, True, "sigmoid")
        start_time = time.time()
        result = educationcomplete(trainList, network, epoch, eth, dimension, False)
        education_time += time.time() - start_time
        print("Education time for layer1iteration, %d iteration: %f" % (i+1, (time.time() - start_time)))
        education_iteration += result[0]
        testresult = testcomplete(testList, network, dimension)
        print("Made %d errors out of %d tries" % (testresult[0], testresult[1]))
        test_accuracy += testresult[0]/testresult[1]
    layer1iteration[i,0] = education_iteration/10
    layer1iteration[i,1] = education_time/10
    layer1iteration[i,2] = test_accuracy/10
print("layer1iteration is done")

layer2iteration = np.zeros((10,3), dtype=float)
for i in range(10):
    education_iteration = 0
    test_accuracy = 0
    education_time = 0
    for j in range(10):
        network4 = network_2layer_create(dimension, 15, (i+1)*5, 5, 5e-1, True, "sigmoid")
        start_time = time.time()
        result = educationcomplete(trainList, network4, epoch, eth, dimension, False)
        education_time += time.time() - start_time
        print("Education time for layer2iteration, %d iteration: %f" % (i+1, (time.time() - start_time)))
        education_iteration += result[0]
        testresult = testcomplete(testList, network4, dimension)
        print("Made %d errors out of %d tries" % (testresult[0], testresult[1]))
        test_accuracy += testresult[0]/testresult[1]
    layer2iteration[i,0] = education_iteration/10
    layer2iteration[i,1] = education_time/10
    layer2iteration[i,2] = test_accuracy/10
print("layer2iteration is done")

# Results are written to an excel file
excelresults = np.concatenate((resultMomentum, resultNoMomentum, layer1iteration, layer2iteration), axis=0)
excelFile = os.path.join(__location__, 'ANN_Results.xls')
saveResults(excelresults, excelFile)