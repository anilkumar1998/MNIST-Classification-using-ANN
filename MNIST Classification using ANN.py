# Import the necessary modules

import numpy as np 
import matplotlib.pyplot as plt 
import pandas as pd 


"""
 Usage: This function will return the parameters i.e, Weigths & Biases by initializing randomly
 Params: from_file  (If we need to read the weights from the file (True or False))
         dimensions (It takes the number of nodes in each layer including the input layer)
         file_name  (If True, specify th file_name)
 
 Description: Here we are multiplying random Weights by 0.01 so that it will be close to 0 and then for a given 
              activation function it will not be close to the numbers whose derivative will be 0.
"""
def initalize_parameters(from_file, dimensions, file_name):
    if from_file == False:
        np.random.seed(1)
        parameters = {}
        N_layers = len(dimensions)
        for i in range(1,N_layers):
            parameters["W" + str(i)] = np.random.randn(dimensions[i],dimensions[i-1]) * 0.01
            parameters["b" + str(i)] = np.random.randn(dimensions[i],1)
    else:
        parameters = read_parameters_from_json_file(file_name)
    return parameters  

"""
 Usage: These functions will read the parameters from the json file and save the parameters to the json file.
"""
def save_parameter_as_json_file(params, name):
    # json can only store specific datatypes in to a json file. Unfortunately numpy array is not in that list.
    # So we need to convert each key's value from numpy array to a list
    for key in params.keys():       
        params[key] = params[key].tolist()
    json.dump(params,open(name+".json","w"))

def read_parameters_from_json_file(name):
    params = json.load(open(name+".json"))
    # We need to convert back the key's value from list to numpy array
    for key in params.keys():
        params[key] = np.array(params[key])
    return params


"""
    Usage: The below 4 functions are the different activation functions that we
           use to make our Neural Network non-linear.
    Params: It takes number (or) a vector (or) even a numpy array and gives the respective functions output (element wise if arrays are passed)
"""
def sigmoid(Z):
    return 1/(1+np.exp(-Z))

def tanh(Z):
    return np.tanh(Z)

def relu(Z):
    return np.maximum(0,Z)

def leaky_relu(Z):
    return np.maximum(0.01*Z, Z)

#************* Forward Propagation ************#

def computeZ(A_prev, W, b):
    Z = np.dot(W,A_prev) + b
    cache = (A_prev, W, b)
    return Z,cache

def computeA(A_prev, W, b, activation_function):
    
    assert activation_function.lower() == 'sigmoid' or activation_function.lower() == 'tanh' or activation_function.lower() == 'relu' or "leaky" in activation_function.lower()
    
    if activation_function.lower() == 'sigmoid':
        Z, cache = computeZ(A_prev,W,b)
        A = sigmoid(Z)
    
    elif activation_function.lower() == 'tanh':
        Z, cache = computeZ(A_prev,W,b)
        A = tanh(Z)
    elif activation_function.lower() == 'relu':
        Z, cache = computeZ(A_prev,W,b)
        A = relu(Z)
    else:
        Z, cache = computeZ(A_prev,W,b)
        A = leaky_relu(Z)

    assert A.shape == (W.shape[0], A_prev.shape[1])

    cache_with_Z = (cache, Z)
    return A,cache_with_Z

"""
    Usage: used to do the forward propagation of a neural network
    Params: It takes the input(X), parameters after initialization & activation function we need to use for the layers

    Description: Here for all the hidden layers, we are using the function as specified by the user(default is relu) 
                 but for the last layer, we are using sigmoid.

"""
def feed_forward(X, parameters, activation_function='relu'):
    A = X
    caches = []
    L = len(parameters)//2  # Dividing with 2 because it has both W & b.
    for l in range(1,L+1):
        A_prev = A
        if(l == L):
            activation_function = "sigmoid"
        A, cache = computeA(A_prev, parameters["W" + str(l)], parameters["b" + str(l)], activation_function)
        caches.append(cache)
    
    return A, caches


#*************** Cost Function ****************#
"""
    Usage: This is the cost function we are using to minimize the error.
    Params: It the predicted output vs actual output

    Description: Here we are using binary Cost-Entropy function.(logistic cost function)
                 L(y_hat,y) = (-1/m)*sigma(y*log(y_hat) + (1-y)*(1-log(y_hat)))
"""
def cost_function(y_hat, y):
    assert(y.shape == y_hat.shape)
    m = y.shape[1]
    cost = (-1/m)*np.sum(y*np.log(y_hat) + (1-y)*np.log(1-y_hat))
    return cost


#**************** Backward Propagation ********************#
"""
    Usage: Finding the derivates of activation functions
    Params: It take the post activation gradient and the current layer Z
"""
def sigmoid_derivative(dA, Z):
    A = sigmoid(Z)
    dZ = dA*A*(1-A)
    return dZ

def tanh_derivative(dA, Z):
    A = tanh(Z)
    dZ = dA*(1-A*A)
    return dZ

def relu_derivative(dA, Z):
    A = relu(Z)
    dZ = dA*(np.int64(A>0))
    return dZ

def leaky_relu_derivative(dA, Z):
    A = leaky_relu(Z)
    A[A >=0 ] = 1
    A[A < 0] = 0.01
    dZ = dA*A
    return dZ

def compute_derivative_wrt_params(dZ, cache):
    A_prev, W, b = cache
    
    m = A_prev.shape[1]
    dW = (1/m)*(np.dot(dZ, A_prev.T))
    db = (1/m)*(np.sum(dZ, axis=1, keepdims=True))
    dA_prev = np.dot(W.T, dZ)

    assert dW.shape == W.shape
    assert db.shape == b.shape
    assert dA_prev.shape == A_prev.shape
    return dA_prev, dW, db

def compute_derivative_cur_layer(dA, cache, activation_function):
    small_cache, Z = cache
    if activation_function.lower() == "sigmoid":
        dZ = sigmoid_derivative(dA, Z)
        dA_prev, dW, db = compute_derivative_wrt_params(dZ, small_cache)
    elif activation_function.lower() == "tanh":
        dZ = tanh_derivative(dA, Z)
        dA_prev, dW, db = compute_derivative_wrt_params(dZ, small_cache)
    elif activation_function.lower() == "relu":
        dZ = relu_derivative(dA, Z)
        dA_prev, dW, db = compute_derivative_wrt_params(dZ, small_cache)
    else:
        dZ = leaky_relu_derivative(dA,Z)
        dA_prev, dW, db = compute_derivative_wrt_params(dZ, small_cache)
    
    return dA_prev, dW, db

"""
    Usage: This will do the back-propagation of neural network so as to learn the parameters such that it minimizes the error.
    Params: It takes the predicted output, actual output, caches and the activation function
"""
def backward_propagation(AL, y, caches, activation_function):
    y = y.reshape(AL.shape)
    gradients = {}
    L = len(caches)

    dAL = np.divide((AL-y), AL*(1-AL))
    gradients["A" + str(L-1)], gradients["W" + str(L)], gradients["b" + str(L)] = compute_derivative_cur_layer(dAL, caches[L-1], "sigmoid")

    for l in range(L-1,0,-1):
        gradients["A" + str(l-1)], gradients["W" + str(l)], gradients["b" + str(l)] = compute_derivative_cur_layer(gradients["A"+str(l)], caches[l-1], activation_function)

    return gradients

"""
    Usage: This is where the update of the weights & biases will happen
    Params: parameters , gradients and learning rate.
"""
def update_parameters(parameters, gradients, learning_rate):
    L = len(parameters) // 2
    for l in range(1,L+1):
        parameters["W" + str(l)] = parameters["W" + str(l)] - (learning_rate)*(gradients["W" + str(l)])
        parameters["b" + str(l)] = parameters["b" + str(l)] - (learning_rate)*(gradients["b" + str(l)])
    return parameters

#******************** Load the data *******************#
"""
    Usage: This is where we will import the data and convert it as per our requirements
    Params: It takes the path where our file resides
"""

def load_data(path_train, path_test):
    train_data = pd.read_csv(path_train)
    test_data = pd.read_csv(path_test)
    # Convert pandas dataframe to numpy array
    train_data = train_data.values
    test_data = test_data.values
    # train_data has shape of (60000, 785) where the first column represent correct label & remaining 784 columns represent image(28*28 = 784)
    # Seperate input and ouptut for train data
    X_train = train_data[:,1:]
    y_train = train_data[:,0].reshape(1,-1) 
    # Seperate input and output for test data
    X_test = test_data[:,1:]
    y_test = test_data[:,0].reshape(1,-1) 
    return X_train, y_train, X_test, y_test

"""
    Usage: Train the neural network as per the parameters
    Params: input data, output data, layers dimensions, learning rate, number of iterations, activation function, 
"""

def neural_network(X, y, dimensions, from_file, file_name, learning_rate=0.01, epochs=1500, activation_function = 'relu'):
    parameters = initalize_parameters(from_file, dimensions, file_name)
    cost_list = []

    for i in range(1,epochs+1):
        AL, caches = feed_forward(X, parameters, activation_function)
        cost = cost_function(AL,y)
        if(i%20 == 1):
            print("Cost after {} epochs : {}".format(i,cost));
            cost_list.append(cost)
        gradients = backward_propagation(AL, y, caches, activation_function)
        parameters = update_parameters(parameters, gradients, learning_rate)
    return parameters

def one_hot_encoding(y):
    size = y.shape[1]
    print(size)
    one_hot_y = np.zeros((size,10))
    one_hot_y[np.arange(size), y] = 1
    return one_hot_y.T

def test(X_test, parameters, y_test):
    correctly_classified = 0
    misclassified_indices = []
    A,cac = feed_forward(X_test.T, parameters)
    equality = np.argmax(A,axis=0)==np.argmax(y_test,axis=0)
    correctly_classified = equality.sum()
    total = X_test.shape[0]
    accuracy = (ans/total)*100
    print("Accuracy is :" + str(accuracy) )
    misclassified_indices = np.where(equality == False)
    return misclassified_indices

def display_misclassified(X_test, misclassified_indices, parameters):
    for i in range(len(misclassified_indices)):
        A,cac = feed_forward(X_test[misclassified_indices[i]].reshape(-1,1), parameters)
        print("Predicted Label is: ", np.argmax(A))
        print("Correct Label is: ", np.argmax(y_test[:,misclassified_indices[i]]))
        plt.imshow(X_test[misclassified_indices[i]].reshape(28,28),cmap="gray")
        plt.show()

# """
#     Usage: This is where we need to start.
#     Params: No params (this is the start)
# """

def start():
    path_train = "mnist_train.csv"
    path_test = "mnist_test.csv"
    X_train, y_train, X_test, y_test = load_data(path_train, path_test)
    # Need to one hot encoding for y_train & y_test
    y_train = one_hot_encoding(y_train)
    y_test = one_hot_encoding(y_test)
    print(X_train.shape)
    print(y_train.shape)
    print(X_test.shape)
    print(y_test.shape)
    dimensions = [784, 256, 128, 64, 10]
    epochs = 1500
    learning_rate = 0.01
    activation_function = "relu"
    parameters = neural_network(X_train.T, y_train, dimensions,False, '' ,learning_rate, epochs,activation_function)
    misclassified_indices = test(X_test, parameters, y_test)
    
    


start()