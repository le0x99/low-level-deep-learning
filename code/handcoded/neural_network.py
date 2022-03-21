from code.handcoded.algebra import *
from code.handcoded.functional import *
from numpy import random

def initialize_parameters():
   
    W1 = random.uniform(size=(2,3))
    b1 = random.randn(3,1) * 0
    W2 = random.uniform(size=(3,2))
    b2 = random.randn(2,1) * 0
    
    parameters = {"W1": W1.tolist(),
                  "b1": transpose(b1.tolist()),
                  "W2": W2.tolist(),
                  "b2": transpose(b2.tolist())}
    
    return parameters
def forward_propagation(X, parameters):
    
    W1, b1 = parameters["W1"],parameters["b1"]
    W2, b2 = parameters["W2"],parameters["b2"]
    # Linear
    Z1 = add(dot(X, W1), b1)
    # Activate
    A1 = sigmoid(Z1)
    # Linear
    Z2 = add(dot(A1, W2), b2)
    # Activate
    A2 = [softmax(Z2)]
    
    cache = {"Z1": Z1,
             "A1": A1,
             "Z2": Z2,
             "A2": A2}
    
    return A2, cache

def compute_cost(A2, Y, parameters):
   

    cost = cross_entropy(A2, Y)
 

    
    
    assert(isinstance(cost, float))
    
    return cost

def backward_propagation(parameters, cache, X, Y):
   

    W1, W2 = parameters["W1"], parameters["W2"]
    A1, A2 = cache["A1"], cache["A2"]
   
    
    dL_dA2 = [[-1/A2[0][_] if Y[0][_]==1. else 0 for _ in range(len(Y[0]))]]
    #dA2_dZ2 = [[A2[0][0]*(1 - A2[0][0]), -A2[0][1]*A2[0][0] ],
    #         [-A2[0][0]*A2[0][1],A2[0][1]*(1 - A2[0][1])]]
    dA2_dZ2 = [[None, None],[None, None]]
    for i in range(2):
        for j in range(2):
            dA2_dZ2[i][j] = A2[0][i]*(1 - A2[0][i]) if i==j else - A2[0][i]*A2[0][j]
    dL_dZ2 = dL_db2 = dot(dL_dA2, dA2_dZ2) 
    dL_dW2 = transpose(dot(transpose(dL_dZ2), A1))
    dZ2_dA1 = transpose(W2)
    dL_dA1 = dot(dL_dZ2, dZ2_dA1)
    dA1_dZ1 = [[A1[0][_]*(1-A1[0][_]) for  _ in range(len(A1[0]))]]
    dL_dZ1 = [[dL_dA1[0][_]*dA1_dZ1[0][_] for  _ in range(len(dA1_dZ1[0]))]]
    dL_db1 = dL_dZ1
    dZ1_dW1 = X
    dL_dW1 = transpose(dot(transpose(dL_dZ1), dZ1_dW1))
    
    grads = {"dW1": dL_dW1,
             "db1": dL_db1,
             "dW2": dL_dW2,
             "db2": dL_db2}
    
    return grads

def update_parameters(parameters, grads, learning_rate = 0.01):
 
    W1,W2,b1,b2 = parameters["W1"], parameters["W2"], parameters["b1"], parameters["b2"]
    dW1, db1, dW2, db2 = grads["dW1"], grads["db1"], grads["dW2"], grads["db2"]
   
    
    for i in range(len(W1)):
        for j in range(len(W1[i])):
            W1[i][j] =  W1[i][j] - learning_rate * dW1[i][j]  
  

    for i in range(len(W2)):
        for j in range(len(W2[i])):
            W2[i][j] =  W2[i][j] - learning_rate * dW2[i][j]  
            
    for i in range(len(b2)):
        for j in range(len(b2[i])):
            b2[i][j] =  b2[i][j] - learning_rate * db2[i][j]  

    for i in range(len(b1)):
        for j in range(len(b1[i])):
            b1[i][j] =  b1[i][j] - learning_rate * db1[i][j] 

        
    
    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2}
    
    return parameters


