import numpy as np

def sigmoid(Z): return 1./(1.+np.exp(-Z))
def softmax(Z): return np.exp(Z)/np.exp(Z).sum()
def softmax_batched(Z): return np.exp(Z) / np.sum(np.exp(Z), axis=1, keepdims=True)

def initialize_parameters():
   
    W1 = np.random.randn(300,784) * 0.01
    b1 = np.zeros((300,1)) 
    W2 = np.random.randn(10,300) * 0.01
    b2 = np.zeros((10,1)) 
    
    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2}
    
    return parameters

def forward_propagation(X, parameters, batched):
    
    W1, b1 = parameters["W1"],parameters["b1"]
    W2, b2 = parameters["W2"],parameters["b2"]
    
    Z1 = W1@X + b1
    A1 = sigmoid(Z1)
    Z2 = W2@A1 + b2
    if batched:
        A2 = softmax_batched(Z2)
    else:
        A2 = softmax(Z2)
    
    cache = {"Z1": Z1,
             "A1": A1,
             "Z2": Z2,
             "A2": A2}
    
    return A2, cache

def compute_cost(A2, Y):
    return (- np.log(A2)@Y.T).flatten()[0]

def compute_cost_stable(A2, Y, batched):
    A2 = np.clip(A2, 1e-12, 1. - 1e-12)
    m = A2.shape[0] if batched == True else 1.
    ce = -np.sum(Y*np.log(A2+1e-9))/m
    return ce


def backward_propagation(parameters, cache, X, Y, batched):
   
    m = X.shape[0] if batched == True else 1.
    
   
    W1, W2 = parameters["W1"], parameters["W2"]
   
    A1, A2 = cache["A1"], cache["A2"]
   
    db2 = (A2-Y).mean(keepdims=True)
    if batched:
        dW2 = (1/m)*(A2-Y)@A1.reshape(m,1,300)
    else:
        dW2 = (1/m)*(A2-Y)@A1.T
    dZ2 = (A2-Y)
    dgZ1 = A1*(1-A1)
    dZ1 = W2.T@dZ2*dgZ1
    db1 = dZ1.mean(axis=1, keepdims=True)#-1
    
    if batched:
        dW1 = ((1/m)*dZ1)@X.reshape(m,1,784)
    else:
        dW1 = ((1/m)*dZ1)@X.T

    
    grads = {"dW1": dW1,
             "db1": db1,
             "dW2": dW2,
             "db2": db2}
    
    return grads

def update_parameters(parameters, grads, batched, learning_rate = 0.01):
   
    W1,W2,b1,b2 = parameters["W1"], parameters["W2"], parameters["b1"], parameters["b2"]
   
    dW1, db1, dW2, db2 = grads["dW1"], grads["db1"], grads["dW2"], grads["db2"]
    if batched:
        W1 -= learning_rate * grads["dW1"].mean(axis=0)
        W2 -= learning_rate * grads["dW2"].mean(axis=0)
        b1 = b1 - learning_rate * grads["db1"].mean(axis=0)
        b2 -= learning_rate * grads["db2"].mean(axis=0)
    else:
        W1 -= learning_rate * grads["dW1"]
        W2 -= learning_rate * grads["dW2"]
        b1 = b1 - learning_rate * grads["db1"]
        b2 -= learning_rate * grads["db2"]
        

    
    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2}
    
    return parameters
