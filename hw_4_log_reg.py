import numpy as np

LR = 0.01
NUM_ITER = 1000
THRESHOLD = 0.5
LAMBDA = 0.5

def sigmoid(z):
    z = np.array(z)
    return 1/(1 + 1/np.exp(z))


def cost_function(theta, X, y, lambda_):
    m = y.size  # number of training examples
    J = 0
    grad = np.zeros(theta.shape)
    h = sigmoid(np.dot(X, theta))
    J = 1/m * np.sum(
        -np.multiply(y, np.log(h)) - np.multiply((1-y), np.log(1 - h)))+\
        lambda_/(2*m)*np.sum(theta[1:]*theta[1:])
    transp = X.T

    grad = 1/m * np.dot(transp, (h - y)) + lambda_/m * theta
    grad[0] = 1/m * np.sum(np.multiply((h - y), transp[0]))
    return J, grad


class Logistic_Regression:
    def __init__(self, lr=LR, num_iter=NUM_ITER, lambda_=LAMBDA):
        self.lr = lr
        self.num_iter = num_iter
        self.lambda_ = lambda_

    def sigmoid(self, z):
        z = np.array(z)
        return 1/(1 + 1/np.exp(z))

        
    def cost_function(self, theta, X, y):
        '''
        theta :  (n+1, ).
        X : (m x n+1)
        y : (m, ).
        
        Returns
        -------
        J : cost
        grad : (n+1, )
        '''
        
        m = y.size  # number of training examples
        J = 0
        grad = np.zeros(theta.shape)
        h = self.sigmoid(np.dot(X, theta))
        J = 1/m * np.sum(
            -np.multiply(y, np.log(h)) - np.multiply((1-y), np.log(1 - h)))+\
            self.lambda_/(2*m)*np.sum(theta[1:]*theta[1:])
        transp = X.T
        
        grad = 1/m * np.dot(transp, (h - y)) + self.lambda_/m * theta
        grad[0] = 1/m * np.sum(np.multiply((h - y), transp[0]))
        return J, grad


    def fit(self, X, y):
        self.theta = np.random.uniform(size=(X.shape[1],))
        loss_array = []
        for i in range(self.num_iter):
            loss, grad = self.cost_function(self.theta, X, y)
            self.theta -= self.lr * grad
            print("step: {}, loss: {} ".format(i, loss))
            loss_array.append(loss)
        

    def predict_proba(self, X):
        return self.sigmoid(np.dot(X, self.theta))

    
    def predict(self, X, threshold=THRESHOLD):

        m = X.shape[0] # Number of training examples
        p = np.zeros(m)
        for i in range(m):
            if self.predict_proba(X) >= 1/2:
                p[i] = 1
        return p
        # return self.predict_proba(X) >= threshold

            

