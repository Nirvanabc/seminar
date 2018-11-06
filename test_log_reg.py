import os
import sys
sys.path.append('/home/nirvana/venv/lib/python3.5/site-packages')
import numpy as np
from matplotlib import pyplot
from scipy import optimize
import utils

def plotData(X, y):
    fig = pyplot.figure()
    pos = y == 1
    neg = y == 0
    pyplot.plot(X[pos, 0], X[pos, 1], 'k*', lw=2, ms=10)
    pyplot.plot(X[neg, 0], X[neg, 1], 'ko', mfc='r', ms=8, mec='k', mew=1)

def sigmoid(z):
    z = np.array(z)
    g = 1/(1 + np.exp(-z))
    return g

def costFunction(theta, X, y):
    m = y.size  # number of training examples
    grad = np.zeros(theta.shape)
    h = sigmoid(np.dot(X, theta))
    J = 1/m * np.sum(-np.multiply(y, np.log(h)) - np.multiply((1-y), np.log(1 - h)))
    transp = X.T
    grad = np.array(1/m * np.dot(X.T, (h - y)))
    return J, grad


# data = np.loadtxt(os.path.join('Data', 'ex2data1.txt'), delimiter=',')
data = np.loadtxt(os.path.join('Data', 'ex2data1.txt'), delimiter=',')
X, y = data[:, 0:2], data[:, 2]

# Add intercept term to X
m, n = X.shape
initial_theta = np.zeros(n+1)
X = (X - np.mean(X, axis=0))/X.std(axis=0)
X = np.concatenate([np.ones((m, 1)), X], axis=1)

LR = 0.5
NUM_ITER = 30
THRESHOLD = 0.5
LAMBDA = 0.5
BETTA = 0.9

class Logistic_Regression:
    def __init__(self, lr=LR, num_iter=NUM_ITER, lambda_=LAMBDA, b=BETTA):
        self.lr = lr
        self.num_iter = num_iter
        self.lambda_ = lambda_
        self.b = b
        self.loss_list = []

    def sigmoid(self, z):
        z = np.array(z)
        g = 1/(1 + np.exp(-z))
        return g

    def cost_function(self, theta, X, y):
        m = y.size  # number of training examples
        J = 0
        grad = np.zeros(theta.shape)
        h = self.sigmoid(np.dot(X, theta))
        J = 1/m * np.sum(
            -np.multiply(y, np.log(h)) - np.multiply((1-y), np.log(1 - h)))+\
            self.lambda_/(2*m)*np.sum(theta[1:]*theta[1:])
        transp = X.T
        grad = 1.0/m * np.dot(transp, (h - y)) + self.lambda_ / m * theta
        grad[0] = 1/m * np.sum(np.multiply((h - y), transp[0]))
        return J, grad

    def fit(self, X_data, y, theta):
        m = y.size
        theta = np.random.uniform(size=(X.shape[1],))
        loss_array = []
        for epoch in range(self.num_iter*10):
            loss, grad = self.cost_function(self.theta, X, y)
            # print(grad.shape)
            self.theta -= self.lr * grad
            # print("step: {}, loss: {} ".format(i, loss))
            loss_array.append(loss)
        self.loss_list = loss_array
    
    def predict_proba(self, X):
        return self.sigmoid(np.dot(X, self.theta))

    def predict(self, X, threshold=THRESHOLD):
        m = X.shape[0] # Number of training examples
        p = np.zeros(m)
        for i in range(m):
            if self.predict_proba(X[i]) >= threshold:
                p[i] = 1
        return p

    
model = Logistic_Regression(lr=LR,
                            num_iter=NUM_ITER,
                            lambda_=LAMBDA,
                            b=BETTA)     
model.fit(X, y)
y_pred = model.predict(X)
print(model.theta)


def gradientDescent(X, y, theta, alpha, num_iters):
    m = y.shape[0]  # number of training examples
    theta = theta.copy()
    J_history = []
    for _ in range(num_iters):
        J, grad = costFunction(theta, X, y)
        theta = theta - alpha * grad
        J_history.append(computeCost(X, y, theta))
    return theta, J_history
