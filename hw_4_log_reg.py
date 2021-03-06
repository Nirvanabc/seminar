import numpy as np

LR = 0.5
NUM_ITER = 1000
THRESHOLD = 0.5
LAMBDA = 0.01
BETTA = 0.98


class Logistic_Regression:
    def __init__(self, lr=LR, num_iter=NUM_ITER, lambda_=LAMBDA, b=BETTA):
        self.lr = lr
        self.num_iter = num_iter
        self.lambda_ = lambda_
        self.b = b
        self.loss_list = []
        
        
    def sigmoid(self, z):
        z = np.array(z)
        # print(z)
        g = 1/(1 + np.exp(-z))
        return g

        
    def cost_function(self, theta, X, y):
        '''
        m samples, n features
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
        
        grad = 1.0/m * np.dot(transp, (h - y)) + self.lambda_ / m * theta
        grad[0] = 1/m * np.sum(np.multiply((h - y), transp[0]))
        return J, grad

    
    def fit_SGD(self, X, y):
        """
        Stochastic gradient descent
        """
        m = y.size
        self.theta = np.random.uniform(size=(X.shape[1],))
        loss_array = []
        tmp_grad = 0
        # for epoch in range(self.num_iter):
        for sample in range(X.shape[0]):
            X_i = X[sample,:].reshape(1, X.shape[1])
            y_i = y[sample]
            loss, grad = self.cost_function(self.theta, X_i, y_i)
            grad = self.b * tmp_grad + (1 - self.b) * grad
            tmp_grad = grad
            self.theta -= self.lr * grad
            # print("step: {}, loss: {} ".format(epoch, loss))
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


#     def fit(self, X, y):
#         """
#         valilla gradient descent
#         """
#         m = y.size
#         self.theta = np.random.uniform(size=(X.shape[1],))
#         loss_array = []
#         for epoch in range(self.num_iter*10):
#             loss, grad = self.cost_function(self.theta, X, y)
#             self.theta -= self.lr * grad
#             # print("step: {}, loss: {} ".format(epoch, loss))
#             loss_array.append(loss)
#         self.loss_list = loss_array

