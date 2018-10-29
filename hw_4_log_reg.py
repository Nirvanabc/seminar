
def sigmoid(z):
    # convert input to a numpy array
    z = np.array(z)
    return 1/(1 + 1/np.exp(z))

def costFunction(theta, X, y):
    m = y.size  # number of training examples
    grad = np.zeros(theta.shape)
    h = sigmoid(np.dot(X, theta))
    J = 1/m * np.sum(-np.multiply(y, np.log(h)) -
                     np.multiply((1-y),
                                 np.log(1 - h)))
    transp = X.T
    grad = np.array(1/m * np.dot(X.T, (h - y)))
    return J, grad

class LogisticRegression:
     def fit():
         print ("fit, not written\n")
     def predict_proba():
         print ("predict_proba, not written\n")
     def predict():
         print ("predict, not written\n")
