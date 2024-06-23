

import numpy as np

class LinearRegression:
    def __init__(self, learning_rate=0.01, num_iterations=1000):
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations

    def cost_function(self, X, y):
        # cost_fucntion = sum(yi*log(hi) + (1-yi)log(1-hi))
        m = X.shape[0]
        h = np.dot(X, self.weights) + self.bias
        return (-1/m)*(np.sum(y*np.log(h) + (1-y)*np.log(1-h)))
        

    def gradient(self, X, y):
        # dw = dL/dw = (1/m)*(predictions - y).X
        m = X.shape[0]
        preds = np.dot(X, self.weights) + self.bias
        dw = (1/m)*(np.dot(X.T, preds - y))
        db = (1/m)*(np.sum(preds - y))
        return dw, db


    def train(self, X, y):
        

        self.weights = np.zeros(X.shape[1])
        self.bias = 0
        for i in range(self.num_iterations):
            dw, db = self.gradient(X, y)
            self.weights -= self.learning_rate*dw
            self.bias -= self.learning_rate*dw

            

    def predict(self, X):
        return np.dot(X, self.weights) + self.bias
    
    def mean_squared_error(X, y):
        y_pred = self.predict(X)
        return np.mean((y_pred - y)**2)