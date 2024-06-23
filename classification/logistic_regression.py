import numpy as np

class LogisticRegression:
    def __init__(self,learning_rate=0.01, num_iterations=1000):
        self.learning_rate=learning_rate
        self.num_iterations = num_iterations
    
    def sigmoid(self, z):
        """
        Returns the sigmoid function value for the prediction
        """
        return 1 / (1 + np.exp(-(z)))
    def gradient(self, X, y):
        """
        return updates for weights and bias
        graident = dL/dw = (h - y).X
        """
        m = X.shape[0]
        preds = self.sigmoid(np.dot(X, self.weights) + self.bias)
        dw = (1/m)*np.dot(X.T, (preds - y))
        db = (1/m)*np.sum((preds - y))
        return dw, db

    def cost_function(self, X, y):
        
        """
        Predict the total loss value of the model
        cost function = sum((yi)log(hi) + (1-yi)log(1-hi))
        """
        m = X.shape[0]
        h = self.sigmoid(np.dot(X.shape, self.weights) + self.bias)
        cost = (-1/m)*np.sum(y*np.log(h) + (1 - y)*np.log(1 - h))
        return cost

    def fit(self, X, y):
        
        """
        Fit method fit the logistic regression model to the trainin data
        """
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0
        for i in range(self.num_iterations):
            dw, db = self.gradient(X, y)
            self.weights -= dw * self.learning_rate
            self.bias -= db * self.learning_rate
            if i%100 == 0:
                print(f"Num iters: {i} | Current cost function: {self.cost_function(X, y)} | | Current Accuracy: {self.accuracy_score(X, y)}")

            

    def predict(self, X):
        """
        Returns the predictions for the given dataset
        """
        preds =  self.sigmoid(np.dot(X, self.weights) + self.bias)
        return (preds > 0.5).astype(int)

        

    def accuracy_score(self, X, y):
        y_score = self.predict(X)

        return np.mean(y_score == y)

## Exmaple Usage
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split

X, y = make_blobs(n_samples=1000, centers=2, n_features=2, random_state=0) # create the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)  # Split the data

print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)
# Train the model
model = LogisticRegression()
model.fit(X_train, y_train)


# Evaluate the model
print("Training Accuracy: ", model.accuracy_score(X_train, y_train))
print("Test Accuracy: ", model.accuracy_score(X_test, y_test))



