import numpy as np

class KNNClassifier:
    def __init__(self, k):
        self.k = k
        self.X_train = None
        self.y_train = None
    
    def euclidean_distance(self, x1, x2):
        return np.sum(np.dot((x1 - x2).T, (x1 - x2)))
    
    def fit(self, X, y):
        self.X_train = X 
        self.y_train = y

    def predict(self, X):
        predictions = []

        for x in X:
            distances = [self.euclidean_distance(x, X[i, :]) for i in range(X.shape[0])]
            
            # get top k indices by ordered by distances
            top_k_indices = np.argsort(distances)[:self.k]

            k_nearest_labels = [self.y_train[i] for i in top_k_indices]                

            # setting the prediction as the label having the majority vote
            predictions.append(max(set(k_nearest_labels), key = k_nearest_labels.count))
        return predictions
    def evaluate(self, X, y):
        y_pred = self.predict(X)
        return np.mean(y == y_pred)
    
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

# Load the iris dataset
iris = load_iris()
X = iris.data
y = iris.target
# Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the KNN Classifier
model = KNNClassifier(k=5)
model.fit(X_train, y_train)

# Evaluate the model
print("Accuracy:", model.evaluate(X_test, y_test))



        