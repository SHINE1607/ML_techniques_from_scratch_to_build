"""
Here's a summary of how the decision tree algorithm works:
* Root Node: The algorithm starts with the entire dataset at the root node. The root node represents the complete dataset.
* Feature Selection: The algorithm selects the best feature to split the data. This is done by evaluating a metric like Gini impurity or information gain for each feature and selecting the one that provides the maximum information gain or minimum impurity.
* Splitting: Based on the selected feature, the dataset is split into subsets. Each subset corresponds to a branch of the tree.
* Recursive Splitting: The splitting process is then repeated recursively on each of the subsets. The algorithm selects the best feature to split each subset and creates new branches.
* Stopping Criteria: The recursive splitting continues until a stopping criterion is met, such as:
    * All instances in a node belong to the same class (for classification problems).
    * The maximum depth of the tree is reached.
    * The number of instances in a node is less than a minimum threshold.
    * No further split can improve the impurity measure.
* Leaf Nodes: When the stopping criterion is met, the algorithm creates a leaf node. Leaf nodes represent the final predictions or class labels.
* Prediction: To make a prediction for a new instance, the algorithm traverses the tree from the root node to a leaf node, making decisions at each internal node based on the feature values of the new instance.
"""


# train(X, y) -> should define the root node 
# build_tree(X, y) -> 

import numpy as np

class DecisionTreeClassifier:

    def __init__(self, max_depth=5):
        self.max_depth = max_depth
        self.root = None
    
    def gini(self):
        """
        Gini impurity computes the gini impurity of a set of labels
        gini impurity = 1 - sum(perc_countof value**2)
        """
        n = len(y)
        p = [y.count(i) / n for i in range(set(y))]
        return 1 - sum([p[i] ** 2 for i in set(y)])

    def entropy(self, y):
        """
        Entropy function impurity computes the gini impurity of a set of labels
        entropy = sum((perc_count of value)*log2((perc_count of value)))
        """
        n = len(y)
        p = [y.count(i) / n for i in range(set(y))]
        return -sum(p[i]*np.log2(p[i]) for i in set(y))


    def best_split(self, X, y):
        # traverse through each feature and through each threshold values and check which split is having the max gini and entropy metric

        n_samples, n_features = X.shape
        best_gini = float('inf')
        best_entropy = float('inf')

        for i in range(n_features):
            values = np.unique(X[:, i])
            for j in range(len(values) - 1):
                threshold = (values[i] + values[i + 1])/2
                
                left_X, left_y, right_X, right_y = self.split(X, y, i, threshold)
                gini_left = self.gini(left_X, left_y)
                entropy_left = self.entropy(left_X, left_y)

                gini_right = self.gini(right_X, right_y)
                entropy_right = self.entropy(right_X, right_y)

                if gini_left + gini_right < best_gini:
                    best_gini = gini_left + gini_right
                    best_split = (i, threshold)
                
                if entropy_left + entropy_right < best_gini:
                    best_gini = entropy_left + entropy_right
                    best_split = (i, threshold)
        
        return best_split
        


    def split(self, X, y, feature_idx, threshold):
        """
        The split function splits the data based on a agiven feature and threshold
        """
        
        left_X = X[X[:, feature_idx] <= threshold]
        left_y = y[X[:, feature_idx] <= threshold]
        right_X = X[X[:, feature_idx] > threshold]
        right_y = X[X[:, feature_idx] > threshold]
        return left_X, left_y, right_X, right_y

    def build_tree(self, X, y):
        """
        The build Tree function builds the decision tree
        """
        if len(set(y)) == 1:
            return y[0]
    
        if self.max_depth == 0:
            return np.argmax(np.bincount(y))
        
        best_split = self.best_split(X, y)  # find the best split for the given data and target
        feature, threshold = best_split

        left_X, left_y, right_X, right_y = self.split(X, y, feature, threshold)
        # recursively calling the build tree functions for the left and right splits
        left_tree = self.build_tree(left_X, left_y)
        right_tree = self.build_tree(right_X, right_y)
        
        return {
            "feature": feature, 
            "threshold": threshold, 
            "left": left_tree, 
            "right": right_tree, 
        }


    def predict(self, X, y):
        """
        Predict function makes predictions using the decision tree
        """
        predictions = []

        for sample in X:
            node = self.root # starting the predictions from the startin node
            while isinstance(node, dict):

                if sample[node['feature'] <= node['threshold']]:
                    node = node["left"]
                else:
                    node = node["right"]
            predictions.append(node)
        return predictions
        

    def train(self, X, y):
        """
        train function trains the decision tree using the input dataset
        """
        self.root = self.build_tree(X, y)

    def evaluate(self, X):
        """
        Evaluate function evaluates the decision tree
        """
        predictions = self.predict(X)
        accuracy = np.mean([p == y for p, y in zip(predictions, y)])
        return accuracy