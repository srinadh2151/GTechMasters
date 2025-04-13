import numpy as np

class RTLearner:
    def __init__(self, leaf_size=1, verbose=False):
        self.leaf_size = leaf_size
        self.verbose = verbose
        self.tree = None
        self.na = -1

    def author(self):
        """
        :return: The GT username of the student
        :rtype: str
        """
        return "snidadana3"  # replace tb34 with your Georgia Tech username

    def add_evidence(self, data_x, data_y):
        """ Add training data to learner """
        data = np.concatenate((data_x, data_y[:, None]), axis=1)
        self.tree = self.build_tree(data)

    def build_tree(self, data):
        """ Build the random tree """
        if self.verbose:
            print(f"Building tree with {data.shape[0]} samples")

        # Base case: if data is less than or equal to leaf_size
        if data.shape[0] <= self.leaf_size:
            if self.verbose:
                print("Creating a leaf node due to leaf size")
            return np.array([[-1, np.mean(data[:, -1]), self.na, self.na]])

        # Base case: if all target values are the same
        if np.all(data[:, -1] == data[0, -1]):
            if self.verbose:
                print("Creating a leaf node due to uniform target values")
            return np.array([[-1, data[0, -1], self.na, self.na]])

        # Randomly select a feature to split on
        best_feature = np.random.randint(0, data.shape[1] - 1)
        split_val = np.median(data[:, best_feature])

        # Check if a valid split is possible
        if np.all(data[:, best_feature] == split_val):
            if self.verbose:
                print("Creating a leaf node due to invalid split")
            return np.array([[-1, np.mean(data[:, -1]), self.na, self.na]])

        # Recursive case: build left and right subtrees
        left_tree = self.build_tree(data[data[:, best_feature] <= split_val])
        right_tree = self.build_tree(data[data[:, best_feature] > split_val])

        root = np.array([[best_feature, split_val, 1, left_tree.shape[0] + 1]])
        return np.vstack((root, left_tree, right_tree))

    def query(self, points):
        """ Estimate a set of test points given the model we built """
        y_pred = np.array([self.query_point(point) for point in points])
        return y_pred
    
    def query_point(self, point):
        """ Estimate a single point given the model we built """
        node = 0
        while self.tree[node, 0] != -1:
            feature = int(self.tree[node, 0])
            split_val = self.tree[node, 1]
            if point[feature] <= split_val:
                node += int(self.tree[node, 2])
            else:
                node += int(self.tree[node, 3])
        return self.tree[node, 1]