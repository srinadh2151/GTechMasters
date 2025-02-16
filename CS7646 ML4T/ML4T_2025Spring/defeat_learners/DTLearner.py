import numpy as np

class DTLearner:
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
        # if data_x.shape[0] <= self.leaf_size or np.all(data_y == data_y[0]):
        #     # Base case: create a leaf node
        #     return np.array([[-1, np.mean(data_y), np.nan, np.nan]])
        #     # ... rest of the code to split the data and recurse

        data = np.concatenate((data_x, data_y[:, None]), axis=1)
        self.tree = self.build_tree(data)

    def build_tree(self, data):
        """ Build the decision tree """
        if data.shape[0] <= self.leaf_size:
            return np.array([[-1, np.mean(data[:, -1]), self.na, self.na]])
        if np.all(data[:, -1] == data[0, -1]):
            return np.array([[-1, data[0, -1], self.na, self.na]])

        # Determine the best feature to split on
        best_feature = self.find_best_feature(data)
        split_val = np.median(data[:, best_feature])

        # Check if the split value actually divides the data
        left_data = data[data[:, best_feature] <= split_val]
        right_data = data[data[:, best_feature] > split_val]

        if left_data.shape[0] == 0 or right_data.shape[0] == 0:
            return np.array([[-1, np.mean(data[:, -1]), self.na, self.na]])

        left_tree = self.build_tree(left_data)
        right_tree = self.build_tree(right_data)

        root = np.array([[best_feature, split_val, 1, left_tree.shape[0] + 1]])
        return np.vstack((root, left_tree, right_tree))

    def find_best_feature(self, data):
        """ Find the best feature to split on """
        features = data.shape[1] - 1
        best_feature = 0
        best_corr = 0
        for feature in range(features):
            corr = np.corrcoef(data[:, feature], data[:, -1])[0, 1]
            if abs(corr) > abs(best_corr):
                best_corr = corr
                best_feature = feature
        return best_feature

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