import numpy as np
from DTLearner import DTLearner
from RTLearner import RTLearner

class BagLearner:
    def __init__(self, learner, kwargs={}, bags=20, boost=False, verbose=False):
        self.learner = learner
        self.kwargs = kwargs
        self.bags = bags
        self.boost = boost
        self.verbose = verbose
        self.learners = [learner(**kwargs) for _ in range(bags)]

    def author(self):
        """
        :return: The GT username of the student
        :rtype: str
        """
        return "snidadana3"  # replace tb34 with your Georgia Tech username

    def add_evidence(self, data_x, data_y):
        """ Add training data to learner """
        for learner in self.learners:
            print("Starting add_evidence in BagLearner for learner: ", learner)
            # Bootstrap sample
            indices = np.random.choice(data_x.shape[0], data_x.shape[0], replace=True)
            bootstrap_x = data_x[indices]
            bootstrap_y = data_y[indices]
            learner.add_evidence(bootstrap_x, bootstrap_y)
        print("Completed add_evidence in BagLearner")
    
    def query(self, points):
        """ Estimate a set of test points given the model we built """
        print("Starting query in BagLearner")
        predictions = np.array([learner.query(points) for learner in self.learners])
        print("Completed query in BagLearner")
        return np.mean(predictions, axis=0)

