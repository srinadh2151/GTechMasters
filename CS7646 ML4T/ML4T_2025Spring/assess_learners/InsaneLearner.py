import BagLearner as bl
import LinRegLearner as lrl
import numpy as np

class InsaneLearner:
    def __init__(self, verbose=False):
        self.verbose = verbose
        self.learners = [bl.BagLearner(learner=lrl.LinRegLearner, kwargs={}, bags=20, boost=False, verbose=verbose) for _ in range(20)]
        self.bags = 20

    def add_evidence(self, data_x, data_y):
        for learner in self.learners:
            learner.add_evidence(data_x, data_y)

    def query(self, points):
        predictions = [learner.query(points) for learner in self.learners]
        return np.mean(predictions, axis=0)