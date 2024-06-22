import numpy as np
from sklearn.ensemble import IsolationForest


class IsolationForestNaiveStream:
    """
    This is the implementation of variant 1 described in the project using isolation forest.
    We trained an isolation forest model on each batch of arriving data and calculated the anomaly score.
    """
    def __init__(self,
                 n_estimators=100,
                 max_samples="auto",
                 contamination=0.1,
                 max_features=1.,
                 bootstrap=False,
                 n_jobs=1,
                 random_state=None):
        self.decision_scores_ = []
        self.contamination = contamination
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.max_features = max_features
        self.bootstrap = bootstrap
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.model_name = 'IForest'
        self.current_time = 0
        self.window = None

    def fit(self, x_input, init_length=None, batch_size=None):
        """
        Builds the model and computes the anomaly score.
        It trains the model in batches and for every batch it calculates its anomaly score.
        """
        self.point_list = list(x_input)
        self.decision_scores_ = []

        if (init_length is None) or (batch_size is None):
            print("You must specify a value for init_length, and batch_size")
            return None

        self.init_length = init_length
        self.batch_size = batch_size

        isolation_forest = IsolationForest(n_estimators=self.n_estimators,
                                           contamination=self.contamination,
                                           max_samples=self.max_samples,
                                           max_features=self.max_features,
                                           bootstrap=self.bootstrap,
                                           n_jobs=self.n_jobs,
                                           random_state=self.random_state)
        self.current_time = self.init_length

        # train IForest based on the first data
        curr_points = np.array([self.point_list[:min(len(self.point_list), self.current_time)]]).reshape(-1, 1)
        isolation_forest.fit(curr_points)
        # initialize decision scores array with their anomaly score
        self.decision_scores_ = isolation_forest.decision_function(curr_points)

        # for every remaining data
        while self.current_time < len(self.point_list) - self.batch_size:

            self.current_time = self.current_time + self.batch_size

            if self.current_time < len(self.point_list) - self.batch_size:
                curr_points = (np.array(
                    [self.point_list[self.current_time - self.batch_size:min(len(self.point_list), self.current_time)]])
                               .reshape(-1, 1))
                self.decision_scores_ = (
                    np.concatenate([self.decision_scores_, isolation_forest.decision_function(curr_points)]))
                isolation_forest.fit(curr_points)  # re-train the model based on the new points
            else:
                curr_points = (np.array([self.point_list[self.current_time - self.batch_size:]]).reshape(-1, 1))
                self.decision_scores_ = (
                    np.concatenate([self.decision_scores_, isolation_forest.decision_function(curr_points)]))
                isolation_forest.fit(curr_points)  # re-train the model based on the new points

        return self
