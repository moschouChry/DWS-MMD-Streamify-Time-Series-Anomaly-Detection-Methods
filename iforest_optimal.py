import numpy as np
from sklearn.ensemble import IsolationForest


class IsolationForestStream:
    """
    This is the implementation of an optimal anomaly detection technique for streaming data using isolation forest.
    We use a sliding window approach. This approach entails defining a window size. Once the window is filled with
    data points, an initial iForest model is trained on that specific window, and the corresponding anomaly score is
    stored. Subsequently, new data is continuously fed into the model until the window reaches its capacity again. At
    this point, the anomaly rate within the sliding window is evaluated against a predetermined threshold. If the
    anomaly rate falls below the threshold, the existing iForest model is retained. Conversely, if the anomaly rate
    surpasses the threshold, a retraining process is initiated using the data contained within the current window.
    This iterative process of training and evaluation continues for the entirety of the incoming data stream.
    """
    def __init__(self,
                 window_size=100,
                 n_estimators=50,
                 max_samples="auto",
                 contamination=0.1,
                 max_features=1.,
                 bootstrap=False,
                 n_jobs=1,
                 random_state=None,
                 verbose=0,
                 anomaly_threshold=0.5, drift_threshold=0.5):
        self.anomaly_rate = None
        self.contamination = contamination
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.max_features = max_features
        self.bootstrap = bootstrap
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.verbose = verbose
        self.model_name = 'IForest'
        self.current_time = 0
        self.window_size = window_size
        self.curr_window = []
        self.anomaly_threshold = anomaly_threshold
        self.drift_threshold = drift_threshold
        self.model = None
        self.prev_window = None
        self.decision_scores_ = []
        self.samples_seen = 0
        self.mode_trained = 0

    def fit(self, x_input, init_length=0):
        """
        This is the function to fit the model. With an initial training if init_length is set
        or if not set there no initial training
        :param x_input: the input points
        :param init_length: optional param, the initial length we want the model to be trained
        """
        isolation_forest = IsolationForest(n_estimators=self.n_estimators,
                                           contamination=self.contamination,
                                           max_samples=self.max_samples,
                                           max_features=self.max_features,
                                           bootstrap=self.bootstrap,
                                           n_jobs=self.n_jobs,
                                           random_state=self.random_state,
                                           verbose=self.verbose)
        self.model = isolation_forest

        # if init_length is set then train the IsolationForest model on the first init_length data from the x_input
        if init_length is not None and init_length > 0:
            initial_points = np.array([x_input[:min(len(x_input), init_length)]]).reshape(-1, 1)

            self.model.fit(initial_points)

            self.decision_scores_ = np.concatenate([self.decision_scores_,
                                                    self.model.decision_function(initial_points)])

            remaining_data = np.array([x_input[min(len(x_input), init_length):]]).flatten()

            for i, x in enumerate(remaining_data):
                self.partial_fit(x, True)
        else:
            for i, x in enumerate(x_input):
                self.partial_fit(x)

        if len(self.curr_window) != 0:
            self.decision_scores_ = np.concatenate([self.decision_scores_,
                                                    self.model.decision_function(
                                                        np.array(self.curr_window).reshape(-1, 1))])
        return self

    def partial_fit(self, x, init_training=False):
        """
        This function is used to partially fit the model. For every input point it checks whether the window
        has been filled and decides if it needs to retrain the model or use the current one to calculate the anomaly
        score of the point.
        :param x: the input point
        :param init_training: if the model already had been trained
        """
        if len(self.curr_window) != 0 and len(self.curr_window) == self.window_size:
            # Update the two windows (previous one and current windows)
            self.prev_window = self.curr_window
            self.curr_window = [x]
        else:
            self.curr_window.append(x)
            # self.curr_window = np.concatenate((self.curr_window, x))

        if self.samples_seen % self.window_size == 0 and self.samples_seen != 0:
            # fit the model when the first window has been filled with data
            if init_training is False and self.mode_trained < 1:
                self.model.fit(np.array(self.prev_window).reshape(-1, 1))
                self.mode_trained += 1

            self.anomaly_rate = self.calculate_anomaly_rate(np.array(self.prev_window))  # calculate anomaly rate

            # update the model if the current anomaly rate is higher than the drift_threshold
            if self.anomaly_rate > self.drift_threshold:
                self.update_model(self.prev_window)

            self.decision_scores_ = np.concatenate([self.decision_scores_,
                                                    self.model.decision_function(
                                                        np.array(self.prev_window).reshape(-1, 1))])

        self.samples_seen += 1

    def update_model(self, window):
        """
        This function is used to retrain the Isolation Forest model based on the given window
        :param window: the window to be used for the training
        """
        self.model.fit(np.array(window).reshape(-1, 1))

    def calculate_anomaly_rate(self, data):
        """
        This function is used to calculate the anomaly rate for a window(the given data parameter).
        It uses the score_samples function of the iForest model to get the anomaly score of its point.
        If the anomaly score of a point is greater than the anomaly_threshold value then the point is
        considered an anomaly.
        Then we find the anomaly rate for all the data in the window and return it
        :param data: the input window
        :return: anomaly rate of the input data
        """

        # check if data array is empty
        if len(data) == 0:
            return 0

        # reshape the window
        data = data.reshape(-1, 1)

        # calculate the anomaly scores for all instances in the window
        # invert scores so that outliers come with higher outlier scores
        anomaly_scores = -self.model.score_samples(data)

        # count anomalies based on the threshold
        # if the score is higher than the threshold then there is an anomaly
        num_anomalies = np.count_nonzero(anomaly_scores > self.anomaly_threshold)

        # find the anomaly rate
        anomaly_rate = num_anomalies / len(anomaly_scores)

        return anomaly_rate
