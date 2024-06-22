import time
from pathlib import Path

import pandas as pd
from sklearn import metrics
from sklearn.preprocessing import MinMaxScaler

import iforest_optimal


def write_list_to_file(data_list):
    with open('results/result.txt', "a+") as f:
        line = ",".join(str(element) for element in data_list) + "\n"
        f.write(line)


def run_without_init_training():
    directory = 'generated_data/'

    for anomaly_threshold in [0.4, 0.5, 0.6]:  # anomaly_threshold
        for window_size in [25, 50, 100, 1000, 1500, 2000]:  # window_size
            for drift_threshold in [0.4, 0.5, 0.6]:
                for filepath in sorted(Path(directory).iterdir()):
                    if filepath.is_file():
                        model = iforest_optimal.IsolationForestStream(anomaly_threshold=anomaly_threshold,
                                                                      window_size=window_size,
                                                                      drift_threshold=drift_threshold)
                        name = filepath.name
                        df = pd.read_csv(filepath, header=None).dropna().to_numpy()

                        max_length = len(df)

                        data = df[:max_length, 0].astype(float)
                        label = df[:max_length, 1].astype(int)

                        # fit the model
                        start_time = time.time()
                        model.fit(data)
                        total_time = time.time() - start_time

                        write_to_file(anomaly_threshold, drift_threshold, label, model, name, total_time, window_size)


def write_to_file(anomaly_threshold, drift_threshold, label, model, name, total_time, window_size, init_training=False):
    score = model.decision_scores_
    score = MinMaxScaler(feature_range=(0, 1)).fit_transform(score.reshape(-1, 1)).ravel()
    auc = metrics.roc_auc_score(label, score)

    write_list_to_file([init_training, name, anomaly_threshold, window_size, drift_threshold, auc, total_time])


def run_with_init_training():
    directory = 'generated_data/'

    for anomaly_threshold in [0.4, 0.5, 0.6]:  # anomaly_threshold
        for window_size in [25, 50, 100, 1000, 1500, 2000]:  # window_size
            for drift_threshold in [0.4, 0.5, 0.6]:
                for filepath in sorted(Path(directory).iterdir()):
                    if filepath.is_file():
                        model = iforest_optimal.IsolationForestStream(anomaly_threshold=anomaly_threshold,
                                                                      window_size=window_size,
                                                                      drift_threshold=drift_threshold)
                        name = filepath.name
                        df = pd.read_csv(filepath, header=None).dropna().to_numpy()

                        max_length = len(df)

                        data = df[:max_length, 0].astype(float)
                        label = df[:max_length, 1].astype(int)

                        # fit the model
                        start_time = time.time()
                        model.fit(data, 1000)  # fit to 1000 first data first
                        total_time = time.time() - start_time

                        write_to_file(anomaly_threshold, drift_threshold, label, model, name, total_time, window_size,
                                      True)
