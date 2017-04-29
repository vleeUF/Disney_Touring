import pandas as pd
import numpy as np


# Import csv and convert to pandas DataFrame then to numpy array
def read_data(file_path):
    df = pd.read_csv(file_path, header=0, low_memory=False)
    df = df.dropna(thresh=0.8 * len(df), axis=1)
    data = df.as_matrix()
    filter(' ', data)
    return data


# Split data into training and test data sets
def split_data(file_path):
    data = read_data(file_path)
    np.random.shuffle(data)
    train, test = data[:80, :], data[80:, :]
    return train, test


def extract_features(batch_size, data):
    features = []
    wait_times = []
    dates = []
    times = []
    for x in data:
        for idx, element in enumerate(x):
            if idx == 0:
                wait_times.append(element)
            elif idx == 3:
                dates.append(element)
            elif idx == 4:
                times.append(element)
            else:
                features.append(element)
    return Batches(batch_size, features, wait_times, dates, times)


class Batches(object):
    def __init__(self, batch_size, features, wait_times, dates, times, shuffle=True):
        assert batch_size <= len(features)
        self.batch_size = batch_size
        self.features = features
        self.wait_times = wait_times
        self.dates = dates
        self.times = times
        self.count = len(self.features)
        self.indexes = list(range(self.count))  # used in shuffling
        self.current_index = 0
        self.num_batches = int(self.count / self.batch_size)
        self.shuffle = shuffle
        self.reset()

    def next_batch(self):
        assert self.has_next_batch()
        from_, to = self.current_index, self.current_index + self.batch_size
        cur_indexes = self.indexes[from_:to]
        features, wait_times, dates, times = zip(*[[self.features[i], self.wait_times[i],
                                                    self.dates[i], self.times[i]] for i in cur_indexes])
        self.current_index += self.batch_size
        return features, wait_times, dates, times

    def has_next_batch(self):
        return self.current_index + self.batch_size <= self.count

    def reset(self):
        self.current_index = 0
        if self.shuffle:
            np.random.shuffle(self.indexes)

