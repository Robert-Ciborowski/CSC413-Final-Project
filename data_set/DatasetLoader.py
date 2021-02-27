import csv

from util.Constants import SAMPLES_OF_DATA_TO_LOOK_AT
import numpy as np

class DatasetLoader:
    def __init__(self):
        pass

    def load(self, path="../data_set/final-dataset.csv", shuffle=False):
        data = []
        labels = []

        with open(path) as csvfile:
            reader = csv.reader(csvfile, delimiter=',')
            skip = True

            for row in reader:
                if skip:
                    skip = False
                    continue

                entries = []

                for i in range(SAMPLES_OF_DATA_TO_LOOK_AT):
                    entries.append((float(row[i]),
                                    float(row[i + SAMPLES_OF_DATA_TO_LOOK_AT]),
                                    float(row[
                                              i + SAMPLES_OF_DATA_TO_LOOK_AT * 2]),
                                    float(row[
                                              i + SAMPLES_OF_DATA_TO_LOOK_AT * 3]),
                                    float(row[
                                              i + SAMPLES_OF_DATA_TO_LOOK_AT * 4]),
                                    float(row[
                                              i + SAMPLES_OF_DATA_TO_LOOK_AT * 5])))

                data.append(np.array(entries))
                label = [float(row[SAMPLES_OF_DATA_TO_LOOK_AT * 6 + i]) for i in
                         range(5)]
                labels.append(label)

        if shuffle:
            indices = [i for i in range(len(data))]
            np.random.shuffle(indices)
            labels = np.array([labels[i] for i in indices])
            data = np.array([data[i] for i in indices])
        else:
            data = np.array(data)
            labels = np.array(labels)

        return data, labels