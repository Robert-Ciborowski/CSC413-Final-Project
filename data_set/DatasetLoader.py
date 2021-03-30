# Name: Dataset Loader
# Author: Robert Ciborowski
# Date: 25/03/2021
# Description: Loads our datasets (either for training or testing) into
#              convenient numpy arrays.

import csv
from util.Constants import INPUT_CHANNELS, SAMPLES_OF_DATA_TO_LOOK_AT
import numpy as np

class DatasetLoader:
    def __init__(self):
        pass

    def load(self, path="../data_set/final-train-dataset.csv", shuffle=False):
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
                    # Get one time point's data, including indicators:
                    entries.append([float(row[i + j * SAMPLES_OF_DATA_TO_LOOK_AT]) for j in range(INPUT_CHANNELS)])

                data.append(np.array(entries))
                label = float(row[SAMPLES_OF_DATA_TO_LOOK_AT * INPUT_CHANNELS])
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
