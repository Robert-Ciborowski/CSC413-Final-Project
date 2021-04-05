# Name: Dataset Loader
# Author: Robert Ciborowski
# Date: 25/03/2021
# Description: Loads our datasets (either for training or testing) into
#              convenient numpy arrays.

import csv
from util.Constants import INPUT_CHANNELS, OUTPUT_CHANNELS, \
    SAMPLES_OF_DATA_TO_LOOK_AT
import numpy as np

class DatasetLoader:
    def __init__(self):
        pass

    def load(self, path="../data_set/final-train-dataset.csv", shuffle=False, onlyLabelToUse=None):
        """
        Load a dataset into a numpy array for features and a numpy array for labels.
        :param path: dataset file path
        :param shuffle: whether the dataset should be shuffled
        :param onlyLabelToUse: if set, the function will only put this label into the dataset.
                                 Set this to 0 if you want to only have the 15th percentile
                                 in the labels, for example.
        :return: two numpy arrays: features and labels.
        """
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

                if onlyLabelToUse is not None:
                    label = [float(row[SAMPLES_OF_DATA_TO_LOOK_AT * INPUT_CHANNELS + onlyLabelToUse])]
                else:
                    label = [float(row[SAMPLES_OF_DATA_TO_LOOK_AT * INPUT_CHANNELS + j]) for j in range(OUTPUT_CHANNELS)]

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
