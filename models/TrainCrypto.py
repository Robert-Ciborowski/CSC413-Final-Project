import numpy as np
from util.Constants import SAMPLES_OF_DATA_TO_LOOK_AT
from models.CnnRnnMlpModel import CnnRnnMlpModel
from models.Hyperparameters import Hyperparameters
import csv

def train():
    print("Loading dataset...")

    data = []
    labels = []

    with open('../data_set/final-dataset.csv') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        skip = True

        for row in reader:
            if skip:
                skip = False
                continue

            entries = []

            for i in range(SAMPLES_OF_DATA_TO_LOOK_AT):
                entries.append((float(row[i]), float(row[i + SAMPLES_OF_DATA_TO_LOOK_AT]),
                                float(row[i + SAMPLES_OF_DATA_TO_LOOK_AT * 2]),
                                float(row[i + SAMPLES_OF_DATA_TO_LOOK_AT * 3]),
                                float(row[i + SAMPLES_OF_DATA_TO_LOOK_AT * 4]),
                                float(row[i + SAMPLES_OF_DATA_TO_LOOK_AT * 5])))

            data.append(np.array(entries))
            labels.append(float(row[SAMPLES_OF_DATA_TO_LOOK_AT * 6]))

    indices = [i for i in range(len(data))]
    np.random.shuffle(indices)
    labels = np.array([labels[i] for i in indices])
    data = np.array([data[i] for i in indices])

    # We want the validation data to contain patterns not in the train data,
    # so we DO NOT shuffle the dataset!
    # labels = np.array(labels)
    # data = np.array(data)

    # Hyperparameters!
    learningRate = 0.003
    epochs = 6000
    batchSize = 40
    decayRate = 0.005
    decayStep = 1.0

    model = CnnRnnMlpModel(tryUsingGPU=True)
    model.setup(Hyperparameters(learningRate, epochs, batchSize,
                                decayRate=decayRate, decayStep=decayStep))
    model.createModel()
    epochs, hist = model.trainModel(data, labels, 0.15)
    list_of_metrics_to_plot = model.listOfMetrics
    model.plotCurve(epochs, hist, list_of_metrics_to_plot)
    model.exportWeights()


if __name__ == "__main__":
    train()
