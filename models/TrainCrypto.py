import numpy as np

from data_set.DatasetLoader import DatasetLoader
from util.Constants import SAMPLES_OF_DATA_TO_LOOK_AT
from models.CnnRnnMlpModel import CnnRnnMlpModel
from models.Hyperparameters import Hyperparameters
import csv

def train():
    print("Loading dataset...")

    datasetLoader = DatasetLoader()
    data, labels = datasetLoader.load(shuffle=False)

    # Hyperparameters!
    learningRate = 0.001
    epochs = 300
    batchSize = 600
    decayRate = 0.03
    decayStep = 1.0
    dropout = 0.1

    model = CnnRnnMlpModel(tryUsingGPU=False)
    model.setup(Hyperparameters(learningRate, epochs, dropout, batchSize,
                                decayRate=decayRate, decayStep=decayStep))
    model.createModel()
    epochs, hist = model.trainModel(data, labels, 0.15)
    list_of_metrics_to_plot = model.listOfMetrics
    model.plotCurve(epochs, hist, list_of_metrics_to_plot)
    model.save()


if __name__ == "__main__":
    train()
