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
    learningRate = 0.0001
    epochs = 600
    batchSize = 250
    decayRate = 0.03
    decayStep = 1.0
    dropout = 0.1

    model = CnnRnnMlpModel(tryUsingGPU=True)
    model.setup(Hyperparameters(learningRate, epochs, dropout, batchSize,
                                decayRate=decayRate, decayStep=decayStep))
    model.createModel()
    # model.load()
    epochs, hist = model.trainModel(data, labels, 0.15)
    listOfMetricsToPlot = model.listOfMetrics
    # model.plotCurve(epochs, hist, listOfMetricsToPlot)
    model.save()

    # Test the model on test data.
    testData, testLabels = datasetLoader.load(shuffle=False, path="../data_set/final-test-dataset.csv")
    model.evaluate(testData, testLabels)
    print(model.predict(testData[627:628,:,:]))
    print(testData.shape[0])

    multiple = model.predictMultiple(testData)
    # print(multiple)


if __name__ == "__main__":
    train()
