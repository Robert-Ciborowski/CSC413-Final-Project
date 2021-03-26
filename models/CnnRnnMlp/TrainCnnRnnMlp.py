# Name: Train Model
# Author: Robert Ciborowski
# Date: 25/03/2021
# Description: A script for training a model.

from data_set.DatasetLoader import DatasetLoader
from models.CnnRnnMlp.CnnRnnMlpModel import CnnRnnMlpModel
from models.Hyperparameters import Hyperparameters

def train():
    print("Loading dataset...")
    datasetLoader = DatasetLoader()
    data, labels = datasetLoader.load(shuffle=False, path="../../data_set/final-test-dataset.csv")

    # Hyperparameters!
    learningRate = 0.0001
    epochs = 600
    batchSize = 250
    dropout = 0.1
    # Not currently in use:
    # decayRate = 0.03
    # decayStep = 1.0

    model = CnnRnnMlpModel(tryUsingGPU=True)
    model.setup(Hyperparameters(learningRate, epochs, dropout, batchSize))
    model.createModel(generateGraph=True)
    # model.load()
    epochs, hist = model.trainModel(data, labels, 0.15)
    listOfMetricsToPlot = model.listOfMetrics
    # model.plotCurve(epochs, hist, listOfMetricsToPlot)
    model.save()

    # Test the model on test data.
    testData, testLabels = datasetLoader.load(shuffle=False, path="../../data_set/final-test-dataset.csv")
    model.evaluate(testData, testLabels)
    print(model.predict(testData[627:628,:,:]))
    print(testData.shape[0])

    # If you want to generate predictions for a whole batch of data, do this:
    # multiple = model.predictMultiple(testData)
    # print(multiple)

if __name__ == "__main__":
    train()
