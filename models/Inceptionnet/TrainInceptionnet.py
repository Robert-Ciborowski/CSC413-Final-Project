# Name: Train Model
# Author: Robert Ciborowski
# Date: 25/03/2021
# Description: A script for training a model.

from data_set.DatasetLoader import DatasetLoader
from models.Inceptionnet.InceptionnetModel import InceptionnetModel
from models.Hyperparameters import Hyperparameters
from util.Constants import OUTPUT_CHANNELS


def train():
    print("Loading dataset...")
    datasetLoader = DatasetLoader()
    # For our outputs, only put in the 15th percentile as the output.
    # Note: THIS IS ONLY USING THE TOP 3 INDICATORS!
    data, labels = datasetLoader.load(shuffle=False, path="../../data_set/final-train-dataset.csv",
                                      onlyLabelToUse=0, useTop3Indicators=True)

    # Hyperparameters!
    learningRate = 0.0002
    epochs = 100
    batchSize = 250
    dropout = 0.0
    # Not currently in use:
    # decayRate = 0.03
    # decayStep = 1.0

    model = InceptionnetModel(tryUsingGPU=True)
    model.setup(Hyperparameters(learningRate, epochs, dropout, batchSize))
    # Predict all percentiles:
    # model.createModel(OUTPUT_CHANNELS, generateGraph=False)
    # Predict one of the percentiles
    model.createModel(1, generateGraph=False)
    # model.load()
    epochs, hist = model.trainModel(data, labels, 0.15)
    # listOfMetricsToPlot = model.listOfMetrics
    model.plotCurve(epochs, hist)
    model.save()

    # Test the model on test data.
    testData, testLabels = datasetLoader.load(shuffle=False, path="../../data_set/final-test-dataset.csv",
                                              onlyLabelToUse=0, useTop3Indicators=True)
    model.evaluate(testData, testLabels)
    print(model.predict(testData[627:628,:,:]))
    print(testData.shape[0])

    # If you want to generate predictions for a whole batch of data, do this:
    # multiple = model.predictMultiple(testData)
    # print(multiple)

if __name__ == "__main__":
    train()
