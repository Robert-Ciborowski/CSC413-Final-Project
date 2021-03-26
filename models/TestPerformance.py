# Name: Test Performance
# Author: Robert Ciborowski
# Date: 25/03/2021
# Description: Uses the PerformanceReporter class on a model.
#
#              Do not use this script until PerformanceReporter is finished!

from data_set.DatasetLoader import DatasetLoader
from models.CnnRnnMlp.CnnRnnMlpModel import CnnRnnMlpModel
from models.Hyperparameters import Hyperparameters
from models.PerformanceReporter import PerformanceReporter


def testPerformance():
    datasetLoader = DatasetLoader()
    trainData, trainLabels = datasetLoader.load(path="../data_set/final-train-dataset.csv", shuffle=False)
    testData, testLabels = datasetLoader.load(path="../data_set/final-test-dataset.csv", shuffle=False)
    model = CnnRnnMlpModel(tryUsingGPU=True)

    # Hyperparameters!
    learningRate = 0.003
    epochs = 500
    batchSize = 40
    decayRate = 0.005
    decayStep = 1.0
    dropout = 0.1
    model.setup(Hyperparameters(learningRate, epochs, dropout, batchSize,
                                decayRate=decayRate, decayStep=decayStep))
    model.createModel()
    model.load()
    predictions = model.predict(trainData, concatenate=True)

    reporter = PerformanceReporter()
    reporter.reportOnPoorPredictions(predictions, trainLabels, errorForPoor=0.01)

    print("The model's performance on the test set.")
    reporter.reportPerformanceOnDataset(model, testData, testLabels)

if __name__ == "__main__":
    testPerformance()
