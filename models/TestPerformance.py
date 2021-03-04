from data_set.DatasetLoader import DatasetLoader
from models.CnnRnnMlpModel import CnnRnnMlpModel
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
    model.loadWeights()
    predictions = model.predict(trainData, concatenate=True)

    reporter = PerformanceReporter()
    reporter.reportOnPoorPredictions(predictions, trainLabels, errorForPoor=0.01)
    reporter.reportPerformanceOnDataset(model, testData, testLabels)

if __name__ == "__main__":
    testPerformance()
