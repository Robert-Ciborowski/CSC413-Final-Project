from data_set.DatasetLoader import DatasetLoader
from models.CnnRnnMlpModel import CnnRnnMlpModel
from models.Hyperparameters import Hyperparameters
from models.PerformanceReporter import PerformanceReporter


def testPerformance():
    datasetLoader = DatasetLoader()
    data, labels = datasetLoader.load(shuffle=False)

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
    predictions = model.predict(data, concatenate=True)

    reporter = PerformanceReporter()
    reporter.reportOnPoorPredictions(predictions, labels, errorForPoor=0.01)


if __name__ == "__main__":
    testPerformance()
