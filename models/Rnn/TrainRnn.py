from data_set.DatasetLoader import DatasetLoader
from models.Rnn.Rnn_Model import RnnModel
from models.Hyperparameters import Hyperparameters
from util.Constants import BINARY_PREDICTION


def train():
    # We train different models to predict different outputs.
    # 0 = 15th percentile, 1 = 25th percentile, 2 = 35th percentile,
    # 3 = 50th percentile, 4 = 65th percentile, 5 = 75th percentile,
    # 6 = 85th percentile
    outputToPredict = 0

    # This represents if we are predicting a binary output (e.g. tomorrow's 50th
    # percentile > today's mean) or if we are predicting a percentage (e.g.
    # tomorrow's 50th percentile is 97% of today's mean)
    binary = BINARY_PREDICTION

    print("Loading dataset...")
    datasetLoader = DatasetLoader()
    # For our outputs, only put in the 15th percentile as the output.
    # Note: THIS IS ONLY USING THE TOP INDICATORS!
    data, labels = datasetLoader.load(shuffle=False, path="../../data_set/final-train-dataset.csv",
                                      onlyLabelToUse=outputToPredict, useOnlyBestIndicators=False,
                                      binary=binary)

    # Hyperparameters!
    learningRate = 0.0005
    epochs = 300
    batchSize = 64
    dropout = 0.02
    # Not currently in use:
    # decayRate = 0.03
    # decayStep = 1.0

    # The model name affects the export file name.
    model = RnnModel(tryUsingGPU=True, binary=binary)
    model.setup(Hyperparameters(learningRate, epochs, dropout, batchSize))
    # If we want to predict all the percentiles at once:
    # model.createModel(OUTPUT_CHANNELS, generateGraph=False)
    # If we only want to predict one of the percentiles:
    model.createModel(generateGraph=False)
    # model.load()
    epochs, hist = model.trainModel(data, labels, 0.1)
    listOfMetricsToPlot = model.listOfMetrics
    model.plotCurve(epochs, hist)
    model.save()

    # Test the model on test data.
    testData, testLabels = datasetLoader.load(shuffle=False, path="../../data_set/final-test-dataset.csv",
                                              onlyLabelToUse=outputToPredict, useOnlyBestIndicators=False)
    model.evaluate(testData, testLabels)
    print(model.predict(testData[50:51, :, :]))
    print(testData.shape[0])

    # If you want to generate predictions for a whole batch of data, do this:
    # multiple = model.predictMultiple(testData)
    # print(multiple)


if __name__ == "__main__":
    train()
