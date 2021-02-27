import numpy as np
from matplotlib import pyplot as plt

class PerformanceReporter:
    def __init__(self):
        pass

    def getEntriesWithErrorGreaterThan(self, predictions, labels, amountGreaterThan=0.01):
        indices = []

        for i in range(predictions.shape[0]):
            error = self._compare(predictions[i, :], labels[i, :])

            if error > amountGreaterThan:
                indices.append(i)

        return indices

    def reportOnPoorPredictions(self, predictions, labels, errorForPoor=0.01):
        poorIndices = self.getEntriesWithErrorGreaterThan(predictions, labels, amountGreaterThan=errorForPoor)
        # badPredictions = predictions[poorIndices, :]
        badLabels = labels[poorIndices, :]
        numBad = badLabels.shape[0]
        total = labels.shape[0]

        mins = np.min(badLabels, axis=0)
        twentyFifths = np.percentile(badLabels, 25, axis=0)
        medians = np.percentile(badLabels, 50, axis=0)
        seventyFifths = np.percentile(badLabels, 75, axis=0)
        maxes = np.max(badLabels, axis=0)

        fig = plt.figure(figsize=(10, 7))
        plt.title("Poorly Predicted Entry Price Min/25/Med/75/Maxes, Error >= "
                  + str(errorForPoor) + ", Poor Predictions=" + str(numBad)
                  + "/" + str(total))
        dataToPlot = []

        for i in range(badLabels.shape[1]):
            dataToPlot.append(badLabels[:, i])

        plt.boxplot(dataToPlot)
        plt.show()

        absDifference = np.abs(badLabels - 1.0)
        fig = plt.figure(figsize=(10, 7))
        plt.title("Poorly Predicted Entry abs(Min/25/Med/75/Max - 1.0), Error >= "
                  + str(errorForPoor) + ", Poor Predictions=" + str(numBad)
                  + "/" + str(total))
        dataToPlot = []

        for i in range(absDifference.shape[1]):
            dataToPlot.append(absDifference[:, i])

        plt.boxplot(dataToPlot)
        plt.show()

        return mins, twentyFifths, medians, seventyFifths, maxes


    def _compare(self, prediction, label):
        totalAbsLoss = np.sum(np.abs(prediction - label))
        return totalAbsLoss / prediction.shape[0]
