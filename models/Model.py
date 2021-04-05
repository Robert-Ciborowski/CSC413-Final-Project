# Name: Model
# Author: Robert Ciborowski
# Date: 25/03/2021
# Description: An abstract class for a price predictor ML model.

from models.Hyperparameters import Hyperparameters

class Model:
    hyperparameters: Hyperparameters

    def setup(self, hyperparameters: Hyperparameters):
        self.hyperparameters = hyperparameters

    def predict(self, data) -> bool:
        pass

    def createModel(self, numberOfOutputs: int):
        """
        Creates the model.
        :param numberOfOutputs: usually either OUTPUT_CHANNELS if predicting
                                all percentiles, or 1 if predicting a single
                                percentile.
        :return:
        """
        pass

    def trainModel(self, features, labels, validationSplit: float):
        pass

    def evaluate(self, features, label):
        pass

    def save(self):
        pass

    def load(self):
        pass
