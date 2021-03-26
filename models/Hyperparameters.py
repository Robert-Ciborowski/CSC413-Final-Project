# Name: Hyperparameters
# Author: Robert Ciborowski
# Date: 18/04/2020
# Description: Stores some hyperparameters for a machine learning model.

class Hyperparameters:
    learningRate: float
    epochs: int
    batchSize: int
    decayRate: float
    decayStep: float
    dropout: float

    def __init__(self, learningRate: float, epochs: int, dropout: float, batchSize: int, decayRate=0.5, decayStep=1.0):
        self.learningRate = learningRate
        self.epochs = epochs
        self.batchSize = batchSize
        self.dropout = dropout
        self.decayRate = decayRate
        self.decayStep = decayStep
