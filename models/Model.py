# Name: CNN + RNN + MLP Model
# Author: Robert Ciborowski
# Date: 18/04/2020
# Description: Determines the characteristics of tomorrow's price distribution.

# from __future__ import annotations
import os
from datetime import datetime
from typing import Dict, List
import numpy as np
import pandas as pd
import tensorflow as tf
from matplotlib import pyplot as plt
from tensorflow.keras import layers
from models.Hyperparameters import Hyperparameters
from util.Constants import SAMPLES_OF_DATA_TO_LOOK_AT

class Model:
    hyperparameters: Hyperparameters

    def setup(self, hyperparameters: Hyperparameters):
        self._buildMetrics()
        self.hyperparameters = hyperparameters

    def predict(self, data) -> bool:
        pass

    def createModel(self):
        pass

    def trainModel(self, features, labels, validationSplit: float):
        pass

    def evaluate(self, features, label):
        pass

    def save(self):
        pass

    def load(self):
        pass
