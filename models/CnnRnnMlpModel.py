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

class CnnRnnMlpModel:
    hyperparameters: Hyperparameters
    listOfMetrics: List
    exportPath: str

    _metrics: List
    _NUMBER_OF_SAMPLES = SAMPLES_OF_DATA_TO_LOOK_AT

    def __init__(self, tryUsingGPU=False):
        super().__init__()

        if not tryUsingGPU:
            os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
        else:
            self._configureForGPU()

        self.exportPath = "./model_exports/cryptopumpanddumpdetector"

        # The following lines adjust the granularity of reporting.
        pd.options.display.max_rows = 10
        pd.options.display.float_format = "{:.1f}".format
        # tf.keras.backend.set_floatx('float32')

    def setup(self, hyperparameters: Hyperparameters):
        self._buildMetrics()
        self.hyperparameters = hyperparameters

    """
    Precondition: prices is a pandas dataframe or series.
    """
    def detect(self, data) -> float:
        # This whole function needs to be remade!!!
        return self._detect(data)

    def _detect(self, data) -> float:
        # This whole function needs to be remade!!!
        time1 = datetime.now()
        data = np.array([data])
        result = self.model.predict(data)[0][0]
        # result = self.model(data).numpy()[0][0]
        time2 = datetime.now()
        print("Gave out a result of " + str(result) + ", took " + str(
            time2 - time1))
        return result

    """
    Creates a brand new neural network for this model.
    """

    def createModel(self):
        # Should go over minutes, not seconds
        input_layer = layers.Input(shape=(SAMPLES_OF_DATA_TO_LOOK_AT, 6))
        # self.model = tf.keras.models.Sequential()
        # self.model.add(input_seq)
        layer = layers.Conv1D(filters=20, kernel_size=3, activation='relu',
                         input_shape=(SAMPLES_OF_DATA_TO_LOOK_AT, 6))(input_layer)
        # layer = layers.AveragePooling1D(pool_size=2)(layer)
        layer = layers.Conv1D(filters=16, kernel_size=5, activation='relu',
                          input_shape=(SAMPLES_OF_DATA_TO_LOOK_AT, 6))(layer)
        layer = layers.AveragePooling1D(pool_size=2)(layer)
        # self.model.add(
        #     layers.Conv1D(filters=2, kernel_size=4, activation='relu',
        #                   input_shape=(SAMPLES_OF_DATA_TO_LOOK_AT, 6)))
        # self.model.add(layers.AveragePooling1D(pool_size=2))
        # layer = layers.Flatten()(layer)
        # layer = tf.keras.layers.LSTM(SAMPLES_OF_DATA_TO_LOOK_AT,
        #                      input_shape=layer.shape)(layer)
        layer = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(SAMPLES_OF_DATA_TO_LOOK_AT, input_shape=layer.shape))(layer)
        # layer = layers.Flatten()(layer)
        layer = layers.Dense(80, activation='relu')(layer)
        layer = layers.Dense(40, activation='relu')(layer)
        layer = layers.Dense(15, activation='relu')(layer)
        layer = layers.Dense(5, activation='relu')(layer)
        layer = tf.keras.layers.Dropout(0.1)(layer)
        # layer = layers.Dense(1, activation='sigmoid')(layer)
        layer = layers.Dense(1, activation='relu')(layer)
        self.model = tf.keras.Model(input_layer, layer)
        self.model.compile(loss='mean_squared_error',
                           optimizer=tf.keras.optimizers.Adam(lr=self.hyperparameters.learningRate),
                           metrics=self._metrics)
        tf.keras.utils.plot_model(self.model,
                                  "crypto_model.png",
                                  show_shapes=True)

    def trainModel(self, features, labels, validationSplit: float):
        """Train the model by feeding it data."""
        history = self.model.fit(x=features, y=labels, batch_size=self.hyperparameters.batchSize,
                                 validation_split=validationSplit, epochs=self.hyperparameters.epochs, shuffle=True)

        # The list of epochs is stored separately from the rest of history.
        epochs = history.epoch

        # To track the progression of training, gather a snapshot
        # of the model's mean squared error at each epoch.
        hist = pd.DataFrame(history.history)
        return epochs, hist

    """
    Evaluates the model on features.
    Returns:
        Scalar test loss (if the model has a single output and no metrics)
        or list of scalars (if the model has multiple outputs
        and/or metrics). The attribute `model.metrics_names` will give you
        the display labels for the scalar outputs.
    """
    def evaluate(self, features, label):
        return self.model.evaluate(features, label, self.hyperparameters.batchSize)

    def plotCurve(self, epochs, hist, metrics):
        """Plot a curve of one or more classification metrics vs. epoch."""
        # list_of_metrics should be one of the names shown in:
        # https://www.tensorflow.org/tutorials/structured_data/imbalanced_data#define_the_model_and_metrics

        plt.figure()
        plt.xlabel("Epoch")
        plt.ylabel("Value")

        for m in metrics:
            x = hist[m]
            plt.plot(epochs[1:], x[1:], label=m)

        plt.legend()
        plt.show()

    def exportWeights(self):
        self.model.save_weights(self.exportPath)

    def loadWeights(self):
        self.model.load_weights(self.exportPath)

    def _buildMetrics(self):
        # Metrics for classifying if the price will be higher or lower tomorrow:
        # self._metrics = [
        #     tf.keras.metrics.BinaryAccuracy(name='accuracy',
        #                                     threshold=self._classificationThreshold),
        #     tf.keras.metrics.Precision(thresholds=self._classificationThreshold,
        #                                name='precision'
        #                                ),
        #     tf.keras.metrics.Recall(thresholds=self._classificationThreshold,
        #                             name="recall"),
        #     tf.keras.metrics.AUC(num_thresholds=100, name='auc')
        # ]
        # self.listOfMetrics = ["accuracy", "precision", "recall", "auc"]

        # Metrics for describing tomorrow's price distribution:
        self._metrics = [
            tf.keras.metrics.MeanAbsoluteError()
        ]
        self.listOfMetrics = ["mean_squared_error"]

    def _configureForGPU(self):
        # https://www.tensorflow.org/guide/gpu
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            try:
                # Currently, memory growth needs to be the same across GPUs
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                logical_gpus = tf.config.experimental.list_logical_devices(
                    'GPU')
                print(len(gpus), "Physical GPUs,", len(logical_gpus),
                      "Logical GPUs")
            except RuntimeError as e:
                # Memory growth must be set before GPUs have been initialized
                print("CryptoPumpAndDumpDetector GPU setup error: " + str(e))
