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
from models.Model import Model
from util.Constants import SAMPLES_OF_DATA_TO_LOOK_AT

class CnnRnnMlpModel(Model):
    hyperparameters: Hyperparameters
    listOfMetrics: List
    exportPath: str

    _metrics: List
    _NUMBER_OF_SAMPLES = SAMPLES_OF_DATA_TO_LOOK_AT
    _numberOfInputChannels = 9

    def __init__(self, tryUsingGPU=False):
        super().__init__()

        if not tryUsingGPU:
            os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
        else:
            self._configureForGPU()

        self.exportPath = "./model_exports/cnnrnnmlp"

        # The following lines adjust the granularity of reporting.
        pd.options.display.max_rows = 10
        pd.options.display.float_format = "{:.1f}".format
        tf.keras.backend.set_floatx('float64')
        self._classificationThreshold = 0.5

    def setup(self, hyperparameters: Hyperparameters):
        self._buildMetrics()
        self.hyperparameters = hyperparameters

    """
    Precondition: prices is a numpy 3d array.
    """
    def predict(self, data, concatenate=False) -> float:
        # This whole function needs to be remade!!!
        # time1 = datetime.now()
        # result = self.model.predict(data)
        # # result = self.model(data).numpy()[0][0]
        # time2 = datetime.now()
        # print("Gave out a result of " + str(result) + ", took " + str(
        #     time2 - time1))
        #
        # if concatenate:
        #     result = np.concatenate(result, axis=1)
        #
        # return result
        pass

    # def makePricePredictionForTommorrow(self, data, meanPriceFifteenDays) -> float:
    #     data = np.array([data])
    #     predictions = self.predict(data)
    #     results = []
    #
    #     for prediction in predictions:
    #         results.append(prediction[0][0] * meanPriceFifteenDays)
    #
    #     return results

    def createModel(self):
        """
        Creates a brand new neural network for this model.
        """
        # Should go over minutes, not seconds
        input_layer = layers.Input(shape=(SAMPLES_OF_DATA_TO_LOOK_AT, self._numberOfInputChannels))
        layer = self._createInceptionLayer(input_layer, 36)
        # layer = self._createInceptionLayer(layer, 16)
        layer = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(SAMPLES_OF_DATA_TO_LOOK_AT, input_shape=layer.shape))(layer)
        # layer = layers.Dense(600, activation='relu')(layer)
        layer = layers.Dense(150, activation='relu')(layer)
        # layer = layers.Dense(20, activation='relu')(layer)
        # layer = layers.Dense(20, activation='relu')(layer)
        # layer = tf.keras.layers.Dropout(self.hyperparameters.dropout)(layer)

        # Median
        medianDense = layers.Dense(20, activation='relu')(layer)
        # medianDense = layers.Dense(20, activation='relu')(medianDense)
        medianDropout = tf.keras.layers.Dropout(self.hyperparameters.dropout)(medianDense)
        medianFinal = layers.Dense(1, activation='relu', name="median")(medianDropout)

        # 35th Percentile
        thirtyFifthConcat = tf.concat([layer, medianFinal], 1)
        thirtyFifthDense = layers.Dense(20, activation='relu')(thirtyFifthConcat)
        # thirtyFifthConcat2 = tf.concat([thirtyFifthDense, medianFinal], 1)
        # thirtyFifthDense = layers.Dense(20, activation='relu')(thirtyFifthConcat2)
        thirtyFifthDropout = tf.keras.layers.Dropout(self.hyperparameters.dropout)(thirtyFifthDense)
        thirtyFifthFinal = layers.Dense(1, activation='relu', name="35th-percentile")(
            thirtyFifthDropout)

        # 25th Percentile
        twentyFifthConcat = tf.concat([layer, thirtyFifthFinal], 1)
        twentyFifthDense = layers.Dense(20, activation='relu')(twentyFifthConcat)
        # twentyFifthConcat2 = tf.concat([twentyFifthDense, thirtyFifthFinal], 1)
        # twentyFifthDense = layers.Dense(20, activation='relu')(twentyFifthConcat2)
        twentyFifthDropout = tf.keras.layers.Dropout(self.hyperparameters.dropout)(twentyFifthDense)
        twentyFifthFinal = layers.Dense(1, activation='relu', name="25th-percentile")(twentyFifthDropout)

        # 15th
        fifteenthConcat = tf.concat([layer, twentyFifthFinal], 1)
        fifteenthDense = layers.Dense(20, activation='relu')(fifteenthConcat)
        # fifteenthConcat2 = tf.concat([fifteenthDense, twentyFifthFinal], 1)
        # fifteenthDense = layers.Dense(20, activation='relu')(fifteenthConcat2)
        fifteenthDropout = tf.keras.layers.Dropout(self.hyperparameters.dropout)(fifteenthDense)
        fifteenthFinal = layers.Dense(1, activation='relu', name="15th-percentile")(fifteenthDropout)

        # 65th Percentile
        sixtyFifthConcat = tf.concat([medianFinal, layer], 1)
        sixtyFifthDense = layers.Dense(20, activation='relu')(sixtyFifthConcat)
        # sixtyFifthConcat2 = tf.concat([medianFinal, sixtyFifthDense], 1)
        # sixtyFifthDense = layers.Dense(20, activation='relu')(sixtyFifthConcat2)
        sixtyFifthDropout = tf.keras.layers.Dropout(self.hyperparameters.dropout)(sixtyFifthDense)
        sixtyFifthFinal = layers.Dense(1, activation='relu', name="65th-percentile")(sixtyFifthDropout)

        # 75th Percentile
        seventyFifthConcat = tf.concat([sixtyFifthFinal, layer], 1)
        seventyFifthDense = layers.Dense(20, activation='relu')(seventyFifthConcat)
        # seventyFifthConcat2 = tf.concat([sixtyFifthFinal, seventyFifthDense], 1)
        # seventyFifthDense = layers.Dense(20, activation='relu')(seventyFifthConcat2)
        seventyFifthDropout = tf.keras.layers.Dropout(self.hyperparameters.dropout)(seventyFifthDense)
        seventyFifthFinal = layers.Dense(1, activation='relu', name="75th-percentile")(seventyFifthDropout)

        # 85th Percentile
        eightyFifthConcat = tf.concat([seventyFifthFinal, layer], 1)
        eightyFifthDense = layers.Dense(20, activation='relu')(eightyFifthConcat)
        # eightyFifthConcat2 = tf.concat([seventyFifthFinal, eightyFifthDense], 1)
        # eightyFifthDense = layers.Dense(20, activation='relu')(eightyFifthConcat2)
        eightyFifthDropout = tf.keras.layers.Dropout(self.hyperparameters.dropout)(eightyFifthDense)
        eightyFifthFinal = layers.Dense(1, activation='relu', name="85th-percentile")(eightyFifthDropout)

        outputs = [fifteenthFinal, twentyFifthFinal, thirtyFifthFinal,
                   medianFinal, sixtyFifthFinal, seventyFifthFinal, eightyFifthFinal]
        lossWeights = {"15th-percentile": 1.0, "25th-percentile": 1.0,
                       "35th-percentile": 1.0, "median": 1.0,
                       "65th-percentile": 1.0, "75th-percentile": 1.0,
                       "85th-percentile": 1.0}
        metrics = {"15th-percentile": self.listOfMetrics, "25th-percentile": self.listOfMetrics,
                   "35th-percentile": self.listOfMetrics, "median": self.listOfMetrics,
                   "65th-percentile": self.listOfMetrics, "75th-percentile": self.listOfMetrics,
                   "85th-percentile": self.listOfMetrics}
        self.model = tf.keras.Model(input_layer, outputs=outputs)
        # self.model.compile(loss="binary_crossentropy", loss_weights=lossWeights,
        #                    optimizer=tf.keras.optimizers.Adam(lr=self.hyperparameters.learningRate),
        #                    metrics=metrics)
        self.model.compile(loss="mean_squared_error", loss_weights=lossWeights,
                           optimizer=tf.keras.optimizers.Adam(
                               lr=self.hyperparameters.learningRate),
                           metrics=metrics)
        tf.keras.utils.plot_model(self.model,
                                  "crypto_model.png",
                                  show_shapes=True)

    def _createInceptionLayer(self, input_layer, filters):
        layer_1 = layers.Conv1D(filters=filters, kernel_size=1, activation='relu',
                                input_shape=(SAMPLES_OF_DATA_TO_LOOK_AT,
                                            self._numberOfInputChannels))(input_layer)
        layer_2_1 = layers.Conv1D(filters=filters, kernel_size=1, activation='relu',
                                input_shape=(SAMPLES_OF_DATA_TO_LOOK_AT,
                                             self._numberOfInputChannels))(input_layer)
        layer_2_2 = layers.Conv1D(filters=filters, kernel_size=4, activation='relu',
                                  input_shape=(SAMPLES_OF_DATA_TO_LOOK_AT,
                                               self._numberOfInputChannels))(layer_2_1)
        layer_3_1 = layers.Conv1D(filters=filters, kernel_size=1, activation='relu',
                                  input_shape=(SAMPLES_OF_DATA_TO_LOOK_AT,
                                               self._numberOfInputChannels))(input_layer)
        layer_3_2 = layers.Conv1D(filters=filters, kernel_size=8, activation='relu',
                                  input_shape=(SAMPLES_OF_DATA_TO_LOOK_AT,
                                               self._numberOfInputChannels))(layer_3_1)
        layer_4_1 = layers.Conv1D(filters=filters, kernel_size=1,
                                  activation='relu',
                                  input_shape=(SAMPLES_OF_DATA_TO_LOOK_AT,
                                               self._numberOfInputChannels))(input_layer)
        layer_4_2 = layers.Conv1D(filters=filters, kernel_size=12,
                                  activation='relu',
                                  input_shape=(SAMPLES_OF_DATA_TO_LOOK_AT,
                                               self._numberOfInputChannels))(layer_4_1)
        layer_5_1 = layers.MaxPooling1D(pool_size=2)(input_layer)
        layer_5_2 = layers.Conv1D(filters=filters, kernel_size=1, activation='relu',
                                  input_shape=(SAMPLES_OF_DATA_TO_LOOK_AT,
                                               self._numberOfInputChannels))(layer_5_1)
        output = tf.concat([layer_1, layer_2_2, layer_3_2, layer_4_2, layer_5_2], 1)
        return output

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

        plt.figure(figsize=(20,10))
        plt.xlabel("Epoch")
        plt.ylabel("Value")

        for m in hist:
            x = hist[m]
            plt.plot(epochs[1:], x[1:], label=m)

        plt.legend()
        plt.show()

    def save(self):
        self.model.save_weights(self.exportPath)

    def load(self):
        self.model.load_weights(self.exportPath)

    def _buildMetrics(self):
        # Metrics for classifying if the price will be higher or lower tomorrow:
        self._metrics = [
            tf.keras.metrics.Accuracy(),
            tf.keras.metrics.Precision(
                thresholds=self._classificationThreshold, class_id=0,
                name="min_precision"
            ),
            tf.keras.metrics.Precision(
                thresholds=self._classificationThreshold, class_id=1,
                name="25th_precision"
            ),
            tf.keras.metrics.Precision(
                thresholds=self._classificationThreshold, class_id=2,
                name="median_precision"
            ),
            tf.keras.metrics.Precision(
                thresholds=self._classificationThreshold, class_id=3,
                name="75th_precision"
            ),
            tf.keras.metrics.Precision(
                thresholds=self._classificationThreshold, class_id=4,
                name="max_precision"
            )
            # tf.keras.metrics.Precision(thresholds=self._classificationThreshold,
            #                            name='precision'),
            # tf.keras.metrics.Recall(thresholds=self._classificationThreshold,
            #                         name="recall")
        ]
        self.listOfMetrics = ["mean_absolute_error"]

        # Metrics for describing tomorrow's price distribution:
        # self._metrics = [
        #     tf.keras.metrics.MeanAbsoluteError()
        # ]
        # self.listOfMetrics = ["mean_squared_error"]

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
