import os
from datetime import datetime
from typing import List

import keras
import numpy as np
import pandas as pd
import tensorflow as tf
from matplotlib import pyplot as plt
from tensorflow.keras import layers
from models.Hyperparameters import Hyperparameters
from models.Model import Model
from util.Constants import INPUT_CHANNELS, OUTPUT_CHANNELS, \
    SAMPLES_OF_DATA_TO_LOOK_AT

class RnnModel(Model):
    hyperparameters: Hyperparameters
    listOfMetrics: List
    exportPath: str
    binary: bool

    _NUMBER_OF_SAMPLES = SAMPLES_OF_DATA_TO_LOOK_AT
    _numberOfInputChannels = INPUT_CHANNELS

    def __init__(self, tryUsingGPU=False, binary=False):
        super().__init__()

        if not tryUsingGPU:
            os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
        else:
            self._configureForGPU()

        self.exportPath = "./model_exports/Rnn"

        # The following lines adjust the granularity of reporting.
        pd.options.display.max_rows = 10
        pd.options.display.float_format = "{:.1f}".format
        tf.keras.backend.set_floatx('float64')
        self._classificationThreshold = 0.5
        self.binary = binary

        # The ability of our model to perform well on the validation set and
        # its ability to perform better on the training set is being affected
        # by weight initialization. Thus, we need to use the same good seed.
        # np.random.seed(1)
        tf.random.set_seed(8008)

    def setup(self, hyperparameters: Hyperparameters):
        self._buildMetrics()
        self.hyperparameters = hyperparameters

    """
    Precondition: prices is a numpy 3d array.
    """
    def predict(self, data, concatenate=False) -> float:
        time1 = datetime.now()
        result = self.model.predict(data)[0][0]
        time2 = datetime.now()
        print("Gave out a result of " + str(result) + ", took " + str(
            time2 - time1))

        if concatenate:
            result = np.concatenate(result, axis=1)

        return result

    def predictMultiple(self, data, concatenate=False) -> float:
        time1 = datetime.now()
        result = self.model.predict(data)
        time2 = datetime.now()
        print("Gave out a result of " + str(result) + ", took " + str(
            time2 - time1))

        if concatenate:
            result = np.concatenate(result, axis=1)

        return result

    def createModel(self, generateGraph=False):
        """
        Creates a brand new neural network for this model.
        """
        outputNames = ["15th-percentile"]
        outputs = []

        # Should go over minutes, not seconds
        input_layer = layers.Input(shape=(SAMPLES_OF_DATA_TO_LOOK_AT, self._numberOfInputChannels))

        # layer = layers.TimeDistributed(layers.Embedding(input_dim=self._numberOfInputChannels, output_dim=64))(input_layer)
        #
        # layer = layers.TimeDistributed(layers.Bidirectional(layers.LSTM(layer.shape[1], return_sequences=True)))(layer)
        #
        # layer = layers.TimeDistributed(layers.Bidirectional(layers.LSTM(64)))(layer)
        print(input_layer.shape)
        forward_lstm = tf.keras.layers.LSTM(input_layer.shape[2], return_sequences=True)
        backward_lstm = tf.keras.layers.LSTM(input_layer.shape[2], activation='tanh', return_sequences=True, go_backwards=True)
        layer = tf.keras.layers.Bidirectional(forward_lstm, backward_layer=backward_lstm, input_shape=input_layer.shape)(input_layer)

        layer = layers.Dense(1, activation='relu')(layer)

        outputs.append(layer)

        lossWeights = {name: 1.0 for name in outputNames}
        metrics = {name: self.listOfMetrics for name in outputNames}

        self.model = tf.keras.Model(input_layer, outputs=outputs)
        self.model.compile(loss="mean_squared_error", loss_weights=lossWeights,
                           optimizer=tf.keras.optimizers.Adam(
                               lr=self.hyperparameters.learningRate),
                           metrics=metrics)

        if self.binary:
            # This compiles the model if we are using binary prediction.
            self.model.compile(loss="binary_crossentropy",
                               optimizer=tf.keras.optimizers.Adam(
                                   lr=self.hyperparameters.learningRate),
                               metrics=self.listOfMetrics)
        else:
            # This compiles the model if we are using percentage prediction.
            self.model.compile(loss="mean_squared_error",
                               optimizer=tf.keras.optimizers.Adam(
                                   lr=self.hyperparameters.learningRate),
                               metrics=self.listOfMetrics)

        if generateGraph:
            tf.keras.utils.plot_model(self.model,
                                      "crypto_model.png",
                                      show_shapes=True)

    def trainModel(self, features, labels, validationSplit: float, earlyStopping=False):
        """Train the model by feeding it data."""
        if earlyStopping:
            earlyStopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                             mode='min', verbose=1,
                                                             patience=15)
            history = self.model.fit(x=features, y=labels,
                                     batch_size=self.hyperparameters.batchSize,
                                     validation_split=validationSplit,
                                     epochs=self.hyperparameters.epochs,
                                     shuffle=True, callbacks=earlyStopping)
        else:
            history = self.model.fit(x=features, y=labels,
                                     batch_size=self.hyperparameters.batchSize,
                                     validation_split=validationSplit,
                                     epochs=self.hyperparameters.epochs,
                                     shuffle=True)

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

    def plotCurve(self, epochs, hist):
        """Plot a curve of one or more classification metrics vs. epoch."""
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
        if self.binary:
            # For binary prediction:
            self.listOfMetrics = [tf.keras.metrics.Precision(thresholds=self._classificationThreshold,
                                                             name='precision'),
                                  tf.keras.metrics.Recall(thresholds=self._classificationThreshold,
                                                          name="recall")]
        else:
            # For percentage prediction:
            self.listOfMetrics = ["mean_absolute_error"]

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
                print("Tensorflow GPU setup error: " + str(e))