# Cryptocurrency Price Distribution Prediction Using Neural Networks
Our final project, a paper which uses neural networks to provide Bitcoin and other cryptocurrency traders with good buy/sell points. This is our final project for CSC413H1, Winter 2021, taught by Jimmy Ba and Bo Wang at the University of Toronto.

## Abstract
Price prediction in the financial market is attractive for both holders and traders due to its volatility and uncertainty. Many have tried to analyze the future of the financial market by using methods and models from machine learning. In this paper, we construct several neural network models to predict sequential data for cryptocurrencies, specifically Convolutional Neural Network (CNN), Recurrent Neural Network (RNN), CNN-RNN Multilayer Perceptron (CNN-RNN-MLP), Long Short-Term Memory with Attention (LSTM-Attention), and InceptionNet. The results showed that CNN-RNN-MLP performs the best among other models.

## Introduction
Financial time series prediction is a major topic within the finance industry. It is a difficult task due to the uncertainty of the financial market. In recent years, the strategies of predicting financial assets have expanded. Researchers have proposed many state-of-the-art methods for stock and cryptocurrency predictions, such as sequential prediction using the CNN and LSTM. However, current research has not proposed models which give traders buying and selling points. Current research also does not explore newer machine learning model types and does not take advantage of technical indicators used by professional human traders. In this paper, we are going to focus on the forecast of cryptocurrencies using several neural network techniques. First, we train our models with Bitcoin data, where the output of our models is a percentile of our choosing of the distribution of Bitcoin prices of the next day, thus providing information for traders on when to buy and sell Bitcoin. Then we use the top performance models to perform extensive experiments on  different cryptocurrencies and on different percentiles. Lastly, we compare our model architectures against those from other research.

## Quick Guide
1. Download data using data_downloads/DownloadBinanceData.py.
2. Generate a data set using data_set/CreateDataSet.py.
3. If you want the model to predict whether a percentile will be higher or lower than the previous day's mean price, set BINARY_PREDICTION to True in util/Constants.py.
4. Train the models in the models folder.
5. Enjoy! 

## Machine Learning Models and Layers Used
- CNN
- LSTM
- CNN + LSTM + MLP
- LSTM + Attention
- InceptionNet

## Technologies Used
TensorFlow 2, Keras, Python
