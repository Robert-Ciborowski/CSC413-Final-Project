# Name: BinanceDataSetCreator
# Author: Robert Ciborowski
# Date: 19/04/2020
# Description: Creates a data set from Binance data.

from datetime import timedelta
from typing import List
import pytz
from stock_pandas import StockDataFrame

from stock_data import HistoricalBinanceDataObtainer
import csv
from scipy import stats

class BinanceDataSetCreator:
    dataObtainer: HistoricalBinanceDataObtainer
    numberOfSamples: int
    inputData: List
    outputData: List

    def __init__(self, dataObtainer: HistoricalBinanceDataObtainer):
        self.dataObtainer = dataObtainer
        # Minutes in a day:
        # self.numberOfSamples = 24 * 60
        self.numberOfSamples = 30

    def exportToCSV(self, path: str, pathPrefix=""):
        if len(self.inputData) == 0:
            return

        try:
            with open(pathPrefix + path, 'w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(["Close-" + str(i) for i in range(self.numberOfSamples)]
                                + ["Volume-" + str(i) for i in range(self.numberOfSamples)]
                                + ["RSI-" + str(i) for i in range(self.numberOfSamples)]
                                + ["MACD-" + str(i) for i in range(self.numberOfSamples)]
                                + ["BollUpper-" + str(i) for i in range(self.numberOfSamples)]
                                + ["BollLower-" + str(i) for i in range(self.numberOfSamples)]
                                + ["Mean"])

                for i in range(len(self.inputData)):
                    data = []

                    for j in self.inputData[i]:
                        data += j

                    for j in self.outputData[i]:
                        data.append(j)

                    writer.writerow(data)

        except IOError as e:
            print("Error writing to csv file! " + str(e))

    def createDataset(self, symbol: str, startDate, endDate):
        timezone = "Etc/GMT-0"
        timezone = pytz.timezone(timezone)
        startDate = timezone.localize(startDate)
        endDate = timezone.localize(endDate)
        self.inputData = []
        self.outputData = []
        df = self.dataObtainer.getHistoricalDataAsDataframe(symbol)

        # First, gather all of the means of the data
        closeMeans = []
        volumeMeans = []
        closeMedians = []
        date = startDate

        while date < endDate:
            startIndex = df.index[df["Timestamp"] == date].tolist()

            if len(startIndex) == 0:
                date += timedelta(days=1)
                continue

            startIndex = startIndex[0]
            endIndex = df.index[df["Timestamp"] == date + timedelta(days=1)].tolist()

            if len(endIndex) == 0:
                date += timedelta(days=1)
                continue

            endIndex = endIndex[0]
            data = df.iloc[startIndex : endIndex]
            closeMeans.append(data["Close"].mean())
            volumeMeans.append(data["Volume"].mean())
            closeMedians.append(data["Close"].median())
            date += timedelta(days=1)

        stock = StockDataFrame({
            'close': closeMeans
        })

        # The standard RSI is 14 day.
        rsis = (stock["rsi:8"] / 100).tolist()
        mas = stock["macd"].tolist()
        bollUppers = stock["boll.upper"].tolist()
        bollLowers = stock["boll.lower"].tolist()

        import math
        rsis = [0 if math.isnan(x) else x for x in rsis]
        mas = [0 if math.isnan(x) else x for x in mas]
        bollUppers = [0 if math.isnan(x) else x for x in bollUppers]
        bollLowers = [0 if math.isnan(x) else x for x in bollLowers]

        for i in range(len(closeMeans) - self.numberOfSamples - 1):
            close = closeMeans[i : i + self.numberOfSamples]
            volume = volumeMeans[i : i + self.numberOfSamples]
            rsi = rsis[i : i + self.numberOfSamples]
            ma = mas[i : i + self.numberOfSamples]
            maxMA = max(mas)
            ma = [((m / maxMA) + 1) / 2 for m in ma]
            bollUpper = bollUppers[i: i + self.numberOfSamples]
            maxBollUpper = max(bollUpper)
            bollUpper = [m / maxBollUpper for m in bollUpper]
            bollLower = bollLowers[i: i + self.numberOfSamples]
            maxBollLower = max(bollLower)
            bollLower = [m / maxBollLower for m in bollLower]
            maxClose = max(close)
            maxVolume = max(volume)
            meanClose = sum(close) / len(close)

            for j in range(len(close)):
                close[j] /= maxClose

            for j in range(len(volume)):
                volume[j] /= maxVolume

            self.inputData.append([close, volume, rsi, ma, bollUpper, bollLower])
            output = closeMedians[i + self.numberOfSamples + 1] / meanClose

            # Use this if you want the model to predict if the price will be
            # higher or lower (instead or predicting the exact mean)
            # if output > 1.0:
            #     output = 1.0
            # else:
            #     output = 0.0

            self.outputData.append([output])
