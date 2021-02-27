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
    medianWithin: float

    _timeInterval = timedelta
    _datapointsPerDay: int

    def __init__(self, dataObtainer: HistoricalBinanceDataObtainer, medianWithin=None, dataInterval="day"):
        self.dataObtainer = dataObtainer
        self.medianWithin = medianWithin

        if dataInterval == "day":
            self._timeInterval = timedelta(days=1)
            self._datapointsPerDay = 1
            self.numberOfSamples = 30
        else:
            # 1 hour
            self._timeInterval = timedelta(hours=1)
            self._datapointsPerDay = 24
            self.numberOfSamples = 30 * 24

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
                                + ["Min", "25th-Percentile", "Median", "75th-Percentile", "Max"])

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
        close75thPercentiles = []
        close25thPercentiles = []
        closeMaxes = []
        closeMins = []
        date = startDate

        while date < endDate:
            print("Processing", date, "/", endDate)
            startIndex = df.index[df["Timestamp"] == date].tolist()

            if len(startIndex) == 0:
                date += self._timeInterval
                continue

            startIndex = startIndex[0]
            endIndex = df.index[df["Timestamp"] == date + timedelta(hours=1)].tolist()

            if len(endIndex) == 0:
                date += self._timeInterval
                continue

            endIndex = endIndex[0]
            data = df.iloc[startIndex : endIndex]
            closeMeans.append(data["Close"].mean())
            volumeMeans.append(data["Volume"].mean())
            closeMedians.append(data["Close"].median())
            close75thPercentiles.append(data["Close"].quantile(0.75))
            close25thPercentiles.append(data["Close"].quantile(0.25))
            closeMaxes.append(data["Close"].max())
            closeMins.append(data["Close"].min())
            date += self._timeInterval

        stock = StockDataFrame({
            'close': closeMeans
        })

        # The standard RSI is 14 day.
        rsis = (stock["rsi:192"] / 100).tolist()
        mas = stock["macd"].tolist()
        bollUppers = stock["boll.upper"].tolist()
        bollLowers = stock["boll.lower"].tolist()

        import math
        rsis = [0 if math.isnan(x) else x for x in rsis]
        mas = [0 if math.isnan(x) else x for x in mas]
        bollUppers = [0 if math.isnan(x) else x for x in bollUppers]
        bollLowers = [0 if math.isnan(x) else x for x in bollLowers]

        for i in range(0, len(closeMeans) - self.numberOfSamples - 1, self._datapointsPerDay):
            print("Creating entry", i, "/", len(closeMeans) - self.numberOfSamples - 1)
            close = closeMeans[i : i + self.numberOfSamples]
            meanClose = sum(close) / len(close)
            outputMedian = closeMedians[
                               i + self.numberOfSamples + 1] / meanClose

            if self.medianWithin is not None and outputMedian + abs(outputMedian - 1.0) > self.medianWithin:
                print("Skipping", outputMedian)
                continue

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


            for j in range(len(close)):
                close[j] /= maxClose

            for j in range(len(volume)):
                volume[j] /= maxVolume

            self.inputData.append([close, volume, rsi, ma, bollUpper, bollLower])
            output75thPercentile = close75thPercentiles[i + self.numberOfSamples + 1] / meanClose
            output25thPercentile = close25thPercentiles[i + self.numberOfSamples + 1] / meanClose
            outputMax = closeMaxes[i + self.numberOfSamples + 1] / meanClose
            outputMin = closeMins[i + self.numberOfSamples + 1] / meanClose

            # Use this if you want the model to predict if the price will be
            # higher or lower (instead or predicting the exact mean)
            # if output > 1.0:
            #     output = 1.0
            # else:
            #     output = 0.0

            self.outputData.append([outputMin, output25thPercentile, outputMedian,
                                    output75thPercentile, outputMax])
