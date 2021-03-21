# Name: BinanceDataSetCreator
# Author: Robert Ciborowski
# Date: 19/04/2020
# Description: Creates a data set from Binance data.

from datetime import timedelta
from typing import List
import pytz
from stock_pandas import StockDataFrame

from stock_data import HistoricalDataObtainer
import csv
from scipy import stats

class BinanceDataSetCreator:
    dataObtainer: HistoricalDataObtainer
    numberOfSamples: int
    inputData: List
    outputData: List
    medianWithin: float
    dayByDay: bool

    _dataTimeInterval = timedelta
    _datapointsPerDay: int

    def __init__(self, dataObtainer: HistoricalDataObtainer, dayByDay=True, medianWithin=None, dataInterval="day"):
        """
        Constructor.

        :param dataObtainer: a HistoricalDataObtainer
        :param dayByDay: True = dataset inputs should be one day (00:00-23:59)
                         False = dataset inputs can be continuous (you can have
                                 an input taken from Mar 11 09:00 to Mar 12 09:00)
        :param medianWithin: we only add entries to the dataset if the output's
                             median price is within this much of the input median
                             price. E.g. for entries only within 35.6% of the input
                             price, set medianWithin to 1.356
        :param dataInterval: How much time passes between each sample of the input
                             data
        """
        self.dataObtainer = dataObtainer
        self.medianWithin = medianWithin
        self.dayByDay = True

        if dataInterval == "day":
            self._dataTimeInterval = timedelta(days=1)
            self._datapointsPerDay = 1
            self.numberOfSamples = 30
        elif dataInterval == "2 hour":
            self._dataTimeInterval = timedelta(hours=2)
            self._datapointsPerDay = 12
            self.numberOfSamples = 15 * 12
        elif dataInterval == "3 hour":
            self._dataTimeInterval = timedelta(hours=3)
            self._datapointsPerDay = 8
            self.numberOfSamples = 15 * 8
        else:
            # 1 hour
            self._dataTimeInterval = timedelta(hours=1)
            self._datapointsPerDay = 24
            self.numberOfSamples = 7 * 24

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
                                + ["15th-Percentile", "25th-Percentile",
                                   "35th-Percentile", "Median", "65th-Percentile",
                                   "75th-Percentile", "85th-Percentile"])

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
        """
        Creates a dataset. Please make sure that the start and end dates are
        the beginnings of days.
        :param symbol: e.g. "BTCUSDT"
        :param startDate: e.g. datetime(year=2020, month=1, day=1)
        :param endDate: e.g. datetime(year=2020, month=2, day=1)
        """
        timezone = "Etc/GMT-0"
        timezone = pytz.timezone(timezone)
        startDate = timezone.localize(startDate)
        endDate = timezone.localize(endDate)
        self.inputData = []
        self.outputData = []
        df = self.dataObtainer.getHistoricalDataAsDataframe(symbol)

        # First, we gather all of the means for our inputs.
        closeMeans = []
        volumeMeans = []

        # Also, we will need to store the outputs, which represent the
        # distributions of the next day prices.
        output15thPercentiles = []
        output25thPercentiles = []
        output35thPercentiles = []
        outputMedians = []
        output65thPercentiles = []
        output75thPercentiles = []
        output85thPercentiles = []
        date = startDate

        while date < endDate:
            print("Processing", date, "/", endDate)
            startIndex = df.index[df["Timestamp"] == date].tolist()

            if len(startIndex) == 0:
                date += self._dataTimeInterval
                closeMeans.append(closeMeans[-1])
                volumeMeans.append(volumeMeans[-1])
                continue

            startIndex = startIndex[0]
            endIndex = df.index[df["Timestamp"] == date + self._dataTimeInterval].tolist()

            if len(endIndex) == 0:
                date += self._dataTimeInterval
                closeMeans.append(closeMeans[-1])
                volumeMeans.append(volumeMeans[-1])
                continue

            endIndex = endIndex[0]
            data = df.iloc[startIndex : endIndex]
            closeMeans.append(data["Close"].mean())
            volumeMeans.append(data["Volume"].mean())
            date += self._dataTimeInterval

        # We will gather the next day distribution characteristics, so we start
        # at tomorrow.
        date = startDate + timedelta(days=self.numberOfSamples // self._datapointsPerDay)

        if self.dayByDay:
            advanceAmount = timedelta(hours=24)
        else:
            advanceAmount = self._dataTimeInterval

        while date < endDate:
            print("Processing", date, "/", endDate)
            startIndex = df.index[df["Timestamp"] == date].tolist()

            if len(startIndex) == 0:
                # date += self._dataTimeInterval * self._datapointsPerDay
                date += advanceAmount
                outputMedians.append(outputMedians[-1])
                output15thPercentiles.append(output15thPercentiles[-1])
                output25thPercentiles.append(output25thPercentiles[-1])
                output35thPercentiles.append(output35thPercentiles[-1])
                output65thPercentiles.append(output65thPercentiles[-1])
                output75thPercentiles.append(output75thPercentiles[-1])
                output85thPercentiles.append(output85thPercentiles[-1])
                continue

            startIndex = startIndex[0]
            endIndex = df.index[
                df["Timestamp"] == date + self._dataTimeInterval * self._datapointsPerDay].tolist()

            if len(endIndex) == 0:
                # date += self._dataTimeInterval * self._datapointsPerDay
                date += advanceAmount
                outputMedians.append(outputMedians[-1])
                output15thPercentiles.append(output15thPercentiles[-1])
                output25thPercentiles.append(output25thPercentiles[-1])
                output35thPercentiles.append(output35thPercentiles[-1])
                output65thPercentiles.append(output65thPercentiles[-1])
                output75thPercentiles.append(output75thPercentiles[-1])
                output85thPercentiles.append(output85thPercentiles[-1])
                continue

            endIndex = endIndex[0]
            data = df.iloc[startIndex: endIndex]["Close"]
            total = data.count()
            median = data.median()
            output15thPercentiles.append(data[data < median * 0.94].count() / total)
            output25thPercentiles.append(data[data < median * 0.96].count() / total)
            output35thPercentiles.append(data[data < median * 0.98].count() / total)
            outputMedians.append(data[data < median].count() / total)
            output65thPercentiles.append(data[data < median * 1.02].count() / total)
            output75thPercentiles.append(data[data < median * 1.04].count() / total)
            output85thPercentiles.append(data[data < median * 1.06].count() / total)
            # outputMedians.append(data.median())
            # output15thPercentiles.append(data.quantile(0.15))
            # output25thPercentiles.append(data.quantile(0.25))
            # output35thPercentiles.append(data.quantile(0.35))
            # output65thPercentiles.append(data.quantile(0.65))
            # output75thPercentiles.append(data.quantile(0.75))
            # output85thPercentiles.append(data.quantile(0.85))
            date += advanceAmount

        stock = StockDataFrame({
            'close': closeMeans
        })

        # The standard RSI is 14 day.
        rsis = (stock["rsi:112"] / 100).tolist()
        rsis2 = (stock["rsi:56"] / 100).tolist()
        rsis3 = (stock["rsi:14"] / 100).tolist()
        mas = stock["macd:96,208"].tolist()
        mas2 = stock["macd:24,52"].tolist()
        bollUppers = stock["boll.upper:160"].tolist()
        bollLowers = stock["boll.lower:160"].tolist()

        import math
        rsis = [0 if math.isnan(x) else x for x in rsis]
        rsis2 = [0 if math.isnan(x) else x for x in rsis2]
        rsis3 = [0 if math.isnan(x) else x for x in rsis3]
        mas = [0 if math.isnan(x) else x for x in mas]
        mas2 = [0 if math.isnan(x) else x for x in mas2]
        bollUppers = [0 if math.isnan(x) else x for x in bollUppers]
        bollLowers = [0 if math.isnan(x) else x for x in bollLowers]

        outputIndex = 0
        entryAmount = int((len(closeMeans) - self.numberOfSamples - 1))

        if self.dayByDay:
            advanceAmount = self._datapointsPerDay
        else:
            advanceAmount = 1

        # for i in range(0, len(closeMeans) - self.numberOfSamples - 1, self._datapointsPerDay):
        for i in range(0, len(closeMeans) - self.numberOfSamples - 1, advanceAmount):
            print("Percent of entries created: " + str(i / entryAmount * 100) + "%")
            close = closeMeans[i : i + self.numberOfSamples]
            meanClose = sum(close) / len(close)
            outputMedian = outputMedians[outputIndex] / meanClose

            if self.medianWithin is not None and outputMedian + abs(outputMedian - 1.0) > self.medianWithin:
                print("Skipping", outputMedian)
                outputIndex += 1
                continue

            volume = volumeMeans[i : i + self.numberOfSamples]
            rsi = rsis[i : i + self.numberOfSamples]
            rsi2 = rsis2[i: i + self.numberOfSamples]
            rsi3 = rsis3[i: i + self.numberOfSamples]
            ma = mas[i : i + self.numberOfSamples]
            ma2 = mas2[i : i + self.numberOfSamples]
            maxMA = max(mas)
            ma = [((m / maxMA) + 1) / 2 for m in ma]
            bollUpper = bollUppers[i: i + self.numberOfSamples]
            maxBollUpper = max(bollUpper)

            if maxBollUpper != 0:
                bollUpper = [m / maxBollUpper for m in bollUpper]

            bollLower = bollLowers[i: i + self.numberOfSamples]
            maxBollLower = max(bollLower)

            if maxBollLower != 0:
                bollLower = [m / maxBollLower for m in bollLower]

            maxClose = max(close)
            maxVolume = max(volume)

            for j in range(len(close)):
                close[j] /= maxClose

            for j in range(len(volume)):
                volume[j] /= maxVolume

            self.inputData.append([close, volume, rsi, rsi2, rsi3, ma, ma2, bollUpper, bollLower])
            # output15thPercentile = output15thPercentiles[outputIndex] / meanClose
            # output25thPercentile = output25thPercentiles[outputIndex] / meanClose
            # output35thPercentile = output35thPercentiles[outputIndex] / meanClose
            # output65thPercentile = output65thPercentiles[outputIndex] / meanClose
            # output7                close[j] /= maxClose

            for j in range(len(volume)):
                volume[j] /= maxVolume

            self.inputData.append([close, volume, rsi, rsi2, rsi3, ma, ma2, bollUpper, bollLower])
            # output15thPercentile = output15thPercentiles[outputIndex] / meanClose
            # output25thPercentile = output25thPercentiles[outputIndex] / meanClose
            # output35thPercentile = output35thPercentiles[outputIndex] / meanClose
            # output65thPercentile = output65thPercentiles[outputIndex] / meanClose
            # output75thPercentile = output75thPercentiles[outputIndex] / meanClose
            # output85thPercentile = output85thPercentiles[outputIndex] / meanClose
            output15thPercentile = output15thPercentiles[outputIndex]
            output25thPercentile = output25thPercentiles[outputIndex]
            output35thPercentile = output35thPercentiles[outputIndex]
            output65thPercentile = output65thPercentiles[outputIndex]
            output75thPercentile = output75thPercentiles[outputIndex]
            output85thPercentile ercentile = output85thPercentiles[outputIndex]

            # Use this if you want the model to predict if the price will be
            # higher or lower (instead or predicting the exact mean)
            # if output > 1.0:
            #     output = output85thPercentiles[outputIndex]

            # Use this if you want the model to predict if the price will be
            # higher or lower (instead or predicting the exact mean)
            # if output > 1.0:
            #     output centile = output75thPercentiles[outputIndex] / meanClose
            # output85thPercentile = output85thPercentiles[outputIndex] / meanClose
            output15thPercentile = output15thPercentiles[outputIndex]
            output25thPercentile = output25thPercentiles[outputIndex]
            output35thPercentile = output35thPercentiles[outputIndex]
            output65thPercentile = output65thPercentiles[outputIndex]
            output75thPercentile = output75thPercentiles[outputIndex]
            output85thPercentile = output85thPercentiles[outputIndex]

            # Use this if you want the model to predict if the price will be
            # higher or lower (instead or predicting the exact mean)
            # if output > 1.0:
            #     output 5thPercentile = output75thPercentiles[outputIndex] / meanClose
            # output85thPercentile = output85thPercentiles[outputIndex] / meanClose
            output15thPercentile = output15thPercentiles[outputIndex]
            output25thPercentile = output25thPercentiles[outputIndex]
            output35thPercentile = output35thPercentiles[outputIndex]
            output65thPercentile = output65thPercentiles[outputIndex]
            output75thPercentile = output75thPercentiles[outputIndex]
            output85thPercentile = output85thPercentiles[outputIndex]

            # Use this if you want the model to predict if the price will be
            # higher or lower (instead or predicting the exact mean)
            # if output > 1.0:
            #     output = 1.0
            # else:
            #     output = 0.0

            self.outputData.append([output15thPercentile,
                                    output25thPercentile,
                                    output35thPercentile,
                                    outputMedian,
                                    output65thPercentile,
                                    output75thPercentile,
                                    output85thPercentile])

            # self.outputData.append([int(output15thPercentile >= 1.0),
            #                         int(output25thPercentile >= 1.0),
            #                         int(output35thPercentile >= 1.0),
            #                         int(outputMedian >= 1.0),
            #                         int(output65thPercentile >= 1.0),
            #                         int(output75thPercentile >= 1.0),
            #                         int(output85thPercentile >= 1.0)])
            outputIndex += 1
