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

from util.Constants import DAYS_IN_AN_INPUT

class DataSetCreator:
    dataObtainer: HistoricalDataObtainer
    _numberOfSamples: int
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
                             data. Can be "day", "hour", "2 hour", "3 hour"
        """
        self.dataObtainer = dataObtainer
        self.medianWithin = medianWithin
        self.dayByDay = dayByDay

        if dataInterval == "day":
            self._dataTimeInterval = timedelta(days=1)
            self._datapointsPerDay = 1
        elif dataInterval == "2 hour":
            self._dataTimeInterval = timedelta(hours=2)
            self._datapointsPerDay = 12
        elif dataInterval == "3 hour":
            self._dataTimeInterval = timedelta(hours=3)
            self._datapointsPerDay = 8
        else:
            # 1 hour
            self._dataTimeInterval = timedelta(hours=1)
            self._datapointsPerDay = 24

        self._numberOfSamples = DAYS_IN_AN_INPUT * self._datapointsPerDay

    def exportToCSV(self, path: str, pathPrefix=""):
        if len(self.inputData) == 0:
            return

        try:
            with open(pathPrefix + path, 'w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(["Close-" + str(i) for i in range(self._numberOfSamples)]
                                + ["Volume-" + str(i) for i in range(self._numberOfSamples)]
                                + ["RSI1-" + str(i) for i in range(self._numberOfSamples)]
                                + ["RSI2-" + str(i) for i in range(self._numberOfSamples)]
                                + ["RSI3-" + str(i) for i in range(self._numberOfSamples)]
                                + ["MACD1-" + str(i) for i in range(self._numberOfSamples)]
                                + ["MACD2-" + str(i) for i in range(self._numberOfSamples)]
                                + ["BollUpper-" + str(i) for i in range(self._numberOfSamples)]
                                + ["BollLower-" + str(i) for i in range(self._numberOfSamples)]
                                + ["15th-Percentile", "25th-Percentile",
                                   "35th-Percentile", "Median", "65th-Percentile",
                                   "75th-Percentile", "85th-Percentile"])

                print("Length of input data", len(self.inputData))
                print("Length of output data", len(self.outputData))
                print("The values above should be the same!")
                assert len(self.inputData) == len(self.outputData)

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
        outputStartDate = startDate
        # We need to go back a little earlier to generate indicators such as RSI.
        startDate -= timedelta(days=DAYS_IN_AN_INPUT + 60)
        endDate = timezone.localize(endDate)
        startDate = timezone.localize(startDate)
        outputStartDate = timezone.localize(outputStartDate)
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
        date = startDate + timedelta(
            days=self._numberOfSamples // self._datapointsPerDay)

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
                df[
                    "Timestamp"] == date + self._dataTimeInterval * self._datapointsPerDay].tolist()

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
            outputMedians.append(data.median())
            output15thPercentiles.append(data.quantile(0.15))
            output25thPercentiles.append(data.quantile(0.25))
            output35thPercentiles.append(data.quantile(0.35))
            output65thPercentiles.append(data.quantile(0.65))
            output75thPercentiles.append(data.quantile(0.75))
            output85thPercentiles.append(data.quantile(0.85))
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
        entryAmount = int((len(closeMeans) - self._numberOfSamples - 1))
        outputIndex = 0

        if self.dayByDay:
            advanceAmount = self._datapointsPerDay
        else:
            advanceAmount = 1

        for i in range(60 * self._datapointsPerDay, len(closeMeans) - self._numberOfSamples - 1, advanceAmount):
            print("Percent of entries created: " + str(i / entryAmount * 100) + "%")
            # Get inputs
            close = closeMeans[i : i + self._numberOfSamples]
            l = len(close)
            volume = volumeMeans[i : i + self._numberOfSamples]
            rsi = rsis[i : i + self._numberOfSamples]
            rsi2 = rsis2[i: i + self._numberOfSamples]
            rsi3 = rsis3[i: i + self._numberOfSamples]
            ma = mas[i : i + self._numberOfSamples]
            ma2 = mas2[i : i + self._numberOfSamples]
            maxMA = max(mas)
            ma = [((m / maxMA) + 1) / 2 for m in ma]
            maxMA2 = max(mas2)
            ma2 = [((m / maxMA2) + 1) / 2 for m in ma2]
            bollUpper = bollUppers[i: i + self._numberOfSamples]
            maxBollUpper = max(bollUpper)

            if maxBollUpper != 0:
                bollUpper = [m / maxBollUpper for m in bollUpper]

            bollLower = bollLowers[i: i + self._numberOfSamples]
            maxBollLower = max(bollLower)

            if maxBollLower != 0:
                bollLower = [m / maxBollLower for m in bollLower]

            maxClose = max(close)
            maxVolume = max(volume)

            for j in range(len(close)):
                close[j] /= maxClose

            for j in range(len(volume)):
                volume[j] /= maxVolume

            # Generate labels
            date = startDate + self._dataTimeInterval * i
            startIndex = df.index[df["Timestamp"] == date].tolist()

            if len(startIndex) == 0:
                continue

            startIndex = startIndex[0]
            endIndex = df.index[df["Timestamp"] == date + self._dataTimeInterval * self._datapointsPerDay].tolist()

            if len(endIndex) == 0:
                continue

            endIndex = endIndex[0]
            # data = df.iloc[startIndex: endIndex]["Close"]

            # 1440 minutes in a day.
            yesterdayCloseMean = df.iloc[startIndex - 1440: endIndex - 1440]["Close"].mean()

            # Final stuff to add to dataset:
            self.inputData.append([close, volume, rsi, rsi2, rsi3, ma, ma2, bollUpper, bollLower])

            output15thPercentile = output15thPercentiles[outputIndex] / yesterdayCloseMean
            output25thPercentile = output25thPercentiles[outputIndex] / yesterdayCloseMean
            output35thPercentile = output35thPercentiles[outputIndex] / yesterdayCloseMean
            outputMedian = outputMedians[outputIndex] / yesterdayCloseMean
            output65thPercentile = output65thPercentiles[outputIndex] / yesterdayCloseMean
            output75thPercentile = output75thPercentiles[outputIndex] / yesterdayCloseMean
            output85thPercentile = output85thPercentiles[outputIndex] / yesterdayCloseMean
            self.outputData.append([output15thPercentile,
                                    output25thPercentile,
                                    output35thPercentile,
                                    outputMedian,
                                    output65thPercentile,
                                    output75thPercentile,
                                    output85thPercentile])
            # self.outputData.append([int(data.mean() >= yesterdayCloseMean)])
            outputIndex += 1
