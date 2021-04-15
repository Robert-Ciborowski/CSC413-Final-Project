# Name: BinanceDataSetCreator
# Author: Robert Ciborowski
# Date: 19/04/2020
# Description: Creates a data set from Binance data.

# Various rando libraries to test out.
from datetime import datetime, timedelta
from random import randint, uniform
from typing import List
import pytz
from stock_pandas import StockDataFrame
import pandas as pd
from ta.utils import dropna

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
        elif dataInterval == "4 hour":
            self._dataTimeInterval = timedelta(hours=4)
            self._datapointsPerDay = 6
        elif dataInterval == "6 hour":
            self._dataTimeInterval = timedelta(hours=6)
            self._datapointsPerDay = 4
        elif dataInterval == "12 hour":
            self._dataTimeInterval = timedelta(hours=12)
            self._datapointsPerDay = 2
        elif dataInterval == "15 min":
            self._dataTimeInterval = timedelta(minutes=15)
            self._datapointsPerDay = 96
        elif dataInterval == "5 min":
            self._dataTimeInterval = timedelta(minutes=5)
            self._datapointsPerDay = 288
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
                                + ["EMA-" + str(i) for i in range(self._numberOfSamples)]
                                + ["EMA2-" + str(i) for i in range(self._numberOfSamples)]
                                + ["MA1-" + str(i) for i in range(self._numberOfSamples)]
                                + ["MA2-" + str(i) for i in range(self._numberOfSamples)]
                                + ["BollUpper-" + str(i) for i in range(self._numberOfSamples)]
                                + ["BollLower-" + str(i) for i in range(self._numberOfSamples)]
                                + ["MFI-" + str(i) for i in range(self._numberOfSamples)]
                                + ["MFI2-" + str(i) for i in range(self._numberOfSamples)]
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

    def createAugmentedDataset(self, symbol: str, augmentationFactor: int, plannedValidationSplit: float, startDate, endDate, useAllIndicators=True):
        inputData = []
        outputData = []
        inputValidationData = []
        outputValidationData = []
        split = 1 - plannedValidationSplit

        for i in range(augmentationFactor):
            print("Augmentation " + str(i) + "...")
            self.createDataset(symbol, startDate, endDate, useAllIndicators)
            size = len(self.inputData)
            inputData += self.inputData[0 : int(split * size)]
            outputData += self.outputData[0 : int(split * size)]
            inputValidationData += self.inputData[int(split * size) : size]
            outputValidationData += self.outputData[int(split * size) : size]

        self.inputData = inputData + inputValidationData
        self.outputData = outputData + outputValidationData

    def createDataset(self, symbol: str, startDate, endDate, useAllIndicators=True, isAugmenting=False):
        """
        Creates a dataset. Please make sure that the start and end dates are
        the beginnings of days.
        :param symbol: e.g. "BTCUSDT"
        :param startDate: e.g. datetime(year=2020, month=1, day=1)
        :param endDate: e.g. datetime(year=2020, month=2, day=1)
        :param useAllIndicators: if False, only uses the minimum indicators
        """
        # These are time-related variables.
        timezone = "Etc/GMT-0"
        timezone = pytz.timezone(timezone)
        outputStartDate = startDate
        # We need to go back a little earlier to generate indicators such as RSI.
        startDate -= timedelta(days=DAYS_IN_AN_INPUT + 60)
        endDate = timezone.localize(endDate)
        startDate = timezone.localize(startDate)
        # outputStartDate = timezone.localize(outputStartDate)

        # We will be collecting our final features and labels in here:
        self.inputData = []
        self.outputData = []

        # This dataframe has all the raw data we need to generate the dataset.
        df = self.dataObtainer.getHistoricalDataAsDataframe(symbol)

        # First, we will gather all of the means for our inputs...
        closeMeans = []
        volumeMeans = []

        # ... also, we will gather the outputs, which represent the
        # distributions of the next day prices.
        output15thPercentiles = []
        output25thPercentiles = []
        output35thPercentiles = []
        outputMedians = []
        output65thPercentiles = []
        output75thPercentiles = []
        output85thPercentiles = []

        # We will use this to normalize our outputs by dividing them by the
        # mean price of the last (latest/most recent) day in our input.
        priceMeansToDivideLabelsBy = []
        volumeMeansToDivideLabelsBy = []
        date = startDate

        # Now we will be collecting the input prices, input volumes, and output
        # percentiles.
        while date < endDate:
            print("Processing", date, "/", endDate)
            # First, we will collect the start and end dates for this input
            # point (which consists of 3 hours of data if that is our input
            # time interval). Then we calculate the mean price and volume for
            # this input data point.
            startIndex = df.index[df["Timestamp"] == date].tolist()

            # If this if condition is true, then we may be missing some data in
            # our dataset. I think this happens during times when Binance was
            # down. In this case, we just use the previous data.
            if len(startIndex) == 0:
                date += self._dataTimeInterval
                closeMeans.append(closeMeans[-1])
                volumeMeans.append(volumeMeans[-1])
                outputMedians.append(outputMedians[-1])
                output15thPercentiles.append(output15thPercentiles[-1])
                output25thPercentiles.append(output25thPercentiles[-1])
                output35thPercentiles.append(output35thPercentiles[-1])
                output65thPercentiles.append(output65thPercentiles[-1])
                output75thPercentiles.append(output75thPercentiles[-1])
                output85thPercentiles.append(output85thPercentiles[-1])
                priceMeansToDivideLabelsBy.append(priceMeansToDivideLabelsBy[-1])
                volumeMeansToDivideLabelsBy.append(volumeMeansToDivideLabelsBy[-1])
                continue

            startIndex = startIndex[0]
            endIndex = df.index[df["Timestamp"] == date + self._dataTimeInterval].tolist()

            if len(endIndex) == 0:
                date += self._dataTimeInterval
                closeMeans.append(closeMeans[-1])
                volumeMeans.append(volumeMeans[-1])
                outputMedians.append(outputMedians[-1])
                output15thPercentiles.append(output15thPercentiles[-1])
                output25thPercentiles.append(output25thPercentiles[-1])
                output35thPercentiles.append(output35thPercentiles[-1])
                output65thPercentiles.append(output65thPercentiles[-1])
                output75thPercentiles.append(output75thPercentiles[-1])
                output85thPercentiles.append(output85thPercentiles[-1])
                priceMeansToDivideLabelsBy.append(priceMeansToDivideLabelsBy[-1])
                volumeMeansToDivideLabelsBy.append(volumeMeansToDivideLabelsBy[-1])
                continue

            endIndex = endIndex[0]
            data = df.iloc[startIndex : endIndex]
            closeMeans.append(data["Close"].mean())
            volumeMeans.append(data["Volume"].mean())

            # Now we get the start and end dates for output data that would
            # be associated with an entry that begins at the data point found
            # above. Then we calculate the percentiles for the output.
            date2 = date + timedelta(days=DAYS_IN_AN_INPUT)
            startIndex = df.index[df["Timestamp"] == date2].tolist()

            if len(startIndex) == 0:
                date += self._dataTimeInterval
                outputMedians.append(outputMedians[-1])
                output15thPercentiles.append(output15thPercentiles[-1])
                output25thPercentiles.append(output25thPercentiles[-1])
                output35thPercentiles.append(output35thPercentiles[-1])
                output65thPercentiles.append(output65thPercentiles[-1])
                output75thPercentiles.append(output75thPercentiles[-1])
                output85thPercentiles.append(output85thPercentiles[-1])
                priceMeansToDivideLabelsBy.append(priceMeansToDivideLabelsBy[-1])
                volumeMeansToDivideLabelsBy.append(volumeMeansToDivideLabelsBy[-1])
                continue

            startIndex = startIndex[0]
            date2 += timedelta(days=1)
            endIndex = df.index[df["Timestamp"] == date2].tolist()

            if len(endIndex) == 0:
                date += self._dataTimeInterval
                outputMedians.append(outputMedians[-1])
                output15thPercentiles.append(output15thPercentiles[-1])
                output25thPercentiles.append(output25thPercentiles[-1])
                output35thPercentiles.append(output35thPercentiles[-1])
                output65thPercentiles.append(output65thPercentiles[-1])
                output75thPercentiles.append(output75thPercentiles[-1])
                output85thPercentiles.append(output85thPercentiles[-1])
                priceMeansToDivideLabelsBy.append(priceMeansToDivideLabelsBy[-1])
                volumeMeansToDivideLabelsBy.append(volumeMeansToDivideLabelsBy[-1])
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

            # Lastly, we need to get the last input day's mean price, which we
            # use to normalize our output percentiles.
            date3 = date + timedelta(days=DAYS_IN_AN_INPUT - 1)
            startIndex = df.index[df["Timestamp"] == date3].tolist()

            if len(startIndex) == 0:
                date += self._dataTimeInterval
                priceMeansToDivideLabelsBy.append(priceMeansToDivideLabelsBy[-1])
                volumeMeansToDivideLabelsBy.append(volumeMeansToDivideLabelsBy[-1])
                continue

            startIndex = startIndex[0]
            date3 = date + timedelta(days=DAYS_IN_AN_INPUT)
            endIndex = df.index[df["Timestamp"] == date3].tolist()

            if len(endIndex) == 0:
                date += self._dataTimeInterval
                priceMeansToDivideLabelsBy.append(priceMeansToDivideLabelsBy[-1])
                volumeMeansToDivideLabelsBy.append(volumeMeansToDivideLabelsBy[-1])
                continue

            endIndex = endIndex[0]
            data = df.iloc[startIndex: endIndex]
            priceMeansToDivideLabelsBy.append(data["Close"].mean())
            volumeMeansToDivideLabelsBy.append(data["Volume"].mean())
            date += self._dataTimeInterval

        # Now that our while loop above collected data for inputs and
        # outputs, we need to generate technical indicators as additional
        # input features. We seem to be getting good performance if we only
        # use close, volume, rsi, ema and mfi, but we also have some other
        # indicators to play around with, such as ma and an additional rsi
        # with a different parameter.
        stock = StockDataFrame({
            "close": closeMeans,
            "volume": volumeMeans
        })

        # The standard RSI is 14 day. Note that if our time interval is 3 hrs,
        # there are 8 data points in a day. Thus, a 14 day RSI is a 112-RSI
        # because 14 * 8 = 112.
        rsis = (stock["rsi:112"] / 100).tolist()
        rsis2 = (stock["rsi:14"] / 100).tolist()
        emas = (stock["ema:21"]).tolist()
        macds = stock["macd:96,208"].tolist()
        macds2 = stock["macd:24,52"].tolist()
        bollUppers = stock["boll.upper:160"].tolist()
        bollLowers = stock["boll.lower:160"].tolist()
        from ta.volume import MFIIndicator
        moneyFlowIndex = MFIIndicator(stock["close"], stock["close"], stock["close"], stock["volume"], window=14)
        mfis = (moneyFlowIndex.money_flow_index().divide(100)).to_list()

        # This gets rid of NANs in our indicators (just in case).
        import math
        rsis = [0 if math.isnan(x) else x for x in rsis]
        rsis2 = [0 if math.isnan(x) else x for x in rsis2]
        emas = [0 if math.isnan(x) else x for x in emas]
        macds = [0 if math.isnan(x) else x for x in macds]
        macds2 = [0 if math.isnan(x) else x for x in macds2]
        bollUppers = [0 if math.isnan(x) else x for x in bollUppers]
        bollLowers = [0 if math.isnan(x) else x for x in bollLowers]
        mfis = [0 if math.isnan(x) else x for x in mfis]

        # Now we will generate our final inputs and outputs! See the for loop
        # below.
        entryAmount = int((len(closeMeans) - self._numberOfSamples - 1))

        if self.dayByDay:
            advanceAmount = self._datapointsPerDay
        else:
            advanceAmount = 1

        def fixWithin0And1(x):
            return min(max(x, 0.0), 1.0)

        for i in range(60 * self._datapointsPerDay, entryAmount, advanceAmount):
            print("Percent of entries created: " + str(i / entryAmount * 100) + "%")
            yesterdayCloseMean = priceMeansToDivideLabelsBy[i]
            yesterdayVolumeMean = volumeMeansToDivideLabelsBy[i]
            # This gets the input features and outputs for this dataset entry.
            close = closeMeans[i : i + self._numberOfSamples]
            volume = volumeMeans[i : i + self._numberOfSamples]
            rsi = rsis[i : i + self._numberOfSamples]
            rsi2 = rsis2[i: i + self._numberOfSamples]
            ema = emas[i: i + self._numberOfSamples]
            macd = macds[i: i + self._numberOfSamples]
            macd2 = macds2[i: i + self._numberOfSamples]
            ema = [fixWithin0And1(m / yesterdayCloseMean / 2) for m in ema]
            macd = [fixWithin0And1(m / yesterdayCloseMean / 2 + 0.5) for m in macd]
            macd2 = [fixWithin0And1(m / yesterdayCloseMean / 2 + 0.5) for m in macd2]
            mfi = mfis[i: i + self._numberOfSamples]
            bollUpper = bollUppers[i: i + self._numberOfSamples]
            bollUpper = [fixWithin0And1(m / yesterdayCloseMean / 2) for m in bollUpper]
            bollLower = bollLowers[i: i + self._numberOfSamples]
            bollLower = [fixWithin0And1(m / yesterdayCloseMean / 2) for m in bollLower]

            for j in range(len(close)):
                close[j] = fixWithin0And1(close[j] / yesterdayCloseMean / 2)

            for j in range(len(volume)):
                volume[j] = fixWithin0And1(volume[j] / yesterdayVolumeMean / 2)

            # Finally, we add the entry to the dataset.
            if useAllIndicators:
                self.inputData.append([close, volume, rsi, rsi2, ema, macd, macd2,
                                       bollUpper, bollLower, mfi])
            else:
                self.inputData.append([close, volume, rsi, ema, mfi])

            # This normalizes our data. 0.5 means that the percentile is the same
            # as the last day's mean. 1.0 means that the percentile is twice the
            # value of the last day's mean. We normalize in this way so that we
            # can use the sigmoid activation function for the outputs, which

            output15thPercentile = output15thPercentiles[i] / yesterdayCloseMean / 2
            output25thPercentile = output25thPercentiles[i] / yesterdayCloseMean / 2
            output35thPercentile = output35thPercentiles[i] / yesterdayCloseMean / 2
            outputMedian = outputMedians[i] / yesterdayCloseMean / 2
            output65thPercentile = output65thPercentiles[i] / yesterdayCloseMean / 2
            output75thPercentile = output75thPercentiles[i] / yesterdayCloseMean / 2
            output85thPercentile = output85thPercentiles[i] / yesterdayCloseMean / 2
            self.outputData.append([
                                    output15thPercentile,
                                    output25thPercentile,
                                    output35thPercentile,
                                    outputMedian,
                                    output65thPercentile,
                                    output75thPercentile,
                                    output85thPercentile
            ])
