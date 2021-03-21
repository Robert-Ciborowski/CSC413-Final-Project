import pytz
from stock_pandas import StockDataFrame

from data_set.BinanceDataSetCreator import BinanceDataSetCreator
from models.CnnRnnMlpModel import CnnRnnMlpModel
from models.Hyperparameters import Hyperparameters
from datetime import datetime, timedelta

from stock_data.HistoricalDataObtainer import HistoricalDataObtainer
import numpy as np

def testPerformance():
    # We are going to need data from 15 days ago up to and including yesterday.
    # Make sure to download all the data using data_downloads/download_binance_data.py!
    # now = datetime.now()
    now = datetime(year=2021, month=3, day=3, hour=23, minute=35)
    timezone = "Etc/GMT-0"
    timezone = pytz.timezone(timezone)
    now = timezone.localize(now)
    now = now.replace(hour=0, minute=0, second=0, microsecond=0)
    startDate = now - timedelta(days=193)
    listOfStocks = ["BTCUSDT"]
    historicalObtainer = HistoricalDataObtainer(startDate, now,
                                                "../data_downloads/")

    print("Reading historical stock data...")
    historicalObtainer.trackStocks(listOfStocks)

    model = CnnRnnMlpModel(tryUsingGPU=False)

    # Hyperparameters!
    learningRate = 0.003
    epochs = 500
    batchSize = 40
    decayRate = 0.005
    decayStep = 1.0
    dropout = 0.1
    model.setup(Hyperparameters(learningRate, epochs, dropout, batchSize,
                                decayRate=decayRate, decayStep=decayStep))
    model.createModel()
    model.load()

    inputData = prepareData(listOfStocks[0], historicalObtainer, startDate, now)
    inputData = np.array(inputData[-1]).T
    df = historicalObtainer.getHistoricalDataAsDataframe(listOfStocks[0])

    startIndex = df.index[df["Timestamp"] == now - timedelta(days=15)].tolist()
    assert len(startIndex) != 0
    startIndex = startIndex[0]

    endIndex = df.index[df["Timestamp"] == now].tolist()
    assert len(endIndex) != 0
    endIndex = endIndex[0]

    meanPrice = df["Close"].iloc[startIndex : endIndex].mean()
    predictions = model.makePricePredictionForTommorrow(inputData, meanPrice)
    print("Predictions: [min, 25th percentile, median, 75th percentile, max]")
    print(predictions)

def prepareData(symbol, dataObtainer, startDate, endDate):
    df = dataObtainer.getHistoricalDataAsDataframe(symbol)

    # We gather all of the means for our inputs.
    closeMeans = []
    volumeMeans = []
    date = startDate

    dataTimeInterval = timedelta(hours=3)
    datapointsPerDay = 8
    numberOfSamples = 15 * 8

    while date < endDate:
        print("Processing", date, "/", endDate)
        startIndex = df.index[df["Timestamp"] == date].tolist()

        if len(startIndex) == 0:
            date += dataTimeInterval
            closeMeans.append(closeMeans[-1])
            volumeMeans.append(volumeMeans[-1])
            continue

        startIndex = startIndex[0]
        endIndex = df.index[
            df["Timestamp"] == date + dataTimeInterval].tolist()

        if len(endIndex) == 0:
            date += dataTimeInterval
            closeMeans.append(closeMeans[-1])
            volumeMeans.append(volumeMeans[-1])
            continue

        endIndex = endIndex[0]
        data = df.iloc[startIndex: endIndex]
        closeMeans.append(data["Close"].mean())
        volumeMeans.append(data["Volume"].mean())
        date += dataTimeInterval

    stock = StockDataFrame({
        'close': closeMeans
    })

    # The standard RSI is 14 day.
    rsis = (stock["rsi:112"] / 100).tolist()
    rsis2 = (stock["rsi:56"] / 100).tolist()
    rsis3 = (stock["rsi:28"] / 100).tolist()
    mas = stock["macd:96,208,72"].tolist()
    mas2 = stock["macd:48,104,36"].tolist()
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
    entryAmount = int((len(closeMeans) - numberOfSamples - 1))
    formattedData = []

    for i in range(0, len(closeMeans) - numberOfSamples, datapointsPerDay):
        print("Percent of entries created: " + str(i / entryAmount * 100) + "%")
        close = closeMeans[i: i + numberOfSamples]
        meanClose = sum(close) / len(close)

        volume = volumeMeans[i: i + numberOfSamples]
        rsi = rsis[i: i + numberOfSamples]
        rsi2 = rsis2[i: i + numberOfSamples]
        rsi3 = rsis3[i: i + numberOfSamples]
        ma = mas[i: i + numberOfSamples]
        ma2 = mas2[i: i + numberOfSamples]
        maxMA = max(mas)
        ma = [((m / maxMA) + 1) / 2 for m in ma]
        bollUpper = bollUppers[i: i + numberOfSamples]
        maxBollUpper = max(bollUpper)
        bollUpper = [m / maxBollUpper for m in bollUpper]
        bollLower = bollLowers[i: i + numberOfSamples]
        maxBollLower = max(bollLower)
        bollLower = [m / maxBollLower for m in bollLower]
        maxClose = max(close)
        maxVolume = max(volume)

        for j in range(len(close)):
            close[j] /= maxClose

        for j in range(len(volume)):
            volume[j] /= maxVolume

        formattedData.append([close, volume, rsi, rsi2, rsi3, ma, ma2, bollUpper, bollLower])
        outputIndex += 1

    return formattedData

if __name__ == "__main__":
    testPerformance()
