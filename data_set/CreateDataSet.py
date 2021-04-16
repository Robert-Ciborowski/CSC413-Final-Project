# Name: Create Data Set
# Author: Robert Ciborowski
# Date: 19/04/2020
# Description: A script which creates the datasets for training and testing.

from data_set.DataSetCreator import DataSetCreator

if __name__ == "__main__":
    from datetime import datetime
    from stock_data.HistoricalDataObtainer import \
        HistoricalDataObtainer

    # What to add to our dataset. We can have multiple assets such as BTC and
    # LTC, but for experimentation purposes it is better to only add one asset
    # for each dataset.
    listOfStocks = [
        # "LTCUSDT"
        "BTCUSDT"
        # "SPY"
        # "ETHUSDT"
        # "EUR"
    ]

    # The training dataset ranges from startDate to trainingEndDate.
    # The testing dataset ranges from trainingEndDate to endDate.

    # startDate = datetime(day=1, month=10, year=2017, hour=0, minute=0)
    # trainingStartDate = datetime(day=1, month=1, year=2018, hour=0, minute=0)
    # trainingEndDate = datetime(day=15, month=3, year=2019, hour=0, minute=0)
    # endDate = datetime(day=7, month=4, year=2019, hour=0, minute=0)

    # Some good dates for if you want a Bitcoin or Ethereum dataset.
    startDate = datetime(day=1, month=10, year=2017, hour=0, minute=0)
    trainingStartDate = datetime(day=1, month=1, year=2018, hour=0, minute=0)
    trainingEndDate = datetime(day=1, month=1, year=2021, hour=0, minute=0)
    endDate = datetime(day=21, month=3, year=2021, hour=0, minute=0)

    # Some good data for if you want a Litecoin dataset.
    # startDate = datetime(day=1, month=1, year=2018, hour=0, minute=0)
    # trainingStartDate = datetime(day=1, month=4, year=2018, hour=0, minute=0)
    # trainingEndDate = datetime(day=1, month=1, year=2021, hour=0, minute=0)
    # endDate = datetime(day=21, month=3, year=2021, hour=0, minute=0)

    # Some good dates for if you want a Euro dataset.
    # startDate = datetime(day=10, month=3, year=2008, hour=0, minute=0)
    # trainingStartDate = datetime(day=1, month=6, year=2008, hour=0, minute=0)
    # trainingEndDate = datetime(day=1, month=1, year=2009, hour=0, minute=0)
    # endDate = datetime(day=1, month=3, year=2009, hour=0, minute=0)

    historicalObtainer = HistoricalDataObtainer(startDate, endDate,
                                                       "../data_downloads/")

    print("Reading historical stock data...")
    historicalObtainer.trackStocks(listOfStocks)

    # You can set how granular you want your input data intervals to be. We
    # recommend using 3 hours as going more granular does not seem to affect
    # model performance.
    # dataSetCreator = DataSetCreator(historicalObtainer, dataInterval="day", medianWithin=1.1)
    # dataSetCreator = DataSetCreator(historicalObtainer, dataInterval="hour", dayByDay=False)
    # dataSetCreator = DataSetCreator(historicalObtainer, dataInterval="2 hour", dayByDay=False)
    dataSetCreator = DataSetCreator(historicalObtainer, dataInterval="3 hour", dayByDay=False)
    # dataSetCreator = DataSetCreator(historicalObtainer, dataInterval="1 hour", dayByDay=False)
    # dataSetCreator = DataSetCreator(historicalObtainer, dataInterval="15 min", dayByDay=False)
    # dataSetCreator = DataSetCreator(historicalObtainer, dataInterval="5 min", dayByDay=False)

    # This creates the training and testing datasets. You can also set the
    # useAllIndicators parameter to False if you only want the best indicators.
    print("Analyzing historical stock data for training dataset...")
    dataSetCreator.createAugmentedDataset(listOfStocks[0], 4, 0.1, trainingStartDate, trainingEndDate, useAllIndicators=True)
    # dataSetCreator.createDataset(listOfStocks[0], trainingStartDate, trainingEndDate, useAllIndicators=True)
    dataSetCreator.exportToCSV("final-train-dataset.csv")

    dataSetCreator = DataSetCreator(historicalObtainer, dataInterval="3 hour", dayByDay=False)
    print("Analyzing historical stock data for testing dataset...")
    dataSetCreator.createDataset(listOfStocks[0], trainingEndDate, endDate, useAllIndicators=True)
    dataSetCreator.exportToCSV("final-test-dataset.csv")
