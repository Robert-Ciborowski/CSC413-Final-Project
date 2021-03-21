# Name: Create Data Set
# Author: Robert Ciborowski
# Date: 19/04/2020
# Description: A script which creates the dataset.

from data_set.BinanceDataSetCreator import BinanceDataSetCreator

if __name__ == "__main__":
    from datetime import datetime, timedelta
    from stock_data.HistoricalDataObtainer import \
        HistoricalDataObtainer

    listOfStocks = [
        # "LTCUSDT"
        "BTCUSDT"
        # "SPY"
        # "EUR"
    ]

    # The training dataset ranges from startDate to trainingEndDate.
    # The testing dataset ranges from trainingEndDate to endDate.

    # Bitcoin
    startDate = datetime(day=1, month=1, year=2018, hour=0, minute=0)
    trainingEndDate = datetime(day=1, month=1, year=2021, hour=0, minute=0)
    endDate = datetime(day=3, month=3, year=2021, hour=0, minute=0)
    historicalObtainer = HistoricalDataObtainer(startDate, endDate,
                                                       "../data_downloads/")

    # SPY
    # startDate = datetime(day=3, month=2, year=2004, hour=0, minute=0)
    # endDate = datetime(day=28, month=1, year=2021, hour=0, minute=0)
    # historicalObtainer = HistoricalBinanceDataObtainer(startDate, endDate + timedelta(days=2),
    #                                                    "../stocks_historical_data/")

    print("Reading historical stock data...")
    historicalObtainer.trackStocks(listOfStocks)

    # You can set how granular you want your data to be.
    # dataSetCreator = BinanceDataSetCreator(historicalObtainer, dataInterval="day", medianWithin=1.1)
    # dataSetCreator = BinanceDataSetCreator(historicalObtainer, dataInterval="hour")
    # dataSetCreator = BinanceDataSetCreator(historicalObtainer, dataInterval="2 hour")
    dataSetCreator = BinanceDataSetCreator(historicalObtainer, dataInterval="hour", dayByDay=False)
    print("Analyzing historical stock data for training dataset...")
    dataSetCreator.createDataset(listOfStocks[0], startDate, trainingEndDate)
    dataSetCreator.exportToCSV("final-train-dataset.csv")

    dataSetCreator = BinanceDataSetCreator(historicalObtainer, dataInterval="hour", dayByDay=False)
    print("Analyzing historical stock data for testing dataset...")
    dataSetCreator.createDataset(listOfStocks[0], trainingEndDate, endDate)
    dataSetCreator.exportToCSV("final-test-dataset.csv")
