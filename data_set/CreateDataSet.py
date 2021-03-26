# Name: Create Data Set
# Author: Robert Ciborowski
# Date: 19/04/2020
# Description: A script which creates the datasets for training and testing.

from data_set.DataSetCreator import DataSetCreator

if __name__ == "__main__":
    from datetime import datetime
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
    startDate = datetime(day=1, month=10, year=2017, hour=0, minute=0)
    trainingStartDate = datetime(day=1, month=1, year=2018, hour=0, minute=0)
    trainingEndDate = datetime(day=1, month=1, year=2021, hour=0, minute=0)
    endDate = datetime(day=21, month=3, year=2021, hour=0, minute=0)
    historicalObtainer = HistoricalDataObtainer(startDate, endDate,
                                                       "../data_downloads/")

    print("Reading historical stock data...")
    historicalObtainer.trackStocks(listOfStocks)

    # You can set how granular you want your data to be.
    # dataSetCreator = BinanceDataSetCreator(historicalObtainer, dataInterval="day", medianWithin=1.1)
    # dataSetCreator = BinanceDataSetCreator(historicalObtainer, dataInterval="hour")
    # dataSetCreator = BinanceDataSetCreator(historicalObtainer, dataInterval="2 hour")
    dataSetCreator = DataSetCreator(historicalObtainer, dataInterval="3 hour", dayByDay=False)
    print("Analyzing historical stock data for training dataset...")
    dataSetCreator.createDataset(listOfStocks[0], trainingStartDate, trainingEndDate)
    dataSetCreator.exportToCSV("final-train-dataset.csv")

    dataSetCreator = DataSetCreator(historicalObtainer, dataInterval="3 hour", dayByDay=False)
    print("Analyzing historical stock data for testing dataset...")
    dataSetCreator.createDataset(listOfStocks[0], trainingEndDate, endDate)
    dataSetCreator.exportToCSV("final-test-dataset.csv")
