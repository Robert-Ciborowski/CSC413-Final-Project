# Name: Create Data Set
# Author: Robert Ciborowski
# Date: 19/04/2020
# Description: A script which creates the dataset.

from data_set.BinanceDataSetCreator import BinanceDataSetCreator

if __name__ == "__main__":
    from datetime import datetime, timedelta
    from stock_data.HistoricalBinanceDataObtainer import \
        HistoricalBinanceDataObtainer

    listOfStocks = [
        # "LTCUSDT"
        "BTCUSDT"
    ]

    startDate = datetime(day=1, month=1, year=2018, hour=0, minute=0)
    endDate = datetime(day=19, month=2, year=2021, hour=0, minute=0)

    historicalObtainer = HistoricalBinanceDataObtainer(startDate, endDate + timedelta(days=2),
        "../binance_historical_data/")
    print("Reading historical stock data...")
    historicalObtainer.trackStocks(listOfStocks)

    # set timeInterval to 60 when creating actual dataset!
    # dataSetCreator = BinanceDataSetCreator(historicalObtainer, medianWithin=1.1)
    dataSetCreator = BinanceDataSetCreator(historicalObtainer, dataInterval="day")
    print("Analyzing historical stock data...")
    dataSetCreator.createDataset(listOfStocks[0], startDate, endDate)
    dataSetCreator.exportToCSV("final-dataset.csv")
