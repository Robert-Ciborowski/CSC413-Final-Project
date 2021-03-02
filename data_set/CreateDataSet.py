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
        # "SPY"
    ]

    # Bitcoin
    startDate = datetime(day=1, month=1, year=2018, hour=0, minute=0)
    # endDate = datetime(day=19, month=3, year=2018, hour=0, minute=0)
    endDate = datetime(day=19, month=2, year=2021, hour=0, minute=0)
    historicalObtainer = HistoricalBinanceDataObtainer(startDate, endDate + timedelta(days=2),
                                                       "../binance_historical_data/")

    # SPY
    # startDate = datetime(day=3, month=2, year=2004, hour=0, minute=0)
    # endDate = datetime(day=28, month=1, year=2021, hour=0, minute=0)
    # historicalObtainer = HistoricalBinanceDataObtainer(startDate, endDate + timedelta(days=2),
    #                                                    "../stocks_historical_data/")

    print("Reading historical stock data...")
    historicalObtainer.trackStocks(listOfStocks)

    # set timeInterval to 60 when creating actual dataset!
    # dataSetCreator = BinanceDataSetCreator(historicalObtainer, dataInterval="day", medianWithin=1.1)
    # dataSetCreator = BinanceDataSetCreator(historicalObtainer, dataInterval="hour")
    # dataSetCreator = BinanceDataSetCreator(historicalObtainer, dataInterval="2 hour")
    dataSetCreator = BinanceDataSetCreator(historicalObtainer, dataInterval="3 hour")
    print("Analyzing historical stock data...")
    dataSetCreator.createDataset(listOfStocks[0], startDate, endDate)
    dataSetCreator.exportToCSV("final-dataset.csv")
