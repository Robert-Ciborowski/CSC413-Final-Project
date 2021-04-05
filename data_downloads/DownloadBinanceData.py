# Name: Download Binance Data
# Author: Robert Ciborowski
# Date: 18/04/2020
# Description: A script which downloads historical data from Binance.
#              It can also update current data with new data.
#
#              See the if __name__ == "__main__" at the bottom of this
#              file.

# credit to https://medium.com/swlh/retrieving-full-historical-data-for-every-cryptocurrency-on-binance-bitmex-using-the-python-apis-27b47fd8137f
# for some of the functions in this script.

from typing import List
import pandas as pd
import math
import os.path
from binance.client import Client
from datetime import timedelta, datetime
from dateutil import parser

# Random API stuff: ignore
binance_api_key = "Redacted"
binance_api_secret = "Redacted"

# Constants and globals.
binsizes = {"1m": 1, "5m": 5, "1h": 60, "1d": 1440}
batch_size = 750
binance_client = Client(api_key=binance_api_key, api_secret=binance_api_secret)

def minutesOfNewData(symbol, kline_size, data):
    if len(data) > 0:  old = parser.parse(data["timestamp"].iloc[-1])
    old = datetime.strptime('1 Jan 2017', '%d %b %Y')
    new = pd.to_datetime(binance_client.get_klines(symbol=symbol, interval=kline_size)[-1][0], unit='ms')
    return old, new

def getAllBinance(symbol, kline_size, save = False, oldest=None):
    filename = '%s-%s-data.csv' % (symbol, kline_size)
    if os.path.isfile(filename): data_df = pd.read_csv(filename)
    else: data_df = pd.DataFrame()
    oldest_point, newest_point = minutesOfNewData(symbol, kline_size, data_df)
    if oldest is not None:
        oldest_point = oldest
    elif len(data_df) > 0:
        oldest_point = pd.to_datetime(data_df["timestamp"].iloc[-1])
    delta_min = (newest_point - oldest_point).total_seconds()/60
    available_data = math.ceil(delta_min/binsizes[kline_size])
    if oldest_point == datetime.strptime('1 Jan 2017', '%d %b %Y'): print('Downloading all available %s data for %s. Be patient..!' % (kline_size, symbol))
    else: print('Downloading %d minutes of new data available for %s, i.e. %d instances of %s data.' % (delta_min, symbol, available_data, kline_size))
    klines = binance_client.get_historical_klines(symbol, kline_size, oldest_point.strftime("%d %b %Y %H:%M:%S"), newest_point.strftime("%d %b %Y %H:%M:%S"))
    data = pd.DataFrame(klines, columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'quote_av', 'trades', 'tb_base_av', 'tb_quote_av', 'ignore' ])
    data['timestamp'] = pd.to_datetime(data['timestamp'], unit='ms')
    if len(data_df) > 0:
        temp_df = pd.DataFrame(data)
        data_df = data_df.append(temp_df)
    else: data_df = data
    data_df.set_index('timestamp', inplace=True)
    if save: data_df.to_csv(filename)
    print('All caught up..!')
    return data_df

def fileExists(file: str) -> bool:
    return os.path.exists(file)

def downloadBinanceDataToCSV():
    tickers = binance_client.get_all_tickers()

    for ticker in tickers:
        symbol = ticker["symbol"]
        print("Obtaining " + symbol + "...")
        getAllBinance(symbol, "1m", save=True)
        print("Obtained " + symbol + ".")

def downloadSpecificBinanceDataToCSV(symbol: str):
    print("Obtaining " + symbol + "...")
    getAllBinance(symbol, "1m", save=True)
    print("Obtained " + symbol + ".")

def downloadSpecificBinanceDataToCSV(tickers: List, oldest=None):
    for symbol in tickers:
        print("Obtaining " + symbol + "...")
        getAllBinance(symbol, "1m", save=True, oldest=oldest)
        print("Obtained " + symbol + ".")

if __name__ == "__main__":
    tickers = [
        "BTCUSDT",
        # "LTCUSDT",
        # "ETHUSDT"
    ]

    # This either downloads or updates our dataset.
    downloadSpecificBinanceDataToCSV(tickers, oldest=datetime(year=2021, month=4, day=1))
    # downloadSpecificBinanceDataToCSV(tickers)
    # downloadBinanceDataToCSV()
    # downloadSpecificBinanceDataToCSV("OAXBTC")
