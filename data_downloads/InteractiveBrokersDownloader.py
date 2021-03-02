# Name: Interactive Brokers Historical Data Downloader
# Author: Robert Ciborowski
# Date: 01/03/2021
# Description: A class for downloading from Interactive Brokers. Requires that
#              you have TraderWorkStation open and that you are subscribed to
#              the correct data.

from ibapi.client import EClient
from ibapi.wrapper import EWrapper
from ibapi.contract import Contract

import threading
import time
import pandas as pd
from datetime import datetime, timedelta

class DataReceiver(EWrapper, EClient):
    def __init__(self):
        EClient.__init__(self, self)
        self.data = []
        self.requestCounter = 0
        self.doneRetrievingHistoricalData = True

    def historicalData(self, requestID, bar):
        if self.requestCounter % 7000 == 0:
            print("Processing data at", pd.to_datetime(bar.date, unit="s"))

        self.requestCounter += 1

        # print(f"Time: {bar.date} Close: {bar.close} Volume: {bar.volume}")
        self.data.append([bar.date, bar.open, bar.high, bar.low, bar.close, bar.volume])
        self.historicalDataRetrievalTime = datetime.now()

    def historicalDataEnd(self, reqId, startDateStr, endDateStr):
        self.doneRetrievingHistoricalData = True

class InteractiveBrokersDownloader:
    dataReceiver: DataReceiver

    def __init__(self):
        self.dataReceiver = DataReceiver()
        self.dataReceiver.connect("127.0.0.1", 7496, 100)

    def download(self, symbol: str, securityType: str, exchange: str, currency: str,
                 startDate: datetime, endDate: datetime):
        """
        Downloads data using InteractiveBroker's API. Examples below are shown
        for download both SPY (a US ETF on NYSE Arca) and EUR (a FIAT currency).
        Data gets stored in "symbol-1m-data.csv".

        :param symbol: e.g. "SPY" or "EUR"
        :param securityType: e.g. "STK" or "CASH"
        :param exchange: e.g. "SMART" or "IDEALPRO"
        :param currency: e.g. "USD" or "USD"
        :param startDate: download from this date and time
        :param endDate: download to this date and time
        """
        print("Downloading " + symbol + "...")

        def run_loop():
            self.dataReceiver.run()

        # Start the socket in a thread
        api_thread = threading.Thread(target=run_loop, daemon=True)
        api_thread.start()

        time.sleep(3)  # Sleep interval to allow time for connection to server

        # Create contract object
        stocks_contract = Contract()
        stocks_contract.symbol = symbol
        stocks_contract.secType = securityType
        stocks_contract.exchange = exchange
        stocks_contract.currency = currency

        delta = timedelta(days=10)
        date = startDate + delta

        while date < endDate + delta:
            self.dataReceiver.doneRetrievingHistoricalData = False
            dateString = date.strftime("%Y%m%d %H:%M:%S")

            # Request historical candles
            try:
                # 1 = use Regular Trading Hours, 0 = don't
                self.dataReceiver.reqHistoricalData(1, stocks_contract, dateString, "10 D",
                                      "1 min", "MIDPOINT", 1, 2,
                                      False, [])
            except:
                print("Error with reqHistoricalData!")

            time.sleep(1)

            while not self.dataReceiver.doneRetrievingHistoricalData:
                pass

            self.dataReceiver.requestCounter = 0
            date += delta

        print("Performing final processing on data...")
        df = pd.DataFrame(self.dataReceiver.data, columns=["timestamp", "open", "high", "low", "close", "volume"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s")
        df = df.drop_duplicates(subset=["timestamp"])
        df.to_csv(symbol + "-1m-data.csv", index=False)
        print("Done downloading " + symbol + "!")
