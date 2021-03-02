from ibapi.client import EClient
from ibapi.wrapper import EWrapper
from ibapi.contract import Contract

import threading
import time
import pandas as pd
from datetime import datetime, timedelta

class IBapi(EWrapper, EClient):
    def __init__(self):
        EClient.__init__(self, self)
        self.data = []  # Initialize variable to store candle
        self.requestCounter = 0
        self.doneRetrievingHistoricalData = True

    def historicalData(self, reqId, bar):
        if self.requestCounter % 7000 == 0:
            print("Processing data at", pd.to_datetime(bar.date, unit="s"))

        self.requestCounter += 1

        # print(f"Time: {bar.date} Close: {bar.close} Volume: {bar.volume}")
        self.data.append([bar.date, bar.open, bar.high, bar.low, bar.close, bar.volume])
        self.historicalDataRetrievalTime = datetime.now()

    def historicalDataEnd(self, reqId, startDateStr, endDateStr):
        self.doneRetrievingHistoricalData = True

def download(symbol: str, securityType: str, exchange: str, currency: str,
             startDate: datetime, endDate: datetime):
    """
    Downloads data using InteractiveBroker's API. Examples below are shown
    for download both SPY (a US ETF on NYSE Arca) and EUR (a FIAT currency)
    :param symbol: e.g. "SPY" or "EUR"
    :param securityType: e.g. "STK" or "CASH"
    :param exchange: e.g. "SMART" or "IDEALPRO"
    :param currency: e.g. "USD" or "USD"
    :param startDate: download from this date and time
    :param endDate: download to this date and time
    :return:
    """
    print("Downloading " + symbol + "...")

    app = IBapi()
    app.connect("127.0.0.1", 7496, 100)

    def run_loop():
        app.run()

    # Start the socket in a thread
    api_thread = threading.Thread(target=run_loop, daemon=True)
    api_thread.start()

    time.sleep(1)  # Sleep interval to allow time for connection to server

    # Create contract object
    stocks_contract = Contract()
    stocks_contract.symbol = symbol
    stocks_contract.secType = securityType
    stocks_contract.exchange = exchange
    stocks_contract.currency = currency

    delta = timedelta(days=10)
    date = startDate + delta

    while date < endDate + delta:
        app.doneRetrievingHistoricalData = False
        dateString = date.strftime("%Y%m%d %H:%M:%S")

        # Request historical candles
        try:
            # 1 = use Regular Trading Hours, 0 = don't
            app.reqHistoricalData(1, stocks_contract, dateString, "10 D",
                                  "1 min", "TRADES", 1, 2,
                                  False, [])
        except:
            print("Error!")

        time.sleep(1)

        while not app.doneRetrievingHistoricalData:
            pass

        app.requestCounter = 0
        date += delta

    print("Performing final processing on data...")
    df = pd.DataFrame(app.data, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s")
    df = df.drop_duplicates(subset=["timestamp"])
    df.to_csv(symbol + "-1m-data.csv", index=False)
    print("Done downloading " + symbol + "!")

if __name__ == "__main__":
    startDate = datetime(day=2, month=2, year=2004, hour=0, minute=0)
    endDate = datetime(day=26, month=2, year=2021, hour=0, minute=0)
    # endDate = datetime(day=28, month=2, year=2021, hour=0, minute=0)
    download("SPY", "STK", "SMART", "USD", startDate, endDate)
