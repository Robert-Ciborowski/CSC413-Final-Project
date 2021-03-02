from ibapi.client import EClient
from ibapi.wrapper import EWrapper
from ibapi.contract import Contract

import threading
import time


class IBapi(EWrapper, EClient):
    def __init__(self):
        EClient.__init__(self, self)
        self.data = []  # Initialize variable to store candle

    def historicalData(self, reqId, bar):
        print(f"Time: {bar.date} Close: {bar.close}")
        self.data.append([bar.date, bar.close])


def run_loop():
    app.run()


app = IBapi()
app.connect("127.0.0.1", 7496, 100)

# Start the socket in a thread
api_thread = threading.Thread(target=run_loop, daemon=True)
api_thread.start()

time.sleep(1)  # Sleep interval to allow time for connection to server

# Create contract object
eurusd_contract = Contract()
# eurusd_contract.symbol = "EUR"
# eurusd_contract.secType = "CASH"
# eurusd_contract.exchange = "IDEALPRO"
# eurusd_contract.currency = "USD"
eurusd_contract.symbol = "SPY"
eurusd_contract.secType = "STK"
eurusd_contract.exchange = "SMART"
eurusd_contract.currency = "USD"

# Request historical candles
try:
    app.reqHistoricalData(1, eurusd_contract, "20040202 12:00:00", "1 D", "1 hour", "BID", 0, 2,
                          False, [])
    # app.reqHistoricalData(1, eurusd_contract, "20040102 12:00:00", "1 D",
    #                       "1 hour", "BID", 0, 2,
    #                       False, [])
except:
    print("Error!")

time.sleep(5)  # sleep to allow enough time for data to be returned

# Working with Pandas DataFrames
import pandas

df = pandas.DataFrame(app.data, columns=["DateTime", "Close", "Volume"])
df["DateTime"] = pandas.to_datetime(df["DateTime"], unit="s")
df.rename({"DateTime": "timestamp", "Close": "close", "Volume": "volume"})
df.to_csv("EURUSD-1h-data.csv")

print(df)
