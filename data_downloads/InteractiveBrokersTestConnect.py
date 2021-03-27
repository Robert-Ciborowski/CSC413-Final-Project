# Name: IB Test Connect
# Author: Robert Ciborowski
# Date: 01/03/2021
# Description: Tests your connection with Interactive Brokers on
#              IP: 127.0.0.1, Port: 7496

from ibapi.client import EClient
from ibapi.wrapper import EWrapper

class IBapi(EWrapper, EClient):
     def __init__(self):
         EClient.__init__(self, self)


if __name__ == "__main__":
    app = IBapi()
    app.connect('127.0.0.1', 7496, 123)
    app.run()

    # If unable to connect, and to prevent errors on a reconnect:
    import time
    time.sleep(2)
    app.disconnect()
