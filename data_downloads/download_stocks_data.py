# Name: Stocks Downloader
# Author: Robert Ciborowski
# Date: 01/03/2021
# Description: Downloads stock data from Interactive Brokers. Requires that
#              you have TraderWorkStation open and that you are subscribed to
#              the correct data (US Securities subscriptions are not free).

from datetime import datetime

from data_downloads.InteractiveBrokersDownloader import \
    InteractiveBrokersDownloader

if __name__ == "__main__":
    # startDate = datetime(day=2, month=2, year=2004, hour=0, minute=0)
    startDate = datetime(day=1, month=10, year=2014, hour=0, minute=0)
    endDate = datetime(day=26, month=2, year=2021, hour=0, minute=0)
    # endDate = datetime(day=28, month=2, year=2021, hour=0, minute=0)
    downloader = InteractiveBrokersDownloader()
    downloader.download("JPXN", "STK", "SMART", "USD", startDate, endDate)
