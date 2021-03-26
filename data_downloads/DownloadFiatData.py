# Name: FIAT Currency Downloader
# Author: Robert Ciborowski
# Date: 01/03/2021
# Description: Downloads FIAT data from Interactive Brokers. Requires that
#              you have TraderWorkStation open and that you are subscribed to
#              the correct data (FIAT subscriptions should be free).

from datetime import datetime

from data_downloads.InteractiveBrokersDownloader import \
    InteractiveBrokersDownloader

if __name__ == "__main__":
    startDate = datetime(day=1, month=3, year=2005, hour=0, minute=0)
    endDate = datetime(day=28, month=2, year=2021, hour=0, minute=0)
    downloader = InteractiveBrokersDownloader()
    downloader.download("EUR", "CASH", "IDEALPRO", "USD", startDate, endDate)
