import pandas as pd
import datetime as dt
import util as ut

class ManualStrategy:
    def __init__(self, verbose=False):
        self.verbose = verbose

    def testPolicy(self, symbol="AAPL", sd=dt.datetime(2010, 1, 1), ed=dt.datetime(2011, 12, 31), sv=100000):
        # Implement your manual strategy logic here
        # Use indicators to generate buy/sell signals
        # Return a DataFrame with trades
        dates = pd.date_range(sd, ed)
        prices_all = ut.get_data([symbol], dates)
        trades = prices_all[[symbol,]]  # only portfolio symbols
        trades.values[:, :] = 0  # set them all to nothing

        # Example logic (replace with your own)
        # trades.values[0, :] = 1000  # BUY signal
        # trades.values[40, :] = -1000  # SELL signal

        if self.verbose:
            print(trades)
        return trades