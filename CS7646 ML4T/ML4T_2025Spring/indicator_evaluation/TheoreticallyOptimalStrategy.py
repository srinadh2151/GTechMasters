import pandas as pd
import numpy as np
import datetime as dt
import util as ut

def author():
    return 'snidadana3'

def testPolicy(symbol="AAPL", sd=dt.datetime(2010, 1, 1), ed=dt.datetime(2011, 12, 31), sv=100000):
    # Get stock prices
    dates = pd.date_range(sd, ed)
    prices_all = ut.get_data([symbol], dates)
    prices = prices_all[symbol]
    
    # Initialize trades DataFrame
    df_trades = pd.DataFrame(index=prices.index, columns=[symbol])
    df_trades.values[:] = 0  # Start with no trades

    # Iterate over prices to determine trades
    for i in range(1, len(prices)):
        if prices[i] > prices[i - 1]:  # Price is going up
            df_trades.iloc[i] = 1000  # Buy
        elif prices[i] < prices[i - 1]:  # Price is going down
            df_trades.iloc[i] = -1000  # Sell

    # Ensure holdings are within constraints
    holdings = df_trades.cumsum()
    df_trades[holdings > 1000] = 0
    df_trades[holdings < -1000] = 0

    return df_trades

if __name__ == "__main__":
    # Example usage
    df_trades = testPolicy(symbol="JPM", sd=dt.datetime(2008, 1, 1), ed=dt.datetime(2009, 12, 31), sv=100000)
    print(df_trades)