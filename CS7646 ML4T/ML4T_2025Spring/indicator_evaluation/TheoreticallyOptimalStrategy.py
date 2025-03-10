import pandas as pd
import numpy as np
import datetime as dt
import util as ut

def author():
    return 'snidadana3'

def get_prices(symbol, dates):
    prices = ut.get_data([symbol], dates)
    prices.fillna(method='ffill', inplace=True)
    prices.fillna(method='bfill', inplace=True)
    return prices[symbol]

def testPolicy(symbol="AAPL", sd=dt.datetime(2010, 1, 1), ed=dt.datetime(2011, 12, 31), sv=100000):
        """
        Computes the best possible trading strategy

        Args:
            - symbol: The stock symbol to use
            - sd: The start date
            - ed: The end date
            - sv: The starting value of the portfolio

        Returns:
            - A dataframe of orders of the form: Order | Date | Symbol | Shares
        """
        prices = get_prices(symbol, pd.date_range(sd, ed))
        tomorrows_prices = prices.shift(-1)
        orders = pd.DataFrame(index=prices.index, columns=['Order', 'Symbol', 'Shares'])

        # Determine buy/sell signals
        orders['Order'] = 'HOLD'
        orders.loc[tomorrows_prices > prices, 'Order'] = 'BUY'
        orders.loc[tomorrows_prices < prices, 'Order'] = 'SELL'

        # Add symbol and shares columns
        orders['Symbol'] = symbol
        orders['Shares'] = 1000

        # Adjust shares for switching positions
        orders.loc[(orders['Order'] == 'BUY') & (orders['Order'].shift(1) == 'SELL'), 'Shares'] = 2000
        orders.loc[(orders['Order'] == 'SELL') & (orders['Order'].shift(1) == 'BUY'), 'Shares'] = 2000

        # Remove 'HOLD' orders
        orders = orders[orders['Order'] != 'HOLD']
        
        print('Orders dataframe:\n', orders)

        return orders

if __name__ == "__main__":
    # Example usage
    df_trades = testPolicy(symbol="JPM", sd=dt.datetime(2008, 1, 1), ed=dt.datetime(2009, 12, 31), sv=100000)
    # print(df_trades)