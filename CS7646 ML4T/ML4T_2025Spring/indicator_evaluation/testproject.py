import pandas as pd
import numpy as np
import datetime as dt
import os
from util import get_data, plot_data
import matplotlib.pyplot as plt
#import marketsimcode as ms
import marketsimcode as mm
from TheoreticallyOptimalStrategy import *
from indicators import *

def author():
	return 'snidadana3'

def testPolicy(symbol = ['JPM'], sd=dt.datetime(2010,1,1), ed=dt.datetime(2011,12,31), sv = 100000):
    orders = []
    lookback = 10
    
    holdings={sym:0 for sym in symbol}
  
    dates = pd.date_range(sd,ed)
    prices_1 = get_data(['JPM'],dates)
    price = prices_1['JPM']
    #prices = prices/prices[0]
    
    
    df_i = get_indicators(price)
    
    sma = df_i['SMA']    
    
    bbp = df_i['bb_value']
    vol = df_i['volatility']
    moment = df_i['momentum']
    cci = df_i['CCI']    
    
    sym = 'JPM'
    orders.append([sd,'JPM','HOLD',0])
    for day in range(lookback+1,df_i.shape[0]):
        
            if (sma.ix[day]<0.5) and (bbp.ix[day]<0.9) and (moment.ix[day]<0):
		
                if holdings[sym]<1000:
                    holdings[sym] += 1000
                    orders.append([price.index[day].date(),sym,'BUY',1000])
            
            
            elif (sma.ix[day]>2) and (bbp.ix[day]>1) and (moment.ix[day]<0):
		
                if holdings[sym]>0:
                    holdings[sym] -= 2000
                    orders.append([price.index[day].date(),sym,'SELL',2000])
                    
            elif (holdings[sym]<=0) and (holdings[sym] >= -1000):
                holdings[sym] -= 1000
                orders.append([price.index[day].date(),sym,'SELL',1000])
                
            elif (sma.ix[day]>1) and (sma.ix[day-1]<1) and (holdings[sym]>0):
                holdings[sym]=0
                orders.append([price.index[day].date(),sym,'SELL',1000])
            
            elif (sma.ix[day]<=1) and (sma.ix[day-1]>1) and (holdings[sym]<0):
                holdings[sym]=0
                orders.append([price.index[day].date(),sym,'BUY',1000])
        
    orders.append([ed,sym,'HOLD',0])
    
    res=pd.DataFrame(orders)
    res.columns=['Date','Symbol','Order','Shares']    
    
    #print res
    p = mm.compute_portvals(res)
    my_colors = ['black', 'blue']
    start_val = 100000
    ben = benchmark_trades('JPM')
    p3 = mm.compute_portvals(ben,start_val)
    
    plt.figure(figsize=(20,10))
    plt.gca().set_color_cycle(['black','blue'])
    plt.legend(loc="upper left")
    p = p/p[0]
    p3 = p3/p3[0]
    pp, = plt.plot(p)
    pb, = plt.plot(p3)
    plt.legend([pp,pb],['Manual','Benchmark'])

    plt.xlabel('Dates')
    plt.ylabel('Prices(normalized)')
        
    plt.show()    
    
    #print port
    return 1


def main():
    testPolicy()
        
    pass


if __name__ == "__main__": main()