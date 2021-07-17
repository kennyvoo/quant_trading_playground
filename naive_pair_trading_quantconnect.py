# https://quantpedia.com/Screener/Details/12

# my first successfully deployed naive algorithm to run backtest on quantconnect. it's to use the hedge ratio calculated from training in backtesting. 

import numpy as np
import pandas as pd
from scipy import stats
from math import floor
from datetime import timedelta
from collections import deque
import itertools as it
from decimal import Decimal
import statsmodels.api as sm
from dateutil.relativedelta import relativedelta
import statsmodels.api as sm
from statsmodels.tsa.stattools import coint

class PairsTradingAlgorithm(QCAlgorithm):
    
    def Initialize(self):
        
        self.SetStartDate(2015,1,1)
        self.SetEndDate(2021,6,1)
        self.SetCash(100000)
        self.UniverseSettings.Resolution = Resolution.Daily
        
        
        # energy,technology,healthcare,finance,consumer_services,transportation
        tickers = self.getTicket("finance")
        
        
        # ['MMM','AXP','AMGN','AAPL', 'BA', 'CAT', 'CVX', 'CSCO', 'KO',  'GS', 'HD', 'HON', 'IBM', 'INTC', 'JNJ', 'JPM', 
        #         'MCD','MRK', 'MSFT', 'NKE', 'PG', 'CRM', 'TRV', 'UNH', 'VZ', 'V',  'WMT', 'DIS']
        
        
        
        
        self.AddEquity("SPY", Resolution.Daily) 
        self.threshold = 2
        self.symbols = []
        for i in tickers:
            self.symbols.append(self.AddEquity(i, Resolution.Daily).Symbol)
        
        self.train_period_start = datetime(2010, 1, 1) # start datetime for history call
        self.train_period_end = datetime(2014, 12, 31) # end datetime for history call
        temp=self.History(["SPY"], self.train_period_start,self.train_period_end, Resolution.Daily)

        self.training_period = len(temp)

        self.history_price = {}
        iterated_symbol=[]
        for symbol in self.symbols:
            iterated_symbol.append(symbol)
            hist = self.History([symbol], self.train_period_start,self.train_period_end, Resolution.Daily)
            if hist.empty or len(hist)!=self.training_period: 
                self.Log(f'removed symbol : {symbol}')
                self.symbols.remove(symbol)
            else:
                df = hist.reset_index().set_index("time")
                self.history_price.update({symbol.Value:df.to_dict()['close']})
        to_be_deleted=set(self.symbols)-set(iterated_symbol) # removed stocks that doesnt exist in this backtesting period 
        for symbol in to_be_deleted:
            self.Log(f'removed symbol 2 : {symbol}')
            self.symbols.remove(symbol)
        
    
        self.symbol_pairs = list(it.combinations(self.symbols, 2))  
        # Add the benchmark
        self.Log(self.history_price.keys())
        #self.PairForming()
        self.Schedule.On(self.DateRules.MonthStart("SPY"), self.TimeRules.AfterMarketOpen("SPY"), self.PairForming)
        self.count = 0
        self.filtered_pairs = []

    def getTicket(self,sector):
        from io import StringIO
        temp=self.Download("https://raw.githubusercontent.com/kennyvoo/quant_trading_playground/main/top_30_cap_stock.csv")
        df=pd.read_csv(StringIO(temp))
        return df[sector].to_list()
        
    def OnData(self, data):
        # Update the price series everyday
        for symbol in self.symbols:
            if data.Bars.ContainsKey(symbol) and str(symbol) in self.history_price.keys():
                self.history_price[symbol.Value][self.Time.now().date()]=data.Bars[symbol].Close
        if self.filtered_pairs is None: return
        history_price_df=pd.DataFrame(self.history_price)
        for pair in self.filtered_pairs:
            # calculate the spread of two price series
            lookback=pair.half_life
            pair.ts_a=history_price_df[pair.symbol_a][-lookback:]
            pair.ts_b=history_price_df[pair.symbol_b][-lookback:]
            pair.static_linear_reg()
            mean = np.mean(pair.spread)
            std = np.std(pair.spread)
            ratio = pair.coef
            
            zScore= (pair.spread[-1]-mean)/std
            
            if zScore > 2 :
                if not self.Portfolio[pair.symbol_a].Invested and not self.Portfolio[pair.symbol_b].Invested:
                    quantity = int(self.CalculateOrderQuantity(pair.symbol_a, 0.1))
                    self.Buy(pair.symbol_a, quantity) 
                    self.Sell(pair.symbol_b,  floor(ratio*quantity))                
                if zScore >3:
                    self.Liquidate(pair.symbol_a) 
                    self.Liquidate(pair.symbol_b)     
            elif zScore <-2 : 
                quantity = int(self.CalculateOrderQuantity(pair.symbol_a, 0.1))
                if not self.Portfolio[pair.symbol_a].Invested and not self.Portfolio[pair.symbol_b].Invested:
                    self.Sell(pair.symbol_a, quantity)  
                    self.Buy(pair.symbol_b, floor(ratio*quantity)) 
                if zScore <-3:
                    self.Liquidate(pair.symbol_a) 
                    self.Liquidate(pair.symbol_b)     
            # the position is closed when prices revert back
            elif self.Portfolio[pair.symbol_a].Invested and self.Portfolio[pair.symbol_b].Invested :
                if zScore>=-0.5 and zScore<=0.5:
                    self.Liquidate(pair.symbol_a) 
                    self.Liquidate(pair.symbol_b)              


    def Rebalance(self):
        # schedule the event to fire every half year to select pairs with the smallest historical distance
        if self.count % 6 == 0:
            distances = {}
            for i in self.symbol_pairs:
                distances[i] = Pair(str(i[0]), str(i[1]), self.history_price[str(i[0])],  self.history_price[str(i[1])]).distance()
                self.sorted_pairs = sorted(distances, key = lambda x: distances[x])[:4]
        self.count += 1
        
    def PairForming(self):
        # schedule the event to fire every half year to select pairs with the smallest historical distance
        # if self.count % 6 == 0:
        if self.count>0:
            return
        for i in self.symbol_pairs:
            pair=Pair(i[0].Value, i[1].Value, pd.DataFrame({i[0].Value:self.history_price[i[0].Value]}),  pd.DataFrame({i[1].Value:self.history_price[i[1].Value]}))
            if pair.cointegration_test():
                self.filtered_pairs.append(pair)
        self.count += 1
        self.filtered_pairs=sorted(self.filtered_pairs, key = lambda x: x.hurst_value,reverse=True)[:10]

class Pair:
    def __init__(self, symbol_a, symbol_b, ts_a, ts_b):
        self.symbol_a = symbol_a
        self.symbol_b = symbol_b
        self.ts_a = ts_a
        self.ts_b = ts_b
        self.coef=None
        self.spread=None # resid
        self.half_life=None
        self.hurst_value=None
        
    def cointegration_test(self):
        if not self.CADF():
            return False
        self.static_linear_reg() #calculate spread and coef
        if self.hurst()>=0.5 or not self.calculate_half_life(): # check hurst<
            return False
        return True
        
    def distance(self):
        # calculate the sum of squared deviations between two normalized price series
        norm_a = np.array(self.ts_a)/self.ts_a[0]
        norm_b = np.array(self.ts_b)/self.ts_b[0]
        return sum((norm_a - norm_b)**2)
        
    def static_linear_reg(self):
        x=sm.add_constant(self.ts_a.values, prepend=False)
        model = sm.OLS(self.ts_b,x)
        result = model.fit()
        self.spread=result.resid
        self.coef=result.params["x1"]

    def hurst(self):
        """Returns the Hurst Exponent of the time series vector ts"""
        # Create the range of lag values
        lags = range(2, 100)
        # Calculate the array of the variances of the lagged differences
        tau = [np.sqrt(np.std(np.subtract(self.spread[lag:].to_list(), self.spread[:-lag].to_list()))) for lag in lags]
     
        # Use a linear fit to estimate the Hurst Exponent
        poly = np.polyfit(np.log(lags),np.log(tau), 1)
        self.hurst_value=poly[0]*2.0
        # Return the Hurst exponent from the polyfit output
        return poly[0]*2.0
    
    def calculate_half_life(self):
        spread_lag = self.spread.shift(1)
        spread_lag.iloc[0] = spread_lag.iloc[1] # some uses 0
        
        spread_ret = self.spread - spread_lag
        spread_ret.iloc[0] = spread_ret.iloc[1] # some uses 0
        
        spread_lag2 = sm.add_constant(spread_lag,prepend=False)
         
        model = sm.OLS(spread_ret,spread_lag2)
        res = model.fit()
        # if cooef is positive, means that it will not mean reverting at all, if close to 0, means half life very long.
        if(res.params[0]>0):
            print("Non Mean Reverting")
            return False
        self.half_life = int(round(-np.log(2) / res.params[0],0))
        return True

    def CADF(self,critical_level=0.05):

        swap=False
        result1 = coint(self.ts_b, self.ts_a,return_results=False) # ts_a is X, ts_b is Y
        result2 = coint(self.ts_a, self.ts_b,return_results=False) 

        if(result1[0]>result2[0]): # choose the one with lower t stat.
            result=result2
            swap=True
        else: 
            result=result1
        score = result[0]
        pvalue = result[1]

        if pvalue < critical_level:  # 95 % confidence that pairs cointegrate
            if(swap):
                self.symbol_a,self.symbol_b=self.symbol_b,self.symbol_a
                self.ts_a,self.ts_b= self.ts_b,self.ts_a
            return True
        return False