import datetime  # For datetime objects
import os.path  # To manage paths
import sys  # To find out the script name (in argv[0])
import pandas as pd
import datetime

from backtesting import Backtest, Strategy
from backtesting.lib import crossover

from backtesting.test import SMA, GOOG

# Create a Stratey
class backTestCustom(Strategy):
    n1 = 20
    n2 = 60

    def init(self):
        price = self.data.df['Close'].values
        self.sigbuy = self.data.df['finalBuy'].values
        self.sigsell = self.data.df['finalSell'].values

        self.pos1 = self.I(self.buy_pos, self.sigbuy)
        self.pos2 = self.I(self.sell_pos, self.sigsell)


    def buy_pos(self, pos):
        return pd.Series(pos)

    def sell_pos(self, pos):
        return pd.Series(pos)


    def next(self):
        ## buy 조건
        if self.pos1[-1] == True:
            self.buy()
        ## sell 조건
        if self.pos2[-1] == True:
            # self.position.close()
            self.sell()





if __name__ == '__main__':
    df = pd.read_csv('test.csv')
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index(['Date'], drop=True, inplace=True)

    bt = Backtest(df, backTestCustom, commission=.015, cash=10000000, exclusive_orders=True)

    stats = bt.run()
    bt.plot()
    print(stats)
