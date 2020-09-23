import unittest
import pandas_datareader.data as web
import pandas as pd
import datetime as dt
from app.distributions import DISTRIBUTIONS
from app.SampleFitting import SampleFitter
import warnings


pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 20)
pd.set_option('display.width', 1000)


class FitterTest(unittest.TestCase):
    def testBestFit(self):
        warnings.filterwarnings('ignore')
        SPY_df = web.DataReader('SPY', 'yahoo', dt.datetime(2018, 3, 29), dt.datetime.today()).reset_index()
        SPY = SPY_df[['Date', 'Close']].set_index('Date', drop=True).copy() * 10  # to convert to index val
        rets = SPY['Close'].pct_change().dropna()
        data = pd.Series(rets)
        fitter = SampleFitter(data)
        fitter.get_best_fit(DISTRIBUTIONS)
        fitter.plot_best_fit("SPY Returns")






