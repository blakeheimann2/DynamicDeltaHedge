import unittest
from app.Securities import Option
from app.Fitter import CurveFitter
import pandas_datareader.data as web
import pandas as pd
import datetime as dt
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as st
from app.DistFitter import make_pdf, best_fit_distribution


pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 20)
pd.set_option('display.width', 1000)


class OptionTest(unittest.TestCase):
    def testBestFit(self):
        SPY_df = web.DataReader('SPY', 'yahoo', dt.datetime(2018, 3, 29), dt.datetime.today()).reset_index()
        SPY = SPY_df[['Date', 'Close']].set_index('Date', drop=True).copy() * 10 #to convert to index val
        data = SPY['Close'].pct_change().dropna()
        best_fit_name, best_fit_params = best_fit_distribution(data, 200, ax=None)
        best_dist = getattr(st, best_fit_name)
        pdf = make_pdf(best_dist, best_fit_params)
        # Display
        plt.figure(figsize=(12, 8))
        ax = pdf.plot(lw=2, label='Fitted PDF', legend=True)
        data.plot(kind='hist', bins=50, density=True, alpha=0.5, label='Raw Data', legend=True, ax=ax)
        param_names = (best_dist.shapes + ', loc, scale').split(', ') if best_dist.shapes else ['loc', 'scale']
        param_str = ', '.join(['{}={:0.2f}'.format(k, v) for k, v in zip(param_names, best_fit_params)])
        dist_str = '{}({})'.format(best_fit_name, param_str)
        ax.set_title(u'SPY Returns Best Fit Distribution:  \n' + dist_str)
        ax.set_xlabel(u'Returns (%)')
        ax.set_ylabel('Frequency')
        ax.set_xlim([-.25,.25])
        dist = CurveFitter(data)
        sample_size = 100000
        resampled = dist.sampledist(st.t, best_fit_params[0], best_fit_params[1], best_fit_params[2], sample_size)
        resampled = pd.Series(resampled)
        bins = int(sample_size/100)
        resampled.plot(kind='hist', bins=bins, density=True, alpha=0.25, label='Resampled Data', legend=True, ax=ax)
        plt.show()

    def testFitter(self):
        SPY_df = web.DataReader('SPY', 'yahoo', dt.datetime(2018, 3, 29), dt.datetime.today()).reset_index()
        SPY = SPY_df[['Date', 'Close']].set_index('Date', drop=True).copy() * 10 #to convert to index val
        rets = SPY['Close'].pct_change().dropna()
        dist = CurveFitter(rets)
        data = []
        #for i in range(100000):
        data = dist.sampledist(st.t, 1.8768166491405975, 0.0, 0.01, 100000)
            #data.append(x)
        data = pd.Series(data)
        pdf = make_pdf(st.t,  (1.8768166491405975, 0.0, 0.01))
        # Display
        #plt.figure(figsize=(12, 8))
        #ax = data.plot(kind='hist', bins=50, normed=True, alpha=0.5)
        # Save plot limits
        #dataYLim = ax.get_ylim()
        #ax.set_ylim(dataYLim)
        plt.figure(figsize=(12, 8))
        ax = pdf.plot(lw=2, label='PDF', legend=True)
        ax.set_xlim([rets.min(), rets.max()])
        bin_size = (len(rets) / 50)
        bins = int(len(data)/bin_size)
        rets.plot(kind='hist', bins=50, density=True, alpha=0.5, label='Raw Data', legend=True, ax=ax)
        data.plot(kind='hist', bins=bins, density=True, alpha=0.5, label='Sampled', legend=True, ax=ax)
        plt.show()

    def testMCDynaDelta(self):
        pf_shares = []
        shares_purchased = []
        trxn_costs = []
        deltas = []
        option_price = []
        underlying_price = []
        chng = []

        #Approx historical returns with SPY
        SPY_df = web.DataReader('GLD', 'yahoo', dt.datetime(2019, 3, 29), dt.datetime.today()).reset_index()
        SPY = SPY_df[['Date', 'Close']].set_index('Date', drop=True).copy() * 10 #to convert to index val
        rets = SPY['Close'].pct_change().dropna()
        dist = CurveFitter(rets)
        #dist.fitGaussian()


        vol = 0.2157
        T = 31/365
        stk = 2000
        timesteps = int(T*365)

        underlyingP = 1933
        underlying_price.append(underlyingP)

        call = Option(underlyingP, stk, T, .05, is_call=True)
        delta_start = call.getGreeks(vol)['delta']
        deltas.append(delta_start)

        opt_cost = call.getPrice(vol)
        option_price.append(opt_cost)

        shares_owned = -round(delta_start,2)*100 #int(round(b/5.0)*5.0)
        pf_shares.append(shares_owned)

        trxn_costs.append(shares_owned*underlyingP)
        shares_purchased.append(shares_owned)
        for i in range(timesteps):
            T = max(T - 1/365, 0)
            if i % 2 == 0:
                pct_chg = np.random.choice(rets)#round(dist.sampledist(st.t, 1.8768166491405975, 0.0, 0.01, 1).item(), 8)  #dist.samplenorm(1).item()
                chng.append(pct_chg)
                underlyingP = underlyingP + underlyingP * pct_chg  # + 30
                underlying_price.append(underlyingP)
            else:
                pct_chg =  np.random.choice(rets) #round(dist.sampledist(st.t, 1.8768166491405975, 0.0, 0.01, 1).item(), 8) #dist.samplenorm(1).item()
                chng.append(pct_chg)
                underlyingP = underlyingP + underlyingP * pct_chg # - 30
                underlying_price.append(underlyingP)
            call.setUnderlyingP(underlyingP)
            call.setTimetoMaturity(T)
            opt_prc = call.getPrice(vol)
            option_price.append(opt_prc)

            delta = call.getGreeks(vol)['delta']
            deltas.append(delta)

            purchased_shares = -100*round(delta,2) - shares_owned
            shares_purchased.append(purchased_shares)

            shares_owned = shares_owned + purchased_shares
            pf_shares.append(shares_owned)

            cost_of_shares = purchased_shares * underlyingP
            trxn_costs.append(cost_of_shares)

        df = pd.DataFrame([underlying_price, chng, deltas, pf_shares, shares_purchased, trxn_costs, option_price]).T
        df.columns = ['underlying_price','pct_chg', 'deltas', 'pf_shares', 'shares_purchased', 'trxn_costs', 'option_price']
        #assume position is closed at mkt close or hedged at mkt close
        df['cum_trxn_costs'] = df['trxn_costs'].cumsum()
        df['earlyexit_share_pnl'] = df['pf_shares'].shift(1) * df['underlying_price'] - df['cum_trxn_costs'].shift(1)
        df['earlyexit_option_pnl'] = (df['option_price'] - opt_cost)*100
        df['earlyexit_pnl'] = df['earlyexit_share_pnl'] + df['earlyexit_option_pnl']
        expiration_pnl = df['earlyexit_pnl'].loc[len(df['earlyexit_pnl'])-1]
        df['underlying_price'].plot()
        plt.axhline(stk, color='red')
        plt.show()
        print("PNL on Expiration: {}".format(expiration_pnl))
        print("Realized Vol: {}".format(pd.Series(underlying_price).pct_change().std()*np.sqrt(252)))
        print(df)


    def testBUYSimpleDynaDeltaCall(self):
        pf_shares = []
        shares_purchased = []
        trxn_costs = []
        deltas = []
        option_price = []
        underlying_price = []

        vol = 0.20
        T = 20/365
        stk = 50
        underlyingP = 49
        underlying_price.append(underlyingP)

        call = Option(underlyingP, stk, T, .05, is_call=True)
        delta_start = call.getGreeks(vol)['delta']
        deltas.append(delta_start)

        opt_cost = call.getPrice(vol)
        option_price.append(opt_cost)

        shares_owned = -delta_start*100
        pf_shares.append(shares_owned)

        trxn_costs.append(shares_owned*underlyingP)
        shares_purchased.append(shares_owned)
        for i in range(20):
            T = max(T - 1/365, 0)
            if i % 2 == 0:
                underlyingP = underlyingP + 0.5
                underlying_price.append(underlyingP)
            else:
                underlyingP = underlyingP - 0.5
                underlying_price.append(underlyingP)
            call.setUnderlyingP(underlyingP)
            call.setTimetoMaturity(T)
            opt_prc = call.getPrice(vol)
            option_price.append(opt_prc)

            delta = call.getGreeks(vol)['delta']
            deltas.append(delta)

            purchased_shares = -100*delta - shares_owned
            shares_purchased.append(purchased_shares)

            shares_owned = shares_owned + purchased_shares
            pf_shares.append(shares_owned)

            cost_of_shares = purchased_shares * underlyingP
            trxn_costs.append(cost_of_shares)

        df = pd.DataFrame([underlying_price, deltas, pf_shares, shares_purchased, trxn_costs, option_price]).T
        df.columns = ['underlying_price', 'deltas', 'pf_shares', 'shares_purchased', 'trxn_costs', 'option_price']
        df['pct_chg'] = df['underlying_price'].pct_change()
        #assume position is closed at mkt close or hedged at mkt close
        df['cum_trxn_costs'] = df['trxn_costs'].cumsum()
        df['earlyexit_share_pnl'] = df['pf_shares'].shift(1) * df['underlying_price'] - df['cum_trxn_costs'].shift(1)
        df['earlyexit_option_pnl'] = (df['option_price'] - opt_cost)*100
        df['earlyexit_pnl'] = df['earlyexit_share_pnl'] + df['earlyexit_option_pnl']
        expiration_pnl = df['earlyexit_pnl'].loc[len(df['earlyexit_pnl'])-1]
        self.assertEquals(round(expiration_pnl,5), 18.85561)

    def testBUYSimpleDynaDeltaPut(self):
        pf_shares = []
        shares_purchased = []
        trxn_costs = []
        deltas = []
        option_price = []
        underlying_price = []

        vol = 0.20
        T = 20/365
        stk = 50
        underlyingP = 51
        underlying_price.append(underlyingP)

        put = Option(underlyingP, stk, T, .05, is_call=False)
        delta_start = put.getGreeks(vol)['delta']
        deltas.append(delta_start)

        opt_cost = put.getPrice(vol)
        option_price.append(opt_cost)

        shares_owned = -delta_start*100
        pf_shares.append(shares_owned)

        trxn_costs.append(shares_owned*underlyingP)
        shares_purchased.append(shares_owned)
        for i in range(20):
            T = max(T - 1/365, 0)
            if i % 2 == 0:
                underlyingP = underlyingP - 0.8
                underlying_price.append(underlyingP)
            else:
                underlyingP = underlyingP + 0.5
                underlying_price.append(underlyingP)
            put.setUnderlyingP(underlyingP)
            put.setTimetoMaturity(T)
            opt_prc = put.getPrice(vol)
            option_price.append(opt_prc)

            delta = put.getGreeks(vol)['delta']
            deltas.append(delta)

            purchased_shares = -100*delta - shares_owned
            shares_purchased.append(purchased_shares)

            shares_owned = shares_owned + purchased_shares
            pf_shares.append(shares_owned)

            cost_of_shares = purchased_shares * underlyingP
            trxn_costs.append(cost_of_shares)

        df = pd.DataFrame([underlying_price, deltas, pf_shares, shares_purchased, trxn_costs, option_price]).T
        df.columns = ['underlying_price', 'deltas', 'pf_shares', 'shares_purchased', 'trxn_costs', 'option_price']
        #assume position is closed at mkt close or hedged at mkt close
        df['cum_trxn_costs'] = df['trxn_costs'].cumsum()
        df['earlyexit_share_pnl'] = df['pf_shares'].shift(1) * df['underlying_price'] - df['cum_trxn_costs'].shift(1)
        df['earlyexit_option_pnl'] = (df['option_price'] - opt_cost)*100
        df['earlyexit_pnl'] = df['earlyexit_share_pnl'] + df['earlyexit_option_pnl']
        expiration_pnl = df['earlyexit_pnl'].loc[len(df['earlyexit_pnl'])-1]
        self.assertEquals(round(expiration_pnl,5), 41.29556)

        df['underlying_price'].plot()
        plt.axhline(stk, color='red')
        plt.show()
        print("PNL on Expiration: {}".format(expiration_pnl))
        print("Realized Vol: {}".format(pd.Series(underlying_price).pct_change().std()*np.sqrt(252)))
        print(df)




    def testSELLSimpleDynaDeltaPut(self):
        pf_shares = []
        shares_purchased = []
        trxn_costs = []
        deltas = []
        option_price = []
        underlying_price = []

        vol = 0.20
        T = 20/365
        stk = 50
        underlyingP = 51
        underlying_price.append(underlyingP)

        put = Option(underlyingP, stk, T, .05, is_call=False)
        delta_start = - put.getGreeks(vol)['delta'] #NEGATIVE BECAUSE SOLD OPTION
        deltas.append(delta_start)

        opt_cost = - put.getPrice(vol) #NEGATIVE BECAUSE SOLD
        option_price.append(opt_cost)

        shares_owned = -delta_start*100
        pf_shares.append(shares_owned)

        trxn_costs.append(shares_owned*underlyingP)
        shares_purchased.append(shares_owned)
        for i in range(20):
            T = max(T - 1/365, 0)
            if i % 2 == 0:
                underlyingP = underlyingP - 0.8
                underlying_price.append(underlyingP)
            else:
                underlyingP = underlyingP + 0.5
                underlying_price.append(underlyingP)
            put.setUnderlyingP(underlyingP)
            put.setTimetoMaturity(T)
            opt_prc = - put.getPrice(vol) #NEG BC SOLD
            option_price.append(opt_prc)

            delta = - put.getGreeks(vol)['delta'] #NEGATIVE BECAUSE SOLD OPTION
            deltas.append(delta)

            purchased_shares = -100*delta - shares_owned
            shares_purchased.append(purchased_shares)

            shares_owned = shares_owned + purchased_shares
            pf_shares.append(shares_owned)

            cost_of_shares = purchased_shares * underlyingP
            trxn_costs.append(cost_of_shares)

        df = pd.DataFrame([underlying_price, deltas, pf_shares, shares_purchased, trxn_costs, option_price]).T
        df.columns = ['underlying_price', 'deltas', 'pf_shares', 'shares_purchased', 'trxn_costs', 'option_price']
        #assume position is closed at mkt close or hedged at mkt close
        df['cum_trxn_costs'] = df['trxn_costs'].cumsum()
        df['earlyexit_share_pnl'] = df['pf_shares'].shift(1) * df['underlying_price'] - df['cum_trxn_costs'].shift(1)
        df['earlyexit_option_pnl'] = (df['option_price'] - opt_cost)*100
        df['earlyexit_pnl'] = df['earlyexit_share_pnl'] + df['earlyexit_option_pnl']
        expiration_pnl = df['earlyexit_pnl'].loc[len(df['earlyexit_pnl'])-1]
        df['underlying_price'].plot()
        plt.axhline(stk, color='red')
        plt.show()
        print("PNL on Expiration: {}".format(expiration_pnl))
        print("Realized Vol: {}".format(pd.Series(underlying_price).pct_change().std()*np.sqrt(252)))
        print(df)
        self.assertEquals(round(expiration_pnl,5), -41.29556)




    def testSELLSimpleDynaDeltaCall(self):
        pf_shares = []
        shares_purchased = []
        trxn_costs = []
        deltas = []
        option_price = []
        underlying_price = []

        vol = 0.20
        T = 20/365
        stk = 50
        underlyingP = 49
        underlying_price.append(underlyingP)

        call = Option(underlyingP, stk, T, .05, is_call=True)
        delta_start = - call.getGreeks(vol)['delta'] #NEGATIVE BECAUSE SOLD OPTION
        deltas.append(delta_start)

        opt_cost = - call.getPrice(vol) #NEGATIVE BECAUSE SOLD
        option_price.append(opt_cost)

        shares_owned = -delta_start*100
        pf_shares.append(shares_owned)

        trxn_costs.append(shares_owned*underlyingP)
        shares_purchased.append(shares_owned)
        for i in range(20):
            T = max(T - 1/365, 0)
            if i % 2 == 0:
                underlyingP = underlyingP + 0.8
                underlying_price.append(underlyingP)
            else:
                underlyingP = underlyingP - 0.5
                underlying_price.append(underlyingP)
            call.setUnderlyingP(underlyingP)
            call.setTimetoMaturity(T)
            opt_prc = - call.getPrice(vol) #NEG BC SOLD
            option_price.append(opt_prc)

            delta = - call.getGreeks(vol)['delta'] #NEGATIVE BECAUSE SOLD OPTION
            deltas.append(delta)

            purchased_shares = -100*delta - shares_owned
            shares_purchased.append(purchased_shares)

            shares_owned = shares_owned + purchased_shares
            pf_shares.append(shares_owned)

            cost_of_shares = purchased_shares * underlyingP
            trxn_costs.append(cost_of_shares)

        df = pd.DataFrame([underlying_price, deltas, pf_shares, shares_purchased, trxn_costs, option_price]).T
        df.columns = ['underlying_price', 'deltas', 'pf_shares', 'shares_purchased', 'trxn_costs', 'option_price']
        #assume position is closed at mkt close or hedged at mkt close
        df['cum_trxn_costs'] = df['trxn_costs'].cumsum()
        df['earlyexit_share_pnl'] = df['pf_shares'].shift(1) * df['underlying_price'] - df['cum_trxn_costs'].shift(1)
        df['earlyexit_option_pnl'] = (df['option_price'] - opt_cost)*100
        df['earlyexit_pnl'] = df['earlyexit_share_pnl'] + df['earlyexit_option_pnl']
        expiration_pnl = df['earlyexit_pnl'].loc[len(df['earlyexit_pnl'])-1]
        df['underlying_price'].plot()
        plt.axhline(stk, color='red')
        plt.show()
        print("PNL on Expiration: {}".format(expiration_pnl))
        print("Realized Vol: {}".format(pd.Series(underlying_price).pct_change().std()*np.sqrt(252)))
        print(df)
        self.assertEquals(round(expiration_pnl,5), -18.85561)






if __name__ == '__main__':
    unittest.main()
