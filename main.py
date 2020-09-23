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


def main():
    ticker = 'TGT'
    SPY_df = web.DataReader(ticker, 'yahoo', dt.datetime(2018, 3, 29), dt.datetime.today()).reset_index()
    SPY = SPY_df[['Date', 'Close']].set_index('Date', drop=True).copy() * 10  # to convert to index val
    data = SPY['Close'].pct_change().dropna()

    best_fit_name, best_fit_params = best_fit_distribution(data, 200, ax=None)
    best_dist = getattr(st, best_fit_name)
    pdf = make_pdf(best_dist, best_fit_params)

    plt.figure(figsize=(12, 8))
    ax = pdf.plot(lw=2, label='Fitted PDF', legend=True)
    data.plot(kind='hist', bins=50, density=True, alpha=0.5, label='Raw Data', legend=True, ax=ax)
    param_names = (best_dist.shapes + ', loc, scale').split(', ') if best_dist.shapes else ['loc', 'scale']
    param_str = ', '.join(['{}={:0.2f}'.format(k, v) for k, v in zip(param_names, best_fit_params)])
    dist_str = '{}({})'.format(best_fit_name, param_str)
    ax.set_title(u'{} Returns Best Fit Distribution:  \n'.format(ticker) + dist_str)
    ax.set_xlabel(u'Returns (%)')
    ax.set_ylabel('Frequency')
    ax.set_xlim([-.25,.25])
    dist = CurveFitter(data)
    sample_size = 100000
    resampled = dist.sampledist(st.nct, best_fit_params[0], best_fit_params[1], best_fit_params[2], best_fit_params[3], sample_size)
    resampled = pd.Series(resampled)
    bins = int(sample_size /100)
    resampled.plot(kind='hist', bins=bins, density=True, alpha=0.25, label='Resampled Data', legend=True, ax=ax)
    plt.savefig("BestFit.png")
    plt.show()

    pf_shares = []
    shares_purchased = []
    trxn_costs = []
    deltas = []
    option_price = []
    underlying_price = []
    chng = []

    vol = 0.30
    T = 123 / 365
    stk = 170
    timesteps = int(T * 365)

    underlyingP = 149.58
    underlying_price.append(underlyingP)

    call = Option(underlyingP, stk, T, .05, is_call=True)
    delta_start = call.getGreeks(vol)['delta']
    deltas.append(delta_start)

    opt_cost = call.getPrice(vol)
    option_price.append(opt_cost)

    shares_owned = -round(delta_start, 2) * 100  # int(round(b/5.0)*5.0)
    pf_shares.append(shares_owned)

    trxn_costs.append(shares_owned * underlyingP)
    shares_purchased.append(shares_owned)
    for i in range(timesteps):
        T = max(T - 1 / 365, 0)
        if i % 2 == 0:
            pct_chg = round(dist.sampledist(best_dist, best_fit_params[0], best_fit_params[1], best_fit_params[2], best_fit_params[3], 1).item(), 8) *2/3  #dist.samplenorm(1).item()
            chng.append(pct_chg)
            underlyingP = underlyingP + underlyingP * pct_chg  # + 30
            underlying_price.append(underlyingP)
        else:
            pct_chg = round(dist.sampledist(best_dist, best_fit_params[0], best_fit_params[1], best_fit_params[2],best_fit_params[3], 1).item(), 8)  #dist.samplenorm(1).item()
            chng.append(pct_chg)
            underlyingP = underlyingP + underlyingP * pct_chg  # - 30
            underlying_price.append(underlyingP)

        call.setUnderlyingP(underlyingP)
        call.setTimetoMaturity(T)
        opt_prc = call.getPrice(vol)
        option_price.append(opt_prc)

        delta = call.getGreeks(vol)['delta']
        deltas.append(delta)

        purchased_shares = -100 * round(delta, 2) - shares_owned
        shares_purchased.append(purchased_shares)

        shares_owned = shares_owned + purchased_shares
        pf_shares.append(shares_owned)

        cost_of_shares = purchased_shares * underlyingP
        trxn_costs.append(cost_of_shares)

    df = pd.DataFrame([underlying_price, chng, deltas, pf_shares, shares_purchased, trxn_costs, option_price]).T
    df.columns = ['underlying_price', 'pct_chg', 'deltas', 'pf_shares', 'shares_purchased', 'trxn_costs',
                  'option_price']
    # assume position is closed at mkt close or hedged at mkt close
    df['cum_trxn_costs'] = df['trxn_costs'].cumsum()
    df['earlyexit_share_pnl'] = df['pf_shares'].shift(1) * df['underlying_price'] - df['cum_trxn_costs'].shift(1)
    df['earlyexit_option_pnl'] = (df['option_price'] - opt_cost) * 100
    df['earlyexit_pnl'] = df['earlyexit_share_pnl'] + df['earlyexit_option_pnl']
    expiration_pnl = df['earlyexit_pnl'].loc[len(df['earlyexit_pnl']) - 1]
    df['underlying_price'].plot(color='blue', linewidth=3.0, )
    plt.axhline(stk, color='red', linewidth=5.0, linestyle='--')
    plt.title("Underlying Price Monte Carlo Simulation")
    plt.ylabel("Price")
    plt.xlabel("Days")
    plt.savefig("UnderlyingPrice.png")
    plt.show()
    print("PNL on Expiration: {}".format(expiration_pnl))
    print("Realized Vol: {}".format(pd.Series(underlying_price).pct_change().std() * np.sqrt(252)))
    print(df)

if __name__=="__main__":
    main()


