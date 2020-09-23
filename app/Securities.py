
from math import log, sqrt, pi, exp
from scipy.stats import norm
from datetime import datetime, date
import numpy as np
import pandas as pd


# Underlying price (per share): S;
# Strike price of the option (per share): K;
# Time to maturity (years): T;
# Continuously compounding risk-free interest rate: r;
# Volatility: sigma;

class Option(object):
    def __init__(self, underlying_price, strike, time_to_maturity, risk_free_rate, is_call):
        self.S = underlying_price
        self.K = strike
        self.T = time_to_maturity
        self.r = risk_free_rate
        self.is_call = bool(is_call)

    def setPut(self):
        self.is_call = False

    def setCall(self):
        self.is_call = True

    def isCall(self):
        return self.is_call

    def isPut(self):
        if self.is_call:
            return False
        if not self.is_call:
            return True

    def setTimetoMaturity(self, years_remaining):
        self.T = years_remaining

    def setUnderlyingP(self, uPrice):
        self.S = uPrice

    def getInfo(self):
        return {'underlying price': self.S,
                'strike' : self.K,
                'time to maturity': self.T,
                'risk free rate (percent)': self.r,
                'is call': self.is_call}

    def getPrice(self, volatility):
        if self.is_call:
            return self._getCallPrice(volatility)
        if not self.is_call:
            return self._getPutPrice(volatility)

    def _d1(self, S,K,T,r,sigma):
        return np.float64(log(np.float64(S)/np.float64(K))+(r+ 0.5 * sigma**2)*T)/(sigma*sqrt(T))

    def _d2(self, S,K,T,r,sigma):
        return np.float64(log(S/K) + (r - 0.5 * sigma ** 2) * T) / np.float64(sigma *sqrt(T))


    def _getCallPrice(self, sigma):
        d1 = self._d1(self.S, self.K, self.T, self.r, sigma)
        d2 = self._d2(self.S, self.K, self.T, self.r, sigma)
        return (self.S * norm.cdf(d1, 0.0, 1.0) - self.K * np.exp(-self.r * self.T) * norm.cdf(d2, 0.0, 1.0))

    def _getPutPrice(self, sigma):
        d1 = self._d1(self.S, self.K, self.T, self.r, sigma)
        d2 = self._d2(self.S, self.K, self.T, self.r, sigma)
        return (self.K * np.exp(-self.r * self.T) * norm.cdf(-d2, 0.0, 1.0) - self.S * norm.cdf(-d1, 0.0, 1.0))

    def getImpliedVol(self, Price):
        sigma = 0.001
        if self.is_call  == True:
            while sigma < 1:
                call_price_implied = self._getCallPrice(sigma)
                if Price - (call_price_implied) < 0.001:
                    return sigma
                sigma += 0.001
            return ArithmeticError("It could not find the right volatility of the call option.")
        else:
            while sigma < 1:
                put_price_implied = self._getPutPrice(sigma)
                if Price - (put_price_implied) < 0.001:
                    return sigma
                sigma += 0.001
            return ArithmeticError("It could not find the right volatility of the put option.")
        return TypeError("Call or Put not specified in the object.")

    def getGreeks(self, volatility):
        if self.is_call:
            return self._getCallGreeks(volatility)
        if not self.is_call:
            return self._getPutGreeks(volatility)

    def _getCallGreeks(self, volatility):
        return {'delta': self._call_delta(volatility),
                'gamma': self._gamma(volatility),
                'vega': self._vega(volatility),
                'theta': self._call_theta(volatility),
                'rho': self._call_rho(volatility)}

    def _getPutGreeks(self, volatility):
        return {'delta': self._put_delta(volatility),
                'gamma': self._gamma(volatility),
                'vega': self._vega(volatility),
                'theta': self._put_theta(volatility),
                'rho': self._put_rho(volatility)}

    def _call_delta(self, sigma):
        d1 = self._d1(self.S, self.K, self.T, self.r, sigma)
        return norm.cdf(d1, 0.0, 1.0)

    def _call_theta(self, sigma):
        d1 = self._d1(self.S, self.K, self.T, self.r, sigma)
        d2 = self._d2(self.S, self.K, self.T, self.r, sigma)
        prob_density = 1 / np.sqrt(2 * np.pi) * np.exp(-d1 ** 2 * 0.5)
        return (-sigma * self.S * prob_density) / (2 * np.sqrt(self.T)) - self.r * self.K * np.exp(-self.r * self.T) * norm.cdf(d2, 0.0, 1.0)

    def _call_rho(self, sigma):
        d2 = self._d2(self.S, self.K, self.T, self.r, sigma)
        return self.T * self.K * np.exp(-self.r * self.T) * norm.cdf(d2, 0.0, 1.0)

    def _put_delta(self, sigma):
        d1 = self._d1(self.S, self.K, self.T, self.r, sigma)
        return -norm.cdf(-d1, 0.0, 1.0)

    def _gamma(self, sigma):
        d1 = self._d1(self.S, self.K, self.T, self.r, sigma)
        prob_density = 1 / np.sqrt(2 * np.pi) * np.exp(-d1 ** 2 * 0.5)
        return prob_density / (self.S * sigma * np.sqrt(self.T))

    def _vega(self, sigma):
        d1 = self._d1(self.S, self.K, self.T, self.r, sigma)
        return self.S * norm.cdf(d1, 0.0, 1.0) * np.sqrt(self.T)

    def _put_theta(self, sigma):
        d1 = self._d1(self.S, self.K, self.T, self.r, sigma)
        d2 = self._d2(self.S, self.K, self.T, self.r, sigma)
        prob_density = 1 / np.sqrt(2 * np.pi) * np.exp(-d1 ** 2 * 0.5)
        return (-sigma * self.S * prob_density) / (2 * np.sqrt(self.T)) + self.r * self.K * np.exp(-self.r * self.T) * norm.cdf(-d2, 0.0, 1.0)

    def _put_rho(self, sigma):
        d2 = self._d2(self.S, self.K, self.T, self.r, sigma)
        return -self.T * self.K * np.exp(-self.r * self.T) * norm.cdf(-d2, 0.0, 1.0)




