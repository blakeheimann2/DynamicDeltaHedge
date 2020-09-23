import warnings
import numpy as np
import pandas as pd
import scipy.stats as st
import statsmodels as sm
import matplotlib
import matplotlib.pyplot as plt
import time
import datetime as dt
import pandas_datareader.data as web

warnings.filterwarnings('ignore')
matplotlib.rcParams['figure.figsize'] = (16.0, 12.0)

class SampleFitter(object):
    def __init__(self, data):
        self.data = data
        self.best_distribution = None
        self.best_params = None
        self.best_sse = np.inf
        self._ax = None
        self._dataYLim = None

    def _set_axes_for_plots(self):
        plt.figure(figsize=(12, 8))
        self._ax = self.data.plot(kind='hist', bins=50, normed=True, alpha=0.5)
        self._dataYLim = self._ax.get_ylim()

    def get_best_fit(self, distribution_list, bins=200, verbose = True):
        self._set_axes_for_plots()
        y, x = np.histogram(self.data, bins=bins, density=True)
        x = (x + np.roll(x, -1))[:-1] / 2.0
        for distribution in distribution_list:
            try:
                with warnings.catch_warnings():
                    # fit dist to data
                    params = distribution.fit(self.data)
                    # Separate parts of parameters
                    arg = params[:-2]
                    loc = params[-2]
                    scale = params[-1]
                    # Calculate fitted PDF and error with fit in distribution
                    pdf = distribution.pdf(x, loc=loc, scale=scale, *arg)
                    sse = np.sum(np.power(y - pdf, 2.0))
                    if verbose:
                        print("Distribution: {0}; Params: {1}; SSE: {2}".format(distribution.name, params, sse))
                    # if axis pass in add to plot
                    try:
                        pd.Series(pdf, x).plot(ax=self._ax)
                    except Exception:
                        pass
                    # identify if this distribution is better
                    if self.best_sse > sse > 0:
                        self.best_distribution = distribution
                        self.best_params = params
                        self.best_sse = sse
            except Exception:
                pass
        if verbose:
            ("BEST Distribution: {0}; Params: {1}; SSE: {2}".format(self.best_distribution.name, self.best_params, self.best_sse))
        return (self.best_distribution.name, self.best_params)

    def _make_pdf(self, size = 10000):
        """Generate distributions's Probability Distribution Function """
        # Separate parts of parameters
        arg = self.best_params[:-2]
        loc = self.best_params[-2]
        scale = self.best_params[-1]
        # Get sane start and end points of distribution
        start = self.best_distribution.ppf(0.01, *arg, loc=loc, scale=scale) if arg else self.best_distribution.ppf(0.01, loc=loc, scale=scale)
        end = self.best_distribution.ppf(0.99, *arg, loc=loc, scale=scale) if arg else self.best_distribution.ppf(0.99, loc=loc, scale=scale)
        # Build PDF and turn into pandas Series
        x = np.linspace(start, end, size)
        y = self.best_distribution.pdf(x, loc=loc, scale=scale, *arg)
        pdf = pd.Series(y, x)
        return pdf


    def plot_best_fit(self, title):
        # Find best fit distribution
        best_dist = getattr(st, self.best_distribution.name)
        # Update plots
        self._ax.set_ylim(self._dataYLim)
        self._ax.set_title(u'{}.\n All Fitted Distributions'.format(title))
        self._ax.set_xlabel(u'Values')
        self._ax.set_ylabel('Frequency')
        # Make PDF with best params
        pdf = self._make_pdf()
        # Display
        plt.figure(figsize=(12, 8))
        ax = pdf.plot(lw=2, label='PDF', legend=True)
        self.data.plot(kind='hist', bins=50, density=True, alpha=0.5, label='Data', legend=True, ax=ax)
        param_names = (best_dist.shapes + ', loc, scale').split(', ') if best_dist.shapes else ['loc', 'scale']
        param_str = ', '.join(['{}={:0.2f}'.format(k, v) for k, v in zip(param_names, self.best_params)])
        dist_str = '{}({})'.format(self.best_distribution.name, param_str)
        ax.set_title(u'{} Best Distribution \n'.format(title) + dist_str)
        ax.set_xlabel(u'Values')
        ax.set_ylabel('Frequency')
        plt.show()
