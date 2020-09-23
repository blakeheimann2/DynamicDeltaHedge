import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.stats import norm

class CurveFitter(object):
    def __init__(self, data):
        self.data = data
        self.mean = None

    def fitGaussian(self):
        mean, std = norm.fit(self.data)
        self.mean = mean
        self.std = std

    def sampledist(self, scipy_dist_fun, *args):
        return scipy_dist_fun.rvs(*args)

    def plot(self):
        plt.hist(self.data, bins=30, normed=True)
        xmin, xmax = plt.xlim()
        x = np.linspace(xmin, xmax, 100)
        y = norm.pdf(x, self.mean, self.std)
        plt.plot(x, y)
        plt.show()

    def getMeanStd(self):
        return self.mean, self.std

    def samplenorm(self, n):
        return np.random.normal(self.mean, self.std, n)




