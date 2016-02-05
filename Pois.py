import numpy as np
from scipy.stats import expon

class prior:

    @staticmethod
    def simul():
        theta = np.random.exponential()
        return theta

    @staticmethod
    def eval(x):
        return expon.pdf(x)

class likelihood:

    @staticmethod
    def simul(theta, n):
        z = np.random.poisson(theta, size = n)
        return z
