import numpy as np
from scipy.stats import expon

class prior:

    @staticmethod
    def simul():
        theta = np.random.exponential()
        return theta

    @staticmethod
    def simulMulti(n):
        theta = np.random.exponential(size = n)
        return theta

    @staticmethod
    def eval(x):
        return expon.pdf(x)

class likelihood:

    @staticmethod
    def simul(theta, n):
        z = np.random.poisson(theta, size = n)
        return z
    
    @staticmethod
    def simulMulti(theta, n):
        m = len(theta)
        z = np.random.poisson(np.tile(theta, n))
        z = z.reshape(n,m)
        return z
