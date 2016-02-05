import numpy as np

class prior:

    @staticmethod
    def simul():
        theta = np.random.uniform()
        return theta

    @staticmethod
    def eval(x):
        return 1 * (0 < x < 1)

class likelihood:

    @staticmethod
    def simul(theta, n):
        z = np.random.geometric(theta, size = n)
        return z

    
