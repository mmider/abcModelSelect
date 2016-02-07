import numpy as np


class prior:

    @staticmethod
    def simul():
        theta = np.random.uniform(-5,5)
        return theta
    
    @staticmethod
    def simulMulti(n):
        theta = np.random.uniform(-5,5, size = n)
        return theta

    @staticmethod
    def eval(theta):
        return 1/10 * (-5 < theta < 5)

class likelihood:

    @staticmethod
    def simul(theta, n):
        prob = np.exp(theta)/(1+np.exp(theta))
        z = np.random.binomial(1,prob,n)
        return z
    
    @staticmethod
    def simulMulti(theta, n):
        m = len(theta)
        prob = np.exp(theta)/(1+np.exp(theta))
        z = np.random.binomial(1,np.tile(prob, n))
        z = z.reshape(n,m)
        return z



