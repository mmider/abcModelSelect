import numpy as np

class prior:
    @staticmethod
    def simul():
        theta = np.random.uniform(-2,2)
        return theta

    @staticmethod
    def simulMulti(n):
        theta = np.random.uniform(-2,2,n)
        return theta

    @staticmethod
    def eval(theta):
        if (-2 < theta < 2):
            return 1/4
        else:
            return 0

class likelihood:

    @staticmethod
    def simul(theta, n):
        u = np.random.normal(0,1,n)
        z = u[1:] + theta*u[:-1]
        return np.array(z)

    @staticmethod
    def simulMulti(theta, n):
        u = np.random.randn(n, len(theta))
        z = u[1:,:] + theta * u[:-1,:]
        return z

