import numpy as np
from scipy.stats import multivariate_normal

class prior:
    """
    prior for the MA(2) model
    """

    @staticmethod
    def simul():
        """
        Simulates one observation from the prior
        """
        theta1,theta2 = (-5,-5)
        while not (-2 < theta1 < 2 and
                   theta1 + theta2 > -1 and
                   theta1 - theta2 < 1):
            theta1 = np.random.uniform(-2,2)
            theta2 = np.random.uniform(-1,1)
        return np.array([theta1, theta2])

    @staticmethod
    def eval(theta):
        """
        Evaluates the value of a prior at point theta
        """
        if (-2 < theta[0] < 2 and
            theta[0] + theta[1] > -1 and
            theta[0] - theta[1] < 1):
            return 1/4
        else:
            return 0

    @staticmethod
    def simulMulti(n):
        """
        Simulates n observations from the prior.
        Returns 2 x n matrix, where each column contains
        one observation.
        """
        theta1 = np.random.uniform(-2,2,n)
        theta2 = np.random.uniform(-1,1,n)
        index = ((theta1 + theta2 <= -1) + (theta1 - theta2 >= 1) > 0)
        theta1[index] = np.sign(theta1[index]) * (2-np.abs(theta1[index]))
        theta2[index] = - theta2[index]
        theta = np.array([theta1, theta2])
        return theta

class likelihood:
    """
    Likelihood for the MA(2) model
    """

    @staticmethod
    def simul(theta,n):
        """
        Simulates one observation from the likelihood
        """
        u = np.random.normal(0,1,n)
        z = u[2:] + theta[0]*u[1:-1] + theta[1]*u[:-2]
        return np.array(z)

    @staticmethod
    def simulMulti(thetas, n):
        """
        Simulate n observations from the likelihood (n = len(theta)).
        Returns (n_full-2) x n matrix of observations, where each
        column contains one observation.
        """
        u = np.random.randn(n,thetas.shape[1])
        z = u[2:,:] + thetas[0] * u[1:-1,:] +\
            thetas[1] * u[:-2,:]
        return z


class proposal:

    @staticmethod
    def simul():
        theta = np.random.multivariate_normal(mean = [0,0],cov = np.diag([2,2]))
        return theta

    @staticmethod
    def eval(theta):
        mvn = multivariate_normal(mean = [0,0], cov = np.diag([2,2]))
        return mvn.pdf(theta)
    
class transKernel:
    """
    Transition Kernel for the MA2 model for the ABC MCMC sampler
    """
    
    @staticmethod
    def simul(theta):
        """
        Given a previous state of theta, simulates a new theta parameter
        """
        theta_prop = np.random.multivariate_normal(theta,np.diag([.1,.1]))
        return theta_prop

    @staticmethod
    def eval(theta1, theta2):
        """
        Evaluates the pdf of the kernel for a transition
        from theta1 to theta2
        """
        mvn = multivariate_normal(mean = theta1, cov = np.diag([.1,.1]))
        return mvn.pdf(theta2)

class densKernel:

    @staticmethod
    def simul(n):
        x = np.random.multivariate_normal(mean = np.zeros(n),
                                          cov = np.diag(np.ones(n)))
        return x

    @staticmethod
    def eval(x):
        n = len(x)
        mvn = multivariate_normal(mean = np.zeros(n),
                                  cov = np.diag(np.ones(n)))
        M = mvn.pdf(np.zeros(n))
        return mvn.pdf(x) / M

