import numpy as np
from scipy.special import binom
import misc
import MA2
import MA1

#   ---   ---   ---   RANDOM FIELDS   ---   ---   ---   #

def pwr(x,k):
    """
    Helper fn. for exclusive use with compute evidence.
    """
    if k == 0:
        return np.log(x)
    else:
        return (x**k)/k

def compute_evidence_RF(s,n,a,b):
    """
    Marginal density for Random Field model with sufficient statistic
    s, number of observations n, and a uniform  prior on (a,b).
    """
    out = 1/(b-a) * np.sum([binom(s-1,k) * (-1)**(s-1-k)*
                            (pwr(1+np.exp(b),k-n+1)-pwr(1+np.exp(a),k-n+1))
                            for k in range(int(s))])
    return out

def RFBayesFactor(y):
    """
    Computes constituents of the Bayes factor:
    P(M == trivial | y) / P(M == MarkovChain | y)
    exactly.
    """
    n = len(y)
    # summary statistics for both models
    s = misc.RFSumStats(y)

    # marginal densities for the two models
    evidence1 = compute_evidence_RF(s[0], n, -5, 5)
    evidence2 = compute_evidence_RF(s[1], n-1, 0, 6)
    
    return (evidence1, evidence2)



#   ---   ---   ---   POISSON / GEOMETRIC   ---   ---   ---   #

def logfact(k):
    """
    Computes log(k!) = log(1) + log(2) + log(3) + ... + log(k)
    """
    return np.sum([np.log(i) for i in range(1,k+1)])

def GeomPoisLogBayesFactor(x):
    """
    Computes constituents of the log Bayes factor:
    log(P(M == Pois | x)) - log(P(M == Geom | x))
    exactly.
    """
    # auxiliary variables
    s = np.sum(x)
    t = np.sum([logfact(xi) for xi in x])
    n = len(x)

    # marginal log densities for the two models
    logEvPois = logfact(s) - t - (s+1) * np.log(n+1)
    logEvGeo = logfact(n) + logfact(s) - logfact(n+s+1)

    return (logEvPois, logEvGeo)

#   ---   ---   ---   MA(q)   ---   ---   ---   #

class MAq_likelihood:
    """
    Object used for calculation of the "exact" Bayes Factor for
    the comparison of the MA(q) models:
    y_k = eps_k + sum_i (theta_i * eps_(k-i))
    """
    def __init__(self, theta, x, q):
        """
        Initialize the MA(q) object with model parameter theta and
        observation x.
        """
        self.theta = theta
        self.T = len(x)
        self.x = x
        self.q = q

    def simulate_noise(self):
        """
        Draw the first q epsilons at random
        """
        self.eps = np.zeros(self.T + self.q)
        self.eps[:self.q] = np.random.normal(0,1,size = self.q)

    def fill_eps(self):
        """
        Recursively fill in the rest of the epsilons.
        """
        for i in range(self.T):
            self.eps[i+self.q] = self.x[i] +\
                            np.sum([self.theta[self.q - 1 - j] * self.eps[i + j]
                                    for j in range(self.q)])
    
    def loglik(self):
        """
        Compute the conditional log-likelihood for the observation x,
        when parameter of the model is theta and the conditioning is
        done on the first q epsilons.
        """
        loglik = np.sum([-(self.x[i] +\
                           np.sum([self.theta[self.q - 1 - j] * self.eps[i+j+1]
                                   for j in range(self.q)]))**2
                         for i in range(self.T)])
        return loglik

    def find_loglik(self, N):
        """
        Compute the log-likelihood for the observation x when the parameter
        of the model is theta. It numerically integrates out the conditioned
        epsilons using the MC sampling.
        """
        results = np.zeros(N)
        for i in range(N):
            self.simulate_noise()
            self.fill_eps()
            results[i] = self.loglik()
        M = np.max(results)
        return np.log(np.mean(np.exp(results - M))) + M

def MALogBayesFactor(x, resolution = (400,200)):
    """
    Numerically approximates the log Bayes Factor for the MA models:
    log(P(M = MA(1) | x)) - log(P(M = MA(2) | x)).
    """
    # define matrix storing the approximations of the log-likelihoods
    ll = np.zeros([2,resolution[0]])

    # Monte Carlo:
    for i in range(resolution[0]):
        print("Beginning loop number: ", i)
        
        # sample parameters
        theta1 = MA1.prior.simul()
        theta2 = MA2.prior.simul()
        l1 = MAq_likelihood([theta1,], x, 1)
        l2 = MAq_likelihood(theta2, x, 2)
        ll[0,i] = l1.find_loglik(resolution[1])
        ll[1,i] = l2.find_loglik(resolution[1])
    M1 = np.max(ll[0,:])
    M2 = np.max(ll[1,:])
    A = np.log(np.mean(np.exp(ll[0,:] - M1)))+M1
    B = np.log(np.mean(np.exp(ll[1,:] - M2)))+M2
    # don't return this value for now, instead go with P(M=0|x)
    logBayesFactor = A - B
    ev0 = np.exp(A)/(np.exp(A) + np.exp(B))
    return ev0

