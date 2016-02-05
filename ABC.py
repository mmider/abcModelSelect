import numpy as np

def sampler(y, iterations, prior, likelihood,
               dist, tolerance, sum_statistics):
    print("Running an ABC sampler.")
    y_stats = sum_statistics(y)
    n_full = len(y)
    thetas = []
    zs = []
    p = progress(iterations)
    for i in range(iterations):
        p.progressStatus(i)
        d = 2 * (tolerance + 1)
        while d > tolerance:
            theta = prior.simul()
            z = likelihood.simul(theta, n_full)
            d = dist(sum_statistics(z), y_stats)
        thetas.append(theta)
        zs.append(z)
    return (thetas, zs)

def MCMCsampler(y, iterations, prior, likelihood,
                   dist, tolerance, sum_statistics,
                   transKernel):
    print("Running an ABC MCMC sampler.")
    y_stats = sum_statistics(y)
    n_full = len(y)
    # zeroth realisation
    thetas, zs = sampler(y = y, iterations = 1, prior = prior,
                          likelihood = likelihood,
                          dist = dist, tolerance = tolerance,
                          sum_statistics = sum_statistics)
    theta = thetas[0]
    z = zs[0]
    
    p = progress(iterations)
    for i in range(iterations):
        p.progressStatus(i)
        theta_prop = transKernel.simul(theta)
        z_prop = likelihood.simul(theta_prop, n_full)
        d = dist(sum_statistics(z_prop), y_stats)
        u = np.random.uniform()
        K = (np.log(prior.eval(theta_prop))+\
            np.log(transKernel.eval(theta_prop, theta)))-\
            (np.log(prior.eval(theta)) +\
             np.log(transKernel.eval(theta, theta_prop)))
        if (u <= np.exp(K) and d <= tolerance):
            theta, z = (theta_prop, z_prop)
        thetas.append(theta)
        zs.append(z)
    return (np.array(thetas), np.array(zs))

def importanceSampler(y, iterations, prior, likelihood, proposal,
                      densKernel, bandwidth, sum_statistics, noisy = False):
    """
    Runs an ABC importance sampler
    """
    y_stats = sum_statistics(y)
    n_full = len(y)
    thetas = []
    zs = []
    weights = []
    
    p = progress(iterations)
    for i in range(iterations):
        p.progressStatus(i)
        theta = proposal.simul()
        z = likelihood.simul(theta, n_full)
        z_stats = sum_statistics(z)
        n = len(z_stats)
        if noisy:
            z_stats += bandwidth * densKernel.simul(n)
        K = densKernel.eval((z_stats - y_stats)/bandwidth)
        if np.random.uniform() < K:
            thetas.append(theta)
            zs.append(z)
            weights.append(prior.eval(theta)/proposal.eval(theta))
    return (thetas, zs, weights)

def kernelMCMCSampler(y, iterations, prior, likelihood,
                      densKernel, bandwidth, transKernel,
                      sum_statistics, dist, tolerance, noisy = False):
    print("Runs an ABC kernelized MCMC sampler.")
    y_stats = sum_statistics(y)
    n_full = len(y)
    thetas, zs = sampler(y = y, iterations = 1, prior = prior,
                          likelihood = likelihood,
                          dist = dist, tolerance = tolerance,
                          sum_statistics = sum_statistics)
    theta = thetas[0]
    z = zs[0]
    
    z_stats = sum_statistics(z)
    n = len(z_stats)
    
    if noisy:
        z_stats += bandwidth * densKernel.simul(n)
        
    p = progress(iterations)
    for i in range(iterations):
        p.progressStatus(i)
        theta_prop =  transKernel.simul(theta)
        z_prop = likelihood.simul(theta_prop, n_full)
        z_prop_stats = sum_statistics(z_prop)
        if noisy:
            z_prop_stats += bandwidth * densKernel.simul(n)
        K = (
            np.log(densKernel.eval((z_prop_stats - y_stats)/bandwidth)) +\
            np.log(prior.eval(theta_prop)) +\
            np.log(transKernel.eval(theta_prop, theta))
            )-\
            (np.log(densKernel.eval((z_stats - y_stats)/bandwidth)) +\
             np.log(prior.eval(theta)) +\
             np.log(transKernel.eval(theta, theta_prop)))
        if np.random.uniform() < np.exp(min(0,K)):
            theta = theta_prop
            z = z_prop
            z_stats = z_prop_stats
        thetas.append(theta)
        zs.append(z)
    return (thetas, zs)

def semiAutomaticSampler(y, iterations, prior, likelihood,
                         densKernel, bandwidth, sum_statistics,
                         dist, tolerance, data_transform):
    raise NotImplementedError

def modelChoiceSampler(y, iterations, paramPriors, likelihoods,
                       modelPrior, dist, tolerance, sum_statistics):
    print("Running an ABC model choice sampler.")
    n_full = len(y)
    thetas = []
    zs = []
    ms = []
    y_stats = sum_statistics(y)
    p = progress(iterations)
    for i in range(iterations):
        p.progressStatus(i)
        d = 2 * (tolerance + 1)
        while d > tolerance:
            m = modelPrior.simul()
            theta = paramPriors[m].simul()
            z = likelihoods[m].simul(theta, n_full)
            d = dist(sum_statistics(z), y_stats)
        thetas.append(theta)
        zs.append(z)
        ms.append(m)
    return (thetas, zs, ms)

def modelChoiceSamplerQuick(y, iterations, paramPriors, likelihoods,
                            modelPrior, dist, tolerance, sum_statistics):
    print("Running a quick ABC model choice sampler.")
    n_full = len(y)
    y_stats = sum_statistics(y)
    p = progress(iterations)
    ms = modelPrior.simulMulti(iterations)
    theta0 = paramPriors[0].simulMulti(ms[0])
    theta1 = paramPriors[1].simulMulti(ms[1])
    z0 = likelihoods[0].simul(theta0, n_full)
    z1 = likelihoods[1].simul(theta1, n_full)
    d0 = dist(sum_statistics(z0), y_stats)
    d1 = dist(sum_statistics(z1), y_stats)
    

    
    for i in range(iterations):
        p.progressStatus(i)
        d = 2 * (tolerance + 1)
        while d > tolerance:
            z = likelihoods[m].simul(theta, n_full)
            d = dist(sum_statistics(z), y_stats)
        thetas.append(theta)
        zs.append(z)
        ms.append(m)
    return (thetas, zs, ms)



def findEpsilon(y, iterations, prior, likelihood,
               dist, perc, sum_statistics):
    print("Searching for the right tolerance level.")
    y_stats = sum_statistics(y)
    n_full = len(y)
    dist_all = []
    p = progress(iterations)
    for i in range(iterations):
        p.progressStatus(i)
        theta = prior.simul()
        z = likelihood.simul(theta, n_full)
        d = dist(sum_statistics(z), y_stats)
        dist_all.append(d)
    n = np.floor(iterations * perc)
    return np.sort(dist_all)[n]

class progress:
    
    def __init__(self, totalSteps):
        if totalSteps < 1:
            raise ValueError("Total number of steps must be at least 1.")
        self.totalSteps = totalSteps
        self.incrementSize = np.ceil(totalSteps/100)
        self.percentageCompleted = 0

    def progressStatus(self, i):
        if i % self.incrementSize == 0:
            print("Completed: {}%.".format(self.percentageCompleted))
            self.percentageCompleted += 1


