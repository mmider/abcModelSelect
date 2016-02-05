import numpy as np
import matplotlib.pyplot as plt
from matplotlib.path import Path
import matplotlib.patches as patches
import ABC
import MA2
import MA1
import Geom
import Pois
import RandFieldM1 as rf1
import RandFieldM2 as rf2

def draw_triangle(ax):
    verts = [
        (0.,-1.),
        (2.,1.),
        (-2.,1.),
        (0.,-1.),
        ]
    codes = [
        Path.MOVETO,
        Path.LINETO,
        Path.LINETO,
        Path.CLOSEPOLY,
        ]
    path = Path(verts, codes)
    patch = patches.PathPatch(path, facecolor = (0.1,0.1,0.1,0.1), lw=0)
    ax.add_patch(patch)

def plot_thetas(thetas):
    thetas = np.array(thetas)
    fig = plt.figure()
    ax = plt.subplot('111')
    ax.set_xlim([-2,2])
    ax.set_ylim([-1,1])
    draw_triangle(ax)
    ax.scatter(thetas.T[0], thetas.T[1])
    plt.show()

class twoModelsPrior:

    @staticmethod
    def simul():
        return np.random.binomial(1,0.5)
    
def randFields_sumStats(y):
    return np.array([np.sum(1 * (y[:-1] == y[1:]))])

def euclidean_dist(y,z):
    """
    Computes the Euclidean distance between two points
    """
    d = np.linalg.norm(y-z)
    return d

def autocov(y):
    """
    Computes the autocovariance for first 2 terms (plus the zeroth one)
    """
    n = len(y)
    autoCov = np.array([np.sum(y[i:]*y[:n-i]) for i in range(3)])
    return autoCov


def test_MA2_model(which = "basic", noisy = False):
    y = MA2.simulate_dataset(theta = [0.6,0.2], epochs = 100)
    N = int(1e6)
    perc = 1e-3
    iterations = int(N * perc)    

    if which == "MCMC":
        eps = 60
        thetas, zs = ABC.MCMCsampler(y = y, iterations = iterations,
                                     prior = MA2.prior,
                                     likelihood = MA2.likelihood,
                                     dist = euclidean_dist,
                                     tolerance = eps,
                                     sum_statistics = autocov,
                                     transKernel = MA2.transKernel)
        plot_thetas(thetas)
        
    elif which == "basic":
        eps = ABC.findEpsilon(y = y, iterations = N, prior = MA2.prior,
                              likelihood = MA2.likelihood,
                              dist = euclidean_dist, perc = perc,
                              sum_statistics = autocov)

        thetas, zs = ABC.sampler(y = y, iterations = iterations,
                                 prior = MA2.prior,
                                 likelihood = MA2.likelihood,
                                 dist = euclidean_dist,
                                 tolerance = eps,
                                 sum_statistics = autocov)
        plot_thetas(thetas)
        
    elif which == "importance":
        thetas, zs, weights = ABC.importanceSampler(y = y,
                                                    iterations = int(1e5),
                                                    prior = MA2.prior,
                                                    likelihood = MA2.likelihood,
                                                    proposal = MA2.proposal,
                                                    densKernel = MA2.densKernel,
                                                    bandwidth = 10,
                                                    sum_statistics = autocov,
                                                    noisy = noisy)
        print(np.sum([t * w for t,w in
                      zip(thetas, weights)],axis = 0)/np.sum(weights))
    elif which == "KernelMCMC":
        thetas, zs = ABC.kernelMCMCSampler(y = y, iterations = int(1e4),
                                           prior = MA2.prior,
                                           likelihood = MA2.likelihood,
                                           densKernel = MA2.densKernel,
                                           bandwidth = 10,
                                           transKernel = MA2.transKernel,
                                           sum_statistics = autocov,
                                           dist = euclidean_dist,
                                           tolerance = 10, noisy = noisy)
        plot_thetas(thetas)
    elif which == "modelChoiceSampler":
        """
        eps1 = ABC.findEpsilon(y = y, iterations = N, prior = MA1.prior,
                              likelihood = MA1.likelihood,
                              dist = MA2.euclidean_dist, perc = perc,
                              sum_statistics = MA2.autocov)
        eps2 = ABC.findEpsilon(y = y, iterations = N, prior = MA2.prior,
                              likelihood = MA2.likelihood,
                              dist = MA2.euclidean_dist, perc = perc,
                              sum_statistics = MA2.autocov)
        """
        thetas, zs, ms = ABC.modelChoiceSampler(y = y, iterations = iterations,
                                                paramPriors = [MA1.prior, MA2.prior],
                                                likelihoods = [MA1.likelihood, MA2.likelihood],
                                                modelPrior = twoModelsPrior,
                                                dist = euclidean_dist,
                                                tolerance = 7,
                                                sum_statistics = autocov)

        print(len([m for m in ms if m == 1])/len(ms))
        print(len([m for m in ms if m == 0])/len(ms))
        
    else:
        raise ValueError("Invalid name of tested model.")

def test_Geom_Pois(trueGener = "Pois"):
    if trueGener == "Pois":
        theta = 0.5
        gen = Pois.likelihood.simul
        t0 = theta
        
    else:
        theta = 0.5
        t0 = 1/theta
        gen = Geom.likelihood.simul
    y = gen(theta = theta, n = 50)
    N = int(1e6)
    perc = 1e-3
    iterations = int(N * perc)
    """
    eps = ABC.findEpsilon(y = y, iterations = N, prior = Geom.prior,
                              likelihood = Geom.likelihood,
                              dist = MA2.euclidean_dist, perc = perc,
                              sum_statistics = np.sum)
    print(eps)
    """
    
    thetas, zs, ms = ABC.modelChoiceSampler(y = y, iterations = iterations,
                                                paramPriors = [Geom.prior, Pois.prior],
                                                likelihoods = [Geom.likelihood, Pois.likelihood],
                                                modelPrior = twoModelsPrior,
                                                dist = euclidean_dist,
                                                tolerance = 0,
                                                sum_statistics = np.sum)
    ev0 = len([m for m in ms if m == 0])/len(ms)
    ev1 = len([m for m in ms if m == 1])/len(ms)
    print("Empirical Bayes, based on sumStats: ",ev1/ev0)
    print("What I think it should be: ",(t0 + 1)**2/t0*np.exp(-t0))

def test_RandField(trueGener = "trivial"):
    if trueGener == "trivial":
        theta = 3
        gen = rf1.likelihood.simul
    else:
        theta = 4
        gen = rf2.likelihood.simul
    y = gen(theta = theta, n = 50)
    N = int(1e6)
    perc = 1e-3
    iterations = int(N * perc)

    thetas, zs, ms = ABC.modelChoiceSampler(y = y, iterations = iterations,
                                                paramPriors = [rf1.prior, rf2.prior],
                                                likelihoods = [rf1.likelihood, rf2.likelihood],
                                                modelPrior = twoModelsPrior,
                                                dist = euclidean_dist,
                                                tolerance = 0,
                                                sum_statistics = randFields_sumStats)
    ev0 = len([m for m in ms if m == 0])/len(ms)
    ev1 = len([m for m in ms if m == 1])/len(ms)
    print(ev1/ev0)
    

# test_RandField()
# test_Geom_Pois("Pois")

test_MA2_model("KernelMCMC")
