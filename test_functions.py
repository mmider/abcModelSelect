import numpy as np
import ABC_naive as ABC
import ABC_vectorized as ABCv
import MA2
import MA1
import Geom
import Pois
import RandFieldM1 as rf1
import RandFieldM2 as rf2
import exact_bayes_factor as bf
import misc

def test_MA2_samplers(which = "basic", noisy = False,
                      generator = "MA(2)", naive = True,
                      exactBF = False):
    # simulate observation
    if generator == "MA(2)":
        y = MA2.likelihood.simul(theta = [0.6,0.2], n = 100+2)
    elif generator == "MA(1)":
        y = MA1.likelihood.simul(theta = 0.6, n = 100+1)
    else:
        raise ValueError("Invalid name of the generator, "
                         "only MA(1) and MA(2) are allowed.")
    # total number of simulations
    N = int(1e6)

    # expected proportion of accepted simulations
    perc = 1e-3

    # effective number of simulations:
    iterations = int(N * perc)

    if which == "MCMC":
        eps = 60
        thetas, zs = ABC.MCMCsampler(y = y, iterations = iterations,
                                     prior = MA2.prior,
                                     likelihood = MA2.likelihood,
                                     dist = misc.euclidean_dist,
                                     tolerance = eps,
                                     sum_statistics = misc.autocov,
                                     transKernel = MA2.transKernel)
        misc.plot_thetas(thetas)
        
    elif which == "basic":
        eps = ABC.findEpsilon(y = y, iterations = N, prior = MA2.prior,
                              likelihood = MA2.likelihood,
                              dist = misc.euclidean_dist, perc = perc,
                              sum_statistics = misc.autocov)

        thetas, zs = ABC.sampler(y = y, iterations = iterations,
                                 prior = MA2.prior,
                                 likelihood = MA2.likelihood,
                                 dist = misc.euclidean_dist,
                                 tolerance = eps,
                                 sum_statistics = misc.autocov)
        misc.plot_thetas(thetas)
        
    elif which == "importance":
        thetas, zs, weights = ABC.importanceSampler(y = y,
                                                    iterations = int(1e5),
                                                    prior = MA2.prior,
                                                    likelihood = MA2.likelihood,
                                                    proposal = MA2.proposal,
                                                    densKernel = MA2.densKernel,
                                                    bandwidth = 10,
                                                    sum_statistics = misc.autocov,
                                                    noisy = noisy)
        out = np.sum([t * w for t,w in
                      zip(thetas, weights)],axis = 0)/np.sum(weights)
        print("Evaluation of E[theta] using importance sampler gives: ", out)
        
    elif which == "KernelMCMC":
        thetas, zs = ABC.kernelMCMCSampler(y = y, iterations = int(1e4),
                                           prior = MA2.prior,
                                           likelihood = MA2.likelihood,
                                           densKernel = MA2.densKernel,
                                           bandwidth = 10,
                                           transKernel = MA2.transKernel,
                                           sum_statistics = misc.autocov,
                                           dist = misc.euclidean_dist,
                                           tolerance = 10, noisy = noisy)
        misc.plot_thetas(thetas)
        
    elif which == "modelChoiceSampler":
        eps = ABCv.findEpsilon(y = y, iterations = N, prior = MA2.prior,
                                  likelihood = MA2.likelihood,
                                    dist = misc.euclidean_dist_multi,
                                    perc = np.array([0.01,0.001, 0.0001]),
                                  sum_statistics = misc.autocov)

        if naive:
            thetas, zs, ms = ABC.modelChoiceSampler(y = y, iterations = iterations,
                                                    paramPriors = [MA1.prior, MA2.prior],
                                                    likelihoods = [MA1.likelihood, MA2.likelihood],
                                                    modelPrior = misc.twoModelsPrior,
                                                    dist = misc.euclidean_dist,
                                                    tolerance = 7,
                                                    sum_statistics = misc.autocov)

            ev0 = len([m for m in ms if m == 0])/len(ms)
        else:
            evs = []
            for e in eps:
                ms = ABCv.modelChoiceSampler(y = y, iterations = N,
                                     paramPriors = [MA1.prior, MA2.prior],
                                     likelihoods = [MA1.likelihood, MA2.likelihood],
                                     modelPrior = misc.twoModelsPrior,
                                     dist = misc.euclidean_dist_multi,
                                     tolerance = e,
                                     sum_statistics = misc.autocov)
                ev0 = ms[0]/(ms[0]+ms[1])
                evs.append(ev0)
        if exactBF:
            print("Finding actual Bayes Factor...\n")
            exact = bf.MALogBayesFactor(y, resolution=(100,200))
        for e in evs:
            misc.printModelChoice("MA(1)", "MA(2)", e, logBF = exact)
    else:
        raise ValueError("Invalid name of tested model.")

def test_Geom_Pois(trueGener = "Pois", naive = True, numBF = False):
    if trueGener == "Pois":
        theta = 4
        gen = Pois.likelihood.simul
        t0 = theta        
    elif trueGener == "Geom":
        theta = 0.5
        t0 = 1/theta
        gen = Geom.likelihood.simul
    else:
        raise ValueError("Invalid value for the generator."
                         " Only Pois and Geom are allowed.")
    # generate the data
    y = gen(theta = theta, n = 50)

    # total number of samples
    N = int(1e6)

    # percentage of accepted samples
    perc = 1e-3

    # expected number of accepted samples
    iterations = int(N * perc)
    """
    eps = ABC.findEpsilon(y = y, iterations = N, prior = Geom.prior,
                              likelihood = Geom.likelihood,
                              dist = MA2.euclidean_dist, perc = perc,
                              sum_statistics = np.sum)
    print(eps)
    """
    if naive:
        thetas, zs, ms = ABC.modelChoiceSampler(y = y, iterations = iterations,
                                                paramPriors = [Pois.prior, Geom.prior],
                                                likelihoods = [Pois.likelihood, Geom.likelihood],
                                                modelPrior = misc.twoModelsPrior,
                                                dist = misc.euclidean_dist,
                                                tolerance = 0,
                                                sum_statistics = np.sum)
        ev0 = len([m for m in ms if m == 0])/len(ms)
    else:
        ms = ABCv.modelChoiceSampler(y = y, iterations = N,
                                     paramPriors = [Pois.prior, Geom.prior],
                                     likelihoods = [Pois.likelihood, Geom.likelihood],
                                     modelPrior = misc.twoModelsPrior,
                                     dist = misc.scalar_dist_multi,
                                     tolerance = 0,
                                     sum_statistics = misc.sumMulti)
        ev0 = ms[0]/(ms[0]+ms[1])
    if numBF:
        poislogev, geomlogev = bf.GeomPoisLogBayesFactor(y)
    misc.printModelChoice("Pois", "Geom", ev0, logBF = poislogev-geomlogev)
    print("What I think it should be: ",2 * np.log(t0 + 1) - np.log(t0) - t0)
    

def test_RandField(trueGener = "trivial", theta = 4, naive = True, exactBF = False):
    theta = theta
    if trueGener == "trivial":
        gen = rf1.likelihood.simul
    else:
        gen = rf2.likelihood.simul
    y = gen(theta = theta, n = 50)
    N = int(1e6)
    perc = 1e-3
    iterations = int(N * perc)

    if naive:
        thetas, zs, ms = ABC.modelChoiceSampler(y = y, iterations = iterations,
                                                    paramPriors = [rf1.prior, rf2.prior],
                                                    likelihoods = [rf1.likelihood, rf2.likelihood],
                                                    modelPrior = misc.twoModelsPrior,
                                                    dist = misc.euclidean_dist,
                                                    tolerance = 0,
                                                    sum_statistics = misc.RFSumStats)
        ev0 = len([m for m in ms if m == 0])/len(ms)
    if naive:
        ms = ABCv.modelChoiceSampler(y = y, iterations = N,
                                         paramPriors = [rf1.prior, rf2.prior],
                                         likelihoods = [rf1.likelihood, rf2.likelihood],
                                         modelPrior = misc.twoModelsPrior,
                                         dist = misc.euclidean_dist_multi,
                                         tolerance = 0,
                                         sum_statistics = misc.RFSumStats)
        ev0v = ms[0]/(ms[0]+ms[1])

    if exactBF:
        ev0_true, ev1_true = bf.RFBayesFactor(y)
    misc.printModelChoice("trivial", "MarkovChain", ev0)
    misc.printModelChoice("trivial", "MarkovChain", ev0v, ev0_true)

