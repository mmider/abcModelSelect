import numpy as np
import matplotlib.pyplot as plt
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
import Normal as nrm
import time


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
                                    perc = np.array([0.01,0.001, 0.0001, 0.00001]),
                                  sum_statistics = misc.autocov)

        if naive:
            evs = []
            for e in eps:
                thetas, zs, ms = ABC.modelChoiceSampler(y = y, iterations = iterations,
                                                    paramPriors = [MA1.prior, MA2.prior],
                                                    likelihoods = [MA1.likelihood, MA2.likelihood],
                                                    modelPrior = misc.twoModelsPrior,
                                                    dist = misc.euclidean_dist,
                                                    tolerance = 7,
                                                    sum_statistics = misc.autocov)

                ev0 = len([m for m in ms if m == 0])/len(ms)
                evs.append(ev0)
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
        misc.plotMAModSel(evs)
        if exactBF:
            print("Finding actual Bayes Factor...\n")
            ev0exact = bf.MALogBayesFactor(y, resolution=(100,200))
        for e in evs:
            misc.printModelChoice("MA(1)", "MA(2)", e, ev0exact)
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
    print("What I think it should be: ", 2 * np.log(t0 + 1) - np.log(t0) - t0)
    return (np.log(ev0) - np.log(1-ev0), poislogev - geomlogev)

def plotGeomVsPois(trueGener = "Pois"):
    approxs = []
    exacts = []
    for i in range(300):
        approx, exact = test_Geom_Pois(trueGener, naive = False, numBF = True)
        approxs.append(approx)
        exacts.append(exact)
    misc.plotGeomVsPois(exacts, approxs)

def test_RandField(trueGener = "trivial", theta = 4, naive = True, exactBF = False):
    if trueGener == "trivial":
        gen = rf1.likelihood.simul
    else:
        gen = rf2.likelihood.simul
    y = gen(theta = theta, n = 50)
    if (np.sum(y) == 50 or np.sum(y) == 0):
        return (None,None)
    N = int(1e6)
    perc = 1e-3
    iterations = int(N * perc)

    if naive:
        thetas, zs, ms = ABC.modelChoiceSampler(y = y, iterations = int(1e2),
                                                    paramPriors = [rf1.prior, rf2.prior],
                                                    likelihoods = [rf1.likelihood, rf2.likelihood],
                                                    modelPrior = misc.twoModelsPrior,
                                                    dist = misc.euclidean_dist,
                                                    tolerance = 0,
                                                    sum_statistics = misc.RFSumStats)
        print(len([m for m in ms if m == 0]))
        print(len(ms))
        ev0 = len([m for m in ms if m == 0])/len(ms)
    else:
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
    #misc.printModelChoice("trivial", "MarkovChain", ev0)
    misc.printModelChoice("trivial", "MarkovChain", ev0, ev0_true)
    return (ev0, ev0_true)

def plot_RandField():
    probs = []
    probs_true = []
    for gen in ("trivial",):
        for i in range(100):
            theta = rf1.prior.simul()
            prob, prob_true = test_RandField(gen, theta = theta,
                naive = True, exactBF = True)
            if prob is not None:
                probs.append(prob)
                probs_true.append(prob_true)
    misc.plotRF(probs, probs_true)


def testNormal(plot = False, perc = 1e-3):
    N = int(1e6)

    # effective number of simulations:
    iterations = int(1e2)

    y = nrm.likelihood.simul(3,100)
    eps = ABC.findEpsilon(y = y, iterations = N, prior = nrm.prior,
                              likelihood = nrm.likelihood,
                              dist = misc.euclidean_dist, perc = perc,
                              sum_statistics = misc.meanVar)
    thetas_all = []

    t0 = time.clock()
    thetas, zs = ABC.sampler(y = y, iterations = iterations,
                                 prior = nrm.prior,
                                 likelihood = nrm.likelihood,
                                 dist = misc.euclidean_dist,
                                 tolerance = eps,
                                 sum_statistics = misc.meanVar)
    thetas_all.append(thetas)
    t1 = time.clock()
    if plot:
        misc.plotDensity(thetas_all, meanPrior = 0, sigmaPrior = 100,
            empMean = np.mean(y), sigma = 4, n = 100)
    print(np.mean(thetas))
    return(t1-t0)

def testNormalCompTime():
    perc = np.array([1e-1, 1e-2, 1e-3, 1e-4])
    compTime = np.zeros([len(perc), 5])
    for i in range(len(perc)):
        for j in range(5):
            compTime[i,j] = testNormal(perc = perc[i])
    misc.plotCompTimeNormal(compTime.T)
