import numpy as np

def modelChoiceSampler(y, iterations, paramPriors, likelihoods,
                       modelPrior, dist, tolerance, sum_statistics):
    print("Running a vectorized version of ABC model choice sampler...")
    n_full = len(y)
    y_stats = sum_statistics(y)
    ms = modelPrior.simulMulti(iterations)
    thetas = []
    zs = []
    ds = []
    for i in range(2):
        print("Evaluating model ", i)
        thetas.append(paramPriors[i].simulMulti(ms[i]))
        zs.append(likelihoods[i].simulMulti(thetas[i], n_full))
        ds.append(dist(sum_statistics(zs[i], True), y_stats))
        index = ds[i] <= tolerance
        # remnants of non-vectorized code: ignore for the moment
        #thetas[i] = thetas[i][index]
        #zs[i] = zs[i][:,index]
        ms[i] = np.sum(1 * index)
    #zs = [val for sublist in zs for val in sublist]
    #theta = [val for sublist in zs for val in sublist]
    return ms

def findEpsilon(y, iterations, prior, likelihood,
               dist, perc, sum_statistics):
    print("Searching for the right tolerance level.")
    y_stats = sum_statistics(y)
    n = len(y)
    dist_all = []
    theta = prior.simulMulti(iterations)
    z = likelihood.simulMulti(theta, n)
    d = dist(sum_statistics(z, True), y_stats)
    index = np.array(np.floor(iterations * perc), dtype=int)
    d = np.sort(d)
    return np.array([d[i] for i in index])

