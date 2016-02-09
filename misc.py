import numpy as np
import matplotlib.pyplot as plt
from matplotlib.path import Path
import matplotlib.patches as patches
import matplotlib.mlab as mlab

#   ---   ---   ---   AUXILIARY FUNCTIONS   ---   ---   ---   #

class twoModelsPrior:
    
    @staticmethod
    def simul():
        return np.random.binomial(1,0.5)

    @staticmethod
    def simulMulti(n):
        mod_0 = np.random.binomial(n,0.5)
        return np.array([mod_0, n - mod_0])

def euclidean_dist(y,z):
    """
    Computes the Euclidean distance between two points,
    assumes y and z are vectors of the same length.
    """
    d = np.linalg.norm(y-z)
    return d

def euclidean_dist_multi(z_mat, y):
    """
    Computes the Euclidean distance between each observation
    in z_mat matrix (assumed to be stored column-wise) and y vector.
    """
    yt = y.reshape(len(y),1)
    d = np.linalg.norm(z_mat - yt, axis = 0)
    return d

def scalar_dist_multi(z_mat, y):
    """
    Computes the distance between each observation in z_mat
    vector (where each observation is a scalar) and y
    (also assumed to be a scalar).
    """
    d = np.abs(z_mat - y)
    return d

def autocov(y, multi = False):
    """
    Computes the autocovariance for first 2 terms (excluding the zeroth one)
    """
    if multi:
        n = y.shape[0]
        autoCov = np.array([np.sum(y[i:,:]*y[:n-i,:], axis = 0) for i in range(3)])
        return autoCov[1:,:]
    else:        
        n = len(y)
        autoCov = np.array([np.sum(y[i:]*y[:n-i]) for i in range(3)])
        return autoCov[1:]

def RFSumStats(y, multi = False):
    """
    Computes the summary statistics for the Random Filed model:
    sum of nodes which are on and sum of pairs of Neighbours which
    are both in the same state.
    """
    if multi:
        return np.array([np.sum(y, axis = 0), np.sum(1 * (y[:-1,:] == y[1:,:]), axis = 0)])
    else:
        return np.array([np.sum(y), np.sum(1 * (y[:-1] == y[1:]))])

def sumMulti(y, multi = False):
    if multi:
        return np.sum(y, axis = 0)
    else:
        return np.sum(y)

def printModelChoice(M1_name, M2_name, ev0, ev0true=None, logBF = None):
    print("Using an ABC model choice sampler gives the following estimates:")
    print("P(M == {}| x) = {}".format(M1_name, ev0))
    print("P(M == {}| x) = {}".format(M2_name, 1 - ev0))
    print("Log-Bayes Factor log(B_(12)) = ", np.log(ev0) - np.log(1-ev0))
    if ev0true:
        print("The exact calculations yield:")
        print("P(M == {}| x) = {}".format(M1_name, ev0true))
        print("P(M == {}| x) = {}".format(M2_name, 1 - ev0true))
    elif logBF:
        print("The exact calculations yield:")
        print("Log-Bayes Factor log(B_(12)) = ", logBF)

def meanVar(y):
    return np.array([np.mean(y), np.var(y)])

#   ---   ---   ---   PLOTTING FUCNTIONS   ---   ---   ---   #

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

def plotGeomVsPois(exact, approximate):
    fig = plt.figure()
    ax = plt.subplot('111')
    ax.scatter(exact, approximate)
    plt.xlabel("Exact log-Bayes Factor")
    plt.ylabel("ABC approx. of log-Bayes Factor")
    plt.title("Comparison of log-Bayes Factor for the data ~ Poisson")
    plt.show()

def setSubplot(ax, prob):
    ax.bar([1,2], [prob,1-prob])
    ax.set_xticks([1,2], ("MA(1), MA(2)"))

def plotMAModSel(prob_MA1):
    ax = np.empty(4)
    fig, (ax0,ax1,ax2,ax3) = plt.subplots(1, 4, sharey = True)
    setSubplot(ax0, prob_MA1[0])
    setSubplot(ax1, prob_MA1[1])
    setSubplot(ax2, prob_MA1[2])
    setSubplot(ax3, prob_MA1[3])
    ax1.set_xlabel("Model 1 vs Model 2")
    ax0.set_ylabel("P(M=i|x)")
    ax1.set_title("ABC model comparison for tolerance = (1%, 0.1%, 0.01%, 0.001%)")
    plt.show()

def plotRF(probs, probs_true):
    fig = plt.figure()
    ax = plt.subplot('111')
    ax.plot([0,1],[0,1],'r')
    ax.scatter(probs_true, probs)
    ax.set_xlabel("Exact P(M=0|x)")
    ax.set_ylabel("ABC approx P(M=0|x)")
    ax.set_title("Comparison of ABC model choice and exact values for Gibbs Fields")
    plt.show()

def plotDensity(thetas, meanPrior, sigmaPrior, empMean, sigma, n):
    fig, axarr = plt.subplots(2,2)
    titles = ['Tolerance: 10%','Tolerance: 1%','Tolerance: 0.1%','Tolerance: 0.01%']
    for i in range(2):
        for j in range(2):
            axarr[i,j].hist(thetas[2 * i + j], bins = 20, normed = True)
            axarr[i,j].set_title(titles[2 * i + j])
    sigmaPost = 1 / (n/sigma + 1/sigmaPrior)
    meanPost = sigmaPost * (meanPrior / sigmaPrior + n * empMean / sigma)
    x = np.linspace(2,4,100)
    for i in range(2):
        for j in range(2):
            axarr[i,j].plot(x, mlab.normpdf(x,meanPost, np.sqrt(sigmaPost)))
    plt.show()

def plotCompTimeNormal(compTime):
    labels = ["1e-1","1e-2","1e-3","1e-4"]
    fig = plt.figure()
    ax = plt.subplot('111')
    ax.boxplot(compTime)
    ax.set_xlabel("tolerance level")
    ax.set_ylabel("execution time (in secs)")
    ax.set_title("Exec. time vs. tolerance level, num. iterations = 100")
    ax.set_yscale('log')
    ax.set_xticklabels(labels)
    plt.show()

