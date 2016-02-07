import numpy as np
import matplotlib.pyplot as plt
from matplotlib.path import Path
import matplotlib.patches as patches

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

