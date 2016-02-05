import numpy as np

class prior:

    @staticmethod
    def simul():
        theta = np.random.uniform(0,6)
        return theta

    @staticmethod
    def eval(theta):
        return 1/6 * (0 < theta < 6)

class likelihood:

    @staticmethod
    def simul(theta, n):
        prob_change_state = 1/(1+np.exp(theta))
        z = np.zeros(n)
        z[0] = np.random.binomial(1,0.5)
        change_state = np.random.binomial(1,prob_change_state, n)
        for i in range(1,n):
            if change_state[i]:
                z[i] = 1-z[i-1]
            else:
                z[i] = z[i-1]
        return z
