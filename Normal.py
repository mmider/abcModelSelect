import numpy as np
from scipy.stats import norm

class prior:
	@staticmethod
	def simul():
		theta = np.random.normal(0,10)
		return theta

	@staticmethod
	def eval(theta):
		return norm.pdf(theta, loc = 0, scale = 10)

class likelihood:

	@staticmethod
	def simul(theta, n):
		z = np.random.normal(theta,2,n)
		return z