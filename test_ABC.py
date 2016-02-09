import test_functions as tf
import matplotlib.pyplot as plt
import numpy as np

if False:
    tf.test_MA2_samplers(which = "basic",
                         noisy = True,
                         generator = "MA(2)",
                         naive = True,
                         exactBF = True)

if False:
    for i in range(10):
         tf.test_Geom_Pois(trueGener = "Pois",
                      naive = False,
                      numBF = True)

if False:
  tf.test_RandField(trueGener = "trivial",
                      theta = 2,
                      naive = False,
                      exactBF = True)
if False:
    tf.plotGeomVsPois()

if True:
  tf.plot_RandField()

if False:
  tf.testNormalCompTime()
  #tf.testNormal(True)