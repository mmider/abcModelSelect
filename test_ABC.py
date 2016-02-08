import test_functions as tf

if False:
    tf.test_MA2_samplers(which = "modelChoiceSampler",
                         noisy = True,
                         generator = "MA(2)",
                         naive = False,
                         exactBF = True)

if False:
    for i in range(10):
         tf.test_Geom_Pois(trueGener = "Pois",
                      naive = False,
                      numBF = True)

if True:
    tf.test_RandField(trueGener = "trivial",
                      theta = 4,
                      naive = True,
                      exactBF = True)
