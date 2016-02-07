import test_functions as tf

if True:
    tf.test_MA2_samplers(which = "modelChoiceSampler",
                         noisy = True,
                         generator = "MA(2)",
                         naive = False,
                         exactBF = True)

if False:
    tf.test_Geom_Pois(trueGener = "Pois",
                      naive = True,
                      numBF = False)

if False:
    tf.test_RandField(trueGener = "trivial",
                      theta = 4,
                      naive = True,
                      exactBF = False)
