import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import optimizers
import matplotlib.pyplot as plt
import plot_set
from sklearn.metrics import accuracy_score
import scikitplot as skplt
from Datahandler import DataHandler


# for custom activation function
from keras import backend as K
from keras.utils.generic_utils import get_custom_objects


get_custom_objects().update(
    {"leakyrelu": tf.keras.layers.LeakyReLU(alpha=0.01)})
get_custom_objects().update({"self_val": tf.keras.layers.LeakyReLU(alpha=1.0)})

# Fetch data
seed = tf.random.set_seed(1)

mc_data = "../data/mctest.csv"
test_data = "../data/datatest.csv"

DH = DataHandler(mc_data, test_data)
#Read inn training data

X_back, X_test_val = DH()

print("Fetching optimal parameters...")
name = "hypermodel_ae"
hypermodel = tf.keras.models.load_model(f"../tf_models/model_{name}.h5")
tf.keras.utils.plot_model(hypermodel, to_file="../figures/results/ae_model_plot.png",
                          show_shapes=True, show_layer_names=True, expand_nested=True)


#with tf.device("/CPU:0"):
#    hypermodel.fit(
#        X_train, X_train, epochs=50, batch_size=4000, validation_data=(X_back_test, X_back_test)
#    )

recon_val = hypermodel(X_test_val)
err_val = tf.keras.losses.msle(recon_val, X_test_val).numpy()


b = err_val/np.max(err_val)

binsize = 100
plt.figure(num=0, dpi=80, facecolor='w', edgecolor='k')
n_b, bins_b, patches_b = plt.hist(b, bins=binsize, histtype="stepfilled", facecolor="b",
                                  label="Background", density=True)

plt.xlabel("Output", fontsize=15)
plt.ylabel("#Events", fontsize=15)
plt.title("Autoencoder output distribution", fontsize=15, fontweight="bold")
plt.legend(fontsize=16, loc="lower right")

plt.savefig("../figures/results/AE_output.pdf", bbox_inches="tight")
plt.show()
