import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import optimizers
import matplotlib.pyplot as plt
import plot_set
from sklearn.metrics import accuracy_score
import scikitplot as skplt


# for custom activation function
from keras import backend as K
from keras.utils.generic_utils import get_custom_objects


get_custom_objects().update(
    {"leakyrelu": tf.keras.layers.LeakyReLU(alpha=0.01)})
get_custom_objects().update({"self_val": tf.keras.layers.LeakyReLU(alpha=1.0)})

# Fetch data
tf.random.set_seed(1)

print("Fetching optimal parameters...")
name = "hypermodel_ae"
hypermodel = tf.keras.models.load_model(f"tf_models/model_{name}.h5")
tf.keras.utils.plot_model(hypermodel, to_file="../figures/results/ae_model_plot.png",
                          show_shapes=True, show_layer_names=True, expand_nested=True)


with tf.device("/CPU:0"):
    hypermodel.fit(
        X_train, X_train, epochs=50, batch_size=4000, validation_data=(X_back_test, X_back_test)
    )
