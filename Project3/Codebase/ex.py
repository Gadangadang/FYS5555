import tensorflow as tf


model = tf.keras.models.load_model("ex_model.h5")

tf.keras.utils.plot_model(model, to_file="../figures/results/ae_ex_model_plot.png",
                          show_shapes=True, show_layer_names=True, expand_nested=True)