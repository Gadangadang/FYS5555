import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV, train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import PredefinedSplit
from tensorflow.keras import optimizers
import keras_tuner as kt
from Functions import *

# for custom activation function
from keras import backend as K
from keras.utils.generic_utils import get_custom_objects


get_custom_objects().update(
    {"leakyrelu": tf.keras.layers.LeakyReLU(alpha=0.01)})
get_custom_objects().update({"self_val": tf.keras.layers.LeakyReLU(alpha=1.0)})


def gridautoencoder(X_b, X_back_test):
    

    start_time = timer(None)
    tuner = kt.Hyperband(
        AE_model_builder,
        objective=kt.Objective("val_mse", direction="min"),
        max_epochs=50,
        factor=3,
        directory="GridSearches",
        project_name="AE",
        overwrite=True,
    )

    tuner.search(X_b, X_b, epochs=50, batch_size=4000,
                 validation_data=(X_back_test, X_back_test))
    timer(start_time)
    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

    print(
        f"""
    For Encoder: \n 
    First layer has {best_hps.get('num_of_neurons0')} with activation {best_hps.get('0_act')} \n
    Second layer has {best_hps.get('num_of_neurons1')} with activation {best_hps.get('1_act')} \n
    
    Latent layer has {best_hps.get("lat_num")} with activation {best_hps.get('2_act')} \n
    \n
    For Decoder: \n 
    First layer has {best_hps.get('num_of_neurons5')} with activation {best_hps.get('5_act')}\n
    Second layer has {best_hps.get('num_of_neurons6')} with activation {best_hps.get('6_act')}\n
    Third layer has activation {best_hps.get('7_act')}\n
    \n
    with learning rate = {best_hps.get('learning_rate')}
    """
    )

    state = True
    while state == True:
        answ = input("Do you want to save model? (y/n) ")
        if answ == "y":
            name = input("name: ")
            tuner.hypermodel.build(best_hps).save(
                f"../tf_models/model_{name}.h5")
            state = False
            print("Model saved")
        elif answ == "n":
            state = False
            print("Model not saved")


def AE_model_builder(hp):
    inputs = tf.keras.layers.Input(shape=30, name="encoder_input")
    x = tf.keras.layers.Dense(
        units=hp.Int("num_of_neurons0", min_value=17, max_value=30, step=1),
        activation=hp.Choice(
            "0_act", ["relu", "tanh", "leakyrelu", "self_val"]),
    )(inputs)
    x1 = tf.keras.layers.Dense(
        units=hp.Int("num_of_neurons1", min_value=9, max_value=16, step=1),
        activation=hp.Choice(
            "1_act", ["relu", "tanh", "leakyrelu", "self_val"]),
    )(x)
    val = hp.Int("lat_num", min_value=1, max_value=8, step=1)
    x2 = tf.keras.layers.Dense(
        units=val, activation=hp.Choice(
            "2_act", ["relu", "tanh", "leakyrelu", "self_val"])
    )(x1)
    encoder = tf.keras.Model(inputs, x2, name="encoder")

    latent_input = tf.keras.layers.Input(shape=val, name="decoder_input")
    x = tf.keras.layers.Dense(
        units=hp.Int("num_of_neurons5", min_value=9, max_value=16, step=1),
        activation=hp.Choice(
            "5_act", ["relu", "tanh", "leakyrelu", "self_val"]),
    )(latent_input)
    x1 = tf.keras.layers.Dense(
        units=hp.Int("num_of_neurons6", min_value=17, max_value=30, step=1),
        activation=hp.Choice(
            "6_act", ["relu", "tanh", "leakyrelu", "self_val"]),
    )(x)
    output = tf.keras.layers.Dense(
        30, activation=hp.Choice("7_act", ["relu", "tanh", "leakyrelu", "sigmoid", "self_val"])
    )(x1)
    decoder = tf.keras.Model(latent_input, output, name="decoder")

    outputs = decoder(encoder(inputs))
    AE_model = tf.keras.Model(inputs, outputs, name="AE_model")

    hp_learning_rate = hp.Choice("learning_rate", values=[
                                 9e-2, 9.5e-2, 1e-3, 1.5e-3])
    optimizer = optimizers.Adam(hp_learning_rate)
    AE_model.compile(loss="mse", optimizer=optimizer, metrics=["mse"])

    return AE_model


if __name__ == "__main__":
    seed = tf.random.set_seed(1)

    #Read inn training data 
    
    with tf.device("/CPU:0"):
        # gridNN()
        gridautoencoder(X_b, X_back_test)
