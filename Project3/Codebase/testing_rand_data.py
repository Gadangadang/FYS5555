import numpy as np
import tensorflow as tf
import plot_set
import matplotlib.pyplot as plt 

def reconstructionError(pred, real):
    diff = pred - real
    err = np.power(diff, 2)
    err = np.sum(err, 1)
    err = np.log10(err)
    return err


def custom_loss(y_true, y_pred):
    mse = tf.keras.losses.MeanSquaredError()
    mse_loss = mse(y_true, y_pred)
    return mse_loss


from keras.utils.generic_utils import get_custom_objects


get_custom_objects().update(
    {"custom_loss": custom_loss})


data = np.random.randn(int(1e6), 36)


model = tf.keras.models.load_model(f"../tf_models/1_epoch_trained_big_wreg.h5", compile=True)

recon = model.predict(data) 


err = reconstructionError(recon, data)



fig, ax = plt.subplots()

n_bins= 50

ax.hist(err, 
        n_bins, 
        density=False, 
        stacked=False, 
        alpha=0.5, 
        histtype='bar', 
        color="green", 
        label="Test", )

ax.legend(prop={'size': 10})
ax.set_title('Random data test', fontsize=15)
ax.set_xlabel('Log10 Reconstruction Error', fontsize=15)
ax.set_ylabel('#Events', fontsize=15)
ax.set_yscale('log')

fig.tight_layout()
#plt.savefig("b_data_recon_big.pdf")
plt.show()