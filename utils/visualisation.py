import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

def plot_np_img(img_np, cmap=None):
    '''
    Plot image given as a numpy array
    '''
    plt.imshow(img_np, cmap=cmap)
    plt.show()


def visualisation_experiment(vae, imgs):
    '''
    Plot images in 'imgs' using the 'vae' reconstructions and original images
    '''
    kwargs = {"decode":True}

    for img in imgs:
        # Plot original image
        plot_np_img(img)
        # Retrieve reconstructed image from vae
        reconstruction = vae(np.expand_dims(img, axis=0), **kwargs)
        reconstruction = reconstruction.numpy()[0]
        reconstruction = stats.logistic.cdf(reconstruction)
        # Plot reconstructed image
        plot_np_img(reconstruction)

