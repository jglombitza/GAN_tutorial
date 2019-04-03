import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import numpy as np


def plot_images(images, figsize=(10, 10), fname=None):
    """ Plot some images """
    n_examples = len(images)
    dim = np.ceil(np.sqrt(n_examples))
    plt.figure(figsize=figsize)
    for i in range(n_examples):
        plt.subplot(dim, dim, i + 1)
        img = np.squeeze(images[i])
        plt.imshow(img, cmap=plt.cm.Greys)
        plt.axis('off')
    plt.tight_layout()
    if fname is not None:
        plt.savefig(fname)
    plt.close()
