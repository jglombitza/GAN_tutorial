# GAN_tutorial
Train different types of Generative Adversarial Networks using tf.contrib.gan
Provided types:
- Vanilla GAN (with label switching)
- Wasserstein GAN with gradient penalty (WGAN-GP)
- Spectral normalization GAN (SNGAN)
- WGAN-GP with spectral normalization

### Datasets
You can traim GANs on the classical keras datasets:
- CIFAR10 / CIFAR100
- MNIST

Furthermore, you can use

- CMS prototype Calorimeter dataset, 100 GEV electrons

This physics dataset is available for a CMS prototype calorimeter, simulated using Geant4. (100 GeV electrons)
For further details see https://link.springer.com/article/10.1007%2Fs41781-018-0019-7.
In contrast to the publication, this datsets contains only data of 3 layers and a single energy bin (for simplicity reasons).

For starting the training just run one of the GAN scripts in the main folder.
For performance reason the usage of a GPU is highly recommended, otherwise the training lasts >10h, before producing reasonable results.
