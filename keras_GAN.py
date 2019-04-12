import numpy as np
import tensorflow as tf
from tensorflow import keras
from gan import make_trainable
from utils import make_dir, plot_images, CookbookInit

layers = keras.layers
models = keras.models
optimizers = keras.optimizers

log_dir = CookbookInit("color_cifar_keras_vanilla_GAN")

# prepare CIFAR10 dataset
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
# x_train = utils.grayscale(x_train)
x_train = 2 * (x_train / 255. - 0.5)
# plot some real images
plot_images(255 * (x_train[:16] / 2. + 0.5), fname=log_dir + '/real_images.png')

# --------------------------------------------------
# Set up generator, discriminator and GAN (stacked generator + discriminator)
# Feel free to modify eg. :
# - the provided models (see gan.py)
# - the learning rate
# - the batchsize
# --------------------------------------------------

# Set up generator
print('\nGenerator')
latent_dim = 64
generator_input = layers.Input(shape=[latent_dim])
x = layers.Dense(2 * 2 * 512, activation='relu')(generator_input)
x = layers.Reshape([2, 2, 512])(x)
x = layers.Conv2DTranspose(512, (5, 5), strides=(2, 2), padding='same', activation='relu')(x)
x = layers.BatchNormalization()(x)
x = layers.Conv2DTranspose(256, (5, 5), strides=(2, 2), padding='same', activation='relu')(x)
x = layers.BatchNormalization()(x)
x = layers.Conv2DTranspose(128, (5, 5), strides=(2, 2), padding='same', activation='relu')(x)
x = layers.BatchNormalization()(x)
x = layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', activation='relu')(x)
x = layers.BatchNormalization()(x)
x = layers.Conv2DTranspose(64, (5, 5), padding='same', activation='relu')(x)
x = layers.BatchNormalization()(x)
x = layers.Conv2D(3, (3, 3), padding='same', activation='tanh')(x)
generator = models.Model(inputs=generator_input, outputs=x)
print(generator.summary())


# Set up discriminator
print('\nDiscriminator')
discriminator_input = layers.Input(shape=[32, 32, 3])
x = layers.Conv2D(64, (3, 3), strides=(2, 2), padding='same')(discriminator_input)
x = layers.LeakyReLU(0.2)(x)
x = layers.Conv2D(128, (4, 4), padding='same', strides=(2, 2))(x)
x = layers.LeakyReLU(0.2)(x)
x = layers.Conv2D(256, (4, 4), padding='same', strides=(2, 2))(x)
x = layers.LeakyReLU(0.2)(x)
x = layers.Conv2D(512, (4, 4), padding='same', strides=(2, 2))(x)
x = layers.LeakyReLU(0.2)(x)
x = layers.Flatten()(x)
x = layers.Dropout(0.4)(x)
x = layers.Dense(2, activation='softmax', kernel_regularizer=keras.regularizers.l1_l2(0.001))(x)
discriminator = models.Model(inputs=discriminator_input, outputs=x)


print(discriminator.summary())
d_opt = optimizers.Adam(lr=2e-4, beta_1=0.5, decay=0.0005)
discriminator.compile(loss='binary_crossentropy', optimizer=d_opt, metrics=['accuracy'])

# Set up GAN by stacking the discriminator on top of the generator
print('\nGenerative Adversarial Network')
gan_input = layers.Input(shape=[latent_dim])
gan_output = discriminator(generator(gan_input))
GAN = models.Model(gan_input, gan_output)
print(GAN.summary())
g_opt = optimizers.Adam(lr=2e-4, beta_1=0.5, decay=0.0005)
make_trainable(discriminator, False)  # freezes the discriminator when training the GAN
GAN.compile(loss='binary_crossentropy', optimizer=g_opt)

# Compile saves the trainable status of the model --> After the model is compiled, updating using make_trainable will have no effect

# loss vector
losses = {"d": [], "g": []}
discriminator_acc = []

# main training loop

batch_size = 64
for epoch in range(150):

    # Plot some fake images
    noise = np.random.randn(batch_size, latent_dim)
    generated_images = 255. * (generator.predict(noise) / 2. + 0.5)
    plot_images(generated_images[:16], fname=log_dir + '/generated_images_%.3i' % epoch)

    perm = np.random.choice(50000, size=50000, replace='False')

    for i in range(50000//batch_size):

        # Create a mini-batch of data (X: real images + fake images, y: corresponding class vectors)
        image_batch = x_train[perm[i*batch_size:(i+1)*batch_size], :, :, :]    # real images
        noise_gen = np.random.randn(batch_size, latent_dim)
        generated_images = generator.predict(noise_gen)                     # generated images
        X = np.concatenate((image_batch, generated_images))
        y = np.zeros([2*batch_size, 2])   # class vector
        y[0:batch_size, 1] = 1
        y[batch_size:, 0] = 1

        # Train the discriminator on the mini-batch
        d_loss, d_acc = discriminator.train_on_batch(X, y)
        losses["d"].append(d_loss)
        discriminator_acc.append(d_acc)

        # Create a mini-batch of data (X: noise, y: class vectors pretending that these produce real images)
        noise_tr = np.random.randn(batch_size, latent_dim)
        # change here for label switching
        y2 = np.zeros([batch_size, 2])
        y2[:, 1] = 1  # classical loss

        # Train the generator part of the GAN on the mini-batch
        g_loss = GAN.train_on_batch(noise_tr, y2)
        losses["g"].append(g_loss)
        print(discriminator_acc[-1])

#
# train_for_n(epochs=10, batch_size=128)
#
# # - Plot the loss of discriminator and generator as function of iterations
generator.save(log_dir + "/generator.h5")
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
plt.figure(figsize=(10, 8))
plt.semilogy(losses["d"], label='discriminitive loss')
plt.semilogy(losses["g"], label='generative loss')
plt.legend()
plt.savefig(log_dir + '/loss.png')
#
# # - Plot the accuracy of the discriminator as function of iterations
plt.figure(figsize=(10, 8))
plt.semilogy(discriminator_acc, label='discriminator accuracy')
plt.legend()
plt.savefig(log_dir + '/discriminator_acc.png')
