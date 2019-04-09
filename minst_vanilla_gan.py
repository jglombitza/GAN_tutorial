import tensorflow as tf
import numpy as np
from utils import make_dir, plot_images

# from tensorflow import keras
layers = tf.layers
tfgan = tf.contrib.gan
mnist = tf.keras.datasets.mnist


BATCH_SIZE = 32
LATENT_DIM = 64
GEN_LR = 0.001
DIS_LR = 0.001
ITER = 100
LOG_DIR = "."
GP = 10

dir = make_dir(LOG_DIR, 'MNIST_vanilla_GAN')


# Build the generator and discriminator.
def generator_fn(x, latent_dim=LATENT_DIM):
    x = layers.Dense(7 * 7 * 128, activation='relu', input_shape=(latent_dim,))(x)  #
    x = tf.reshape(x, shape=[BATCH_SIZE, 7, 7, 128])
    x = layers.Conv2DTranspose(128, (5, 5), strides=(2, 2), padding='same', activation='relu')(x)
    x = layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', activation='relu')(x)
    x = layers.Conv2D(1, (5, 5), padding='same', activation='sigmoid')(x)
    return x


def discriminator_fn(x, drop_rate=0.25):
    """ Discriminator network """
    x = layers.Conv2D(32, (5, 5), padding='same', strides=(2, 2), activation='relu', input_shape=(28, 28, 1))(x)
    x = tf.nn.leaky_relu(x, 0.2)
    x = layers.Conv2D(64, (5, 5), padding='same', strides=(2, 2), activation='relu')(x)
    x = tf.nn.leaky_relu(x, 0.2)
    x = layers.Conv2D(128, (5, 5), padding='same', strides=(2, 2), activation='relu')(x)
    x = tf.nn.leaky_relu(x, 0.2)
    x = layers.Flatten()(x)
    x = layers.Dense(256)(x)
    x = tf.nn.leaky_relu(x, 0.2)
    x = layers.Dense(1)(x)
    return x


gan_estimator = tfgan.estimator.GANEstimator(
    dir,
    generator_fn=generator_fn,
    discriminator_fn=discriminator_fn,
    generator_loss_fn=tfgan.losses.modified_generator_loss,
    discriminator_loss_fn=tfgan.losses.modified_discriminator_loss,
    generator_optimizer=tf.train.AdamOptimizer(GEN_LR, 0.5),
    discriminator_optimizer=tf.train.AdamOptimizer(DIS_LR, 0.5),
    use_loss_summaries=True)


def batched_dataset(BATCH_SIZE, LATENT_DIM, generator_fn):
    Dataset = tf.data.Dataset.from_generator(
        lambda: generator_fn(LATENT_DIM), output_types=(tf.float32, tf.float32),
        output_shapes=(tf.TensorShape((LATENT_DIM,)), tf.TensorShape((28, 28, 1))))
    return Dataset.batch(BATCH_SIZE)


def generator(LATENT_DIM):
    while True:
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
        images = (np.expand_dims(x_train, axis=-1)) / 255.
        images = images.astype(np.float32)
        noise = np.random.randn(60000, LATENT_DIM).reshape(60000, LATENT_DIM)
        idx = np.random.permutation(60000)
        noise = noise[idx]
        images = images[idx]
        for i in range(60000):
            yield (noise[i], images[i])


for loop in range(0, 200):
    gan_estimator.train(lambda: batched_dataset(BATCH_SIZE, LATENT_DIM, generator), steps=ITER)
    result = gan_estimator.predict(lambda: batched_dataset(BATCH_SIZE, LATENT_DIM, generator))
    images = []
    for i, image in enumerate(result):
        images.append(image*255.)
        if i == 15:
            images = np.array(images)
            break
    plot_images(images, fname=dir + "/images_%.3i.png" % loop)
