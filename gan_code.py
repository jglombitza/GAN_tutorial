import tensorflow as tf
import numpy as np
from utils import make_dir, plot_images

# from tensorflow import keras
layers = tf.layers
tfgan = tf.contrib.gan


BATCH_SIZE = 32
LATENT_DIM = 64
GEN_LR = 0.001
DIS_LR = 0.0001
ITER = 1000
LOG_DIR = "."
GP = 10
N_CRIT = 5

dir = make_dir(LOG_DIR, "WGAN_GP")
# Set up the input.


def cast_to_float32(list):
    ret = []
    for entry in list:
        ret.append(entry.astype(np.float32))
    return ret


# def get_input_fn(BATCH_SIZE, LATENT_DIM):
#     def train_input_fn():
#         (x_train, y_train), (x_test, y_test) = mnist.load_data()
#         x_train = (np.expand_dims(x_train, axis=-1)) / 255
#         x_train = x_train.astype(np.float32)
#         noise = np.random.randn(60000, LATENT_DIM).reshape(60000, LATENT_DIM)
#         noise = noise.astype(np.float32)
#         print(np.mean(noise))
#         data = tf.data.Dataset.from_tensor_slices((noise, x_train)).repeat(None).shuffle(5)
#         return data.batch(BATCH_SIZE)
#     return train_input_fn


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


def discrimintator_loss(model, add_summaries=True):

    loss = tf.contrib.gan.losses.wasserstein_discriminator_loss(model, add_summaries=add_summaries)
    gp_loss = GP * tf.contrib.gan.losses.wasserstein_gradient_penalty(model, epsilon=1e-10, one_sided=True, add_summaries=add_summaries)
    loss += gp_loss

    if add_summaries:
        tf.summary.scalar('discriminator_loss', loss)

    return loss


gan_estimator = tfgan.estimator.GANEstimator(
    dir,
    generator_fn=generator_fn,
    discriminator_fn=discriminator_fn,
    generator_loss_fn=tfgan.losses.wasserstein_generator_loss,
    discriminator_loss_fn=discrimintator_loss,
    generator_optimizer=tf.train.AdamOptimizer(GEN_LR, 0.5),
    discriminator_optimizer=tf.train.AdamOptimizer(DIS_LR, 0.5),
    get_hooks_fn=tfgan.get_sequential_train_hooks(tfgan.GANTrainSteps(1, N_CRIT)),
    config=tf.estimator.RunConfig(save_summary_steps=10, keep_checkpoint_max=1, save_checkpoints_steps=200),
    use_loss_summaries=True)


# def input():
#     def get_generator(BATCH_SIZE, LATENT_DIM):
#         def generator():
#             while True:
#                 (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
#                 images = (np.expand_dims(x_train, axis=-1)) / 255.
#                 images = images.astype(np.float32)
#                 noise = np.random.randn(60000, LATENT_DIM).reshape(60000, LATENT_DIM)
#                 idx = np.random.permutation(60000)
#                 noise = noise[idx]
#                 images = images[idx]
#                 for i in range(60000):
#                     yield (noise[i], images[i])
#         return generator
#
#     generator = get_generator(BATCH_SIZE, LATENT_DIM)
#     Dataset_2 = tf.data.Dataset.from_generator(
#         generator, output_types=(tf.float32, tf.float32),
#         output_shapes=(tf.TensorShape((LATENT_DIM,)), tf.TensorShape((28, 28, 1))))
#     return Dataset_2.batch(BATCH_SIZE)
# gan_estimator.train(input, max_steps=ITER)
# result = gan_estimator.predict(input)


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

import itertools
test_image = np.array(list(itertools.islice(generator(LATENT_DIM), 1)))

for loop in range(0, 15):
    gan_estimator.train(lambda: batched_dataset(BATCH_SIZE, LATENT_DIM, generator), steps=ITER)
    result = gan_estimator.predict(lambda: batched_dataset(BATCH_SIZE, LATENT_DIM, generator))
    images = []
    for i, image in enumerate(result):
        images.append(image*255.)
        if i == 15:
            images = np.array(images)
            break
    plot_images(images, fname=dir + "/images_%.3i.png" % loop)
