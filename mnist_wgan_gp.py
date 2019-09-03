import tensorflow as tf
import numpy as np
from utils import make_dir, plot_images, CookbookInit

# from tensorflow import keras
layers = tf.layers
tfgan = tf.contrib.gan
mnist = tf.keras.datasets.mnist


BATCH_SIZE = 100
nfilter = 64
LATENT_DIM = 90
LR = 0.0001
ITER = 100
LOG_DIR = "."
N_CRIT = 5  # 10
GP = 10

dir = CookbookInit("MNIST_WGAN-GP")


# Build the generator and discriminator.
def generator_fn(x, latent_dim=LATENT_DIM):
    x = layers.Dense(7 * 7 * nfilter, activation='relu', input_shape=(latent_dim,))(x)  #
    x = tf.reshape(x, shape=[BATCH_SIZE, 7, 7, nfilter])
    x = layers.Conv2DTranspose(nfilter, (5, 5), strides=(2, 2), padding='same', activation='relu')(x)
    x = layers.Conv2D(nfilter // 2, (5, 5), padding='same', activation='relu')(x)
    x = layers.Conv2DTranspose(nfilter // 2, (5, 5), strides=(2, 2), padding='same', activation='relu')(x)
    x = layers.Conv2D(1, (5, 5), padding='same', activation='sigmoid')(x)
    return x


def discriminator_fn(x, drop_rate=0.25):
    """ Discriminator network """
    x = layers.Conv2D(32, (3, 3), padding='same', activation=tf.nn.leaky_relu, input_shape=(28, 28, 1))(x)
    x = layers.Conv2D(64, (3, 3), padding='same', activation=tf.nn.leaky_relu, strides=(2, 2))(x)
    x = layers.Conv2D(128, (3, 3), padding='same', activation=tf.nn.leaky_relu, strides=(2, 2))(x)
    x = layers.Conv2D(128, (3, 3), padding='same', activation=tf.nn.leaky_relu, strides=(2, 2))(x)
    x = layers.Flatten()(x)
    x = layers.Dense(256, activation=tf.nn.leaky_relu)(x)
    x = layers.Dense(1)(x)
    return x


def discriminator_loss(model, add_summaries=True):

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
    discriminator_loss_fn=discriminator_loss,
    generator_optimizer=tf.train.AdamOptimizer(LR, 0.5, 0.9),
    discriminator_optimizer=tf.train.AdamOptimizer(LR, 0.5, 0.9), get_hooks_fn=tfgan.get_sequential_train_hooks(tfgan.GANTrainSteps(1, N_CRIT)),
    config=tf.estimator.RunConfig(save_summary_steps=1, keep_checkpoint_max=3, save_checkpoints_steps=10000),
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
