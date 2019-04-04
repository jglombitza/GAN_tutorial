import tensorflow as tf
import numpy as np
from utils import make_dir, plot_images
layers = tf.layers
tfgan = tf.contrib.gan


BATCH_SIZE = 64
LATENT_DIM = 128
GEN_LR = 0.0001
DIS_LR = 0.0001
ITER = 1000
LOG_DIR = "."
GP = 10
N_CRIT = 5

dir = make_dir(LOG_DIR, "CIFAR_WGAN_GP")


# Build the generator and discriminator.
def generator_fn(x, latent_dim=LATENT_DIM):
    x = layers.Dense(4 * 4 * 256, activation='relu', input_shape=(latent_dim,))(x)  #
    x = tf.reshape(x, shape=[BATCH_SIZE, 4, 4, 256])
    x = layers.Conv2DTranspose(256, (4, 4), strides=(2, 2), padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2DTranspose(128, (4, 4), strides=(2, 2), padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2DTranspose(64, (4, 4), strides=(2, 2), padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(3, (3, 3), padding='same', activation='tanh')(x)
    return x


def discriminator_fn(x, drop_rate=0.25):
    """ Discriminator network """
    x = layers.Conv2D(64, (3, 3), padding='same', input_shape=(32, 32, 3))(x)
    x = tf.contrib.layers.layer_norm(x)
    x = tf.nn.leaky_relu(x, 0.2)
    x = layers.Conv2D(64, (4, 4), padding='same', strides=(2, 2))(x)
    x = tf.contrib.layers.layer_norm(x)
    x = tf.nn.leaky_relu(x, 0.2)
    x = layers.Conv2D(128, (3, 3), padding='same')(x)
    x = tf.contrib.layers.layer_norm(x)
    x = tf.nn.leaky_relu(x, 0.2)
    x = layers.Conv2D(128, (4, 4), padding='same', strides=(2, 2))(x)
    x = tf.contrib.layers.layer_norm(x)
    x = tf.nn.leaky_relu(x, 0.2)
    x = layers.Conv2D(256, (3, 3), padding='same')(x)
    x = tf.contrib.layers.layer_norm(x)
    x = tf.nn.leaky_relu(x, 0.2)
    x = layers.Conv2D(256, (4, 4), padding='same', strides=(2, 2))(x)
    x = tf.contrib.layers.layer_norm(x)
    x = tf.nn.leaky_relu(x, 0.2)
    x = layers.Conv2D(512, (3, 3), padding='same')(x)
    x = tf.contrib.layers.layer_norm(x)
    x = tf.nn.leaky_relu(x, 0.2)
    x = layers.Flatten()(x)
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
    generator_optimizer=tf.train.AdamOptimizer(GEN_LR, 0.5, 0.9),
    discriminator_optimizer=tf.train.AdamOptimizer(DIS_LR, 0.5, 0.9),
    get_hooks_fn=tfgan.get_sequential_train_hooks(tfgan.GANTrainSteps(1, N_CRIT)),
    config=tf.estimator.RunConfig(save_summary_steps=100, keep_checkpoint_max=3, save_checkpoints_steps=10000),
    use_loss_summaries=True)


def batched_dataset(BATCH_SIZE, LATENT_DIM, generator_fn):
    Dataset = tf.data.Dataset.from_generator(
        lambda: generator_fn(LATENT_DIM), output_types=(tf.float32, tf.float32),
        output_shapes=(tf.TensorShape((LATENT_DIM,)), tf.TensorShape((32, 32, 3))))
    return Dataset.batch(BATCH_SIZE)


def generator(LATENT_DIM):
    while True:
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar100.load_data()
        nsamples = x_train.shape[0]
        images = 2 * (x_train / 255. - 0.5)
        images = images.astype(np.float32)
        noise = np.random.randn(nsamples, LATENT_DIM).reshape(nsamples, LATENT_DIM)
        idx = np.random.permutation(nsamples)
        noise = noise[idx]
        images = images[idx]
        for i in range(nsamples):
            yield (noise[i], images[i])


for loop in range(0, 600):
    gan_estimator.train(lambda: batched_dataset(BATCH_SIZE, LATENT_DIM, generator), steps=ITER)
    result = gan_estimator.predict(lambda: batched_dataset(BATCH_SIZE, LATENT_DIM, generator))
    images = []
    for i, image in enumerate(result):
        images.append(255 * (image / 2. + 0.5))
        if i == 15:
            images = np.array(images)
            break
    plot_images(images, fname=dir + "/images_%.3i.png" % loop)
