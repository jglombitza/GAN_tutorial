import tensorflow as tf
import numpy as np
from utils import make_dir, plot_calo_images, conv2d_sn, dense_sn
layers = tf.layers
tfgan = tf.contrib.gan


BATCH_SIZE = 64
LATENT_DIM = 64
GEN_LR = 0.0001
DIS_LR = 0.0001
ITER = 1000
LOG_DIR = "."
GP = 10
N_CRIT = 5

dir = make_dir(LOG_DIR, "calorimeter_snwgan")


# Build the generator and discriminator.
def generator_fn(x, latent_dim=LATENT_DIM):
    x = layers.Dense(3 * 3 * 256, activation='relu', input_shape=(latent_dim,))(x)  #
    x = tf.reshape(x, shape=[BATCH_SIZE, 3, 3, 256])
    x = layers.Conv2DTranspose(256, (4, 4), strides=(2, 2), padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2DTranspose(128, (3, 3), strides=(2, 2), padding='valid', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2DTranspose(64, (3, 3), padding='valid', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(3, (3, 3), padding='same', kernel_initializer=tf.initializers.random_uniform(minval=0, maxval=2.))(x)
    return x

# import tensorflow as tf
# l = tf.keras.layers
# model = tf.keras.models.Sequential()
# model.add(l.Dense(3 * 3 * 256, activation='relu', input_shape=(64,)))
# model.add(l.Reshape((3, 3, 256)))
# model.add(l.Conv2DTranspose(256, (5, 5), strides=(2, 2), padding='same', activation='relu'))
# model.add(l.Conv2DTranspose(256, (3, 3), strides=(2, 2), padding='same', activation='relu'))
# model.add(l.Conv2DTranspose(256, (3, 3), strides=(1, 1), padding='same', activation='relu'))
# model.summary()


def discriminator_fn(x, drop_rate=0.25):
    """ Discriminator network """
    x = conv2d_sn(x, 64, (3, 3), name="sn_conv01", padding='same')
    x = tf.contrib.layers.layer_norm(x)
    x = tf.nn.leaky_relu(x, 0.2)
    x = conv2d_sn(x, 64, (5, 5), name="sn_conv02", padding='same', strides=(2, 2))
    x = tf.contrib.layers.layer_norm(x)
    x = tf.nn.leaky_relu(x, 0.2)
    x = conv2d_sn(x, 128, (3, 3), name="sn_conv03", padding='same')
    x = tf.contrib.layers.layer_norm(x)
    x = tf.nn.leaky_relu(x, 0.2)
    x = conv2d_sn(x, 128, (4, 4), name="sn_conv04", padding='same', strides=(2, 2))
    x = tf.contrib.layers.layer_norm(x)
    x = tf.nn.leaky_relu(x, 0.2)
    x = conv2d_sn(x, 256, (3, 3), name="sn_conv05", padding='same')
    x = tf.contrib.layers.layer_norm(x)
    x = tf.nn.leaky_relu(x, 0.2)
    x = conv2d_sn(x, 512, (3, 3), name="sn_conv06", padding='same')
    x = tf.contrib.layers.layer_norm(x)
    x = tf.nn.leaky_relu(x, 0.2)
    x = layers.Flatten()(x)
    x = dense_sn(x, 1, name='sn_dense01')
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
        output_shapes=(tf.TensorShape((LATENT_DIM,)), tf.TensorShape((15, 15, 3))))
    return Dataset.batch(BATCH_SIZE)


def generator(LATENT_DIM):
    while True:
        calo_ims = np.load("data/3_layer_calorimeter_padded.npz")['data']
        nsamples = calo_ims.shape[0]
        calo_ims = np.log10(calo_ims+1)
        calo_ims = calo_ims.astype(np.float32)
        noise = np.random.randn(nsamples, LATENT_DIM).reshape(nsamples, LATENT_DIM)
        idx = np.random.permutation(nsamples)
        noise = noise[idx]
        calo_ims = calo_ims[idx]
        for i in range(nsamples):
            yield (noise[i], calo_ims[i])


import itertools
images = np.array(list(itertools.islice(generator(LATENT_DIM), 4)))[:, 1]
images = 10**images - 1
plot_calo_images(images, fname=dir + "/original_calorimeter_images.png")


for loop in range(0, 600):
    gan_estimator.train(lambda: batched_dataset(BATCH_SIZE, LATENT_DIM, generator), steps=ITER)
    result = gan_estimator.predict(lambda: batched_dataset(BATCH_SIZE, LATENT_DIM, generator))
    images = []
    for i, image in enumerate(result):
        images.append(10**image - 1)
        if i == 15:
            images = np.array(images)
            break
        plot_calo_images(images, fname=dir + "/generated_images_%.3i.png" % loop)
