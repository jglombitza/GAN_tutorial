import tensorflow as tf
import numpy as np
from utils import make_dir, plot_calo_images, plot_average_image, conv2d_sn, dense_sn, conv2d_transpose_sn, CookbookInit
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

dir = CookbookInit("calorimeter_snwgan")


# Build the calo_data_generator and discriminator.
def generator_fn(x, latent_dim=LATENT_DIM):
    x = layers.Dense(3 * 3 * 256, activation='relu')(x)  #
    x = tf.reshape(x, shape=[BATCH_SIZE, 3, 3, 256])
    x = conv2d_transpose_sn(x, 256, (4, 4), strides=(2, 2), padding='same', name='sn_conv_gen_transposed01', activation=tf.nn.relu)
    x = layers.BatchNormalization()(x)
    x = conv2d_transpose_sn(x, 128, (3, 3), strides=(2, 2), padding='valid', name='sn_conv_gen_transposed02', activation=tf.nn.relu)
    x = layers.BatchNormalization()(x)
    x = conv2d_transpose_sn(x, 64, (3, 3), padding='valid', name='sn_conv_gen_transposed03', activation=tf.nn.relu)
    x = layers.BatchNormalization()(x)
    x = conv2d_sn(x, 3, (3, 3), name="sn_gen_conv01", padding='same', kernel_initializer=tf.initializers.random_uniform(minval=0, maxval=2.), activation=tf.nn.relu)
    return x

# import tensorflow as tf
# l = tf.keras.layers
# model = tf.keras.models.Sequential()
# model.add(l.Dense(3 * 3 * 256, activation='relu', input_shape=(64,)))
# model.add(l.Reshape((3, 3, 256)))
# model.add(l.Conv2DTranspose(256, (4, 4), strides=(2, 2), padding='same', activation='relu'))
# model.add(l.Conv2DTranspose(128, (3, 3), strides=(2, 2), padding='valid', activation='relu'))
# model.add(l.Conv2DTranspose(64, (3, 3), strides=(1, 1), padding='same', activation='relu'))
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
    generator_optimizer=tf.train.AdamOptimizer(GEN_LR, 0.5, 0.9),
    discriminator_optimizer=tf.train.AdamOptimizer(DIS_LR, 0.5, 0.9),
    get_hooks_fn=tfgan.get_sequential_train_hooks(tfgan.GANTrainSteps(1, N_CRIT)),
    config=tf.estimator.RunConfig(save_summary_steps=100, keep_checkpoint_max=3, save_checkpoints_steps=10000),
    use_loss_summaries=True)


def calo_batched_dataset(BATCH_SIZE, LATENT_DIM, generator_fn):
    Dataset = tf.data.Dataset.from_generator(
        lambda: generator_fn(LATENT_DIM), output_types=(tf.float32, tf.float32),
        output_shapes=(tf.TensorShape((LATENT_DIM,)), tf.TensorShape((15, 15, 3))))
    return Dataset.batch(BATCH_SIZE)


def calo_data_generator(LATENT_DIM):
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
images = np.array(list(itertools.islice(calo_data_generator(LATENT_DIM), 10000)))[:, 1]
images = 10**images - 1
plot_calo_images(images[0:3], fname=dir + "/original_calorimeter_images.png")
plot_average_image(np.mean(images, axis=0), fname=dir + "/average_over_10000_images.png")


for loop in range(0, 100):
    gan_estimator.train(lambda: calo_batched_dataset(BATCH_SIZE, LATENT_DIM, calo_data_generator), steps=ITER)
    result = gan_estimator.predict(lambda: calo_batched_dataset(BATCH_SIZE, LATENT_DIM, calo_data_generator))
    images = []
    for i, image in zip(range(16), result):
        images.append(10**image - 1)
    images = np.array(images)
    plot_calo_images(images, fname=dir + "/generated_images_%.3i.png" % loop)


generated = []
for i, image in zip(range(10000), result):
    generated.append(10**image - 1)
generated = np.array(generated)
plot_average_image(np.mean(generated, axis=0), fname=dir + "/generated_average_over_10000_images.png")

from utils import plot_layer_correlations, plot_cell_number_histo
calo_ims = np.load("data/3_layer_calorimeter_padded.npz")['data']
plot_layer_correlations(np.sum(calo_ims, axis=(1, 2)), datatype='data', fname=dir + "/original_corr.png")
plot_layer_correlations(np.sum(generated, axis=(1, 2)), datatype='generated', fname=dir + "/generated_corr.png")

plot_cell_number_histo(fake=np.sum(generated > 0, axis=(1, 2, 3)), data=np.sum(calo_ims > 0, axis=(1, 2, 3)), log_dir=dir)
