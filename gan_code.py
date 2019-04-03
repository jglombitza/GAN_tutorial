import tensorflow as tf
import numpy as np
# from tensorflow import keras
layers = tf.layers
tfgan = tf.contrib.gan
mnist = tf.keras.datasets.mnist


BATCH_SIZE = 32
LATENT_DIM = 10
GEN_LR = 0.001
DIS_LR = 0.001
ITER = 100
LOG_DIR = "."
# Set up the input.
noise = tf.random_normal([BATCH_SIZE, LATENT_DIM])
place = tf.placeholder(tf.float32, shape=[BATCH_SIZE, 28, 28, 1])


def cast_to_float32(list):
    ret = []
    for entry in list:
        ret.append(entry.astype(np.float32))
    return ret


def get_input_fn(BATCH_SIZE, LATENT_DIM):
    def train_input_fn():
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        x_train = (np.expand_dims(x_train, axis=-1)-128) / 128.
        x_train = x_train.astype(np.float32)
        noise = np.random.randn(60000, LATENT_DIM).reshape(60000, LATENT_DIM)
        noise = noise.astype(np.float32)
        data = tf.data.Dataset.from_tensor_slices((noise, x_train))
        # image_batch = images.batch(BATCH_SIZE)
        # noise_batch = tf.random_normal([BATCH_SIZE, LATENT_DIM])
        # # noise = tf.data.Dataset.from_tensor_slices(noise)
        # # noise_batch = images.batch(BATCH_SIZE)
        # image_batch = tf.random_normal([BATCH_SIZE, 28, 28, 1])
        return data.batch(BATCH_SIZE)
        # noise_batch = tf.random_normal([BATCH_SIZE, LATENT_DIM])
        # image_batch = tf.random_normal([BATCH_SIZE, 28, 28, 1])
        # return (noise_batch, image_batch)
    return train_input_fn


# Build the generator and discriminator.
def generator_fn(x, latent_dim=LATENT_DIM):
    x = layers.Dense(7 * 7 * 128, activation='relu', input_shape=(latent_dim,))(x)  #
    x = layers.BatchNormalization()(x)
    x = tf.reshape(x, shape=[BATCH_SIZE, 7, 7, 128])
    x = layers.Conv2DTranspose(128, (5, 5), strides=(2, 2), padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(1, (5, 5), padding='same', activation='sigmoid')(x)
    return x


def discriminator_fn(x, drop_rate=0.25):
    """ Discriminator network """
    x = layers.Conv2D(32, (5, 5), padding='same', strides=(2, 2), activation='relu', input_shape=(28, 28, 1))(x)
    x = tf.nn.leaky_relu(x, 0.2)
    x = layers.Dropout(drop_rate)(x)
    x = layers.Conv2D(64, (5, 5), padding='same', strides=(2, 2), activation='relu')(x)
    x = tf.nn.leaky_relu(x, 0.2)
    x = layers.Dropout(drop_rate)(x)
    x = layers.Conv2D(128, (5, 5), padding='same', strides=(2, 2), activation='relu')(x)
    x = tf.nn.leaky_relu(x, 0.2)
    x = layers.Dropout(drop_rate)(x)
    x = layers.Flatten()(x)
    x = layers.Dense(256)(x)
    x = tf.nn.leaky_relu(x, 0.2)
    x = layers.Dropout(drop_rate)(x)
    x = layers.Dense(2, activation='softmax')(x)
    return x


gan_estimator = tfgan.estimator.GANEstimator(
    generator_fn=generator_fn,
    discriminator_fn=discriminator_fn,
    generator_loss_fn=tfgan.losses.wasserstein_generator_loss,
    discriminator_loss_fn=tfgan.losses.wasserstein_discriminator_loss,
    generator_optimizer=tf.train.AdamOptimizer(0.001, 0.5),
    discriminator_optimizer=tf.train.AdamOptimizer(0.0001, 0.5))

input_fn = get_input_fn(BATCH_SIZE, LATENT_DIM)

gan_estimator.train(input_fn, max_steps=ITER)


# gan_model = tfgan.gan_model(
#     generator_fn=generator_fn,  # you define
#     discriminator_fn=discriminator_fn,  # you define
#     real_data=images,
#     generator_inputs=noise)
#
# # Build the GAN loss.
# gan_loss = tfgan.gan_loss(
#     gan_model,
#     generator_loss_fn=tfgan.losses.wasserstein_generator_loss,
#     discriminator_loss_fn=tfgan.losses.wasserstein_discriminator_loss)
#
# # Create the train ops, which calculate gradients and apply updates to weights.
# train_ops = tfgan.gan_train_ops(
#     gan_model,
#     gan_loss,
#     generator_optimizer=tf.train.AdamOptimizer(GEN_LR, 0.5),
#     discriminator_optimizer=tf.train.AdamOptimizer(DIS_LR, 0.5))
#
# # Run the train ops in the alternating training scheme.
# tfgan.gan_train(
#     train_ops,
#     hooks=[tf.train.StopAtStepHook(num_steps=ITER)],
#     logdir=LOG_DIR)


# def generator_fn(x, latent_dim=LATENT_DIM):
#     x = layers.Dense(7 * 7 * 128, input_shape=(latent_dim,))(x)  #
#     x = layers.BatchNormalization()(x)
#     x = layers.Activation('relu')(x)
#     x = layers.Reshape([7, 7, 128])(x)
#     x = layers.UpSampling2D(size=(2, 2))(x)
#     x = layers.Conv2D(128, (5, 5), padding='same')(x)
#     x = layers.BatchNormalization()(x)
#     x = layers.Activation('relu')(x)
#     x = layers.UpSampling2D(size=(2, 2))(x)
#     x = layers.Conv2D(64, (5, 5), padding='same')(x)
#     x = layers.BatchNormalization()(x)
#     x = layers.Activation('relu')(x)
#     x = layers.Conv2D(1, (5, 5), padding='same', activation='sigmoid')(x)
#     return x
#
#
# def discriminator_fn(x, drop_rate=0.25):
#     """ Discriminator network """
#     x = layers.Conv2D(32, (5, 5), padding='same', strides=(2, 2), activation='relu', input_shape=(28, 28, 1))(x)
#     # x = layers.LeakyReLU(0.2)(x)
#     x = layers.Dropout(drop_rate)(x)
#     x = layers.Conv2D(64, (5, 5), padding='same', strides=(2, 2), activation='relu')(x)
#     x = layers.LeakyReLU(0.2)(x)
#     x = layers.Dropout(drop_rate)(x)
#     x = layers.Conv2D(128, (5, 5), padding='same', strides=(2, 2), activation='relu')(x)
#     x = layers.LeakyReLU(0.2)(x)
#     x = layers.Dropout(drop_rate)(x)
#     x = layers.Flatten()(x)
#     x = layers.Dense(256)(x)
#     x = layers.LeakyReLU(0.2)(x)
#     x = layers.Dropout(drop_rate)(x)
#     x = layers.Dense(2, activation='softmax')(x)
#     return x
