import tensorflow as tf
import numpy as np
from utils import plot_images

# from tensorflow import keras
layers = tf.layers
tfgan = tf.contrib.gan
mnist = tf.keras.datasets.mnist


BATCH_SIZE = 32
LATENT_DIM = 64
GEN_LR = 0.001
DIS_LR = 0.001
ITER = 10000
LOG_DIR = "."


def make_dir(LOG_DIR):
    import os
    import time
    import datetime
    daytime = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S')
    folder = LOG_DIR + "/train_" + daytime.replace(" ", "_")
    os.makedirs(folder)
    return(folder)


dir = make_dir(LOG_DIR)
# Set up the input.


def cast_to_float32(list):
    ret = []
    for entry in list:
        ret.append(entry.astype(np.float32))
    return ret


def get_input_fn(BATCH_SIZE, LATENT_DIM):
    def train_input_fn():
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        x_train = (np.expand_dims(x_train, axis=-1)) / 255
        x_train = x_train.astype(np.float32)
        noise = np.random.randn(60000, LATENT_DIM).reshape(60000, LATENT_DIM)
        noise = noise.astype(np.float32)
        print(np.mean(noise))
        data = tf.data.Dataset.from_tensor_slices((noise, x_train)).repeat(None).shuffle(5)
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
    gp_loss = 10 * tf.contrib.gan.losses.wasserstein_gradient_penalty(model, epsilon=1e-10, one_sided=True, add_summaries=add_summaries)
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
    generator_optimizer=tf.train.AdamOptimizer(0.001, 0.5),
    discriminator_optimizer=tf.train.AdamOptimizer(0.0001, 0.5),
    use_loss_summaries=True)

input_fn = get_input_fn(BATCH_SIZE, LATENT_DIM)

def input():
    def get_generator(BATCH_SIZE, LATENT_DIM):
        def generator():
            while True:
                (x_train, y_train), (x_test, y_test) = mnist.load_data()
                images = (np.expand_dims(x_train, axis=-1)) / 255.
                images = images.astype(np.float32)
                noise = np.random.randn(60000, LATENT_DIM).reshape(60000, LATENT_DIM)
                idx = np.random.permutation(60000)
                noise = noise[idx]
                images = images[idx]
                for i in range(60000):
                    yield (noise[i], images[i])
        return generator


    generator = get_generator(BATCH_SIZE, LATENT_DIM)
    Dataset_2 = tf.data.Dataset.from_generator(
        generator,
        (tf.float32, tf.float32),
        output_shapes=(tf.TensorShape((LATENT_DIM,)), tf.TensorShape((28, 28, 1))),
        args=None)
    return Dataset_2.batch(BATCH_SIZE)

#   gan_estimator.train(input_fn, max_steps=ITER)
gan_estimator.train(input, max_steps=ITER)
result = gan_estimator.predict(input)
images = []
for i, image in enumerate(result):
    images.append(image*255.)
    if i == 15:
        images = np.array(images)
        break
plot_images(images, fname=dir + "/images.png")
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
