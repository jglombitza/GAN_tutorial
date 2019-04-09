import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import tensorflow as tf
import numpy as np


def make_dir(LOG_DIR, name=""):
    import os
    import time
    import datetime
    daytime = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S')
    folder = LOG_DIR + "/" + name + "_train_" + daytime.replace(" ", "_")
    os.makedirs(folder)
    return(folder)


def BGR2RGB(image):
    image[:, :, [0, 2]] = image[:, :, [2, 0]]
    return image


def plot_images(images, figsize=(10, 10), fname=None):
    """ Plot some images """
    n_examples = len(images)
    dim = np.ceil(np.sqrt(n_examples))
    plt.figure(figsize=figsize)
    for i, img in enumerate(images):
        plt.subplot(dim, dim, i + 1)
        if img.shape[-1] == 3:
            img = img.astype(np.uint8)
            plt.imshow(img)
        else:
            img = np.squeeze(img)
            plt.imshow(img, cmap=plt.cm.Greys)
        plt.axis('off')
        plt.imshow(img)
    plt.tight_layout()
    if fname is not None:
        plt.savefig(fname)
    plt.close()


def rectangular_array(n=15):
    """ Return x,y coordinates for rectangular array with n^2 stations. """
    n0 = (n - 1) / 2
    return (np.mgrid[0:n, 0:n].astype(float) - n0)


def plot_signal_map(footprint, axis, label, event=None):
    """Plot a map *footprint* for an detector array specified by *v_stations*. """
    xd, yd = rectangular_array()
#    xd, yd = triangular_array()
    filter = footprint != 0
    axis.scatter(xd[~filter], yd[~filter], c='grey', s=70, alpha=0.1, label="silent")
    axis.set_title("Layer %i" % (label+1), loc='right')
    if event is not None:
        axis.set_title('Event %i' % (event+1), loc='left')
    circles = axis.scatter(xd[filter], yd[filter], c=footprint[filter], s=80, alpha=1, label="loud", norm=matplotlib.colors.LogNorm(vmin=None, vmax=500))
    # cbar = plt.colorbar(circles, ax=axis)
    # cbar.set_label('signal [a.u.]')
    axis.set_aspect('equal')
    return circles


def plot_calo_images(images, fname):
    fig = plt.figure(figsize=(11, 10))
    grid = matplotlib.gridspec.GridSpec(3, 1)
    for event, (image, sub_grid) in enumerate(zip(images, grid)):
        layers_grid = matplotlib.gridspec.GridSpecFromSubplotSpec(1, 3, subplot_spec=sub_grid)
        for id, layer in enumerate(layers_grid):
            ax = plt.subplot(layer)
            scat = plot_signal_map(image[:, :, id], ax, label=id, event=event)
            plt.tight_layout()
    fig.subplots_adjust(bottom=0.05, top=0.95, left=0.05, right=0.85)
    cbar_ax = fig.add_axes([0.9, 0.05, 0.05, 0.9])
    cbar = fig.colorbar(scat, cax=cbar_ax)
    cbar.set_label('signal [a.u.]')
    fig.savefig(fname, dpi=120)


def plot_average_image(image, fname):
    fig, axis = plt.subplots(1, 3, figsize=(11, 4))
    for id, ax in enumerate(axis):
        scat = plot_signal_map(image[:, :, id], ax, label=id)
        plt.tight_layout()
    fig.subplots_adjust(bottom=0.05, top=0.95, left=0.05, right=0.85)
    cbar_ax = fig.add_axes([0.9, 0.05, 0.05, 0.9])
    cbar = fig.colorbar(scat, cax=cbar_ax)
    cbar.set_label('signal [a.u.]')
    fig.suptitle("Average calorimeter images")
    fig.savefig(fname, dpi=120)


def spectral_norm(W, use_gamma=False, factor=None, name='sn'):
    shape = W.get_shape().as_list()
    if len(shape) == 1:
        sigma = tf.reduce_max(tf.abs(W))
    else:
        if len(shape) == 4:
            _W = tf.reshape(W, (-1, shape[3]))
            shape = (shape[0] * shape[1] * shape[2], shape[3])
        elif len(shape) == 5:
            _W = tf.reshape(W, (-1, shape[4]))
            shape = (shape[0] * shape[1] * shape[2] * shape[3], shape[4])
        else:
            _W = W
        u = tf.get_variable(
            name=name + "_u",
            shape=(1, shape[0]),
            initializer=tf.random_normal_initializer,
            trainable=False
        )

        _u = u
        for _ in range(1):
            _v = tf.nn.l2_normalize(tf.matmul(_u, _W), 1)
            _u = tf.nn.l2_normalize(tf.matmul(_v, tf.transpose(_W)), 1)
        _u = tf.stop_gradient(_u)
        _v = tf.stop_gradient(_v)
        sigma = tf.reduce_mean(tf.reduce_sum(_u * tf.transpose(tf.matmul(_W, tf.transpose(_v))), 1))
        update_u_op = tf.assign(u, _u)
        with tf.control_dependencies([update_u_op]):
            sigma = tf.identity(sigma)

    if factor:
        sigma = sigma / factor

    if use_gamma:
        s = tf.svd(tf.transpose(_W), compute_uv=False)[0]
        gamma = tf.get_variable(name=name + "_gamma", initializer=s)
        return gamma * W / sigma
    else:
        return W / sigma


def _conv_sn(conv, inputs, filters, kernel_size, name,
             strides=1,
             padding='valid',
             activation=None,
             use_bias=True,
             kernel_initializer=tf.glorot_uniform_initializer(),
             bias_initializer=tf.zeros_initializer(),
             use_gamma=False,
             factor=None):
    input_shape = inputs.get_shape().as_list()
    input_dim = int(input_shape[-1])  # channels_last

    with tf.variable_scope(name):
        kernel_shape = kernel_size + (input_dim, filters)
        kernel = tf.get_variable('kernel', shape=kernel_shape, initializer=kernel_initializer)
        outputs = conv(inputs, spectral_norm(kernel, use_gamma=use_gamma, factor=factor), strides=(1, *strides, 1), padding=padding.upper())
        if use_bias is True:
            bias = tf.get_variable('bias', shape=(filters,), initializer=bias_initializer)
            outputs = tf.nn.bias_add(outputs, bias)
        if activation is not None:
            outputs = activation(outputs)

    return outputs


def dense_sn(inputs, units, name,
             activation=None,
             use_bias=True,
             kernel_initializer=tf.glorot_uniform_initializer(),
             bias_initializer=tf.zeros_initializer(),
             use_gamma=False,
             factor=None):

    input_shape = inputs.get_shape().as_list()

    with tf.variable_scope(name):
        kernel = tf.get_variable('kernel', shape=(input_shape[-1], units), initializer=kernel_initializer)
        outputs = tf.matmul(inputs, spectral_norm(kernel, use_gamma=use_gamma, factor=factor))
        if use_bias is True:
            bias = tf.get_variable('bias', shape=(units,), initializer=bias_initializer)
            outputs = tf.nn.bias_add(outputs, bias)
        if activation is not None:
            outputs = activation(outputs)

    return outputs


def conv2d_sn(inputs, filters, kernel_size, name,
              strides=(1, 1),
              padding='valid',
              activation=None,
              use_bias=True,
              kernel_initializer=tf.glorot_uniform_initializer(),
              bias_initializer=tf.zeros_initializer(),
              use_gamma=False,
              factor=None):
    return _conv_sn(tf.nn.conv2d, inputs, filters, kernel_size, name,
                    strides=strides,
                    padding=padding,
                    activation=activation,
                    use_bias=use_bias,
                    kernel_initializer=kernel_initializer,
                    bias_initializer=bias_initializer,
                    use_gamma=use_gamma,
                    factor=factor)


def conv2d_tanspose_sn(inputs, filters, kernel_size, name,
                       strides=(1, 1),
                       padding='valid',
                       activation=None,
                       use_bias=True,
                       kernel_initializer=tf.glorot_uniform_initializer(),
                       bias_initializer=tf.zeros_initializer(),
                       use_gamma=False,
                       factor=None):
    return _conv_sn(tf.nn.conv2d_transpose, inputs, filters, kernel_size, name,
                    strides=strides,
                    padding=padding,
                    activation=activation,
                    use_bias=use_bias,
                    kernel_initializer=kernel_initializer,
                    bias_initializer=bias_initializer,
                    use_gamma=use_gamma,
                    factor=factor)
