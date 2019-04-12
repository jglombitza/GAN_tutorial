import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import tensorflow as tf
from tensorflow.python.layers.utils import deconv_output_length
import numpy as np
import argparse


def is_interactive():
    try:
        __IPYTHON__
    except NameError:
        return False
    else:
        return True


def make_dir(LOG_DIR, name=""):
    import os
    import time
    import datetime
    daytime = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S')
    folder = LOG_DIR + "/" + name + "_train_" + daytime.replace(" ", "_")
    os.makedirs(folder)
    return(folder)


def CookbookInit(path):
    if is_interactive() is False:
        parser = argparse.ArgumentParser(description='Configuration Flags')
        parser.add_argument('-log_dir', '--log_dir', default=".", type=str)
        args = parser.parse_args()
        return args.log_dir
    else:
        return make_dir(".", path)


def BGR2RGB(image):
    image[:, :, [0, 2]] = image[:, :, [2, 0]]
    return image


def grayscale(data, dtype='float32'):
    # luma coding weighted average in video systems
    r, g, b = np.asarray(.3, dtype=dtype), np.asarray(.59, dtype=dtype), np.asarray(.11, dtype=dtype)
    rst = r * data[:, :, :, 0] + g * data[:, :, :, 1] + b * data[:, :, :, 2]
    # add channel dimension
    rst = np.expand_dims(rst, axis=3)
    return rst


def plot_images(images, labels=None, figsize=(10, 10), fname=None):
    """ Plot some images """
    n_examples = len(images)
    dim = np.ceil(np.sqrt(n_examples))
    plt.figure(figsize=figsize)
    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    for i, img in enumerate(images):
        ax = plt.subplot(dim, dim, i + 1)
        if img.shape[-1] == 3:
            img = img.astype(np.uint8)
            plt.imshow(img)
            if labels is not None:
                u = int(labels[i][1])
                ax.set_title(class_names[u])
        else:
            img = np.squeeze(img)
            plt.imshow(img, cmap=plt.cm.Greys)
        plt.axis('off')
    plt.tight_layout()
    if fname is not None:
        plt.savefig(fname)
    plt.close()


def plot_cond_images(images, labels=None, epoch=None, fname=None):
    """ Plot some images """
    fig, sub = plt.subplots(nrows=3, ncols=10, figsize=(12, 4))
    if epoch is not None:
        plt.suptitle('Iteration %.3i k' % (epoch // 1000), fontsize=12)
    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    idx = 0
    for r, row in enumerate(sub):
        for j, col in enumerate(row):
            col.imshow(images[idx].astype(np.uint8))
            if r == 0:
                col.set_title(class_names[j])
            idx += 1
            col.axis('off')
    # fig.tight_layout()
    if fname is not None:
        plt.savefig(fname, dpi=120)
    plt.close()


def rectangular_array(n=15):
    """ Return x,y coordinates for rectangular array with n^2 stations. """
    n0 = (n - 1) / 2
    return (np.mgrid[0:n, 0:n].astype(float) - n0)


def triangular_array(n=15, offset=True):
    """ Return x,y coordinates for triangular array with n^2 stations. """
    n0 = (n - 1) / 2
    x, y = np.mgrid[0:n, 0:n].astype(float) - n0
    if offset:  # offset coordinates
        x += 0.5 * ((y+1.) % 2)
    else:  # axial coordinates
        x += 0.5 * y
    y *= np.sin(np.pi / 3)
    return x, y


def plot_footprint(footprint, axis, label=None):
    """Plot a map *footprint* for an detector array specified by *v_stations*. """
    xd, yd = rectangular_array(n=9)
    filter = footprint != 0
    # filter[5, 5] = True
    axis.scatter(xd[~filter], yd[~filter], c='grey', s=150, alpha=0.1, label="silent")
    circles = axis.scatter(xd[filter], yd[filter], c=footprint[filter], s=150, alpha=1, label="loud", vmin=.1, vmax=6)
    cbar = plt.colorbar(circles, ax=axis)
    cbar.set_label('signal [a.u.]')
    axis.grid(True)
    if label != None:
        axis.text(0.95, 0.1, "Energy: %.1f EeV" % label, verticalalignment='top', horizontalalignment='right', transform=axis.transAxes, backgroundcolor='w')
    axis.set_aspect('equal')
    axis.set_xlim(-5, 5)
    axis.set_ylim(-5, 5)
    axis.set_xlabel('x [km]')
    axis.set_ylabel('y [km]')


def plot_multiple_footprints(footprint, fname=None, log_dir='.', title='Iteration', epoch=None, nrows=2, ncols=2, labels=None):
    """ Plots the time and signal footprint in one figure """
    fig, sub = plt.subplots(nrows=nrows, ncols=ncols, figsize=(9, 7))
    for i in range(ncols):
        for j in range(nrows):
            idx = np.random.choice(np.arange(footprint.shape[0]))
            plot_footprint(np.squeeze(footprint[idx]), axis=sub[i, j], label=labels[idx] if labels is not None else None)
    plt.tight_layout()
    fig.subplots_adjust(left=0.02, top=0.95)
    if epoch is not None:
        plt.suptitle(title + ' %.3i k' % (epoch // 1000), fontsize=12)
    plt.savefig(fname)
    # plt.show()
    plt.close('all')


def plot_total_signal(fake, data, fname=None, log_dir="."):
    """ histogram of #total signal values """
    fig, ax = plt.subplots(1)
    ax.hist(fake, bins=np.arange(0, 45, 1), normed=True, label='fake', alpha=0.5)
    ax.hist(data, bins=np.arange(0, 45, 1), normed=True, label='data', alpha=0.5)
    ax.set_xlabel('total signal')
    ax.set_ylabel('relative frequency')
    plt.legend(loc='upper right', fancybox=False)
    fig.savefig(log_dir + '/total_signal%s.png' % fname)


def plot_cell_number_histo(fake, data, fname=None, log_dir="."):
    """ histogram of #station values """
    print('Plot cell number distribution')
    fig, ax = plt.subplots(1)
    ax.hist(fake, bins=np.arange(0, 55, 1), normed=True, label='fake', alpha=0.5)
    ax.hist(data, bins=np.arange(0, 55, 1), normed=True, label='data', alpha=0.5)
    ax.set_xlabel('number of cells with signal')
    ax.set_ylabel('relative frequency')
    plt.legend(loc='upper right', fancybox=False)
    fig.savefig(log_dir + '/cell_number%s.png' % fname)


def plot_signal_map(footprint, axis, label, event=None, hex=False):
    """Plot a map *footprint* for an detector array specified by *v_stations*. """
    if hex is True:
        xd, yd = triangular_array()
    else:
        xd, yd = rectangular_array()
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
            scat = plot_signal_map(image[:, :, id], ax, label=id, event=event, hex=True)
    plt.tight_layout()
    fig.subplots_adjust(bottom=0.05, top=0.95, left=0.05, right=0.85)
    cbar_ax = fig.add_axes([0.9, 0.05, 0.05, 0.9])
    cbar = fig.colorbar(scat, cax=cbar_ax)
    cbar.set_label('signal [a.u.]')
    fig.savefig(fname, dpi=120)


def plot_average_image(image, fname):
    fig, axis = plt.subplots(1, 3, figsize=(11, 4))
    for id, ax in enumerate(axis):
        scat = plot_signal_map(image[:, :, id], ax, label=id, hex=True)
        plt.tight_layout()
    fig.subplots_adjust(bottom=0.05, top=0.95, left=0.05, right=0.85)
    cbar_ax = fig.add_axes([0.9, 0.05, 0.05, 0.9])
    cbar = fig.colorbar(scat, cax=cbar_ax)
    cbar.set_label('signal [a.u.]')
    fig.suptitle("Average calorimeter images")
    fig.savefig(fname, dpi=120)


def plot_layer_correlations(image, datatype='', fname=None):
    fig, axis = plt.subplots(1, 3, figsize=(11, 4))
    fig.suptitle(datatype)
    axis[0].hexbin(image[:, 0], image[:, 1], linewidth=0.3, mincnt=1, gridsize=50)
    axis[0].set_xlabel("Total signal layer 1")
    axis[0].set_ylabel("Total signal layer 2")
    axis[1].hexbin(image[:, 0], image[:, 2], linewidth=0.3, mincnt=1, gridsize=50)
    axis[1].set_xlabel("Total signal layer 1")
    axis[1].set_ylabel("Total signal layer 3")
    axis[2].hexbin(image[:, 1], image[:, 2], linewidth=0.3, mincnt=1, gridsize=50)
    axis[2].set_xlabel("Total signal layer 2")
    axis[2].set_ylabel("Total signal layer 3")
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
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
             factor=None, transposed=False):
    input_shape = inputs.get_shape().as_list()
    c_axis, h_axis, w_axis = 3, 1, 2  # channels last
    input_dim = input_shape[c_axis]
    with tf.variable_scope(name):
        if transposed is True:
            kernel_shape = kernel_size + (filters, input_dim)
            kernel = tf.get_variable('kernel', shape=kernel_shape, initializer=kernel_initializer)
            height, width = input_shape[h_axis], input_shape[w_axis]
            kernel_h, kernel_w = kernel_size
            stride_h, stride_w = strides
            out_height = deconv_output_length(height, kernel_h, padding, stride_h)
            out_width = deconv_output_length(width, kernel_w, padding, stride_w)
            output_shape = (input_shape[0], out_height, out_width, filters)
            outputs = conv(inputs, spectral_norm(kernel, use_gamma=use_gamma, factor=factor), tf.stack(output_shape), strides=(1, *strides, 1), padding=padding.upper())
        else:
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


def conv2d_transpose_sn(inputs, filters, kernel_size, name,
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
                    factor=factor, transposed=True)
