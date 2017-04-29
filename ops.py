import math
import tensorflow as tf
import numpy as np


def weight(name, shape, init='he'):
    """ Initializes weight.
    :param name: Variable name
    :param shape: Tensor shape
    :param init: Init mode. xavier / normal / uniform / he (default is 'he')
    :param range:
    :return: Variable
    """
    initializer = tf.constant_initializer()
    if init == 'xavier':
        fan_in, fan_out = _get_dims(shape)
        r = math.sqrt(6.0 / (fan_in + fan_out))
        initializer = tf.random_uniform_initializer(-r, r)

    elif init == 'he':
        fan_in, _ = _get_dims(shape)
        std = math.sqrt(2.0 / fan_in)
        initializer = tf.random_normal_initializer(stddev=std)

    elif init == 'normal':
        initializer = tf.random_normal_initializer(stddev=0.1)

    var = tf.get_variable(name, shape, initializer=initializer)
    tf.add_to_collection('l2', tf.nn.l2_loss(var))  # Add L2 Loss
    return var


def _get_dims(shape):
    fan_in = shape[0] if len(shape) == 2 else np.prod(shape[:-1])
    fan_out = shape[1] if len(shape) == 2 else shape[-1]
    return fan_in, fan_out


def bias(name, dim, initial_value=0.0):
    """ Initializes bias parameter.
    :param name: Variable name
    :param dim: Tensor size (list or int)
    :param initial_value: Initial bias term
    :return: Variable
    """
    dims = dim if isinstance(dim, list) else [dim]
    return tf.get_variable(name, dims, initializer=tf.constant_initializer(initial_value))


def batch_norm(x, is_training):
    with tf.variable_scope('BatchNorm'):
        inputs_shape = x.get_shape()
        axis = list(range(len(inputs_shape) - 1))
        param_shape = inputs_shape[-1:]

        beta = tf.get_variable('beta', param_shape, initializer=tf.constant_initializer(0.))
        gamma = tf.get_variable('gamma', param_shape, initializer=tf.constant_initializer(1.,0.02))
        batch_mean, batch_var = tf.nn.moments(x, axis)
        ema = tf.train.ExponentialMovingAverage(decay=0.5)

        def mean_var_with_update():
            ema_apply_op = ema.apply([batch_mean, batch_var])
            with tf.control_dependencies([ema_apply_op]):
                return tf.identity(batch_mean), tf.identity(batch_var)

        mean, var = tf.cond(is_training,
                            mean_var_with_update,
                            lambda: (ema.average(batch_mean), ema.average(batch_var)))
        bn = tf.nn.batch_normalization(x, mean, var, beta, gamma, 1e-3)
    return bn


def cnn(x, feature_maps, kernels, x_dim, is_training, bn = True):
    layers = []
    for idx, kernel_dim in enumerate(kernels):
        reduced_length = x.get_shape()[1] - kernel_dim + 1

        # [batch_size x seq_length x embed_dim x feature_map_dim]
        convolution = conv(x, feature_maps[idx], kernel_dim, x_dim,
                           name="kernel%d" % idx)

        # [batch_size x 1 x 1 x feature_map_dim]
        pool = tf.nn.max_pool(tf.tanh(convolution), [1, reduced_length, 1, 1], [1, 1, 1, 1], 'VALID')

        layers.append(tf.squeeze(pool))

    if len(kernels) > 1:
        output = tf.concat(1, layers)
    else:
        output = layers[0]
    if bn:
        output = batch_norm(output, is_training)
    return tf.nn.relu(output)


def conv(x, output_dim, k_h, k_w, stddev=0.02, name="convolution2d"):
    with tf.variable_scope(name):
        w = tf.get_variable('w', [k_h, k_w, x.get_shape()[-1], output_dim],
                            initializer=tf.truncated_normal_initializer(stddev=stddev))
        conv_layer = tf.nn.conv2d(x, w, strides=[1, 1, 1, 1], padding='VALID')
    return conv_layer


def dropout(x, d_rate, is_training):
    return tf.cond(is_training, lambda: tf.nn.dropout(x, d_rate), lambda: x)


def fc(x, num_n, name, is_training):
    input_size = x.get_shape()[1]
    w = weight(name, [input_size, num_n], init='he')
    x = tf.matmul(x,w)
    x = batch_norm(x, is_training)
    return tf.nn.relu(x)


def _linear(self, x, h, bias_default=0.0):
    i, d = x.get_shape().as_list()[1], self._num_units
    w = weight('W', [i, d])
    u = weight('U', [d, d])
    b = bias('b', d, bias_default)

    if True:
        with tf.variable_scope('Linear1'):
            x_w = batch_norm(tf.matmul(x, w), is_training=self.is_training)
        with tf.variable_scope('Linear2'):
            h_u = batch_norm(tf.matmul(h, u), is_training=self.is_training)
        return x_w + h_u + b
    else:
        return tf.matmul(x, w) + tf.matmul(h, u) + b





