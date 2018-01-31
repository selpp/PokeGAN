import tensorflow as tf
import numpy as np
import random
import os

from utils import *

def get_images_batch(dir, width, height, channels, batch_size):
    images = []
    for path in os.listdir(dir):
        images.append(os.path.join(dir, path))

    all_images = tf.convert_to_tensor(images, dtype = tf.string)
    images_queue = tf.train.slice_input_producer([all_images])
    content = tf.read_file(images_queue[0])

    image = tf.image.decode_jpeg(content, channels = channels)
    # image = tf.image.random_flip_left_right(image)
    # image = tf.image.random_brightness(image, max_delta = 0.1)
    # image = tf.image.random_contrast(image, lower = 0.9, upper = 1.1)

    size = [height, width]
    image = tf.image.resize_images(image, size)
    image.set_shape([height, width, channels])

    image = tf.cast(image, tf.float32)
    image = image / 255.0
    image = (image - 0.5) * 2.0

    images_batch = tf.train.shuffle_batch(
        [image],
        batch_size = batch_size,
        num_threads = 4,
        capacity = 919,
        min_after_dequeue = 200
    )
    count = len(images)

    return images_batch, count

def generator_block(scope, x, out_channels, kernel, stride, padding = 'SAME', trainable = True,
                    epsilon = 1e-5, decay = 0.9, batch_norm = True, activation = leaky_relu):
    with tf.variable_scope(scope):
        conv = deconvolution(x, out_channels, kernel, stride, padding = padding, trainable = trainable)
        bn = batch_normalization(conv, trainable = trainable, epsilon = 1e-5, decay = 0.9) if batch_norm else conv
        act = activation(bn) if activation is not None else bn
    return act

def generator_block_1(input, random_dim, trainable):
    with tf.variable_scope('block_1'):
        w = tf.get_variable(
            'weights',
            shape = [random_dim, 4 * 4 * 512],
            dtype = tf.float32,
            trainable = trainable,
            initializer = tf.contrib.layers.xavier_initializer()
        )
        b = tf.get_variable(
            'biases',
            shape = [4 * 4 * 512],
            dtype = tf.float32,
            trainable = trainable,
            initializer = tf.constant_initializer(0.0)
        )
        flat_conv = tf.add(tf.matmul(input, w), b)
        conv = tf.reshape(flat_conv, shape = [-1, 4, 4, 512])
        bn = batch_normalization(conv, trainable = trainable, epsilon = 1e-5, decay = 0.9)
        act = leaky_relu(bn)
    return act

def generator(input, channels, random_dim, is_train, reuse = False):
    with tf.variable_scope('generator') as scope:
        if reuse:
            scope.reuse_variables()
        '''act1 = generator_block_1(input, random_dim, is_train)
        act2 = generator_block('block_2', act1, 256, [5, 5], [2, 2], padding = 'SAME', trainable = is_train, epsilon = 1e-5, decay = 0.9, batch_norm = True, activation = leaky_relu)
        act3 = generator_block('block_3', act2, 128, [5, 5], [2, 2], padding = 'SAME', trainable = is_train, epsilon = 1e-5, decay = 0.9, batch_norm = True, activation = leaky_relu)
        act4 = generator_block('block_4', act3, 64, [5, 5], [2, 2], padding = 'SAME', trainable = is_train, epsilon = 1e-5, decay = 0.9, batch_norm = True, activation = leaky_relu)
        act5 = generator_block('block_5', act4, 32, [5, 5], [2, 2], padding = 'SAME', trainable = is_train, epsilon = 1e-5, decay = 0.9, batch_norm = True, activation = leaky_relu)
        act6 = generator_block('block_6', act5, channels, [5, 5], [2, 2], padding = 'SAME', trainable = is_train, epsilon = 1e-5, decay = 0.9, batch_norm = False, activation = tanh)
        return act6'''
        act1 = generator_block_1(input, random_dim, is_train)
        act2 = generator_block('block_2', act1, 256, [5, 5], [2, 2], padding = 'SAME', trainable = is_train, epsilon = 1e-5, decay = 0.9, batch_norm = True, activation = relu)
        act3 = generator_block('block_3', act2, 128, [5, 5], [2, 2], padding = 'SAME', trainable = is_train, epsilon = 1e-5, decay = 0.9, batch_norm = True, activation = relu)
        act4 = generator_block('block_4', act3, 64, [5, 5], [2, 2], padding = 'SAME', trainable = is_train, epsilon = 1e-5, decay = 0.9, batch_norm = True, activation = leaky_relu)
        act5 = generator_block('block_5', act4, channels, [5, 5], [2, 2], padding = 'SAME', trainable = is_train, epsilon = 1e-5, decay = 0.9, batch_norm = True, activation = tanh)
        return act5

def discriminator_block(scope, x, out_channels, kernel, stride, padding = 'SAME', trainable = True,
                    epsilon = 1e-5, decay = 0.9, batch_norm = True, activation = leaky_relu):
    with tf.variable_scope(scope):
        conv = convolution(x, out_channels, kernel, stride, padding = padding, trainable = trainable)
        bn = batch_normalization(conv, trainable = trainable, epsilon = 1e-5, decay = 0.9) if batch_norm else conv
        act = activation(bn) if activation is not None else bn
    return act

def discriminator(input, is_train, reuse = False):
    with tf.variable_scope('discriminator') as scope:
        if reuse:
            scope.reuse_variables()
        '''act1 = discriminator_block('block_1', input, 64, [5, 5], [2, 2], padding = 'SAME', trainable = is_train, epsilon = 1e-5, decay = 0.9, batch_norm = True, activation = leaky_relu)
        act2 = discriminator_block('block_2', act1, 128, [5, 5], [2, 2], padding = 'SAME', trainable = is_train, epsilon = 1e-5, decay = 0.9, batch_norm = True, activation = leaky_relu)
        act3 = discriminator_block('block_3', act2, 256, [5, 5], [2, 2], padding = 'SAME', trainable = is_train, epsilon = 1e-5, decay = 0.9, batch_norm = True, activation = leaky_relu)
        act4 = discriminator_block('block_4', act3, 512, [5, 5], [2, 2], padding = 'SAME', trainable = is_train, epsilon = 1e-5, decay = 0.9, batch_norm = True, activation = leaky_relu)
        flat = flatten(act4)
        logits = dense(flat, 1, trainable = is_train)
        return logits'''
        act1 = discriminator_block('block_1', input, 64, [5, 5], [2, 2], padding = 'SAME', trainable = is_train, epsilon = 1e-5, decay = 0.9, batch_norm = True, activation = leaky_relu)
        act2 = discriminator_block('block_2', act1, 256, [5, 5], [2, 2], padding = 'SAME', trainable = is_train, epsilon = 1e-5, decay = 0.9, batch_norm = True, activation = leaky_relu)
        act3 = discriminator_block('block_3', act2, 512, [5, 5], [2, 2], padding = 'SAME', trainable = is_train, epsilon = 1e-5, decay = 0.9, batch_norm = True, activation = leaky_relu)
        flat = flatten(act3)
        logits = dense(flat, 1, trainable = is_train)
        return logits, sigmoid(logits)
