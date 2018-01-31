import tensorflow as tf

def relu(x):
    return tf.nn.relu(x)

def leaky_relu(x, alpha = 0.2):
    return tf.maximum(alpha * x, x)

def parametric_relu(x, trainable = True):
    alpha = tf.get_variable(
        name = 'alpha',
        shape = x.het_shape()[-1],
        dtype = tf.float32,
        initializer = tf.constant_initializer(0.0),
        trainable = trainable
    )

def sigmoid(x):
    return tf.nn.sigmoid(x)

def tanh(x):
    return tf.nn.tanh(x)

def convolution(x, out_channels, kernel, stride, padding = 'SAME', trainable = True):
    in_channels = x.get_shape()[-1]
    w = tf.get_variable(
        name = 'weights',
        shape = [kernel[0], kernel[1], in_channels, out_channels],
        initializer = tf.contrib.layers.xavier_initializer(),
        trainable = trainable
    )
    b = tf.get_variable(
        name = 'biases',
        shape = [out_channels],
        initializer = tf.constant_initializer(0.0),
        trainable = trainable
    )
    x = tf.nn.conv2d(x, w, [1, stride[0], stride[1], 1], padding = padding)
    x = tf.nn.bias_add(x, b)
    return x

def deconvolution(x, out_channels, kernel, stride, padding, trainable):
    conv = tf.layers.conv2d_transpose(
        x,
        out_channels,
        kernel_size = kernel,
        strides = stride,
        padding = padding,
        kernel_initializer = tf.contrib.layers.xavier_initializer(),
        trainable = trainable
    )
    return conv

def pooling(x, kernel, stride, padding = 'SAME', mode = 'MAX'):
    if mode == 'MAX':
        x = tf.nn.max_pool(x, kernel, strides = stride, padding = padding)
    elif mode == 'AVG':
        x = tf.nn.avg_pool(x, kernel, strides = stride, padding = padding)
    elif mode == 'MIN':
        x = tf.nn.min_pool(x, kernel, strides = stride, padding = padding)
    return x

def flatten(x):
    shape = x.get_shape()
    size = shape[1].value * shape[2].value * shape[3].value
    flat_x = tf.reshape(x, [-1, size])
    return flat_x

def dense(x, out_nodes, trainable = True):
    shape = x.get_shape()
    size = shape[-1].value
    w = tf.get_variable(
        name = 'weights',
        shape = [size, out_nodes],
        initializer = tf.contrib.layers.xavier_initializer()
    )
    b = tf.get_variable(
        name = 'biases',
        shape = [out_nodes],
        initializer = tf.constant_initializer(0.0)
    )
    x = tf.nn.bias_add(tf.matmul(x, w), b)
    return x

def dropout(x, rate = 0.5, trainable = False):
    return tf.layers.dropout(x, rate = rate, training = trainable)

def batch_normalization(x, epsilon = 1e-5, decay = 0.9, trainable = True):
    return tf.contrib.layers.batch_norm(x, is_training = trainable, epsilon = epsilon, decay = decay,  updates_collections = None)

def pixel_shuffle(x, r, color = True):
    def _phase_shift(I, r):
        bsize, a, b, c = I.get_shape().as_list()
        X = tf.reshape(I, (bsize, a, b, r, r))
        X = tf.transpose(X, (0, 1, 2, 4, 3))
        X = tf.split(1, a, X)
        X = tf.concat(2, [tf.squeeze(x) for x in X])
        X = tf.split(1, b, X)
        X = tf.concat(2, [tf.squeeze(x) for x in X])
        bsize, a*r, b*r
        return tf.reshape(X, (bsize, a*r, b*r, 1))

    if color:
      Xc = tf.split(3, 3, X)
      X = tf.concat(3, [_phase_shift(x, r) for x in Xc])
    else:
      X = _phase_shift(X, r)
    return X

def cross_entropy_logit(logits, labels):
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits = logits, labels = labels)
    loss = tf.reduce_mean(cross_entropy)
    return loss

def accuracy_logit(logits, labels):
    correct = tf.equal(tf.arg_max(logits, 1), tf.arg_max(labels, 1))
    correct = tf.cast(correct, tf.float32)
    accuracy = tf.reduce_mean(correct) * 100.0
    return accuracy
