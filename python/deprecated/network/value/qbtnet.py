import tensorflow as tf

def build_network(LAMBDA, LR):
    x = tf.placeholder(tf.float32, shape=[None, 225], name="x")
    y_ = tf.placeholder(tf.float32, shape=[None, 1], name="y_")
    board = tf.reshape(x, [-1, 15, 15, 1])

    conv_layer_1 = tf.contrib.layers.conv2d(
        inputs=board, num_outputs=64, kernel_size=3, stride=1, padding="VALID",
        activation_fn=tf.nn.relu, trainable=True,
        weights_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
        weights_regularizer=tf.contrib.layers.l2_regularizer(scale=LAMBDA))
    conv_layer_2 = tf.contrib.layers.conv2d(
        inputs=conv_layer_1, num_outputs=64, kernel_size=3, stride=1, padding="VALID",
        activation_fn=tf.nn.relu, trainable=True,
        weights_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
        weights_regularizer=tf.contrib.layers.l2_regularizer(scale=LAMBDA))
    conv_layer_3 = tf.contrib.layers.conv2d(
        inputs=conv_layer_2, num_outputs=64, kernel_size=3, stride=1, padding="VALID",
        activation_fn=tf.nn.relu, trainable=True,
        weights_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
        weights_regularizer=tf.contrib.layers.l2_regularizer(scale=LAMBDA))
    conv_layer_4 = tf.contrib.layers.conv2d(
        inputs=conv_layer_3, num_outputs=64, kernel_size=3, stride=1, padding="VALID",
        activation_fn=tf.nn.relu, trainable=True,
        weights_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
        weights_regularizer=tf.contrib.layers.l2_regularizer(scale=LAMBDA))
    conv_layer_5 = tf.contrib.layers.conv2d(
        inputs=conv_layer_4, num_outputs=64, kernel_size=3, stride=1, padding="VALID",
        activation_fn=tf.nn.relu, trainable=True,
        weights_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
        weights_regularizer=tf.contrib.layers.l2_regularizer(scale=LAMBDA))
    conv_layer_6 = tf.contrib.layers.conv2d(
        inputs=conv_layer_5, num_outputs=64, kernel_size=3, stride=1, padding="VALID",
        activation_fn=tf.nn.relu, trainable=True,
        weights_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
        weights_regularizer=tf.contrib.layers.l2_regularizer(scale=LAMBDA))
    conv_layer_7 = tf.contrib.layers.conv2d(
        inputs=conv_layer_6, num_outputs=1, kernel_size=3, stride=1, padding="VALID",
        activation_fn=None, trainable=True,
        weights_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
        weights_regularizer=tf.contrib.layers.l2_regularizer(scale=LAMBDA))
    y = tf.tanh(tf.reshape(conv_layer_7, [-1, 1]), name="y")

    loss = tf.identity(tf.losses.mean_squared_error(labels=y_, predictions=y), name="loss")
    # loss = tf.identity(tf.losses.mean_squared_error(labels=y_, predictions=y) +
    #     tf.reduce_sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)), name="loss")
    tf.train.AdamOptimizer(LR).minimize(loss, name="train_step")
    return "qbtnet"
