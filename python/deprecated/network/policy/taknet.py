import tensorflow as tf

def build_network(LAMBDA, EPSILON, LR):
    x = tf.placeholder(tf.float32, shape=[None, 225], name="x")
    y_ = tf.placeholder(tf.float32, shape=[None, 225], name="y_")
    is_training = tf.placeholder(tf.bool, name="is_training")
    board = tf.reshape(x, [-1, 15, 15, 1])

    conv_layer_1 = tf.contrib.layers.conv2d(
        inputs=board, num_outputs=64, kernel_size=7, stride=1, padding="VALID",
        activation_fn=None, biases_initializer=None, trainable=True,
        weights_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
        weights_regularizer=tf.contrib.layers.l2_regularizer(scale=LAMBDA))
    norm_layer_1 = tf.contrib.layers.batch_norm(
        inputs=conv_layer_1, decay=0.9, center=True, scale=True, epsilon=EPSILON,
        is_training=is_training, activation_fn=tf.nn.relu, trainable=True)

    conv_layer_2 = tf.contrib.layers.conv2d(
        inputs=norm_layer_1, num_outputs=16, kernel_size=1, stride=1, padding="VALID",
        activation_fn=None, biases_initializer=None, trainable=True,
        weights_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
        weights_regularizer=tf.contrib.layers.l2_regularizer(scale=LAMBDA))
    norm_layer_2 = tf.contrib.layers.batch_norm(
        inputs=conv_layer_2, decay=0.9, center=True, scale=True, epsilon=EPSILON,
        is_training=is_training, activation_fn=tf.nn.relu, trainable=True)

    flatten = tf.reshape(norm_layer_2, [-1, 9 * 9 * 16])
    fc_layer_3 = tf.contrib.layers.fully_connected(
        inputs=flatten, num_outputs=256, activation_fn=None, trainable=True,
        weights_initializer=tf.contrib.layers.xavier_initializer(),
        weights_regularizer=tf.contrib.layers.l2_regularizer(scale=LAMBDA))
    norm_layer_3 = tf.contrib.layers.batch_norm(
        inputs=fc_layer_3, decay=0.9, center=True, scale=True, epsilon=EPSILON,
        is_training=is_training, activation_fn=tf.nn.relu, trainable=True)

    y = tf.contrib.layers.fully_connected(
        inputs=norm_layer_3, num_outputs=225, activation_fn=None, trainable=True,
        weights_initializer=tf.contrib.layers.xavier_initializer(),
        weights_regularizer=tf.contrib.layers.l2_regularizer(scale=LAMBDA))

    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y), name="loss")
    tf.train.AdamOptimizer(LR).minimize(loss, name="train_step")
    tf.reduce_mean(tf.cast(tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1)), tf.float32), name="accuracy")
    return "taknet"