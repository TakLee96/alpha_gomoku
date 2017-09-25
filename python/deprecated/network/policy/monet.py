import tensorflow as tf

def build_network(LAMBDA, EPSILON, LR):
    x = tf.placeholder(tf.float32, shape=[None, 15, 15, 2], name="x")
    y_ = tf.placeholder(tf.float32, shape=[None, 225], name="y_")

    conv_layer_1_1 = tf.contrib.layers.conv2d(
        inputs=x, num_outputs=64, kernel_size=3, stride=1, padding="SAME",
        activation_fn=tf.nn.relu, trainable=True,
        weights_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
        weights_regularizer=tf.contrib.layers.l2_regularizer(scale=LAMBDA))
    conv_layer_1_2 = tf.contrib.layers.conv2d(
        inputs=conv_layer_1_1, num_outputs=64, kernel_size=3, stride=1, padding="SAME",
        activation_fn=tf.nn.relu, trainable=True,
        weights_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
        weights_regularizer=tf.contrib.layers.l2_regularizer(scale=LAMBDA))
    conv_layer_1_3 = tf.contrib.layers.conv2d(
        inputs=conv_layer_1_2, num_outputs=64, kernel_size=3, stride=1, padding="SAME",
        activation_fn=tf.nn.relu, trainable=True,
        weights_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
        weights_regularizer=tf.contrib.layers.l2_regularizer(scale=LAMBDA))
    
    conv_layer_2_1 = tf.contrib.layers.conv2d(
        inputs=conv_layer_1_3, num_outputs=64, kernel_size=3, stride=1, padding="SAME",
        activation_fn=tf.nn.relu, trainable=True,
        weights_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
        weights_regularizer=tf.contrib.layers.l2_regularizer(scale=LAMBDA))
    conv_layer_2_2 = tf.contrib.layers.conv2d(
        inputs=conv_layer_2_1, num_outputs=64, kernel_size=3, stride=1, padding="SAME",
        activation_fn=tf.nn.relu, trainable=True,
        weights_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
        weights_regularizer=tf.contrib.layers.l2_regularizer(scale=LAMBDA))
    conv_layer_2_3 = tf.contrib.layers.conv2d(
        inputs=conv_layer_2_2, num_outputs=64, kernel_size=3, stride=1, padding="SAME",
        activation_fn=tf.nn.relu, trainable=True,
        weights_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
        weights_regularizer=tf.contrib.layers.l2_regularizer(scale=LAMBDA))

    conv_layer_3_1 = tf.contrib.layers.conv2d(
        inputs=conv_layer_2_3, num_outputs=64, kernel_size=3, stride=1, padding="SAME",
        activation_fn=tf.nn.relu, trainable=True,
        weights_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
        weights_regularizer=tf.contrib.layers.l2_regularizer(scale=LAMBDA))
    conv_layer_3_2 = tf.contrib.layers.conv2d(
        inputs=conv_layer_3_1, num_outputs=64, kernel_size=3, stride=1, padding="SAME",
        activation_fn=tf.nn.relu, trainable=True,
        weights_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
        weights_regularizer=tf.contrib.layers.l2_regularizer(scale=LAMBDA))
    conv_layer_3_3 = tf.contrib.layers.conv2d(
        inputs=conv_layer_3_2, num_outputs=64, kernel_size=3, stride=1, padding="SAME",
        activation_fn=tf.nn.relu, trainable=True,
        weights_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
        weights_regularizer=tf.contrib.layers.l2_regularizer(scale=LAMBDA))

    conv_layer_last = tf.contrib.layers.conv2d(
        inputs=tf.concat([conv_layer_1_3, conv_layer_2_3, conv_layer_3_3], axis=3),
        num_outputs=1, kernel_size=3, stride=1, padding="SAME",
        activation_fn=None, trainable=True,
        weights_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
        weights_regularizer=tf.contrib.layers.l2_regularizer(scale=LAMBDA))

    y = tf.reshape(conv_layer_last, [-1, 225], name="y")

    loss = tf.identity(tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y)) + \
        tf.reduce_sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)), name="loss")
    tf.train.AdamOptimizer(LR).minimize(loss, name="train_step")
    tf.reduce_mean(tf.cast(tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1)), tf.float32), name="accuracy")
    return "monet"
