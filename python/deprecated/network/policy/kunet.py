import tensorflow as tf

def build_network(LAMBDA, EPSILON, LR):
    x = tf.placeholder(tf.float32, shape=[None, 15, 15, 2], name="x")
    y = tf.placeholder(tf.float32, shape=[None, 225], name="y")
    f = tf.placeholder(tf.float32, shape=[None], name="f")

    conv_layer_1_1 = tf.contrib.layers.conv2d(
        inputs=x, num_outputs=64, kernel_size=3, stride=1, padding="SAME",
        activation_fn=tf.nn.relu, trainable=True,
        weights_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
        biases_initializer=tf.random_normal_initializer(mean=0.0, stddev=0.01))
    conv_layer_1_2 = tf.contrib.layers.conv2d(
        inputs=conv_layer_1_1, num_outputs=64, kernel_size=3, stride=1, padding="SAME",
        activation_fn=tf.nn.relu, trainable=True,
        weights_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
        biases_initializer=tf.random_normal_initializer(mean=0.0, stddev=0.01))
    conv_layer_1_3 = tf.contrib.layers.conv2d(
        inputs=conv_layer_1_2, num_outputs=64, kernel_size=3, stride=1, padding="SAME",
        activation_fn=tf.nn.relu, trainable=True,
        weights_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
        biases_initializer=tf.random_normal_initializer(mean=0.0, stddev=0.01))
    
    conv_layer_2_1 = tf.contrib.layers.conv2d(
        inputs=conv_layer_1_3, num_outputs=64, kernel_size=3, stride=1, padding="SAME",
        activation_fn=tf.nn.relu, trainable=True,
        weights_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
        biases_initializer=tf.random_normal_initializer(mean=0.0, stddev=0.01))
    conv_layer_2_2 = tf.contrib.layers.conv2d(
        inputs=conv_layer_2_1, num_outputs=64, kernel_size=3, stride=1, padding="SAME",
        activation_fn=tf.nn.relu, trainable=True,
        weights_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
        biases_initializer=tf.random_normal_initializer(mean=0.0, stddev=0.01))
    conv_layer_2_3 = tf.contrib.layers.conv2d(
        inputs=conv_layer_2_2, num_outputs=64, kernel_size=3, stride=1, padding="SAME",
        activation_fn=tf.nn.relu, trainable=True,
        weights_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
        biases_initializer=tf.random_normal_initializer(mean=0.0, stddev=0.01))

    conv_layer_3_1 = tf.contrib.layers.conv2d(
        inputs=conv_layer_2_3, num_outputs=64, kernel_size=3, stride=1, padding="SAME",
        activation_fn=tf.nn.relu, trainable=True,
        weights_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
        biases_initializer=tf.random_normal_initializer(mean=0.0, stddev=0.01))
    conv_layer_3_2 = tf.contrib.layers.conv2d(
        inputs=conv_layer_3_1, num_outputs=64, kernel_size=3, stride=1, padding="SAME",
        activation_fn=tf.nn.relu, trainable=True,
        weights_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
        biases_initializer=tf.random_normal_initializer(mean=0.0, stddev=0.01))
    conv_layer_3_3 = tf.contrib.layers.conv2d(
        inputs=conv_layer_3_2, num_outputs=64, kernel_size=3, stride=1, padding="SAME",
        activation_fn=tf.nn.relu, trainable=True,
        weights_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
        biases_initializer=tf.random_normal_initializer(mean=0.0, stddev=0.01))

    conv_layer_last = tf.contrib.layers.conv2d(
        inputs=tf.concat([x, conv_layer_1_3, conv_layer_2_3, conv_layer_3_3], axis=3),
        num_outputs=1, kernel_size=3, stride=1, padding="SAME", activation_fn=None, trainable=True,
        weights_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
        biases_initializer=tf.random_normal_initializer(mean=0.0, stddev=0.01))

    prob = tf.nn.softmax(tf.reshape(conv_layer_last, [-1, 225]), name="prob")
    likelihood = tf.reduce_mean(tf.multiply(f, tf.reduce_sum(tf.multiply(prob, y), axis=1)), name="likelihood")
    tf.train.AdamOptimizer(LR).minimize(-likelihood, name="train_step")
    return "kunet"
