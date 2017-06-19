import tensorflow as tf

def build_network(LAMBDA, EPSILON, LR):
    x = tf.placeholder(tf.float32, shape=[None, 15, 15, 5], name="x")
    y = tf.placeholder(tf.float32, shape=[None, 225], name="y")
    f = tf.placeholder(tf.float32, shape=[None], name="f")

    conv_layer_1_1 = tf.contrib.layers.conv2d(
        inputs=x, num_outputs=64, kernel_size=3, stride=1, padding="SAME",
        activation_fn=tf.nn.relu, trainable=True,
        weights_initializer=tf.contrib.layers.xavier_initializer_conv2d())
    conv_layer_1_2 = tf.contrib.layers.conv2d(
        inputs=conv_layer_1_1, num_outputs=64, kernel_size=3, stride=1, padding="SAME",
        activation_fn=tf.nn.relu, trainable=True,
        weights_initializer=tf.contrib.layers.xavier_initializer_conv2d())
    conv_layer_1_3 = tf.contrib.layers.conv2d(
        inputs=conv_layer_1_2, num_outputs=64, kernel_size=3, stride=1, padding="SAME",
        activation_fn=tf.nn.relu, trainable=True,
        weights_initializer=tf.contrib.layers.xavier_initializer_conv2d())
    
    conv_layer_2_1 = tf.contrib.layers.conv2d(
        inputs=conv_layer_1_3, num_outputs=64, kernel_size=3, stride=1, padding="SAME",
        activation_fn=tf.nn.relu, trainable=True,
        weights_initializer=tf.contrib.layers.xavier_initializer_conv2d())
    conv_layer_2_2 = tf.contrib.layers.conv2d(
        inputs=conv_layer_2_1, num_outputs=64, kernel_size=3, stride=1, padding="SAME",
        activation_fn=tf.nn.relu, trainable=True,
        weights_initializer=tf.contrib.layers.xavier_initializer_conv2d())
    conv_layer_2_3 = tf.contrib.layers.conv2d(
        inputs=conv_layer_2_2, num_outputs=64, kernel_size=3, stride=1, padding="SAME",
        activation_fn=tf.nn.relu, trainable=True,
        weights_initializer=tf.contrib.layers.xavier_initializer_conv2d())

    conv_layer_3_1 = tf.contrib.layers.conv2d(
        inputs=conv_layer_2_3, num_outputs=64, kernel_size=3, stride=1, padding="SAME",
        activation_fn=tf.nn.relu, trainable=True,
        weights_initializer=tf.contrib.layers.xavier_initializer_conv2d())
    conv_layer_3_2 = tf.contrib.layers.conv2d(
        inputs=conv_layer_3_1, num_outputs=64, kernel_size=3, stride=1, padding="SAME",
        activation_fn=tf.nn.relu, trainable=True,
        weights_initializer=tf.contrib.layers.xavier_initializer_conv2d())
    conv_layer_3_3 = tf.contrib.layers.conv2d(
        inputs=conv_layer_3_2, num_outputs=64, kernel_size=3, stride=1, padding="SAME",
        activation_fn=tf.nn.relu, trainable=True,
        weights_initializer=tf.contrib.layers.xavier_initializer_conv2d())

    conv_layer_last_1 = tf.contrib.layers.conv2d(
        inputs=tf.concat([x, conv_layer_1_3, conv_layer_2_3, conv_layer_3_3], axis=3),
        num_outputs=64, kernel_size=3, stride=1, padding="SAME",
        activation_fn=tf.nn.relu, trainable=True,
        weights_initializer=tf.contrib.layers.xavier_initializer_conv2d())
    conv_layer_last_2 = tf.contrib.layers.conv2d(
        inputs=conv_layer_last_1, num_outputs=64, kernel_size=3, stride=1, padding="SAME",
        activation_fn=tf.nn.relu, trainable=True,
        weights_initializer=tf.contrib.layers.xavier_initializer_conv2d())
    conv_layer_last_3 = tf.contrib.layers.conv2d(
        inputs=conv_layer_last_2, num_outputs=1, kernel_size=3, stride=1, padding="SAME",
        activation_fn=None, trainable=True,
        weights_initializer=tf.contrib.layers.xavier_initializer_conv2d())

    bias = tf.Variable(initial_value=tf.truncated_normal(shape=[225]), trainable=True)
    prob = tf.nn.log_softmax(tf.add(tf.reshape(conv_layer_last_3, [-1, 225]), bias), name="prob")
    likelihood = tf.reduce_mean(tf.multiply(f, tf.reduce_sum(tf.multiply(prob, y), axis=1)))
    tf.exp(likelihood, name="likelihood")
    tf.train.AdamOptimizer(LR).minimize(-likelihood, name="train_step")
    return "alphanet-5"
