""" define policy network and export its graph as meta file """

import os
import tensorflow as tf

def tf_conv2d(inputs, filters, kernel_size, name, activation=tf.nn.relu):
    return tf.layers.conv2d(
        inputs=inputs, filters=filters, kernel_size=kernel_size, strides=(1, 1),
        padding="SAME", activation=activation, use_bias=True, name=name,
        bias_initializer=tf.truncated_normal_initializer(),
        kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d())

def export_meta(model_folder, model_name):
    with tf.Graph().as_default():
        sy_x_b = tf.placeholder(tf.float32, shape=[None, 15, 15, 11], name="x_b")
        sy_y_b = tf.placeholder(tf.float32, shape=[None, 225], name="y_b")

        conv_1_1 = tf_conv2d(sy_x_b  , 256, 5, "conv_1_1")
        conv_1_2 = tf_conv2d(conv_1_1, 256, 3, "conv_1_2")
        conv_1_3 = tf_conv2d(conv_1_2, 256, 3, "conv_1_3")

        conv_2_1 = tf_conv2d(conv_1_3, 256, 3, "conv_2_1")
        conv_2_2 = tf_conv2d(conv_2_1, 256, 3, "conv_2_2")
        conv_2_3 = tf_conv2d(conv_2_2, 256, 3, "conv_2_3")

        conv_3_1 = tf_conv2d(conv_2_3, 256, 3, "conv_3_1")
        conv_3_2 = tf_conv2d(conv_3_1, 256, 3, "conv_3_2")
        conv_3_3 = tf_conv2d(conv_3_2, 256, 3, "conv_3_3")

        conv_4_1 = tf_conv2d(conv_3_3, 256, 3, "conv_4_1")
        conv_4_2 = tf_conv2d(conv_4_1, 256, 3, "conv_4_2")
        conv_4_3 = tf_conv2d(conv_4_2, 256, 3, "conv_4_3")

        concated = tf.concat([sy_x_b, conv_1_3, conv_2_3, conv_3_3, conv_4_3], axis=3)
        conv_n_1 = tf_conv2d(concated, 256, 3, "conv_n_1")
        conv_n_2 = tf_conv2d(conv_n_1, 256, 3, "conv_n_2")
        conv_n_3 = tf_conv2d(conv_n_2,   1, 3, "conv_n_3", None)
        
        bias = tf.get_variable("bias", shape=[225], dtype=tf.float32,
            initializer=tf.truncated_normal_initializer(), trainable=True)
        logits = tf.add(bias, tf.reshape(conv_n, shape=[-1, 225]), name="logits")
        sy_y_p = tf.nn.softmax(logits, name="y_p")
        sy_accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(sy_y_p, 1), tf.argmax(sy_y_b, 1)), tf.float32), name="accuracy")
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=sy_y_b, logits=logits), name="loss")
        step = tf.train.AdamOptimizer(1e-3).minimize(loss, name="step")

        if not os.path.exists(model_folder):
            os.makedirs(model_folder)
        saver = tf.train.Saver(max_to_keep=99999999)
        saver.export_meta_graph(model_folder + "/" + model_name + ".meta")
