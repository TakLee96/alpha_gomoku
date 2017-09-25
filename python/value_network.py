""" define value network and export its graph as meta file """

import os
import tensorflow as tf

def tf_conv2d(inputs, filters, kernel_size, name, activation=tf.nn.relu):
    return tf.layers.conv2d(
        inputs=inputs, filters=filters, kernel_size=kernel_size, strides=(1, 1),
        padding="SAME", activation=activation, use_bias=True, name=name,
        bias_initializer=tf.truncated_normal_initializer(),
        kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d())

def tf_dense(inputs, units, name, activation=tf.nn.relu):
    return tf.layers.dense(
        inputs=inputs, units=units, activation=activation, use_bias=True, name=name,
        kernel_initializer=tf.contrib.layers.xavier_initializer(),
        bias_initializer=tf.truncated_normal_initializer(), trainable=True)

def export_meta(model_folder, model_name):
    with tf.Graph().as_default():
        sy_x_b = tf.placeholder(tf.float32, shape=[None, 15, 15, 11], name="x_b")
        sy_v_b = tf.placeholder(tf.float32, shape=[None], name="v_b")

        conv_1_1 = tf_conv2d(sy_x_b  , 128, 5, "conv_1_1")
        conv_1_2 = tf_conv2d(conv_1_1, 128, 3, "conv_1_2")
        conv_1_3 = tf_conv2d(conv_1_2, 128, 3, "conv_1_3")

        conv_2_1 = tf_conv2d(conv_1_3, 128, 3, "conv_2_1")
        conv_2_2 = tf_conv2d(conv_2_1, 128, 3, "conv_2_2")
        conv_2_3 = tf_conv2d(conv_2_2, 128, 3, "conv_2_3")

        conv_3_1 = tf_conv2d(conv_2_3, 128, 3, "conv_3_1")
        conv_3_2 = tf_conv2d(conv_3_1, 128, 3, "conv_3_2")
        conv_3_3 = tf_conv2d(conv_3_2, 128, 3, "conv_3_3")

        conv_4_1 = tf_conv2d(conv_3_3, 128, 3, "conv_4_1")
        conv_4_2 = tf_conv2d(conv_4_1, 128, 3, "conv_4_2")
        conv_4_3 = tf_conv2d(conv_4_2, 128, 3, "conv_4_3")

        concated = tf.concat([sy_x_b, conv_1_3, conv_2_3, conv_3_3, conv_4_3], axis=3)
        conv_n_1 = tf_conv2d(concated, 128, 3, "conv_n_1")
        conv_n_2 = tf_conv2d(conv_n_1,   1, 3, "conv_n_2")
        
        fc_1 = tf_dense(tf.reshape(conv_n_2, shape=[-1, 225]), 256, "fc_1")
        sy_v_p = tf.reshape(tf_dense(fc_1, 1, "fc_2", tf.tanh), shape=[-1], name="v_p")
        
        loss = tf.reduce_mean(tf.square(sy_v_b - sy_v_p))
        step = tf.train.AdamOptimizer(1e-3).minimize(loss)

        if not os.path.exists(model_folder):
            os.makedirs(model_folder)
        saver = tf.train.Saver(max_to_keep=99999999)
        saver.export_meta_graph(model_folder + "/" + model_name + ".meta")
