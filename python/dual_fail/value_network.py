""" define value network and export its graph as meta file """

import os
import tensorflow as tf

REGULARIZATION_SCALE = 1e-3

def conv_bn(inputs, filters, kernel_size, name, training, activation=tf.nn.relu):
    conv = tf.layers.conv2d(
        inputs=inputs, filters=filters, kernel_size=kernel_size, strides=(1, 1),
        padding="SAME", activation=None, use_bias=False, name=name,
        kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=1e-3),
        kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d())
    conv = tf.layers.batch_normalization(conv, axis=3, training=training)
    if activation is not None:
        conv = activation(conv)
    return conv

def res_block(inputs, filters, kernel_size, name, training):
    with tf.variable_scope(name):
        conv = conv_bn(inputs, filters, kernel_size, "conv_1", training)
        conv = conv_bn(inputs, filters, kernel_size, "conv_2", training, None)
    return tf.nn.relu(conv + inputs)

def export_meta(model_name):
    with tf.Graph().as_default():
        """ neural network for logit computing """
        training = tf.placeholder(tf.bool, name="training")
        sy_x_b = tf.placeholder(tf.float32, shape=[None, 15, 15, 11], name="x_b")
        sy_v_b = tf.placeholder(tf.float32, shape=[None], name="v_b")

        conv = conv_bn(sy_x_b, 192, 3, "conv_initial", training)
        for i in range(9):
            conv = res_block(conv, 192, 3, "res_%d" % i, training)
        
        value = conv_bn(conv, 1, 1, "conv_value", training)
        value = tf.layers.dense(tf.reshape(value, shape=[-1, 225]), 256, name="fc_value",
            kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=1.0))
        sy_v_p = tf.tanh(tf.reshape(tf.layers.dense(value, 1, name="fc_out_value",
            kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=1.0)), [-1]), name="v_p")

        value_loss = tf.reduce_mean(tf.square(sy_v_p - sy_v_b), name="value_loss")
        regularization_loss = tf.multiply(tf.reduce_sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)), REGULARIZATION_SCALE, name="regularization_loss")
        loss = tf.add(value_loss, regularization_loss, name="loss")
        # loss = tf.identity(value_loss, name="loss")

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            step = tf.train.AdamOptimizer(1e-5).minimize(loss, name="step")

        if not os.path.exists(model_name):
            os.makedirs(model_name)
        saver = tf.train.Saver(max_to_keep=99999999)
        saver.export_meta_graph(model_name + "/" + model_name + ".meta")
