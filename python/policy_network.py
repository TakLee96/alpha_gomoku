""" define policy network and export its graph as meta file """

import os
import tensorflow as tf


def conv_bn(inputs, filters, kernel_size, name, training, activation=tf.nn.relu):
    conv = tf.layers.conv2d(
        inputs=inputs, filters=filters, kernel_size=kernel_size, strides=(1, 1),
        padding="SAME", activation=None, use_bias=False, name=name,
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
        sy_y_b = tf.placeholder(tf.float32, shape=[None, 226], name="y_b")

        conv = conv_bn(sy_x_b, 192, 3, "conv_initial", training)
        for i in range(9):
            conv = res_block(conv, 192, 3, "res_%d" % i, training)
        
        policy = conv_bn(conv, 2, 1, "conv_policy", training)
        logits = tf.layers.dense(tf.reshape(policy, shape=[-1, 450]), 226, name="fc_out_logits")
        sy_y_p = tf.nn.softmax(logits, name="y_p")
        accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(sy_y_p, 1), tf.argmax(sy_y_b, 1)), tf.float32), name="accuracy")

        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=sy_y_b, logits=logits), name="loss")

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            step = tf.train.AdamOptimizer(1e-4).minimize(loss, name="step")

        if not os.path.exists(model_name):
            os.makedirs(model_name)
        saver = tf.train.Saver(max_to_keep=99999999)
        saver.export_meta_graph(model_name + "/" + model_name + ".meta")
