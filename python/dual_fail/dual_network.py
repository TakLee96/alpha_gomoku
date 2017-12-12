""" define dual network and export its graph as meta file """

import os
import tensorflow as tf

VALUE_LOSS_SCALE = 4
REGULARIZATION_SCALE = 1e-3

def conv_bn(inputs, filters, kernel_size, name, training, activation=tf.nn.relu):
    conv = tf.layers.conv2d(
        inputs=inputs, filters=filters, kernel_size=kernel_size, strides=(1, 1),
        padding="SAME", activation=None, use_bias=True, name=name,
        bias_initializer=tf.truncated_normal_initializer(),
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
        sy_y_b = tf.placeholder(tf.float32, shape=[None, 226], name="y_b")
        sy_v_b = tf.placeholder(tf.float32, shape=[None], name="v_b")

        conv = conv_bn(sy_x_b, 128, 3, "conv_initial", training)
        for i in range(9):
            conv = res_block(conv, 128, 3, "res_%d" % i, training)
        
        policy = conv_bn(conv, 2, 1, "conv_policy", training)
        logits = tf.layers.dense(tf.reshape(policy, shape=[-1, 450]), 226, name="fc_out_logits",
            kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=1.0))
        sy_y_p = tf.nn.softmax(logits, name="y_p")
        accuracy = tf.reduce_mean(
            tf.cast(tf.equal(tf.argmax(sy_y_p, 1), tf.argmax(sy_y_b, 1)), tf.float32), name="accuracy")

        value = conv_bn(conv, 1, 1, "conv_value", training)
        value = tf.layers.dense(tf.reshape(value, shape=[-1, 225]), 256, name="fc_value",
            kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=1.0))
        sy_v_p = tf.tanh(tf.layers.dense(value, 1, name="fc_out_value",
            kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=1.0)), name="v_p")

        policy_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=sy_y_b, logits=logits), name="policy_loss")
        value_loss = tf.multiply(tf.reduce_mean(tf.square(sy_v_p - sy_v_b)), VALUE_LOSS_SCALE, name="value_loss")
        regularization_loss = tf.multiply(tf.reduce_sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)), REGULARIZATION_SCALE, name="regularization_loss")
        loss = tf.add(tf.add(policy_loss, value_loss), regularization_loss, name="loss")

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            step = tf.train.AdamOptimizer(1e-3).minimize(loss, name="step")

        if not os.path.exists(model_name):
            os.makedirs(model_name)
        saver = tf.train.Saver(max_to_keep=99999999)
        saver.export_meta_graph(model_name + "/" + model_name + ".meta")
