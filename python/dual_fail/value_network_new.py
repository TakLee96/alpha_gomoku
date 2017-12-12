""" define value network and export its graph as meta file """

import os
import tensorflow as tf

REGULARIZATION_SCALE = 1e-4

def dense(input_tensor, hidden_units, name, training, activation=tf.nn.relu):
    fc = tf.layers.dense(input_tensor, hidden_units, activation=activation, name=name,
            kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=1.0))
    return tf.layers.batch_normalization(fc, training=training)

def export_meta(model_name):
    with tf.Graph().as_default():
        """ neural network for logit computing """
        training = tf.placeholder(tf.bool, name="training")
        sy_x_b = tf.placeholder(tf.float32, shape=[None, 50], name="x_b")
        sy_v_b = tf.placeholder(tf.float32, shape=[None], name="v_b")

        fc = dense(sy_x_b, 256, "fc_init", training)
        for i in range(1):
            fc = dense(fc, 256, "fc_%d" % i, training)
        sy_v_p = tf.identity(tf.layers.dense(fc, 1, activation=tf.tanh, name="fc_last",
            kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=1.0)), name="v_p")

        value_loss = tf.reduce_mean(tf.square(sy_v_p - 0.9 * sy_v_b), name="value_loss")
        regularization_loss = tf.multiply(tf.reduce_sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)), REGULARIZATION_SCALE, name="regularization_loss")
        loss = tf.add(value_loss, regularization_loss, name="loss")

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            step = tf.train.AdamOptimizer(5e-5).minimize(loss, name="step")

        if not os.path.exists(model_name):
            os.makedirs(model_name)
        saver = tf.train.Saver(max_to_keep=99999999)
        saver.export_meta_graph(model_name + "/" + model_name + ".meta")
