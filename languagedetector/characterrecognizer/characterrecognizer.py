#!/usr/bin/env python3

import os
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# Global constants
BATCH_SIZE = 64
LEARNING_RATE = 1e-3
DECAY_RATE = 0.95

# Initialization functions
# Initialize with noise
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

# Layer functions
def conv_layer(x, channels_in, channels_out, name='conv'):
    with tf.name_scope(name):
        w = weight_variable([5, 5, channels_in, channels_out])
        b = bias_variable([channels_out])
        conv = tf.nn.conv2d(x, w, strides=[1, 1, 1, 1], padding='SAME')
        act = tf.nn.relu(conv + b)

        tf.summary.histogram('weights', w)
        tf.summary.histogram('biases', b)
        tf.summary.histogram('activations', act)

        return tf.nn.max_pool(act, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
            padding='SAME')

def fc_layer(x, channels_in, channels_out, name='fc', apply_act=True):
    with tf.name_scope(name):
        w = weight_variable([channels_in, channels_out])
        b = bias_variable([channels_out])
        h = tf.matmul(x, w) + b

        return tf.nn.relu(h) if apply_act else h

class Model(object):
    def __init__(self, base_learning_rate, train_size):
        self.base_learning_rate = base_learning_rate
        self.batch_step = tf.Variable(0, dtype=tf.float32)
        self.train_size = train_size

        self.x = tf.placeholder(tf.float32, [None, 784], name='x')
        self.y = tf.placeholder(tf.float32, [None, 10], name='labels')
        self.keep_prob = tf.placeholder(tf.float32, name='dropout')

        y_predict = self.inference(self.x)
        self.loss = self.loss(self.y, y_predict)
        self.train_step = self.optimize(self.loss)

        self.accuracy = self.accuracy(self.y, y_predict)
        self.summary = tf.summary.merge_all()
        self.saver = tf.train.Saver()

    def train(self, batch_xs, batch_ys):
        self.train_step.run(feed_dict={self.x: batch_xs, self.y: batch_ys,
            self.keep_prob: 0.5})

    def inference(self, x):
        """Make prediction on input x.

        Args:
            x (tf.placeholder): Input batch.

        Returns:
            y_predict (tf.Variable): Prediction.
        """

        # TODO This reshapes to 28 by 28
        x_image = tf.reshape(x, [-1, 28, 28, 1])
        tf.summary.image('input', x_image, 3)

        conv1 = conv_layer(x_image, 1, 32, 'conv1')
        conv2 = conv_layer(conv1, 32, 64, 'conv2')
        conv2_flat = tf.reshape(conv2, [-1, 7 * 7 * 64])

        fc1 = fc_layer(conv2_flat, 7 * 7 * 64, 1024, 'fc1')
        fc1_drop = tf.nn.dropout(fc1, self.keep_prob)
        fc2 = fc_layer(fc1_drop, 1024, 10, 'fc2', apply_act=False)

        y_predict = fc2
        return y_predict

    def loss(self, labels, logits):
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
            labels=labels, logits=logits), name='loss')
        tf.summary.scalar('loss', loss)
        return loss

    def optimize(self, loss):
        optimizer = tf.train.AdamOptimizer(self.learning_rate)
        return optimizer.minimize(loss, name='optimizer')

    @property
    def learning_rate(self):
        return tf.train.exponential_decay(self.base_learning_rate,
            self.batch_step * BATCH_SIZE,
            self.train_size,
            DECAY_RATE,
            staircase=True)

    def accuracy(self, labels, logits):
        correct = tf.equal(tf.argmax(logits, 1), tf.argmax(labels, 1))
        correct = tf.cast(correct, tf.float32)
        accuracy = tf.reduce_mean(correct, name='accuracy')
        tf.summary.scalar('accuracy', accuracy)
        return accuracy

    def accuracy_eval(self, inputs, labels):
        return self.accuracy.eval(feed_dict={self.x: inputs, self.y: labels,
            self.keep_prob: 1.0})

    def summarize(self, inputs, labels):
        return self.summary.eval(feed_dict={self.x: inputs, self.y: labels,
            self.keep_prob: 1.0})

    def save(self, filename):
        sess = tf.get_default_session()
        self.saver.save(sess, filename)

def get_log_dir(dir_path):
    i = 1
    while os.path.isdir(dir_path + str(i)):
        i += 1
    return dir_path + str(i)

if __name__ == "__main__":
    mnist = input_data.read_data_sets('MNIST_data/', one_hot=True)
    model = Model(LEARNING_RATE, len(mnist.train.images))
    writer = tf.summary.FileWriter(get_log_dir('/tmp/characterrecognizer/'))

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        writer.add_graph(sess.graph)

        for i in range(2000):
            batch_xs, batch_ys = mnist.train.next_batch(BATCH_SIZE)
            model.train(batch_xs, batch_ys)

            if i % 100 == 0:
                accuracy = model.accuracy_eval(batch_xs, batch_ys)
                summary = model.summarize(batch_xs, batch_ys)
                writer.add_summary(summary, i)
                model.save('./model/model')

        print(model.accuracy_eval(mnist.test.images, mnist.test.labels))

# TODO
# name_scope
# restore/loading train
# decorator...
# Keep training data separate from testing data
# One-hot encoded a, b, c, ...

