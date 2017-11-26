#!/usr/bin/env python3

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# Initialization functions
# Initialize with noise
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

# Convolution and pooling functions
def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
        padding='SAME')

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
    def __init__(self):
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

        x_image = tf.reshape(x, [-1, 28, 28, 1])

        # Define layer 1
        # Convolution: 5x5 kernel, 32 features
        W_conv1 = weight_variable([5, 5, 1, 32])
        b_conv1 = bias_variable([32])

        h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
        h_pool1 = max_pool_2x2(h_conv1)

        # Define layer 2
        # Convolution: 5x5 kernel, 64 features
        W_conv2 = weight_variable([5, 5, 32, 64])
        b_conv2 = bias_variable([64])

        h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
        h_pool2 = max_pool_2x2(h_conv2)

        # Define layer 3
        # Fully connected: 1024 neurons
        W_fc1 = weight_variable([7 * 7 * 64, 1024])
        b_fc1 = bias_variable([1024])

        h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
        h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

        # Dropout
        h_fc1_drop = tf.nn.dropout(h_fc1, self.keep_prob)

        # Define Layer 4
        # Fully connected: 10 neurons
        W_fc2 = weight_variable([1024, 10])
        b_fc2 = bias_variable([10])
        y_predict = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

        return y_predict

    def accuracy(self, labels, logits):
        correct = tf.equal(tf.argmax(logits, 1), tf.argmax(labels, 1))
        correct = tf.cast(correct, tf.float32)
        accuracy = tf.reduce_mean(correct, name='accuracy')
        tf.summary.scalar('accuracy', accuracy)
        return accuracy

    def accuracy_eval(self, inputs, labels):
        return self.accuracy.eval(feed_dict={self.x: inputs, self.y: labels,
            self.keep_prob: 1.0})

    def loss(self, labels, logits):
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
            labels=labels, logits=logits), name='loss')
        tf.summary.scalar('loss', loss)
        return loss

    def optimize(self, loss):
        optimizer = tf.train.AdamOptimizer(1e-4)
        return optimizer.minimize(loss, name='optimizer')

    def save(self, filename):
        sess = tf.get_default_session()
        self.saver.save(sess, filename)

if __name__ == "__main__":
    mnist = input_data.read_data_sets('MNIST_data/', one_hot=True)
    model = Model()
    writer = tf.summary.FileWriter('/tmp/characterrecognizer/1')

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        writer.add_graph(sess.graph)

        for i in range(2000):
            batch_xs, batch_ys = mnist.train.next_batch(50)
            model.train(batch_xs, batch_ys)

            if i % 100 == 0:
                accuracy = model.accuracy_eval(batch_xs, batch_ys)
                summary = sess.run(model.summary, feed_dict={model.x: batch_xs, model.y: batch_ys, model.keep_prob: 1.0})  # TODO Not encapsulated
                writer.add_summary(summary, i)
                model.save('./model/model')

        print(model.accuracy_eval(mnist.test.images, mnist.test.labels))

if __name__ == "__main__":
    pass

    # # Define placeholders
    # x = tf.placeholder(tf.float32, [None, 784])
    # y_ = tf.placeholder(tf.float32, [None, 10])
    # # keep_prob

    # # Build graph
    # model = Model()
    # y_predict, keep_prob = model.inference(x)

    # # Define loss and optimizer
    # def loss(labels, logits):
    #     return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
    #         labels=labels, logits=logits))

    # cross_entropy = loss(y_, y_predict)
    # train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

    # def train(batch_xs, batch_ys):
    #     train_step.run(feed_dict={x: batch_xs, y_: batch_ys, keep_prob: 0.5})

    # # Run session
    # with tf.Session() as sess:
    #     sess.run(tf.global_variables_initializer())

    #     # Train
    #     for i in range(2000):
    #         batch_xs, batch_ys = mnist.train.next_batch(50)
    #         train(batch_xs, batch_ys)

# TODO
# decay learning rate
# summary
# name_scope
# restore/loading train
# decorator...
# Keep training data separate from testing data

