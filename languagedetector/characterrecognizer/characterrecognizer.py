#!/usr/bin/env python3

import math
import os
import string
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow.contrib.lookup as tfl

# Global constants
DATA_ROOT = '../../data/alphabet'

BATCH_SIZE = 64
DECAY_RATE = 0.95
LEARNING_RATE = 1e-3

IMG_WIDTH = 128
IMG_HEIGHT = 128
INPUT_WIDTH = 28
INPUT_HEIGHT = 28

NUM_CLASSES = 26

# Initialization functions
def weight_variable(shape):
    """Initialize weight variable with noise."""
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    """Initialize bias variable with noise."""
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

        self.x = tf.placeholder(tf.float32, [None, INPUT_HEIGHT, INPUT_WIDTH, 1], name='x')
        self.y = tf.placeholder(tf.float32, [None, NUM_CLASSES], name='labels')
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

        tf.summary.image('input', x, 3)

        conv1 = conv_layer(x, 1, 32, 'conv1')
        conv2 = conv_layer(conv1, 32, 64, 'conv2')
        conv2_flat = tf.reshape(conv2, [-1, 7 * 7 * 64])

        fc1 = fc_layer(conv2_flat, 7 * 7 * 64, 1024, 'fc1')
        fc1_drop = tf.nn.dropout(fc1, self.keep_prob)
        fc2 = fc_layer(fc1_drop, 1024, NUM_CLASSES, 'fc2', apply_act=False)

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

def get_image_data(filename, label):
    print(filename, label)
    raw = tf.read_file(filename)
    img = tf.image.decode_png(raw, channels=1)
    # img = tf.image.resize_image_with_crop_or_pad(img, IMG_HEIGHT, IMG_WIDTH)
    img = tf.image.resize_images(img, tf.constant([INPUT_HEIGHT, INPUT_WIDTH], tf.int32))
    # one_hot = tf.one_hot(label, NUM_CLASSES)
    # return img, one_hot
    return img, label

def load_dataset(dir_path, label_encoding):
    df = pd.read_csv(os.path.join(dir_path, 'dataset.csv'), sep='\t')
    filenames = df['filename']
    labels = df['label']

    file_paths = [os.path.join(dir_path, f) for f in filenames]
    tf_filenames = tf.constant(file_paths, dtype=tf.string)

    labels = [label_encoding[x] for x in labels]
    tf_labels = tf.one_hot(labels, len(label_encoding))
    print('beep\n\n\n')
    # tf_labels = tf.constant(labels.tolist(), dtype=tf.string)
    # tf_labels = tf.constant(labels, dtype=tf.int32)

    dataset = tf.data.Dataset.from_tensor_slices((tf_filenames, tf_labels))
    return dataset

def load_datasets(root, directories, label_encoding):
    datasets = {}

    for dir_name in directories:
        dir_path = os.path.join(root, dir_name)
        print('Loading {}...'.format(dir_path))
        datasets[dir_name] = load_dataset(dir_path, label_encoding)

    return datasets

def get_dataset_size(root, dataset_name):
    df = pd.read_csv(os.path.join(root, dataset_name, 'dataset.csv'), sep='\t')
    return df.shape[0]

def init_dataset_test(dataset, batch_size):
    return (dataset.repeat()
        .map(get_image_data)
        .batch(batch_size))

def init_dataset_train(dataset, shuffle_buffer_size):
    return (dataset.repeat()
        .shuffle(shuffle_buffer_size)
        .map(get_image_data)
        .batch(BATCH_SIZE))

if __name__ == "__main__":
    alphabet = string.ascii_lowercase
    codes = list(range(1, len(alphabet) + 1))
    label_encoding = dict(zip(alphabet, codes))
    # label_encoding = tfl.HashTable(tfl.KeyValueTensorInitializer(alphabet, codes), -1)

    directories = [ 'train', 'test', 'validation' ]
    datasets = load_datasets(DATA_ROOT, directories, label_encoding)
    dataset_sizes = { d: get_dataset_size(DATA_ROOT, d) for d in directories }

    datasets['train'] = init_dataset_train(datasets['train'], dataset_sizes['train'])
    datasets['test'] = init_dataset_test(datasets['test'], dataset_sizes['test'])
    datasets['validation'] = init_dataset_test(datasets['validation'], dataset_sizes['validation'])

    it_train = datasets['train'].make_one_shot_iterator()
    it_test = datasets['test'].make_one_shot_iterator()
    it_validation = datasets['validation'].make_one_shot_iterator()

    model = Model(LEARNING_RATE, dataset_sizes['train'])
    writer = tf.summary.FileWriter(get_log_dir('/tmp/characterrecognizer/'))

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        # sess.run(tf.tables_initializer())
        writer.add_graph(sess.graph)

        validation_batch = sess.run(it_validation.get_next())
        test_batch = sess.run(it_test.get_next())

        # TODO stop training when no improvement on validation set
        for i in range(2000):
            batch = it_train.get_next()
            batch_xs, batch_ys = sess.run(batch)

            model.train(batch_xs, batch_ys)

            if i % math.ceil(dataset_sizes['train'] / BATCH_SIZE) == 0:
                accuracy = model.accuracy_eval(*validation_batch)
                summary = model.summarize(*validation_batch)
                writer.add_summary(summary, i)
                model.save('./model/model')

        print(model.accuracy_eval(*test_batch))

# TODO

# Scale network params (28 -> 128, 10 -> 26)
# Load PNG files
# tf.decode_csv instead of pandas
# one_hot = tf.one_hot(label, NUM_CLASSES)
# output_shape for images
# Shuffle
# name_scope
# restore/loading train
# Split source code into multiple files if too long

# Augmentation:
#     downsample, upscale (simulate different size data)
#     rotation
#     noise
#     contrast

# https://stackoverflow.com/questions/44132307/tf-contrib-data-dataset-repeat-with-shuffle-notice-epoch-end-mixed-epochs
# If you want to perform some computation between epochs, and avoid mixing data from different epochs, it is probably easiest to avoid repeat() and catch the OutOfRangeError at the end of each epoch.


# https://stackoverflow.com/questions/37454932/tensorflow-train-step-feed-incorrect

