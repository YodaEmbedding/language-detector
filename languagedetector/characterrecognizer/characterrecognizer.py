#!/usr/bin/env python3

import argparse
import math
import operator
import os
import string

from collections import Counter
from functools import partial

import tensorflow as tf
import tensorflow.contrib.lookup as tfl
import numpy as np
import pandas as pd

# Parse arguments
parser = argparse.ArgumentParser(description='Character recognition module.')
parser.add_argument('--load', dest='load', action='store_true')
parser.add_argument('--no-load', dest='load', action='store_false')
parser.add_argument('--test', dest='test', action='store_true')
parser.add_argument('--no-test', dest='test', action='store_false')
parser.add_argument('--train', dest='train', action='store_true')
parser.add_argument('--no-train', dest='train', action='store_false')
parser.add_argument('--file', dest='file', action='store')
parser.set_defaults(load=False)
parser.set_defaults(test=True)
parser.set_defaults(train=False)
parser.set_defaults(file='')
args = parser.parse_args()

# Global constants
ENABLE_FILE = bool(len(args.file))
ENABLE_LOAD = args.load
ENABLE_TEST_STATS = args.test
ENABLE_TRAIN = args.train

DATA_ROOT = '../../data/alphabet'
MODEL_PATH = './model/model'

MAX_EPOCHES = 50
EPOCHES_PER_SAVE = 100

BATCH_SIZE = 64
DECAY_RATE = 0.95
LEARNING_RATE = 1e-4
DROPOUT = 0.5

# Actual image widths and resized widths
SOURCE_WIDTH = 128
SOURCE_HEIGHT = 128
WIDTH = 128
HEIGHT = 128

NUM_CLASSES = 26

# Augmention parameters
ANGLE_RANGE = 0.00 * np.pi
NOISE_STD = 0.0

# Decorators
def wrap_with_counter(fn, counter):
    def wrapped_fn(*args, **kwargs):
        # control_dependencies forces the assign op to be run even if we don't use the result
        with tf.control_dependencies([tf.assign_add(counter, 1)]):
            return fn(*args, **kwargs)
    return wrapped_fn

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

        self.x = tf.placeholder(tf.float32, [None, HEIGHT, WIDTH, 1], name='x')
        self.y = tf.placeholder(tf.float32, [None, NUM_CLASSES], name='labels')
        self.keep_prob = tf.placeholder(tf.float32, name='dropout')

        self.y_predict = self.inference(self.x)
        self.loss = self.loss(self.y, self.y_predict)
        self.train_step = self.optimize(self.loss)

        self.test = tf.argmax(tf.nn.softmax(self.y_predict), 1)
        self.accuracy = self.accuracy(self.y, self.y_predict)
        self.summary = tf.summary.merge_all()
        self.saver = tf.train.Saver()

    def train(self, batch_xs, batch_ys):
        self.train_step.run(feed_dict={self.x: batch_xs, self.y: batch_ys,
            self.keep_prob: DROPOUT})

    def inference(self, x):
        """Make prediction on input x.

        Args:
            x (tf.placeholder): Input batch.

        Returns:
            y_predict (tf.Variable): Prediction.
        """

        tf.summary.image('input', x, 1)

        layer = conv_layer(x, 1, 8, 'conv1')
        layer = conv_layer(layer, 8, 16, 'conv2')
        # layer = conv_layer(layer, 16, 32, 'conv3')
        flat_shape = int(np.prod(layer.get_shape()[1:]))
        layer = tf.reshape(layer, [-1, flat_shape])

        layer = fc_layer(layer, flat_shape, 1024, 'fc1')
        layer = tf.nn.dropout(layer, self.keep_prob)
        layer = fc_layer(layer, 1024, NUM_CLASSES, 'fc2', apply_act=False)

        return layer

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
        with tf.name_scope('accuracy'):
            correct = tf.equal(tf.argmax(logits, 1), tf.argmax(labels, 1))
            correct = tf.cast(correct, tf.float32)
            accuracy = tf.reduce_mean(correct)
            tf.summary.scalar('accuracy', accuracy)
            return accuracy

    def accuracy_eval(self, inputs, labels):
        return self.accuracy.eval(feed_dict={self.x: inputs, self.y: labels,
            self.keep_prob: 1.0})

    def run(self, inputs, labels=None):
        if labels is None:
            labels = np.repeat([np.zeros(NUM_CLASSES)], inputs.shape[0], axis=0)
        return self.test.eval(feed_dict={self.x: inputs, self.y: labels,
            self.keep_prob: 1.0})

    def summarize(self, inputs, labels):
        return self.summary.eval(feed_dict={self.x: inputs, self.y: labels,
            self.keep_prob: 1.0})

    def load(self, filename):
        sess = tf.get_default_session()
        self.saver.restore(sess, filename)

    def save(self, filename):
        sess = tf.get_default_session()
        self.saver.save(sess, filename)

# counter = tf.Variable(0, tf.int32)
# counter = tf.get_variable(dtype=tf.int32, shape=(), name='counter',
#     initializer=tf.zeros_initializer())
# wrap_with_counter_ = partial(wrap_with_counter, counter=counter)
# @wrap_with_counter_

def get_image_data(filename, label):
    raw = tf.read_file(filename)
    img = tf.image.decode_png(raw, channels=1)
    # img.set_shape([None, SOURCE_HEIGHT, SOURCE_WIDTH, 1])
    # img = tf.image.resize_image_with_crop_or_pad(img, HEIGHT, WIDTH)
    img = tf.image.resize_images(img, tf.constant([HEIGHT, WIDTH], tf.int32))
    # tf.Print(img, [img.get_shape()])
    return img, label

# TODO
def augment_image(img, angle):
    shape = tf.shape(img)
    # shape = img.get_shape()

    # Invert to ensure background is black.
    # Then, zero padded values are of same color.
    max_intensity = tf.constant(255.0, dtype=tf.float32)
    img = max_intensity - img

    img = tf.contrib.image.rotate(img, angle, interpolation='BILINEAR')

    # TODO translations, zoom

    # Re-invert or uninvert the inverted image so that it is no longer inverted
    img = max_intensity - img

    # Discolorations
    fine_noise = tf.truncated_normal(shape=shape, mean=0.0, stddev=NOISE_STD)
    # coarse_noise = tf.truncated_normal(shape=shape, mean=0.0, stddev=NOISE_STD)
    img = img + fine_noise
    img = tf.clip_by_value(img, 0, 255)

    return img

def augment(img, label):
    with tf.name_scope('augmentation'):
        # angle = tf.random_uniform([1], minval=-ANGLE_RANGE, maxval=ANGLE_RANGE)
        angle = tf.truncated_normal([1], mean=0.0, stddev=0.5 * ANGLE_RANGE)
        # angle = tf.where(angle >= 0.0, angle, 2.0 * np.pi - angle)
        # tf.Print(angle, [angle])
        img = augment_image(img, angle)
        return img, label

def get_log_dir(dir_path):
    i = 1
    while os.path.isdir(dir_path + str(i)):
        i += 1
    return dir_path + str(i)

# Dataset management
def load_dataset(dir_path, label_encoding):
    df = pd.read_csv(os.path.join(dir_path, 'dataset.csv'), sep='\t')
    filenames = df['filename']
    labels = df['label']

    file_paths = [os.path.join(dir_path, f) for f in filenames]
    tf_filenames = tf.constant(file_paths, dtype=tf.string)

    labels = [label_encoding[x] for x in labels]
    tf_labels = tf.one_hot(labels, NUM_CLASSES)

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
        # .cache()
        # .map(augment)
        .batch(batch_size))

def init_dataset_train(dataset, shuffle_buffer_size):
    return (dataset.repeat()
        .shuffle(shuffle_buffer_size)
        .map(get_image_data)
        # .cache()
        .map(augment)
        .batch(BATCH_SIZE))

def print_letter_accuracy(test_batch, label_decoding, alphabet):
    y_predict = model.run(*test_batch)
    predictions = np.array([y for y in y_predict])
    labels = np.array([y for y in np.argmax(test_batch[1], 1)])
    correct = predictions == labels

    labels_dec = [label_decoding[x] for x in labels]
    predictions_dec = [label_decoding[x] for x in predictions]
    pairs = list(zip(labels_dec, correct))

    # pairs_ = list(zip(labels_dec, predictions_dec))
    # print(pairs_)

    same = [x[0] for x in pairs if x[1]]
    different = [x[0] for x in pairs if not x[1]]

    same_counts = Counter(same)
    different_counts = Counter(different)
    accuracy = {}

    for letter in alphabet:
        total = same_counts[letter] + different_counts[letter]
        accuracy[letter] = (same_counts[letter] / total if total != 0
            else float('nan'))

    strs = ['{} {}'.format(k, v) for k, v in accuracy.items()]
    print('\n'.join(strs))

if __name__ == "__main__":
    alphabet = string.ascii_lowercase
    codes = list(range(1, len(alphabet) + 1))
    label_encoding = dict(zip(alphabet, codes))
    label_encoding['None'] = 0
    label_decoding = {v: k for k, v in label_encoding.items()}

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
        print('\n\n')
        sess.run(tf.global_variables_initializer())
        writer.add_graph(sess.graph)

        if ENABLE_LOAD:
            model.load(MODEL_PATH)

        if ENABLE_TRAIN:
            iterations_per_epoch = dataset_sizes['train'] / BATCH_SIZE
            iterations_max = math.ceil(MAX_EPOCHES * iterations_per_epoch)

            # TODO stop training when no improvement on validation set
            for i in range(iterations_max):
                batch = it_train.get_next()
                batch_xs, batch_ys = sess.run(batch)

                model.train(batch_xs, batch_ys)

                if i % iterations_per_epoch < 1.0:
                    validation_batch = sess.run(it_validation.get_next())
                    accuracy = model.accuracy_eval(*validation_batch)
                    summary = model.summarize(*validation_batch)
                    # accuracy = model.accuracy_eval(batch_xs, batch_ys)
                    # summary = model.summarize(batch_xs, batch_ys)
                    epoch = int(i / iterations_per_epoch)
                    print('epoch {}, step {}, accuracy {}'.format(epoch, i, accuracy))
                    writer.add_summary(summary, i)

                if i % (EPOCHES_PER_SAVE * iterations_per_epoch) < 1.0 and i > 0:
                    model.save(MODEL_PATH)

            model.save(MODEL_PATH)

        if not ENABLE_LOAD and not ENABLE_TRAIN:
            print('Warning!\nNo model loaded or trained!\n')

        if ENABLE_TRAIN or ENABLE_TEST_STATS:
            test_batch = sess.run(it_test.get_next())
            accuracy = model.accuracy_eval(*test_batch)
            print('test accuracy {}'.format(accuracy))
            print_letter_accuracy(test_batch, label_decoding, alphabet)

        if ENABLE_FILE:
            img, _ = get_image_data(args.file, None)
            x = tf.reshape(img, [1, HEIGHT, WIDTH, 1])
            img = sess.run(x)
            result = model.run(img)
            print('Result: {}'.format(label_decoding[result[0]]))

# TODO

# Experiment with different hyperparameters

# Augmentation:
#     downsample, upscale (simulate different size data)

# Write language string distance thing/lookup
# Write test program (i.e. input one single image)
# Join everything up together

# Nitpicking
# name_scope
# tf.decode_csv instead of pandas

