#!/usr/bin/env python3

import math
import os
import string
import numpy as np
import pandas as pd
import tensorflow as tf

# Global constants
BATCH_SIZE = 64
DECAY_RATE = 0.95
IMG_WIDTH = 128
IMG_HEIGHT = 128
LEARNING_RATE = 1e-3
DATA_ROOT = '../../data/alphabet'

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

def get_image_data(filename, label):
    print(filename, label)
    raw = tf.read_file(filename)
    img = tf.image.decode_png(raw, channels=1)
    img = tf.image.resize_image_with_crop_or_pad(img, IMG_HEIGHT, IMG_WIDTH)
    return img, label

def load_dataset(dir_path, label_encoding):
    df = pd.read_csv(os.path.join(dir_path, 'dataset.csv'), sep='\t')
    filenames = df['filename']
    labels = df['label']

    file_paths = [os.path.join(dir_path, f) for f in filenames]
    tf_filenames = tf.constant(file_paths, dtype=tf.string)

    # labels = [label_encoding[x] for x in labels]
    # labels = tf.one_hot(labels, len(label_encoding))
    tf_labels = tf.constant(labels.tolist(), dtype=tf.string)

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

def init_dataset_train(dataset, shuffle_buffer_size):
    return (dataset.repeat()
        .shuffle(shuffle_buffer_size)
        .map(get_image_data)  # TODO
        .batch(BATCH_SIZE))

if __name__ == "__main__":
    alphabet = string.ascii_lowercase
    label_encoding = dict(zip(alphabet, range(1, len(alphabet) + 1)))

    directories = [ 'train', 'test', 'validation' ]
    datasets = load_datasets(DATA_ROOT, directories, label_encoding)
    train_size = get_dataset_size(DATA_ROOT, 'train')
    datasets['train'] = init_dataset_train(datasets['train'], train_size)
    it_train = datasets['train'].make_one_shot_iterator()

    model = Model(LEARNING_RATE, train_size)
    writer = tf.summary.FileWriter(get_log_dir('/tmp/characterrecognizer/'))


    # image_list, label_list = read_csv_stuff()
    # input_queue = tf.train.slice_input_producer([image_list, label_list],
    #     num_epochs=10,  # TODO
    #     shuffle=True)

    # image, label = read_images_from_disk(input_queue)

    # # `image_batch` and `label_batch` represent the "next" batch
    # # read from the input queue.
    # image_batch, label_batch = tf.train.batch([image, label], batch_size=2)

    # x = image_batch
    # y_ = label_batch

    # # Define your model in terms of `x` and `y_` here....
    # train_step = ...

    # # N.B. You must run this function after creating your graph.
    # init = tf.initialize_all_variables()
    # sess.run(init)

    # # N.B. You must run this function before `sess.run(train_step)` to
    # # start the input pipeline.
    # tf.train.start_queue_runners(sess)

    # for i in range(100):
    #     # No need to feed, because `x` and `y_` are already bound to
    #     # the next input batch.
    #     sess.run(train_step)


    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        writer.add_graph(sess.graph)

        # TODO stop training when no improvement on validation set
        for i in range(2000):
            batch = it_train.get_next()
            batch_xs, batch_ys = sess.run(batch)

            model.train(batch_xs, batch_ys)

            if i % math.ceil(train_size / BATCH_SIZE) == 0:
                accuracy = model.accuracy_eval(
                    datasets['validation'].images,
                    datasets['validation'].labels)
                summary = model.summarize(batch_xs, batch_ys)
                writer.add_summary(summary, i)
                model.save('./model/model')

        print(model.accuracy_eval(datasets['test'].images, datasets['test'].labels))

# TODO

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

