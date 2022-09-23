from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import pickle
import sys
import tensorflow as tf
version = sys.version_info

import numpy as np

class ImageNetData(object):
    def __init__(self, path):
        train_filenames = ['train_data_batch_{}'.format(ii + 1) for ii in range(10)]
        x_list = []
        y_list = []
        for ii, fname in enumerate(train_filenames):
            cur_images, cur_labels = self._load_datafile(os.path.join(path, fname))
            x_list.append(cur_images)
            y_list.append(cur_labels)
        train_images = np.concatenate(x_list, axis=0)
        train_labels = np.concatenate(y_list, axis=0)
        self.train_data = DataSubset(train_images, train_labels)

    @staticmethod
    def unpickle(file):
        with open(file, 'rb') as fo:
            dict = pickle.load(fo)
        return dict
    
    def _load_datafile(self, data_file, img_size=32):
        d = self.unpickle(data_file)
        x = d['data']
        y = d['labels']
        # Labels are indexed from 1, shift it so that indexes start at 0
        y = [i-1 for i in y]
        data_size = x.shape[0]
        img_size2 = img_size * img_size
        x = np.dstack((x[:, :img_size2], x[:, img_size2:2*img_size2], x[:, 2*img_size2:]))
        x = x.reshape((x.shape[0], img_size, img_size, 3))
        y = np.array(y)
        assert x.dtype == np.uint8
        return x, y


class AugmentedImageNetData(object):
    def __init__(self, raw_imagenetdata, sess, model, subset=False):
        assert isinstance(raw_imagenetdata, ImageNetData) or isinstance(raw_imagenetdata[0], ImageNetData)
        self.image_size = 32
        # create augmentation computational graph
        self.x_input_placeholder = tf.placeholder(tf.float32, shape=[None, 32, 32, 3])
        padded = tf.map_fn(lambda img: tf.image.resize_image_with_crop_or_pad(
            img, self.image_size + 4, self.image_size + 4),
            self.x_input_placeholder)
        cropped = tf.map_fn(lambda img: tf.random_crop(img, [self.image_size,
                                                             self.image_size,
                                                             3]), padded)
        flipped = tf.map_fn(lambda img: tf.image.random_flip_left_right(img), cropped)
        self.augmented = flipped
        self.train_data = AugmentedDataSubset(raw_imagenetdata.train_data, sess,
                                             self.x_input_placeholder,
                                              self.augmented, subset)

class DataSubset(object):
    def __init__(self, xs, ys):
        self.xs = xs
        self.n = xs.shape[0]
        self.ys = ys
        self.batch_start = 0
        self.cur_order = np.random.permutation(self.n)

    def get_next_batch(self, batch_size, multiple_passes=False, reshuffle_after_pass=True):
        if self.n < batch_size:
            raise ValueError('Batch size can be at most the dataset size')
        if not multiple_passes:
            actual_batch_size = min(batch_size, self.n - self.batch_start)
            if actual_batch_size <= 0:
                raise ValueError('Pass through the dataset is complete.')
            batch_end = self.batch_start + actual_batch_size
            batch_xs = self.xs[self.cur_order[self.batch_start : batch_end], ...]
            batch_ys = self.ys[self.cur_order[self.batch_start : batch_end], ...]
            self.batch_start += actual_batch_size
            return batch_xs, batch_ys
        actual_batch_size = min(batch_size, self.n - self.batch_start)
        if actual_batch_size < batch_size:
            if reshuffle_after_pass:
                self.cur_order = np.random.permutation(self.n)
            self.batch_start = 0
        batch_end = self.batch_start + batch_size
        batch_xs = self.xs[self.cur_order[self.batch_start : batch_end], ...]
        batch_ys = self.ys[self.cur_order[self.batch_start : batch_end], ...]
        self.batch_start += batch_size
        return batch_xs, batch_ys


class AugmentedDataSubset(object):
    def __init__(self, raw_datasubset, sess, x_input_placeholder, 
                 augmented, subset):
        self.sess = sess
        self.raw_datasubset = raw_datasubset
        self.x_input_placeholder = x_input_placeholder
        self.augmented = augmented
        self.subset = subset

    def get_next_batch(self, batch_size, multiple_passes=False, reshuffle_after_pass=True):
        if self.subset:
            raw_batches = [datasubset.get_next_batch(batch_size, multiple_passes,
                                                           reshuffle_after_pass) for datasubset in self.raw_datasubset]
            raw_x_batches = [i[0] for i in raw_batches]
            raw_y_batches = [i[1] for i in raw_batches]
            raw_batch = []
            raw_batch.append(np.concatenate(raw_x_batches, axis=0))
            raw_batch.append(np.concatenate(raw_y_batches, axis=0))
        else:
            raw_batch = self.raw_datasubset.get_next_batch(batch_size, multiple_passes,
                                                           reshuffle_after_pass)
        images = raw_batch[0].astype(np.float32)
        return self.sess.run(self.augmented, feed_dict={self.x_input_placeholder:
                                                    raw_batch[0]}), raw_batch[1]

