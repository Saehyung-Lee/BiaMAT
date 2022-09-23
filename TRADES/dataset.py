from torch.utils.data import Dataset
import sys
import numpy as np
from PIL import Image
import os
import pickle
import tensorflow as tf
version = sys.version_info

class Auxiliary(Dataset):
    def __init__(self, img, target, transform=None):
        self.len = img.shape[0]
        self.img = img
        self.target = target
        self.transform = transform

    def __len__(self):
        return self.len

    def __getitem__(self, index):

        img = (Image.fromarray((self.img[index]).astype(np.uint8)))

        if self.transform is not None:
            img = self.transform(img)
        return img, self.target[index]

class ImageNet(object):
    def __init__(self, path):
        self.image, self.label = self._load_data(path)
        self.num_classes = 1000

    def _load_data(self, path):
        train_filenames = ['train_data_batch_{}'.format(ii + 1) for ii in range(10)]
        x_list = []
        y_list = []
        for ii, fname in enumerate(train_filenames):
            cur_images, cur_labels = self._load_datafile(os.path.join(path, fname))
            x_list.append(cur_images)
            y_list.append(cur_labels)
        image = np.concatenate(x_list, axis=0)
        label = np.concatenate(y_list, axis=0)
        return image, label

    def _unpickle(self, file):
        with open(file, 'rb') as fo:
            dict = pickle.load(fo)
        return dict
    
    def _load_datafile(self, data_file, img_size=32):
        d = self._unpickle(data_file)
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

def Load_dataset(path):
    if 'Imagenet' in path:
        return ImageNet(path)
