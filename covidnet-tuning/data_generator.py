import numpy as np
from tensorflow import keras
import cv2
import os
from sklearn.utils import shuffle


# TODO(Mike) - Should use inheritance here


class BalanceDataGenerator(keras.utils.Sequence):
    def __init__(
        self,
        dataset,
        mapping,
        input_shape,
        batch_size,
        augmentation_translation_magnitude,
        augmentation_rotation_magnitude,
        augmentation_brightness_magnitude,
        is_training=True,
        num_channels=3,
        shuffle_data=True,
        augmentation=True,
        datadir='data',
        debug=True,
    ):
        'Initialization'
        self.datadir = datadir
        self.dataset = dataset
        self.is_training = is_training
        self.batch_size = batch_size
        self.N = len(self.dataset)
        self.input_shape = input_shape
        self.num_channels = num_channels
        self.mapping = mapping
        self.shuffle_data = shuffle_data
        self.debug = debug

        self.augmentation = None
        if augmentation:
            self.augmentation = keras.preprocessing.image.ImageDataGenerator(
                featurewise_center=False,
                featurewise_std_normalization=False,
                rotation_range=augmentation_rotation_magnitude,
                width_shift_range=augmentation_translation_magnitude / self.input_shape[0],
                height_shift_range=augmentation_translation_magnitude / self.input_shape[1],
                horizontal_flip=True,
                brightness_range=(1 - augmentation_brightness_magnitude, 1 + augmentation_brightness_magnitude),
                fill_mode='constant',
                cval=0.,
            )

        datasets = {'normal': [], 'pneumonia': [], 'COVID-19': []}
        for l in dataset:
            datasets[l.split()[-1]].append(l)
        self.datasets = [
            datasets['normal'] + datasets['pneumonia'],
            datasets['COVID-19'],
        ]

        self.on_epoch_end()

        if self.debug:
            print(f'Initialization of {self} completed, {len(self.datasets[0])}, {len(self.datasets[1])}')

    @property
    def num_classes(self):
        return len(self.mapping)

    def __len__(self):
        return int(np.ceil(len(self.datasets[0]) / float(self.batch_size)))

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        if self.shuffle_data:
            for v in self.datasets:
                np.random.shuffle(v)

    def __getitem__(self, idx):
        batch_x, batch_y = np.zeros((self.batch_size, *self.input_shape, self.num_channels)), np.zeros(self.batch_size)

        batch_files = self.datasets[0][idx * self.batch_size: (idx + 1) * self.batch_size]
        batch_files[np.random.randint(self.batch_size)] = np.random.choice(self.datasets[1])

        for i in range(self.batch_size):
            sample = batch_files[i].split()

            if self.is_training:
                folder = 'train'
            else:
                folder = 'test'

            x = cv2.imread(os.path.join(self.datadir, folder, sample[1]))
            x = cv2.resize(x, self.input_shape)

            if self.is_training and self.augmentation:
                x = self.augmentation.random_transform(x)

            x = x.astype('float32') / 255
            y = self.mapping[sample[2]]

            batch_x[i] = x
            batch_y[i] = y

        return batch_x, keras.utils.to_categorical(batch_y, num_classes=self.num_classes)


class DataGenerator(keras.utils.Sequence):
    def __init__(
        self,
        dataset,
        mapping,
        input_shape,
        batch_size,
        is_training=True,
        num_channels=3,
    ):
        # Initialization
        self.dataset = dataset
        self.is_training = is_training
        self.batch_size = batch_size
        self.N = len(self.dataset)
        self.input_shape = input_shape
        self.num_channels = num_channels
        self.mapping = mapping

        self.on_epoch_end()

    @property
    def num_classes(self):
        return len(self.mapping)

    def __len__(self):
        return int(np.ceil(self.N / float(self.batch_size)))

    def on_epoch_end(self):
        self.dataset = shuffle(self.dataset, random_state=0)

    def __getitem__(self, idx):
        batch_x, batch_y = np.zeros((self.batch_size, *self.input_shape, self.num_channels)), np.zeros(self.batch_size)
        for i in range(self.batch_size):
            index = min((idx * self.batch_size) + i, self.N - 1)

            sample = self.dataset[index].split()

            if self.is_training:
                folder = 'train'
            else:
                folder = 'test'

            x = cv2.imread(os.path.join('data', folder, sample[1]))
            x = cv2.resize(x, self.input_shape)

            x = x.astype('float32') / 255
            y = self.mapping[sample[2]]

            batch_x[i] = x
            batch_y[i] = y

        return batch_x, keras.utils.to_categorical(batch_y, num_classes=self.num_classes)
