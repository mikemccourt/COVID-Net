import numpy as np
import keras
import cv2
import os
from sklearn.utils import shuffle
from keras.preprocessing.image import ImageDataGenerator


# This is created through a blatant copy-paste to minimize deviations from the original repo while allowing tuning
class BalanceDataGeneratorForTuning(keras.utils.Sequence):
    mapping = {'normal': 0, 'pneumonia': 1, 'COVID-19': 2}
    
    def __init__(self,
                 dataset,
                 is_training=True,
                 batch_size=8,
                 input_shape=(224, 224),
                 n_classes=3,
                 num_channels=3,
                 mapping=None,
                 shuffle=True,
                 augmentation=True,
                 datadir='data',
                 augmentation_brightness=.1,
                 augmentation_rotation=10,
                 augmentation_translation=20,
                 augmentation_zoom=0,
                 covid_percent=.25,
                 ):
        'Initialization'
        self.datadir = datadir
        self.dataset = dataset
        self.is_training = is_training
        self.batch_size = batch_size
        self.N = len(self.dataset)
        self.input_shape = input_shape
        self.n_classes = n_classes
        self.num_channels = num_channels
        self.mapping = mapping or self.mapping
        self.shuffle = shuffle
        self.covid_percent = covid_percent

        if augmentation:
            self.augmentation = ImageDataGenerator(
                featurewise_center=False,
                featurewise_std_normalization=False,
                rotation_range=augmentation_rotation,
                width_shift_range=augmentation_translation / self.input_shape[0],
                height_shift_range=augmentation_translation / self.input_shape[1],
                horizontal_flip=True,
                brightness_range=(1 - augmentation_brightness, 1 + augmentation_brightness),
                zoom_range=(1 - augmentation_zoom, 1 + augmentation_zoom),
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
        print(len(self.datasets[0]), len(self.datasets[1]))

        self.on_epoch_end()

    def __len__(self):
        return int(np.ceil(len(self.datasets[0]) / float(self.batch_size)))

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        if self.shuffle == True:
            for v in self.datasets:
                np.random.shuffle(v)

    def upsample_covid_cases(self, batch_files):
        covid_size = max(int(len(batch_files) * self.covid_percent), 1)
        covid_inds = np.random.choice(np.arange(len(batch_files)), size=covid_size, replace=False)
        covid_files = np.random.choice(self.datasets[1], size=covid_size, replace=False)
        for i, covid_file in zip(covid_inds, covid_files):
            batch_files[i] = covid_file

    def __getitem__(self, idx):
        batch_files = self.datasets[0][idx*self.batch_size : (idx+1)*self.batch_size]
        self.upsample_covid_cases(batch_files)

        batch_size = len(batch_files)  # Since it could be less than the full batch size
        batch_x, batch_y = np.zeros((batch_size, *self.input_shape, self.num_channels)), np.zeros(batch_size)

        for i, batch_file in enumerate(batch_files):
            sample = batch_file.split()

            if self.is_training:
                folder = 'train'
            else:
                folder = 'test'

            x = cv2.imread(os.path.join(self.datadir, folder, sample[1]))
            h, _, _ = x.shape
            x = x[int(h / 6):, :, :]  # This trims hospital symbols from the top of each x-ray
            x = cv2.resize(x, self.input_shape)

            if self.is_training and hasattr(self, 'augmentation'):
                x = self.augmentation.random_transform(x)

            x = x.astype('float32') / 255.0
            y = self.mapping[sample[2]]

            batch_x[i] = x
            batch_y[i] = y

        return batch_x, keras.utils.to_categorical(batch_y, num_classes=self.n_classes)


class BalanceDataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self,
                 dataset,
                 is_training=True,
                 batch_size=8,
                 input_shape=(224,224),
                 n_classes=3,
                 num_channels=3,
                 mapping={'normal': 0, 'pneumonia': 1, 'COVID-19': 2},
                 shuffle=True,
                 augmentation=True,
                 datadir='data',
                 ):
        'Initialization'
        self.datadir = datadir
        self.dataset = dataset
        self.is_training = is_training
        self.batch_size = batch_size
        self.N = len(self.dataset)
        self.input_shape = input_shape
        self.n_classes = n_classes
        self.num_channels = num_channels
        self.mapping = mapping
        self.shuffle = True

        if augmentation:
            self.augmentation = ImageDataGenerator(
                featurewise_center=False,
                featurewise_std_normalization=False,
                rotation_range=10,
                width_shift_range=0.1,
                height_shift_range=0.1,
                horizontal_flip=True,
                brightness_range=(0.9, 1.1),
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
        print(len(self.datasets[0]), len(self.datasets[1]))

        self.on_epoch_end()


    def __len__(self):
        return int(np.ceil(len(self.datasets[0]) / float(self.batch_size)))

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        if self.shuffle == True:
            for v in self.datasets:
                np.random.shuffle(v)

    def __getitem__(self, idx):
        batch_x, batch_y = np.zeros((self.batch_size, *self.input_shape, self.num_channels)), np.zeros(self.batch_size)

        batch_files = self.datasets[0][idx*self.batch_size : (idx+1)*self.batch_size]
        batch_files[np.random.randint(self.batch_size)] = np.random.choice(self.datasets[1])

        for i in range(self.batch_size):
            sample = batch_files[i].split()

            if self.is_training:
                folder = 'train'
            else:
                folder = 'test'

            x = cv2.imread(os.path.join(self.datadir, folder, sample[1]))
            x = cv2.resize(x, self.input_shape)

            if self.is_training and hasattr(self, 'augmentation'):
                x = self.augmentation.random_transform(x)

            x = x.astype('float32') / 255.0
            y = self.mapping[sample[2]]

            batch_x[i] = x
            batch_y[i] = y

        return batch_x, keras.utils.to_categorical(batch_y, num_classes=self.n_classes)


class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self,
                 dataset,
                 is_training=True,
                 batch_size=8,
                 input_shape=(224,224),
                 n_classes=3,
                 num_channels=3,
                 mapping={'normal': 0, 'pneumonia': 1, 'COVID-19': 2},
                 shuffle=True):
        'Initialization'
        self.dataset = dataset
        self.is_training = is_training
        self.batch_size = batch_size
        self.N = len(self.dataset)
        self.input_shape = input_shape
        self.n_classes = n_classes
        self.num_channels = num_channels
        self.mapping = mapping
        self.shuffle = True
        self.on_epoch_end()

    def __len__(self):
        return int(np.ceil(self.N / float(self.batch_size)))

    def on_epoch_end(self):
        self.dataset = shuffle(self.dataset, random_state=0)

    def __getitem__(self, idx):
        batch_x, batch_y = np.zeros((self.batch_size, *self.input_shape, self.num_channels)), np.zeros(self.batch_size)
        for i in range(self.batch_size):
            index = min((idx * self.batch_size) + i, self.N-1)

            sample = self.dataset[index].split()

            if self.is_training:
                folder = 'train'
            else:
                folder = 'test'

            x = cv2.imread(os.path.join('data', folder, sample[1]))
            x = cv2.resize(x, self.input_shape)
            x = x.astype('float32') / 255.0
            #y = int(sample[1])
            y = self.mapping[sample[2]]

            batch_x[i] = x
            batch_y[i] = y

        return batch_x, keras.utils.to_categorical(batch_y, num_classes=self.n_classes)
