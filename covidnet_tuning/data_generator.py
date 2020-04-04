import numpy as np
import keras
import cv2
import os
from sklearn.utils import shuffle


class DataGenerator(keras.utils.Sequence):
    def __init__(
        self,
        dataset,
        mapping,
        input_shape,
        batch_size,
        is_training=True,
        num_channels=3,
        data_directory='data',
    ):
        # Initialization
        self.data_directory = data_directory
        self.dataset = dataset
        self.is_training = is_training
        self.batch_size = batch_size
        self.input_shape = input_shape
        self.num_channels = num_channels
        self.mapping = mapping

        self.on_epoch_end()

    @property
    def num_samples(self):
        return len(self.dataset)

    @property
    def num_classes(self):
        return len(self.mapping)

    def __len__(self):
        return int(np.ceil(self.num_samples / self.batch_size))

    def on_epoch_end(self):
        self.dataset = shuffle(self.dataset, random_state=0)

    def _generate_batch_files_from_index(self, idx):
        return self.dataset[idx * self.batch_size: (idx + 1) * self.batch_size]

    def _postprocess_image_in_batch(self, image):
        return image.astype('float32') / 255

    def __getitem__(self, idx):
        batch_files = self._generate_batch_files_from_index(idx)

        batch_size = len(batch_files)  # Could be less than self.batch_size depending on divisibility
        batch_x, batch_y = np.empty((batch_size, *self.input_shape, self.num_channels)), np.zeros(batch_size)

        for sample_num, this_file_in_batch in enumerate(batch_files):
            _, filename, class_name = this_file_in_batch.split()

            folder = 'train' if self.is_training else 'test'
            x = cv2.imread(os.path.join(self.data_directory, folder, filename))
            x = cv2.resize(x, self.input_shape)

            x = self._postprocess_image_in_batch(x)
            y = self.mapping[class_name]

            batch_x[sample_num] = x
            batch_y[sample_num] = y

        return batch_x, keras.utils.to_categorical(batch_y, num_classes=self.num_classes)


class BalanceDataGenerator(DataGenerator):
    """
    This object guarantees that at least 1 element of each batch is of the COVID class
    """
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
        data_directory='data',
        shuffle_data=True,
        augmentation=True,
        debug=False,
    ):
        # Have to define this before super to allow on_epoch_end to get called
        self.shuffle_data = shuffle_data
        self.debug = debug

        mapping_to_datasets = {'normal': [], 'pneumonia': [], 'COVID-19': []}
        for l in dataset:
            mapping_to_datasets[l.split()[-1]].append(l)
        self.datasets = [
            mapping_to_datasets['normal'] + mapping_to_datasets['pneumonia'],
            mapping_to_datasets['COVID-19'],
        ]

        super().__init__(
            dataset,
            mapping,
            input_shape,
            batch_size,
            is_training,
            num_channels,
            data_directory,
        )

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

        self.on_epoch_end()

        if self.debug:
            print(f'Initialization of {type(self)} completed, {len(self.datasets[0])}, {len(self.datasets[1])}')

    # To reach the balance, we pretend that the data is only as long as the normal + pneumonia cases
    def __len__(self):
        return int(np.ceil(len(self.datasets[0]) / float(self.batch_size)))

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        if self.shuffle_data:
            for v in self.datasets:
                np.random.shuffle(v)

    def _generate_batch_files_from_index(self, idx):
        batch_files = self.datasets[0][idx * self.batch_size: (idx + 1) * self.batch_size]
        batch_files[np.random.randint(len(batch_files))] = np.random.choice(self.datasets[1])
        return batch_files

    def _postprocess_image_in_batch(self, image):
        image = super()._postprocess_image_in_batch(image)
        if self.is_training and self.augmentation:
            image = self.augmentation.random_transform(image)
        return image
