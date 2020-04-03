import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau

import numpy as np
import os, pathlib, time, cv2
from sklearn.metrics import confusion_matrix
from data_generator import DataGenerator, BalanceDataGenerator
from form_model_structure import form_COVIDNet_structure

os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # which gpu to train on

DEFAULT_MAPPING = {'normal': 0, 'pneumonia': 1, 'COVID-19': 2}
DEFAULT_INPUT_SHAPE = (224, 224)


# Data is assumed to be stored in <base_dir>/data/train and <base_dir>/data/test


# TODO(Mike) - Add training monitor checkpoints in here
def get_callbacks(save_path, factor, patience):
    callbacks = []
    lr_schedule = ReduceLROnPlateau(
        monitor='val_loss',
        factor=factor,
        patience=patience,
        min_lr=0.0000005,  # Is this something that we want to consider tuning?
        min_delta=1e-2,
    )
    callbacks.append(lr_schedule)  # reduce learning rate when stuck

    checkpoint_path = os.path.join(save_path, f'cp-{epoch:02d}-{val_loss:.2f}.hdf5')
    checkpoint = keras.callbacks.ModelCheckpoint(
        checkpoint_path,
        verbose=1,
        save_best_only=False,
        save_weights_only=True,
        mode='min',
        period=1,
    )
    callbacks.append(checkpoint)

    class SaveAsCKPT(keras.callbacks.Callback):
        def __init__(self):
            self.saver = tf.train.Saver()
            self.sess = keras.backend.get_session()

        def on_epoch_end(self, epoch, logs=None):
            checkpoint_path = os.path.join(save_path, f'cp-{epoch:02d}.ckpt')
            self.saver.save(self.sess, checkpoint_path)
    callbacks.append(SaveAsCKPT())

    return callbacks


def form_data_generators(
    training_file,
    testing_file,
    mapping,
    input_shape,
    batch_size,
    augmentation_translation_magnitude,
    augmentation_rotation_magnitude,
    augmentation_brightness_magnitude,
):
    file = open(training_file, 'r')
    train_data_locations = file.readlines()
    file = open(testing_file, 'r')
    test_data_locations = file.readlines()
    train_generator = BalanceDataGenerator(
        train_data_locations,
        mapping,
        input_shape,
        batch_size,
        augmentation_translation_magnitude,
        augmentation_rotation_magnitude,
        augmentation_brightness_magnitude,
        is_training=True,
    )
    test_generator = DataGenerator(test_data_locations, mapping, input_shape, batch_size, is_training=False)
    return train_generator, test_generator


def form_confusion_matrix(test_file, mapping, model):
    y_test = []
    pred = []
    with open(test_file, 'r') as f:
        for line in f:
            split_line = line.split()
            x = cv2.imread(os.path.join('data', 'test', split_line[1]))
            x = cv2.resize(x, (224, 224))
            x = x.astype('float32') / 255.0
            y_test.append(mapping[split_line[2]])
            pred.append(np.array(model.predict(np.expand_dims(x, axis=0))).argmax(axis=1))
    y_test = np.array(y_test)
    pred = np.array(pred)

    matrix = confusion_matrix(y_test, pred)
    return matrix.astype('float')

    # Need to move this to later
    # class_acc = [matrix[i, i] / np.sum(matrix[i, :]) if np.sum(matrix[i, :]) else 0 for i in range(len(matrix))]
    # print('Sens Normal: {0:.3f}, Pneumonia: {1:.3f}, COVID-19: {2:.3f}'.format(class_acc[0],
    #                                                                            class_acc[1],
    #                                                                            class_acc[2]))
    # ppvs = [matrix[i, i] / np.sum(matrix[:, i]) if np.sum(matrix[:, i]) else 0 for i in range(len(matrix))]
    # print('PPV Normal: {0:.3f}, Pneumonia {1:.3f}, COVID-19: {2:.3f}'.format(ppvs[0],
    #                                                                          ppvs[1],
    #                                                                          ppvs[2]))


def train_model(
    train_file,
    test_file,
    mapping=None,
    input_shape=None,
    covid_class_weight=25,
    batch_size=8,
    epochs=10,
    learning_rate=2e-5,
    factor=0.7,
    patience=5,
    augmentation_translation_magnitude=20,
    augmentation_rotation_magnitude=10,
    augmentation_brightness_magnitude=.1,
    main_output_directory='./output/',
    debug=True,
):
    mapping = DEFAULT_MAPPING if mapping is None else mapping
    input_shape = DEFAULT_INPUT_SHAPE if input_shape is None else input_shape
    class_weight = {
        class_num: covid_class_weight if diagnosis != 'COVID-19' else 1.0
        for diagnosis, class_num in mapping.items()
    }

    run_id = str(int(time.time() * 1e6))
    save_path = os.path.join(main_output_directory, run_id)
    pathlib.Path(save_path).mkdir(parents=True, exist_ok=True)

    callbacks = get_callbacks(save_path, factor, patience)
    train_generator, test_generator = form_data_generators(
        train_file,
        test_file,
        mapping,
        input_shape,
        batch_size,
        augmentation_translation_magnitude,
        augmentation_rotation_magnitude,
        augmentation_brightness_magnitude,
    )
    if debug:
        print(f'Data generators created, {len(train_generator)} training, {len(test_generator)} testing')

    optimizer = Adam(learning_rate=learning_rate, amsgrad=True)
    model = form_COVIDNet_structure(len(mapping))
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    if debug:
        print('Model compiled, ready for training')

    stuff = model.fit_generator(
        train_generator,
        callbacks=callbacks,
        validation_data=test_generator,
        epochs=epochs,
        shuffle=True,
        class_weight=class_weight,
        use_multiprocessing=False,
    )
    if debug:
        print('Model training completed')
        print(stuff)

    return form_confusion_matrix(test_file, mapping, model)
