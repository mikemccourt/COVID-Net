import keras
from keras.optimizers import Adam
from keras.callbacks import ReduceLROnPlateau

import numpy as np
import os, pathlib, time, cv2, json
from sklearn.metrics import confusion_matrix
from covidnet_tuning.data_generator import DataGenerator, BalanceDataGenerator
from covidnet_tuning.form_model_structure import form_COVIDNet_structure

os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # which gpu to train on


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

    # Need to figure out how the epoch/val_loss thing works (no idea right now)
    checkpoint_path = os.path.join(save_path, 'cp-{epoch:02d}-{val_loss:.2f}.hdf5')
    checkpoint = keras.callbacks.ModelCheckpoint(
        checkpoint_path,
        verbose=1,
        save_best_only=False,
        save_weights_only=True,
        mode='min',
        period=1,
    )
    callbacks.append(checkpoint)

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
    data_directory='data',
):
    file = open(training_file, 'r')
    train_data_locations = file.readlines()
    train_generator = BalanceDataGenerator(
        train_data_locations,
        mapping,
        input_shape,
        batch_size,
        augmentation_translation_magnitude,
        augmentation_rotation_magnitude,
        augmentation_brightness_magnitude,
        is_training=True,
        data_directory=data_directory,
    )

    file = open(testing_file, 'r')
    test_data_locations = file.readlines()
    test_generator = DataGenerator(
        test_data_locations,
        mapping,
        input_shape,
        batch_size,
        is_training=False,
        data_directory=data_directory,
    )
    return train_generator, test_generator


def form_confusion_matrix(test_generator, model):
    y_test = []
    pred = []
    for batch_num in range(len(test_generator)):
        for x, y_one_hot in zip(*test_generator[batch_num]):
            y_test.append(np.argmax(y_one_hot))
            pred.append(np.array(model.predict(np.expand_dims(x, axis=0))).argmax(axis=1)[0])

    return confusion_matrix(y_test, pred).astype('float')


def train_model(
    train_file,
    test_file,
    mapping,
    input_shape,
    covid_class_weight=25,
    batch_size=8,
    epochs=10,
    learning_rate=2e-5,
    factor=0.7,
    patience=5,
    augmentation_translation_magnitude=20,
    augmentation_rotation_magnitude=10,
    augmentation_brightness_magnitude=.1,
    main_output_directory='output',
    data_directory='data',
    debug=True,
):
    class_weight = {
        class_num: covid_class_weight if diagnosis != 'COVID-19' else 1.0
        for diagnosis, class_num in mapping.items()
    }

    run_id = str(int(time.time() * 1e6))
    save_path = os.path.join(main_output_directory, run_id)
    pathlib.Path(save_path).mkdir(parents=True, exist_ok=True)
    tunable_parameters = {
        'covid_class_weight': covid_class_weight,
        'batch_size': batch_size,
        'epochs': epochs,
        'learning_rate': learning_rate,
        'factor': factor,
        'patience': patience,
        'augmentation_translation_magnitude': augmentation_translation_magnitude,
        'augmentation_rotation_magnitude': augmentation_rotation_magnitude,
        'augmentation_brightness_magnitude': augmentation_brightness_magnitude,
    }
    model_meta_path = os.path.join(save_path, 'meta.json')
    with open(model_meta_path, 'w') as f:
        json.dump(tunable_parameters, f)

    train_generator, test_generator = form_data_generators(
        train_file,
        test_file,
        mapping,
        input_shape,
        batch_size,
        augmentation_translation_magnitude,
        augmentation_rotation_magnitude,
        augmentation_brightness_magnitude,
        data_directory=data_directory,
    )
    if debug:
        print(f'Data generators created, {len(train_generator)} training, {len(test_generator)} testing')

    optimizer = Adam(learning_rate=learning_rate, amsgrad=True)
    model = form_COVIDNet_structure(mapping, input_shape)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    if debug:
        print('Model compiled, ready for training')

    model.fit_generator(
        train_generator,
        callbacks=get_callbacks(save_path, factor, patience),
        validation_data=test_generator,
        epochs=epochs,
        shuffle=True,
        class_weight=class_weight,
        use_multiprocessing=False,
    )
    if debug:
        print('Model training completed')

    return form_confusion_matrix(test_generator, model)
