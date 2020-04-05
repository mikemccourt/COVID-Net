from keras.optimizers import Adam
from keras.callbacks import ReduceLROnPlateau
import keras
from model import build_COVIDNet
from sigopt import Connection

import numpy as np
import os, pathlib, argparse, time, cv2
from sklearn.metrics import confusion_matrix
from data import DataGenerator
from sklearn.utils import shuffle
from keras.preprocessing.image import ImageDataGenerator

os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # which gpu to train on


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
                 augmentation_brightness=.1,
                 augmentation_rotation=10,
                 augmentation_translation=20,
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
                rotation_range=augmentation_rotation,
                width_shift_range=augmentation_translation / self.input_shape[0],
                height_shift_range=augmentation_translation / self.input_shape[1],
                horizontal_flip=True,
                brightness_range=(1 - augmentation_brightness, 1 + augmentation_brightness),
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


def basic_training(
    epochs,
    trainfile,
    testfile,
    checkpoint,
    learning_rate,
    batch_size,
    patience,
    factor,
    class_weighting,
    augmentation_brightness,
    augmentation_rotation,
    augmentation_translation,
):
    mapping = {'normal': 0, 'pneumonia': 1, 'COVID-19': 2}
    class_weight = {0: 1., 1: 1., 2: class_weighting}

    outputPath = './output/'
    run_id = str(int(time.time() * 1e6))
    runPath = outputPath + run_id
    pathlib.Path(runPath).mkdir(parents=True, exist_ok=True)
    print('Output: ' + runPath)

    # load data
    file = open(trainfile, 'r')
    trainfiles = file.readlines()
    file = open(testfile, 'r')
    testfiles = file.readlines()

    train_generator = BalanceDataGenerator(
        trainfiles,
        batch_size=batch_size,
        is_training=True,
        augmentation_brightness=augmentation_brightness,
        augmentation_rotation=augmentation_rotation,
        augmentation_translation=augmentation_translation,
    )
    test_generator = DataGenerator(testfiles, batch_size=batch_size, is_training=False)

    def get_callbacks(runPath):
        callbacks = []
        lr_schedule = ReduceLROnPlateau(monitor='val_loss', factor=factor, patience=patience, min_lr=0.000001, min_delta=1e-2)
        callbacks.append(lr_schedule)  # reduce learning rate when stuck

        checkpoint_path = runPath + '/checkpoint.hdf5'
        callbacks.append(keras.callbacks.ModelCheckpoint(checkpoint_path,
                                                         verbose=1, save_best_only=False, save_weights_only=True,
                                                         mode='min', period=1))

        return callbacks

    model = build_COVIDNet(checkpoint=checkpoint)

    opt = Adam(learning_rate=learning_rate, amsgrad=True)
    callbacks = get_callbacks(runPath)
    model.compile(loss='categorical_crossentropy',
                  optimizer=opt,
                  metrics=['accuracy']) # TO-DO: add additional metrics for COVID-19
    print('Ready for training!')

    model.fit_generator(train_generator, callbacks=callbacks, validation_data=test_generator, epochs=epochs, shuffle=True, class_weight=class_weight, use_multiprocessing=False)

    y_test = []
    pred = []
    for i in range(len(testfiles)):
        line = testfiles[i].split()
        x = cv2.imread(os.path.join('data', 'test', line[1]))
        x = cv2.resize(x, (224, 224))
        x = x.astype('float32') / 255.0
        y_test.append(mapping[line[2]])
        pred.append(np.array(model.predict(np.expand_dims(x, axis=0))).argmax(axis=1))
    y_test = np.array(y_test)
    pred = np.array(pred)[:, 0]

    matrix = confusion_matrix(y_test, pred)
    matrix = matrix.astype('float')
    print(matrix)
    class_acc = [matrix[i,i]/np.sum(matrix[i,:]) if np.sum(matrix[i,:]) else 0 for i in range(len(matrix))]
    print('Sens Normal: {0:.3f}, Pneumonia: {1:.3f}, COVID-19: {2:.3f}'.format(class_acc[0],
                                                                               class_acc[1],
                                                                               class_acc[2]))
    ppvs = [matrix[i,i]/np.sum(matrix[:,i]) if np.sum(matrix[:,i]) else 0 for i in range(len(matrix))]
    print('PPV Normal: {0:.3f}, Pneumonia {1:.3f}, COVID-19: {2:.3f}'.format(ppvs[0],
                                                                             ppvs[1],
                                                                             ppvs[2]))

    return matrix


def generate_metrics_from_confusion_matrix(matrix, secret):
    mapping = {'normal': 0, 'pneumonia': 1, 'COVID-19': 2}
    class_accuracy = np.diag(matrix) / (np.sum(matrix, axis=1) + 1e-10)  # Add a fudge factor to avoid 1/0
    ppv = np.diag(matrix) / (np.sum(matrix, axis=0) + 1e-10)
    metrics = {}
    for class_name, index in mapping.items():
        identifier = f'class_{index}' if secret else class_name
        metrics[f'{identifier} sensitivity'] = class_accuracy[index]
        metrics[f'{identifier} PPV'] = ppv[index]
    return metrics


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='COVID-Net Training')
    parser.add_argument('--trainfile', required=True, type=str, help='Name of train file')
    parser.add_argument('--testfile', required=True, type=str, help='Name of test file')
    parser.add_argument('--checkpoint', default='', type=str, help='Start training from existing weights')
    parser.add_argument('--exp-id', required=True, type=int, help='SigOpt experiment id')
    parser.add_argument('--sigopt-api-token', required=True, type=str, help='To access SigOpt')
    parser.add_argument('--num-obs', required=True, type=int, help='Number of observations to run now')
    args = parser.parse_args()

    trainfile = args.trainfile
    testfile = args.testfile
    checkpoint = args.checkpoint
    client_token = args.sigopt_api_token
    experiment_id = args.exp_id
    num_observations = args.num_obs

    conn = Connection(client_token=client_token)
    experiment = conn.experiments(experiment_id).fetch()
    epochs = experiment.metadata['epochs']
    secret = experiment.metadata['secret'] != 'false'

    print(
        f'Experiment {experiment.id} loaded, '
        f'{experiment.progress.observation_count} of {experiment.observation_budget} completed, '
        f'{num_observations} to be conducted now.'
    )

    for k in range(num_observations):
        suggestion = conn.experiments(experiment.id).suggestions().create()

        a = suggestion.assignments
        matrix = basic_training(
            epochs,
            trainfile,
            testfile,
            checkpoint,
            learning_rate=10 ** a['log10_learning_rate'],
            batch_size=2 ** a['log2_batch_size'],
            patience=a['patience'],
            factor=a['factor'],
            class_weighting=a['class_weighting'],
            augmentation_brightness=a['augmentation_brightness'],
            augmentation_rotation=a['augmentation_rotation'],
            augmentation_translation=a['augmentation_translation'],
        )

        metrics = generate_metrics_from_confusion_matrix(matrix, secret)
        values = [{'name': name, 'value': value} for name, value in metrics.items()]

        observation = conn.experiments(experiment.id).observations().create(suggestion=suggestion.id, values=values)
        print(f'{k}th observation completed, id {observation.id}')


# PYTHONPATH=. python sigopt_tuning_with_wrapper.py --exp-id 183510 --sigopt-api-token XXX --num-obs 10 --trainfile faketrain.txt --testfile faketest.txt
