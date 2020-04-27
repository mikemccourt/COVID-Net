import os
import numpy as np

from covidnet_tuning.train_model import train_model


# TODO(Mike) - Probably should box all this up into an object


def fetch_sigopt_api_token(filename=None, env_name='SIGOPT_API_TOKEN'):
    if filename:
        with open(filename, 'r') as f:
            token = f.readline()[:-1]
        return token
    token = os.environ.get(env_name)
    if token:
        return token
    raise AttributeError('No API token found')


def generate_metrics_from_confusion_matrix(matrix, mapping, secret):
    class_accuracy = np.diag(matrix) / (np.sum(matrix, axis=1) + 1e-10)  # Add a fudge factor to avoid 1/0
    ppv = np.diag(matrix) / (np.sum(matrix, axis=0) + 1e-10)
    metrics = {}
    for class_name, index in mapping.items():
        identifier = f'class_{index}' if secret else class_name
        metrics[f'{identifier} sensitivity'] = class_accuracy[index]
        metrics[f'{identifier} PPV'] = ppv[index]
    return metrics


def create_sigopt_experiment_meta(mapping, secret, budget, name=None, exp_type='random'):
    base_metrics_dict = generate_metrics_from_confusion_matrix(np.eye(len(mapping)), mapping, secret)
    metrics = [{'name': m, 'objective': 'maximize', 'strategy': 'store'} for m in base_metrics_dict]
    metrics[0]['strategy'] = 'optimize'

    return dict(
        name=name or 'Random testing',
        type=exp_type,
        parameters=[
            {'name': 'log10_learning_rate', 'type': 'double', 'bounds': {'min': -10, 'max': -1}},
            {'name': 'log2_batch_size', 'type': 'int', 'bounds': {'min': 2, 'max': 5}},
            {'name': 'factor', 'type': 'double', 'bounds': {'min': .5, 'max': .9}},
            {'name': 'patience', 'type': 'int', 'bounds': {'min': 3, 'max': 10}},
            {'name': 'augmentation_translation', 'type': 'double', 'bounds': {'min': 0, 'max': 20}},
            {'name': 'augmentation_rotation', 'type': 'double', 'bounds': {'min': 0, 'max': 10}},
            {'name': 'augmentation_brightness', 'type': 'double', 'bounds': {'min': 0, 'max': .3}},
            {'name': 'class_weighting', 'type': 'double', 'bounds': {'min': 5, 'max': 50}},
        ],
        metrics=metrics,
        observation_budget=budget,
        parallel_bandwidth=1,
        project='initial-testing',
    )


def create_sigopt_observation_dict(
    suggestion,
    train_file,
    test_file,
    mapping,
    input_shape,
    epochs,
    secret,
    data_directory,
    main_output_directory,
):
    x = suggestion.assignments
    confusion_matrix = train_model(
        train_file,
        test_file,
        mapping,
        input_shape,
        covid_class_weight=x['class_weighting'],
        batch_size=2 ** x['log2_batch_size'],
        epochs=epochs,
        learning_rate=10 ** x['log10_learning_rate'],
        factor=x['factor'],
        patience=x['patience'],
        augmentation_translation_magnitude=x['augmentation_translation'],
        augmentation_rotation_magnitude=x['augmentation_rotation'],
        augmentation_brightness_magnitude=x['augmentation_brightness'],
        data_directory=data_directory,
        main_output_directory=main_output_directory,
    )
    metrics = generate_metrics_from_confusion_matrix(confusion_matrix, mapping, secret)

    values = [{'name': name, 'value': value} for name, value in metrics.items()]
    return {'suggestion': suggestion.id, 'values': values}
