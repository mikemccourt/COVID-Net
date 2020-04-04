import argparse
from sigopt import Connection

from covidnet_tuning.tune_model import (
    fetch_sigopt_api_token,
    create_sigopt_experiment_meta,
    create_sigopt_observation_dict,
)


DEFAULT_MAPPING = {'normal': 0, 'pneumonia': 1, 'COVID-19': 2}
DEFAULT_INPUT_SHAPE = (224, 224)


def parse_args():
    parser = argparse.ArgumentParser(description='COVID-Net Training')
    parser.add_argument('--train-file', default='train_COVIDx.txt', type=str, help='Name of train file')
    parser.add_argument('--test-file', default='test_COVIDx.txt', type=str, help='Name of test file')
    parser.add_argument('--data-directory', default='data', type=str, help='Path to train/test directories')
    parser.add_argument('--output-directory', default='output', type=str, help='Where to store checkpoints')
    parser.add_argument('--epochs', required=True, type=int, help='Number of epochs')
    parser.add_argument('--sigopt-api-token', required=False, type=str, help='To access SigOpt')
    parser.add_argument('--sigopt-api-token-file', required=False, type=str, help='File containing key')
    parser.add_argument('--secret', action='store_true', help='File containing key')
    parser.add_argument('--budget', required=True, type=int, help='Number of observations to run now')
    parser.add_argument('--name', required=False, type=str, help='Experiment name')
    parser.add_argument('--exp-type', required=False, default='random', type=str, help='SigOpt experiment type')
    parser.add_argument('--exp-id', required=False, type=int, help='Exp ID to restart (if one exists)')
    parser.add_argument('--api-url', required=False, type=str, help='If accessing special SigOpt server')
    return parser.parse_args()


def main():
    args = parse_args()
    client_token = args.sigopt_api_token or fetch_sigopt_api_token(filename=args.sigopt_api_token_file)
    conn = Connection(client_token=client_token)
    if args.api_url:
        conn.set_api_url(args.api_url)

    if args.exp_id:
        experiment = conn.experiments(args.exp_id).fetch()
    else:
        experiment_meta = create_sigopt_experiment_meta(
            DEFAULT_MAPPING,
            args.secret,
            args.budget,
            name=args.name,
            exp_type=args.exp_type
        )
        experiment_meta.update({'metadata': {'epochs': args.epochs}})
        experiment = conn.experiments().create(**experiment_meta)
    print(
        f'Experiment {experiment.id} loaded, '
        f'{experiment.progress.observation_count}/{experiment.observation_budget} observations completed'
    )

    # This check for open suggestions will only grab the first 100 (hopefully there isn't more than 1)
    open_suggestions = list(conn.experiments(experiment.id).suggestions().fetch(state="open").iterate_pages())
    if len(open_suggestions):
        print(f'{len(open_suggestions)} open suggestions already exist -- those will be resumed first')

    for k in range(args.budget):
        if len(open_suggestions):
            suggestion = open_suggestions.pop(0)
        else:
            suggestion = conn.experiments(experiment.id).suggestions().create()
        observation_dict = create_sigopt_observation_dict(
            suggestion,
            args.train_file,
            args.test_file,
            DEFAULT_MAPPING,
            DEFAULT_INPUT_SHAPE,
            args.epochs,
            args.secret,
            args.data_directory,
            args.output_directory,
        )
        conn.experiments(experiment.id).observations().create(**observation_dict)
        print(f'Observation {k + 1} of {args.budget} completed')


if __name__ == '__main__':
    main()

# To run PYTHONPATH=. python covidnet_tuning/sigopt_test.py --train-file faketrain.txt --test-file faketest.txt --epochs 4 --sigopt-api-token-file sigopt-api-token --secret --budget 10 --name trial --exp-type random
