from covidnet_tuning.train_model import train_model


def main():
    return train_model(
        'faketrain.txt',
        'faketest.txt',
        {'normal': 0, 'pneumonia': 1, 'COVID-19': 2},
        (224, 224),
        covid_class_weight=25,
        batch_size=16,
        epochs=3,
        learning_rate=2e-5,
        factor=0.7,
        patience=5,
        augmentation_translation_magnitude=20,
        augmentation_rotation_magnitude=10,
        augmentation_brightness_magnitude=.1,
        data_directory='data',
    )


if __name__ == '__main__':
    main()

# To run PYTHONPATH=. python covidnet_tuning/basic_test.py
