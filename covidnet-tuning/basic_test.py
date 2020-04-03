from train_model import train_model


def main():
    train_model(
        'train_COVIDx.txt',
        'test_COVIDx.txt',
        covid_class_weight=25,
        batch_size=8,
        epochs=10,
        learning_rate=2e-5,
        factor=0.7,
        patience=5,
        augmentation_translation_magnitude=20,
        augmentation_rotation_magnitude=10,
        augmentation_brightness_magnitude=.1,
    )


if __name__ == '__main__':
    main()

# To run PYTHONPATH=covidnet-tuning python basic_test.py
