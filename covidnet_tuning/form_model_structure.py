from keras.applications.resnet_v2 import ResNet50V2
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D, Flatten


def form_COVIDNet_structure(mapping, input_shape, flatten=True, checkpoint=''):
    input_shape_with_channels = input_shape + (3, )  # Maybe shouldn't hard code this?
    base_model = ResNet50V2(include_top=False, weights='imagenet', input_shape=input_shape_with_channels)

    x = base_model.output
    x = Flatten()(x) if flatten else GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)
    x = Dense(256, activation='relu')(x)

    predictions = Dense(len(mapping), activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=predictions)
    if len(checkpoint):
        model.load_weights(checkpoint)
    return model
