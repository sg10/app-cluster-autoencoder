from keras import Model, Input
from keras.layers import Dense

from appclusters import config

layer_config = [100]


def model_encoder_decoder(num_words):
    global layer_config

    inputs = Input(shape=(num_words,))
    layer = inputs

    if type(layer_config[0]) is float:
        layer_config = [int(num_words*fraction) for fraction in layer_config]

    for neurons in layer_config:
        layer = Dense(neurons, activation='relu')(layer)

    layer = Dense(config.Clustering.latent_size_t2t, activation='linear', name='latent')(layer)
    latent = layer

    for neurons in reversed(layer_config):
        layer = Dense(neurons, activation='relu')(layer)

    outputs = Dense(num_words, activation='linear')(layer)

    return Model(inputs=inputs, outputs=outputs), Model(inputs=inputs, outputs=latent)
