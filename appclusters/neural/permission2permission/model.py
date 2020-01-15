from keras import Input, Model
from keras.layers import Dense
from appclusters import config


layer_config = [100]


def model_encoder_decoder(num_permissions):
    global layer_config

    inputs = Input(shape=(num_permissions,))
    layer = inputs

    #layer_config = [num_permissions//2]

    if type(layer_config[0]) is float:
        layer_config = [int(num_permissions*fraction) for fraction in layer_config]

    for neurons in layer_config:
        layer = Dense(neurons, activation='relu')(layer)

    layer = Dense(config.Clustering.latent_size_p2p, activation='linear', name='latent')(layer)
    latent = layer

    for neurons in reversed(layer_config):
        layer = Dense(neurons, activation='relu')(layer)

    outputs = Dense(num_permissions, activation='sigmoid')(layer)

    return Model(inputs=inputs, outputs=outputs), Model(inputs=inputs, outputs=latent)
