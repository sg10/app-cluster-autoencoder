from math import sqrt

import random

from appclusters import config


def random_encoder_archs(num_archs):
    archs = []

    for _ in range(num_archs):
        a = []
        depth = random.randint(config.Clustering.encoder_depth[0], config.Clustering.encoder_depth[1])
        for _ in range(depth):
            num_neurons_fraction = random.uniform(config.Clustering.hidden_neurons_fraction[0],
                                                  config.Clustering.hidden_neurons_fraction[1])
            num_neurons_fraction /= sqrt(depth)
            a.append(num_neurons_fraction)

        archs.append(a)

    return archs
