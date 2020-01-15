import os
import pickle

import keras
import numpy as np
import tensorflow as tf
from tensorflow._api.v1 import logging

from appclusters import config
from appclusters.neural.text2text import model
from appclusters.neural.text2text.model import model_encoder_decoder
from appclusters.preprocessing.samples_database import SamplesDatabase
from appclusters.util.metrics import tfidf_accuracy, tfidf_f1, tfidf_precision, tfidf_recall
from appclusters.util.random_search import random_encoder_archs
from appclusters.util.tensorboard import create_tensorboard_callback


def get_samples(final=False):
    db = SamplesDatabase.get()

    packages = db.filter(('description_raw', 'len>', 30),
                         ('lang', '==', 'en'))

    tfidf_data = pickle.load(open(config.TFIDFModels.description_data, "rb"))
    packages = list(set(packages).intersection(set(tfidf_data['ids'])))

    #random.seed(42)
    #random.shuffle(packages)
    #if not final:
    #    packages = packages[:50000]

    X = np.empty((len(packages), tfidf_data['data'].shape[1]))

    for i, package in enumerate(packages):
        X[i] = tfidf_data['data'][tfidf_data['ids'].index(package)].todense()

    return X, packages


def train(verbose=True, final=True):
    if not verbose:
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
        os.environ['TF_CPP_MIN_VLOG_LEVEL'] = '2'
        logging.set_verbosity(logging.ERROR)

    model_loss = 'mean_squared_error'
    model_metrics = ['mean_squared_error', tfidf_accuracy, tfidf_precision, tfidf_recall, tfidf_f1]
    model_save_file = config.TrainedModels.text2text

    X, packages = get_samples(final)

    keras.backend.clear_session()

    model_full, model_encoder = model_encoder_decoder(X.shape[1])

    keras.backend.get_session().run(tf.local_variables_initializer())

    model_full.compile(loss=model_loss,
                       optimizer='adam',
                       metrics=model_metrics)

    if verbose:
        # keras.utils.plot_model(model_full, "model.png", show_shapes=True)
        model_full.summary()

    callbacks = [
        keras.callbacks.EarlyStopping(monitor='val_tfidf_f1',
                                      mode="max",
                                      min_delta=0.005,
                                      patience=config.Clustering.early_stopping_patience,
                                      verbose=verbose),
        keras.callbacks.ModelCheckpoint(filepath=model_save_file,
                                        monitor='val_tfidf_f1',
                                        mode="max",
                                        save_best_only=True,
                                        verbose=verbose),
    ]

    if verbose:
        print("starting training")

    model_full.fit(x=X,
                   y=X,
                   batch_size=config.Clustering.batch_size,
                   validation_split=config.Clustering.validation_split,
                   epochs=config.Clustering.max_train_epochs,
                   shuffle=True,
                   callbacks=callbacks,
                   verbose=verbose)

    model_full.load_weights(model_save_file)

    if final:
        print("running TB callback on final model state")
        tb_callback = create_tensorboard_callback(X, packages,
                                                  config.Clustering.tensorboard_descriptions,
                                                  get_metadata,
                                                  'latent')
        model_full.fit(X, X, batch_size=128, callbacks=[tb_callback])

        model_encoder.load_weights(model_save_file, by_name=True)
        y_emb = model_encoder.predict(X)

        save_data = {
            'embeddings': y_emb.tolist(),
            'packages': packages
        }
        pickle.dump(save_data, open(config.Clustering.embeddings_descriptions, "wb"))

        print("saving embeddings")

        print("done.")
    else:
        X_val = X[:int(len(X)*config.Clustering.validation_split)]
        eval = model_full.evaluate(X_val, X_val, verbose=False, batch_size=128)
        eval = zip(model_full.metrics_names, eval)

        return eval


def get_metadata(package_name):
    db = SamplesDatabase.get()

    return {
        "title": db.read(package_name, 'title'),
        "category": str(db.read(package_name, 'category')),
        "description": db.read(package_name, "description_raw")[:300]
    }


def random_search():
    print("Text2Text random search started")

    layer_configs = random_encoder_archs(100)

    first = True

    for enc_config in layer_configs:
        model.layer_config = enc_config

        arch_performance = train(False, False)
        if first:
            header = ["arch"] + [name for name, _ in arch_performance]
            print(";".join(header))
            first = False

        arch_str = ",".join(["%d" % l for l in model.layer_config]) + "," + str(config.Clustering.latent_size_t2t)
        row = [arch_str] + ["%.7f" % value for _, value in arch_performance]
        print(";".join(row))


def final():
    model.layer_config = [1368, 970]
    train(True, True)


if __name__ == "__main__":
    #random_search()
    final()
