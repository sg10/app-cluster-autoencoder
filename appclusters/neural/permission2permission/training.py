import os
import pickle
from collections import Counter

import keras
import numpy as np
from tensorflow._api.v1 import logging

from appclusters import config
from appclusters.neural.permission2permission import model
from appclusters.neural.permission2permission.model import model_encoder_decoder
from appclusters.preprocessing.permissionparser import PermissionParser
from appclusters.preprocessing.samples_database import SamplesDatabase
from appclusters.util.metrics import recall, precision, get_fbeta_micro
from appclusters.util.random_search import random_encoder_archs
from appclusters.util.tensorboard import create_tensorboard_callback


def get_samples(final):
    db = SamplesDatabase.get()
    all_permissions = db.read(None, 'permissions')
    all_permissions = [p for sublist in all_permissions for p in sublist]  # flatten
    permission_names = [name for name, count in Counter(all_permissions).items()
                        if count > config.Clustering.min_permission_occur]

    permission_parser = PermissionParser(mode='single', selected=permission_names)

    all_packages = db.filter(('permissions', 'len>', 0))

    permissions_matrix = np.empty((len(all_packages), len(permission_names)))

    i = 0
    packages_selected = []
    for package in all_packages:
        onehot_enc = permission_parser.transform(db.read(package, 'permissions'))
        if np.sum(onehot_enc) > 0:
            permissions_matrix[i] = onehot_enc
            packages_selected.append(package)
            i += 1

    permissions_matrix = permissions_matrix[:len(packages_selected), :]

    return permission_parser, permissions_matrix, packages_selected


def train(verbose=True, tensor_board=True, final=True):
    if not verbose:
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
        os.environ['TF_CPP_MIN_VLOG_LEVEL'] = '2'
        logging.set_verbosity(logging.ERROR)

    model_loss = 'binary_crossentropy'
    model_metrics = ['binary_accuracy', 'mean_squared_error', precision, recall, get_fbeta_micro()]

    model_save_file = config.TrainedModels.permission2permission

    permission_parser, X, packages = get_samples(final)

    keras.backend.clear_session()

    model_full, model_encoder = model_encoder_decoder(permission_parser.count())

    model_full.compile(loss=model_loss,
                       optimizer='adam',
                       metrics=model_metrics)

    # keras.utils.plot_model(model_full, "model.png", show_shapes=True)
    if verbose:
        model_full.summary()

    callbacks = [
        keras.callbacks.EarlyStopping(monitor='val_fbeta_micro',
                                      mode="max",
                                      patience=config.Clustering.early_stopping_patience,
                                      verbose=verbose),
        keras.callbacks.ModelCheckpoint(filepath=model_save_file,
                                        monitor='val_fbeta_micro',
                                        mode="max",
                                        save_best_only=True,
                                        verbose=verbose)
        ]
    if tensor_board:
        callbacks.append(
            create_tensorboard_callback(X, packages,
                                        config.Clustering.tensorboard_permissions,
                                        get_metadata,
                                        'latent')
        )

    if verbose:
        print("starting training")

    class_weight = np.sum(X, axis=0)
    class_weight = np.ones(class_weight.shape) * np.max(class_weight) - class_weight

    # model_full.load_weights(model_save_file, by_name=True)
    # for _ in range(15):
    #     p = random.choice(packages)
    #     print("#  ", p)
    #     x = X[[packages.index(p)]]
    #     y = model_full.predict(x)
    #     idx_actual = np.argwhere(x[0] > 0).flatten().tolist()
    #     idx_target = np.argwhere(y[0] > 0.5).flatten().tolist()
    #     for i in set(idx_actual).union(set(idx_target)):
    #         f = "ok" if i in idx_actual and i in idx_target else ""
    #         print("%3d  %5d    %.4f   %s" % (i, x[0, i], y[0, i], f))
    #     print()
    #
    # return
    model_full.fit(x=X,
                   y=X,
                   batch_size=config.Clustering.batch_size,
                   epochs=config.Clustering.max_train_epochs,
                   shuffle=True,
                   #class_weight=class_weight,
                   callbacks=callbacks,
                   validation_split=config.Clustering.validation_split,
                   verbose=verbose)

    if final:
        print("running TB callback on final model state")
        tb_callback = create_tensorboard_callback(X, packages,
                                                  config.Clustering.tensorboard_permissions,
                                                  get_metadata,
                                                  'latent')
        model_full.fit(X, X, batch_size=config.Clustering.batch_size, callbacks=[tb_callback])

        model_encoder.load_weights(model_save_file, by_name=True)
        y_emb = model_encoder.predict(X)

        save_data = {
            'embeddings': y_emb.tolist(),
            'packages': packages
        }
        pickle.dump(save_data, open(config.Clustering.embeddings_permissions, "wb"))

        print("saving embeddings")

        print("done.")
    else:
        model_full.load_weights(model_save_file)
        X_val = X[:int(len(X)*config.Clustering.validation_split)]
        eval = model_full.evaluate(X_val, X_val, verbose=False, batch_size=128)

        eval = zip(model_full.metrics_names, eval)

        return eval


def get_metadata(package_name):
    db = SamplesDatabase.get()

    permissions = db.read(package_name, 'permissions')
    permissions_short = [p.split(".")[-1] for p in permissions][:15]
    permissions_str = ", ".join(permissions_short)

    return {
        "title": db.read(package_name, 'title'),
        "category": str(db.read(package_name, 'category')),
        "package": package_name,
        "permissions": permissions_str,
        "description": db.read(package_name, "description_raw")[:100],
    }


def final():
    model.layer_config = [210, 196]
    train(True, True, True)


def random_search():
    print("Permission2Permission random search started")
    layer_configs = random_encoder_archs(101)
    first = True
    for enc_config in layer_configs:
        model.layer_config = enc_config

        arch_performance = train(False, False, False)
        if first:
            header = ["arch"] + [name for name, _ in arch_performance]
            print(";".join(header))
            first = False

        arch_str = ",".join(["%d" % l for l in model.layer_config]) + "," + str(config.Clustering.latent_size_p2p)
        row = [arch_str] + ["%.7f" % value for _, value in arch_performance]
        print(";".join(row))


if __name__ == "__main__":
    #random_search()
    final()


