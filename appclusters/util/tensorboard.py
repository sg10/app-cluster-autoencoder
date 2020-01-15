import os
import shutil

from keras.callbacks import TensorBoard
import numpy as np

from appclusters import config
from appclusters.preprocessing.samples_database import SamplesDatabase


def create_tensorboard_callback(X, all_packages, log_dir, metadata_func, layer_name):
    shutil.rmtree(log_dir, ignore_errors=True)
    os.mkdir(log_dir)

    embeddings_data, embeddings_meta_filepath = assemble_samples(all_packages, X, metadata_func, log_dir)

    return TensorBoard(log_dir=log_dir,
                       embeddings_freq=1,
                       embeddings_layer_names=layer_name,
                       embeddings_metadata=embeddings_meta_filepath,
                       embeddings_data=embeddings_data)


def assemble_samples(all_packages, X, metadata_func, log_dir):
    db = SamplesDatabase.get()

    packages_to_visualize = db.filter(
        ('downloads', '>=', config.Clustering.min_downloads_visualize),
        ('lang', '==', 'en'))
    print("samples to be displayed: %d" % len(packages_to_visualize))

    samples_meta = []
    samples_data = []

    for package_name in packages_to_visualize:
        if package_name not in all_packages:
            continue
        meta = metadata_func(package_name)
        x = X[all_packages.index(package_name)]
        #meta['emb'] = "|".join(["%.2f" % v for v in x.tolist()])
        samples_meta.append(meta)
        samples_data.append(x)
        if len(samples_meta) > config.Clustering.max_visualize:
            break

    samples_data = np.asarray(samples_data)

    column_names = samples_meta[0].keys()
    samples_meta = [list(s.values()) for s in samples_meta]

    embeddings_meta_filepath = save_metadata_file(column_names, samples_meta, log_dir)

    return samples_data, embeddings_meta_filepath


def save_metadata_file(meta_columns, meta_samples, log_dir):
    embeddings_meta_filepath = os.path.join(log_dir, "metadata.tsv")

    with open(embeddings_meta_filepath, 'w', encoding='utf-8') as fp:
        fp.write("%s\n" % "\t".join(meta_columns))

        for meta_sample in meta_samples:
            fp.write("%s\n" % "\t".join(meta_sample))

    return embeddings_meta_filepath


