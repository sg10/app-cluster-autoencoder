data_folder = "/data"


class Samples:

    # contains subfolders (categories)
    app_metadata = data_folder + "/samples/metadata/"

    samples_database = data_folder + "/samples/samples_database.pk"
    list_of_cross_platform_apps = data_folder + "/samples/cross_platform_apps.txt"

    # 50% most downloaded apps, 50% least downloaded apps
    # only english
    num_for_test_set = 1000


class Clustering:

    max_visualize = 5000
    min_downloads_visualize = 4e6
    tensorboard_permissions = data_folder + "/tensorboard/permissions"
    tensorboard_descriptions = data_folder + "/tensorboard/descriptions"
    max_train_epochs = 550
    validation_split = 0.25
    min_permission_occur = 10

    early_stopping_patience = 10

    batch_size = 32

    latex_out_folder = data_folder + "/clustering/latex/"
    embeddings_permissions = data_folder + "/clustering/permissions.pk"
    embeddings_descriptions = data_folder + "/clustering/descriptions.pk"
    tsne_permissions = data_folder + "/clustering/permissions_tsne.pk"
    pca_permissions = data_folder + "/clustering/permissions_pca.pk"
    tsne_descriptions = data_folder + "/clustering/descriptions_tsne.pk"
    pca_descriptions = data_folder + "/clustering/descriptions_pca.pk"

    encoder_depth = (1, 8)
    hidden_neurons_fraction = (0.3, 1.0)
    latent_size_p2p = 5
    latent_size_t2t = 10


class Permissions:

    groups_list = data_folder + "/permission_groups.json"


class TrainedModels:

    models_dir = data_folder + "/trained_models/"

    permission2permission = models_dir + "permission2permission.h5"
    text2text = models_dir + "text2text.h5"


class TFIDFModels:

    tfidf_folder = data_folder + "/tfidfs/"

    min_terms_per_doc = 5

    description_model = tfidf_folder + "descriptions-model.pk"
    description_data = tfidf_folder + "descriptions-data.pk"

    description_model_2 = tfidf_folder + "descriptions-model-2.pk"
    description_data_2 = tfidf_folder + "descriptions-data-2.pk"


class Stemming:

    unstem_file = data_folder + '/unstem.pk'
