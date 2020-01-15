import os
import pickle
import pprint

import numpy as np
from collections import Counter
from matplotlib import cm
from matplotlib import pyplot as plt
from scipy.spatial import KDTree
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from appclusters import config
from appclusters.preprocessing.samples_database import SamplesDatabase
import json


class EmbeddingsVisualizer:

    def __init__(self):
        self.saved_embeddings = {}  # keys: embeddings, packages
        self.packages_to_plot = []
        self.category_colors = {}
        self.package_highlight_colors = {}
        self.all_categories = []
        self.categories_to_highlight = []
        self.X_reduced = []

    def load_embeddings(self, file):
        self.saved_embeddings = pickle.load(open(file, "rb"))

    def load_transformed(self, file):
        save_data = pickle.load(open(file, "rb"))
        (self.X_reduced,
         self.packages_to_plot) = save_data

    def save_transformed(self, file):
        save_data = (self.X_reduced,
                     self.packages_to_plot)
        pickle.dump(save_data, open(file, "wb"))

    def select_packages(self, min_download_count=4e6):
        db = SamplesDatabase.get()
        self.packages_to_plot = db.filter(('lang', '==', 'en'), ('downloads', '>=', min_download_count))
        self.packages_to_plot = set(self.packages_to_plot).intersection(self.saved_embeddings['packages'])
        self.packages_to_plot = list(self.packages_to_plot)

    def select_categories(self, categories_to_highlight):
        db = SamplesDatabase.get()

        self.all_categories = set(db.read(None, 'category'))
        pprint.pprint(self.all_categories)
        color_map = cm.get_cmap('hsv')
        self.categories_to_highlight = categories_to_highlight

        self.category_colors = {}
        for category in self.all_categories:
            if category in categories_to_highlight:
                i = categories_to_highlight.index(category)
                self.category_colors[category] = color_map(i / len(categories_to_highlight))
            else:
                self.category_colors[category] = [0.85, 0.85, 0.85, 1.0]

    def select_apps(self, packages_to_highlight):
        color_map = cm.get_cmap('hsv')

        self.package_highlight_colors = {}
        for package in self.packages_to_plot:
            if package in packages_to_highlight:
                i = packages_to_highlight.index(package)
                i = len(packages_to_highlight) - i
                self.package_highlight_colors[package] = color_map(i / len(packages_to_highlight))

    def transform(self, method, perplexity=None):
        X = [self.saved_embeddings['embeddings'][self.saved_embeddings['packages'].index(p)]
             for p in self.packages_to_plot]

        if method == "tsne":
            fitter = TSNE(n_components=2, perplexity=perplexity)
        else:
            fitter = PCA(n_components=2)

        self.X_reduced = fitter.fit_transform(X)

    def plot(self, title):
        db = SamplesDatabase.get()

        plt.figure()

        plot_x = self.X_reduced[:, 0]
        plot_y = self.X_reduced[:, 1]

        fig, ax = plt.subplots()
        if len(self.categories_to_highlight) > 0:
            plot_categories = np.array([db.read(p, 'category') for p in self.packages_to_plot])

            for category in self.all_categories:
                idx_category = np.argwhere(plot_categories == category)
                color = self.category_colors[category]
                category_name = category if category in self.categories_to_highlight else None
                ax.scatter(plot_x[idx_category], plot_y[idx_category],
                           c=[color] * len(idx_category),
                           label=category_name,
                           s=2)
        else:
            color = [0.85, 0.85, 0.85, 1.0]
            ax.scatter(plot_x, plot_y,
                       c=[color] * len(self.packages_to_plot),
                       label="Other",
                       s=2)

            for package, color in self.package_highlight_colors.items():
                i = self.packages_to_plot.index(package)
                plot_x_1 = self.X_reduced[i:i + 1, 0]
                plot_y_1 = self.X_reduced[i:i + 1, 1]
                label = "%s (%d/%d)" % (package, plot_x_1, plot_y_1)
                ax.scatter(plot_x_1, plot_y_1,
                           c=[color],
                           label=label,
                           s=2)

        plt.legend(loc='upper center',
                   bbox_to_anchor=(0.5, -0.05),
                   markerscale=4,
                   prop={'size': 9},
                   ncol=2)

        plt.title(title)

        plt.show()

    def closest(self, num, packages):
        db = SamplesDatabase.get()
        tree = KDTree(self.X_reduced)
        for package in packages:
            perm_common = []
            print("-- ", db.read(package, "title"))
            vec = self.X_reduced[self.packages_to_plot.index(package)]
            ids = tree.query(vec, k=num)
            for dist, id in zip(ids[0].tolist(), ids[1]):
                if self.packages_to_plot[id] == package: continue
                title = db.read(self.packages_to_plot[id], 'title')
                print("%5d   %s (%.4f)" % (id, title, dist))
                perm_common += db.read(self.packages_to_plot[id], "permissions")
            cnt = Counter(perm_common).items()
            cnt = [(c, v) for c, v in cnt if v < 9]
            print(sorted(cnt, key=lambda k: k[1], reverse=True)[:6])

    def latex_data_export(self, folder, filename):
        file_data = ["x0\ty0\tlabel\n"]

        for x, y, package in zip(self.X_reduced[:, 0].tolist(),
                                 self.X_reduced[:, 1].tolist(),
                                 self.packages_to_plot):
            if len(self.categories_to_highlight) > 0:
                category = SamplesDatabase.get().read(package, "category")
                if category not in self.categories_to_highlight:
                    label = "other"
                else:
                    label = category.replace("_", "")
            else:
                if package in list(self.package_highlight_colors.keys()):
                    label = package.replace(".", "")
                else:
                    label = "other"

            file_data.append("%.2f\t%.2f\t%s\n" % (x, y, label))

        file = os.path.join(folder, filename)
        open(file, "w").writelines(file_data)


def analyze_descriptions():
    generate = True
    num_closest = 10

    tasks = [
        (config.Clustering.embeddings_descriptions,
         45,
         "desc-tsne1.dat",
         "desc-pca1.dat",
         config.Clustering.tsne_descriptions,
         config.Clustering.pca_descriptions,
         ),
    ]

    for emb, perplexity, tsne_latex_file, pca_latex_file, tsne_saved, pca_saved in tasks:
        vis = EmbeddingsVisualizer()
        vis.load_embeddings(emb)

        vis.select_packages(4e6)
        vis.select_categories(['WEATHER', 'MEDIA_AND_VIDEO', 'COMMUNICATION'])

        if generate:
            vis.transform('tsne', perplexity)
            vis.save_transformed(tsne_saved)

            vis.transform('pca')
            vis.save_transformed(pca_saved)

        packages = ['com.whatsapp', 'com.opera.browser', 'com.antivirus']

        print("t-SNE")
        vis.load_transformed(tsne_saved)
        vis.closest(num_closest, packages)
        vis.latex_data_export(config.Clustering.latex_out_folder, tsne_latex_file)
        vis.plot("t-SNE (P %d)" % perplexity)

        print("PCA")
        vis.load_transformed(pca_saved)
        vis.closest(num_closest, packages)
        vis.latex_data_export(config.Clustering.latex_out_folder, pca_latex_file)
        vis.plot("PCA")


def analyze_permissions():
    generate = False
    num_closest = 10

    t = (config.Clustering.embeddings_permissions,
         35,
         "perm-tsne-%d.dat",
         config.Clustering.tsne_permissions
         )
    tasks = [t]
    #for i in range(10, 80, 5):
    #   t1 = (t[0], i, t[2], t[3])
    #   tasks.append(t1)

    for emb, perplexity, tsne_latex_file, tsne_saved in tasks:
        group_packages = [
            [
                'com.kms.free',
                'and.anti',
                'com.avast.android.backup',
                'com.avast.android.mobilesecurity',
                'com.psafe.msuite',
                'com.avira.android',
                'com.qihoo.security',
                'com.nqmobile.antivirus20',
                'com.symantec.mobilesecurity',
                'com.cleanmaster.security',
                'com.qihoo360.mobilesafe',
                'com.antivirus.tablet',
                'com.trustgo.mobile.security',
                'com.antivirus',
                'com.wsandroid.suite',
                'com.iobit.mobilecare',
                'com.estsoft.alyac',
                'com.nqmobile.antivirus20.multilang',
                'com.lookout',
                'com.zrgiu.antivirus',
                'com.drweb',
                'com.drweb.pro'
            ], [
                 'com.opera.browser',
                 'com.opera.browser.classic',
                 'com.ksmobile.cb',
                 'com.appsverse.photon',
                 'com.uc.browser.en',
                 'com.cloudmosa.puffinFree',
                 'com.boatbrowser.free',
                 'com.baidu.browser.inter',
                 'org.mozilla.firefox',
                 'com.jiubang.browser',
                 'com.android.chrome',
                 'com.yandex.browser',
                 'com.uc.browser.hd',
                 'com.opera.mini.android',
                 'com.mx.browser',
                 'mobi.mgeek.TunnyBrowser',
                 'com.UCMobile.intl'
            ]
        ]

        #group_packages.pop(0)

        for i_package, packages in enumerate(group_packages):

            for p in packages:
                name = SamplesDatabase.get().read(p, "title")
                j = json.load(open("/data/samples/metadata/%s.json" % p, "r", encoding="utf-8"))
                version = j['details']['app_details']['version_string']
                dev = j['details']['app_details']['developer_name']
                print("%s (%s) & %s, %s \\\\" % (name, dev, p, version))
            continue

            vis = EmbeddingsVisualizer()
            vis.load_embeddings(emb)

            vis.select_packages(4e6)

            if True:
                db = SamplesDatabase.get()
                apps = {}
                perms = set()
                for p in packages:
                    apps[p] = db.read(p, 'permissions')
                    perms.update(set(apps[p]))

                print(";", ";".join(perms))
                for p in packages:
                    perms1 = ";".join(["x" if pr in apps[p] else "" for pr in perms])
                    print("%s;%s;" % (p, perms1))

                continue

            vis.select_apps(packages)

            if generate:
                vis.transform('tsne', perplexity)
                vis.save_transformed(tsne_saved)

            print("t-SNE 1")
            vis.load_transformed(tsne_saved)
            vis.closest(num_closest, packages)
            vis.latex_data_export(config.Clustering.latex_out_folder, tsne_latex_file % i_package)
            vis.plot("t-SNE (P %d)" % perplexity)


if __name__ == "__main__":
    analyze_permissions()
    # analyze_descriptions()
