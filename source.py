import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from itertools import product
from random import choices
from sklearn.manifold import TSNE
from sklearn.decomposition import TruncatedSVD
from sklearn.decomposition import PCA
from matplotlib.colors import Normalize
from sklearn.metrics import rand_score, homogeneity_score, completeness_score, v_measure_score, fowlkes_mallows_score, \
    silhouette_score
from sklearn.cluster import KMeans


def show(images: np.ndarray):
    for i in range(len(images)):
        plt.subplot(1, len(images), i + 1)
        plt.imshow(images[i].reshape(28, 28), cmap=plt.cm.gray_r)
        plt.gca().axes.get_yaxis().set_visible(False)
        plt.gca().axes.get_xaxis().set_visible(False)
    plt.show()


def random_voice_attack(dataset: np.ndarray, labels: np.ndarray, attacked_label: int, new_label: int, count: int,
                        voiced_pixels: int = 20):
    im_shape = int(dataset[0].shape[0] ** 0.5)
    pixels = list(product(range(im_shape), repeat=2))
    random_voice_pixels = np.array(choices(pixels, k=voiced_pixels))
    random_voice_values = np.random.randint(256, size=voiced_pixels)
    new_dataset = dataset.copy()
    new_labels = labels.copy()
    for i in range(len(dataset)):
        if count == 0:
            return new_dataset, new_labels
        if labels[i] == attacked_label:
            count -= 1
            new_labels[i] = new_label
            for j in range(voiced_pixels):
                x, y = random_voice_pixels[j]
                new_dataset[i, y * im_shape + x] = random_voice_values[j]
    return new_dataset, new_labels


def clusters_plot(x_2d: np.ndarray, labels: np.ndarray, title: str):
    plt.figure(figsize=(16, 9))
    for i in range(10):
        norm = Normalize(vmin=0, vmax=10)
        color = plt.cm.gist_ncar(norm(i))
        plt.scatter(x_2d[:, 0][labels == i],
                    x_2d[:, 1][labels == i],
                    c=[color] * len(x_2d[:, 0][labels == i]), label=i, edgecolor='none', s=50)
    plt.title(title, fontsize=20)
    plt.legend()
    plt.show()


def make_balanced_selection(x: np.ndarray, y: np.ndarray, size_of_each_class):
    sizes = {}
    for i in set(y):
        sizes[i] = 0
    new_x = np.array([])
    new_y = np.array([])
    for i in range(len(y)):
        if sizes[y[i]] < size_of_each_class:
            sizes[y[i]] += 1
            new_x = np.append(new_x, x[i])
            new_y = np.append(new_y, y[i])
    new_x = new_x.reshape(sum(sizes.values()), x.shape[1])
    return new_x, new_y


def get_pca(n, x):
    pca = PCA(n_components=n)
    return pca.fit(x).transform(x)


def get_svd(n, x):
    svd = TruncatedSVD(n_components=n)
    return svd.fit_transform(x)


def get_tsne(n, x):
    tsne = TSNE(n_components=n, n_iter=400)
    return tsne.fit_transform(x)


def count_metrics(x: np.ndarray, y: np.ndarray):
    statistic = pd.DataFrame(
        columns=["rand_score", "homogeneity_score", "completeness_score", "v_measure_score", "fowlkes_mallows_score",
                 "silhouette_score"],
        index=["pca2", "pca3", "svd2", "svd3", "tsne2", "tsne3"])
    all_data = {"pca2": get_pca(2, x), "pca3": get_pca(3, x), "svd2": get_svd(2, x), "svd3": get_svd(3, x),
                "tsne2": get_tsne(2, x), "tsne3": get_tsne(3, x)}
    for i in all_data.keys():
        model = KMeans(n_clusters=len(set(y))).fit(all_data[i])
        y_true, y_pred = y, model.labels_
        metrics = {"rand_score": rand_score(y_true, y_pred),
                   "homogeneity_score": homogeneity_score(y_true, y_pred),
                   "completeness_score": completeness_score(y_true, y_pred),
                   "v_measure_score": v_measure_score(y_true, y_pred),
                   "fowlkes_mallows_score": fowlkes_mallows_score(y_true, y_pred),
                   "silhouette_score": silhouette_score(x, y_pred)}
        statistic.loc[i] = metrics
    return statistic
