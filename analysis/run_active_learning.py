import argparse

import h5py
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from skactiveml.classifier import SklearnClassifier, ParzenWindowClassifier, MixtureModelClassifier
from skactiveml.pool import UncertaintySampling
from skactiveml.utils import MISSING_LABEL, labeled_indices, unlabeled_indices
from skactiveml.visualization import plot_decision_boundary, plot_utilities
from sklearn.decomposition import PCA
from sklearn.svm import SVC


def plot_trajectory(trajectory_file: str, hdf5_file: str):
    figure, ax = plt.subplots(1)

    with h5py.File(hdf5_file) as file:
        farray = file[trajectory_file][()]
    farray = np.transpose(farray, (1, 2, 0))
    if farray.shape[-1] == 2:
        farray = np.append(farray, np.zeros_like(farray)[:, :, 0:1], axis=-1)
    ax.imshow(farray)

    return figure


def main(embeddings_file: str, arrays_file: str):

    edf = pd.read_csv(embeddings_file, sep=";", index_col=0)
    print(edf.shape)
    df = edf.copy()

    embeddings_columns = [col for col in df.columns if "emb_dim" in col]

    pca_decomposer = PCA()
    pca_vectors = pca_decomposer.fit_transform(df.loc[:, embeddings_columns].values)

    xvalues = df.loc[:, embeddings_columns]
    # xvalues = pca_vectors[:, :3]
    centroid = np.expand_dims(xvalues.mean(axis=0), axis=0)

    distances = xvalues - centroid

    abs_dist = np.power(distances, 2).sum(axis=1)

    central_idx = abs_dist.argmin()
    outlier_idx = abs_dist.argmax()

    y = np.full(xvalues.shape[0], fill_value=MISSING_LABEL)
    y[central_idx] = 0
    y[outlier_idx] = 1

    clf = SklearnClassifier(
        SVC(
            probability=True,
            kernel="rbf",
            # C=30.0,
            gamma=0.03,
        ),
        classes=[0, 1],
    )
    print("Starting classifier...")
    # clf = ParzenWindowClassifier(classes=[0,1])
    # clf = MixtureModelClassifier()

    n_cycles = 20
    # n_cycles = 5
    # qs = UncertaintySampling(method="entropy", random_state=42)
    qs = UncertaintySampling(method="least_confident", random_state=42)
    print("First fit of classifier")
    clf.fit(xvalues, y)
    print("Finished fit")

    for cycle in range(n_cycles):
        # print("querying")
        query_idx = qs.query(X=xvalues, y=y, clf=clf, 
                             # batch_size=1
                             )
        # print("finished query")
        query_trajectory = df.loc[query_idx[0], "filename"]

        fig = plot_trajectory(query_trajectory, arrays_file)
        fig.show()

        new_classification = input(
            "Was this trajectory an outlier? 1 for outlier, 0 for regular:"
        )

        while new_classification not in ["0", "1"]:
            new_classification = input(
                "Invalid entry, please use 1 for outlier, 0 for regular, q to quit:"
            )
            if new_classification == "q":
                raise SystemExit()

        new_classification = int(new_classification)

        y[query_idx] = new_classification

        clf.fit(xvalues, y)
        plt.close()
        # plotting
        unlbld_idx = unlabeled_indices(y)
        lbld_idx = labeled_indices(y)

    prediction = clf.predict(xvalues)

    outlier_mask = prediction == 1
    highlight_points = pca_vectors[outlier_mask]
    dehighlight_points = pca_vectors[~outlier_mask]

    df["class"] = "regular"
    df.loc[outlier_mask, "class"] = "outlier"

    df.to_csv("exported/fitted_data.csv", sep=";")

    # xx, yy = np.meshgrid(np.linspace(-6, 3, 200), np.linspace(-.6, .5, 200))
    # Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
    # Z = Z.reshape(xx.shape)



    ax = plt.subplot()
    # ax.pcolormesh(xx, yy, -Z, cmap=plt.cm.RdBu)
    # plot_decision_boundary(clf, [[-6, -0.6], [2.5, 0.5], [-0.5, 0.5]], ax)

    ax.scatter(dehighlight_points[:, 0], dehighlight_points[:, 1], c="tab:blue", s=2)
    ax.scatter(highlight_points[:, 0], highlight_points[:, 1], c="tab:orange")
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--embeddings", help="Embeddings file", required=True)
    parser.add_argument(
        "-a", "--image-archive", help="Trajectory image archive (hdf5)", required=True
    )

    args = parser.parse_args()

    main(args.embeddings, args.image_archive)
