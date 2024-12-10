import argparse
import pickle
from io import StringIO
from time import time
from typing import Dict

import h5py
import numpy as np
import pandas as pd
import plotly.express as px
from dash import Dash, Input, Output, callback, callback_context, dcc, html, no_update
from plotly.subplots import make_subplots
from skactiveml.classifier import SklearnClassifier
from skactiveml.pool import UncertaintySampling
from skactiveml.utils import MISSING_LABEL, labeled_indices, unlabeled_indices
from sklearn.decomposition import PCA
from sklearn.svm import SVC

parser = argparse.ArgumentParser()
parser.add_argument("-e", "--embeddings", help="Embeddings file", required=True)
parser.add_argument(
    "-a", "--image-archive", help="Trajectory image archive (hdf5)", required=True
)

args = parser.parse_args()
embeddings_file = args.embeddings
arrays_file = args.image_archive

filecolumn = "filename"
ENCODING = "utf-16-le"


def _no_matchin_data_message():
    return {
        "layout": {
            "xaxis": {"visible": False},
            "yaxis": {"visible": False},
            "annotations": [
                {
                    "text": "No matching data",
                    "xref": "paper",
                    "yref": "paper",
                    "showarrow": False,
                    "font": {"size": 28},
                }
            ],
        }
    }


def show_hdf5_image(filename):
    with h5py.File(arrays_file) as file:
        farray = file[filename][()]
    farray = np.transpose(farray, (1, 2, 0))
    while farray.shape[-1] < 3:
        farray = np.append(farray, np.zeros_like(farray)[:, :, 0:1], axis=-1)
    return px.imshow(farray)


external_stylesheets = ["https://codepen.io/chriddyp/pen/bWLwgP.css"]

app = Dash(__name__, external_stylesheets=external_stylesheets)

app.layout = html.Div(
    [
        html.Div(
            [
                dcc.Graph(
                    id="trajectories-scatter",
                    style={
                        "width": "100vh",
                        "height": "100vh",
                    },
                )
            ],
            style={
                "width": "60%",
                "display": "inline-block",
                "padding": "0 20",
            },
            id="scatter-plot-container",
        ),
        html.Div(
            [
                dcc.Graph(
                    id="show-trajectory",
                    style={
                        "height": "40%",
                    },
                ),
                html.Div(
                    [
                        dcc.RadioItems(
                            id="label-selector",
                            options=[
                                {"label": "None", "value": -1},
                                {"label": "Regular", "value": 0},
                                {"label": "Outlier", "value": 1},
                            ],
                            inline=True,
                        )
                    ],
                    id="label-prediction-container",
                ),
                dcc.Graph(
                    id="show-related-images",
                    style={
                        "height": "60%",
                    },
                ),
            ],
            style={"display": "inline-block", "vertical-align": "top", "width": "40%"},
        ),
        html.Div(
            [
                html.Button("Like!", id="like-button"),
                dcc.Store(id="stored-data", data={"last-clicked-time": time()}),
                dcc.Store(id="raw-pca-data"),
                dcc.Store(id="fitted-data"),
                dcc.Store(id="x-values"),
                dcc.Store(id="y-labeled"),
                dcc.Store(id="y-predicted"),
            ]
        ),
    ]
)


@callback(
    Output("trajectories-scatter", "clickData"),
    Output("stored-data", "data"),
    Input("stored-data", "data"),
    Input("scatter-plot-container", "n_clicks"),
    Input("trajectories-scatter", "clickData"),
    prevent_initial_call=True,
)
def update_stored_data(stored_data, n_clicks, clicked_data):
    changed_inputs = [x["prop_id"] for x in callback_context.triggered]
    if "scatter-plot-container.n_clicks" in changed_inputs:
        now = time()
        if now - stored_data["last-clicked-time"] < 0.3:
            return None, {"last-clicked-time": now}
        else:
            return no_update, {"last-clicked-time": now}
    else:
        return no_update, no_update


@callback(
    Output("show-trajectory", "figure"),
    Input("trajectories-scatter", "hoverData"),
    Input("trajectories-scatter", "clickData"),
    prevent_initial_call=True,
)
def render_trajectory_image(hoverData: Dict, clickData):
    if clickData is not None:
        filename = clickData["points"][0]["customdata"][0]
    else:
        filename = hoverData["points"][0]["customdata"][0]
    fig = show_hdf5_image(filename)
    fig.update_layout(title=filename.split("/")[-1], margin={"l": 0, "b": 0, "r": 0})
    return fig


@callback(
    Output("trajectories-scatter", "figure"),
    Input("raw-pca-data", "data"),
    Input("y-predicted", "data"),
)
def plot_trajectory_points(plot_json, predicted_labels):
    plot_df = pd.read_json(StringIO(plot_json))
    plot_df = plot_df.drop(columns="class")

    plabels = pd.read_json(StringIO(predicted_labels))

    plot_df = pd.merge(plot_df, plabels, on=filecolumn)
    plot_df[filecolumn] = plot_df[filecolumn].apply(lambda x: x[5:])
    color_column = "class"
    if len(plot_df[color_column].unique() == 1):
        color_column = None

    fig = px.scatter_3d(
        data_frame=plot_df,
        x="xcol",
        y="ycol",
        z="zcol",
        custom_data=filecolumn,
        size="ms",
        opacity=0.5,
        color=color_column,
    )
    fig.update_layout(margin={"l": 0, "b": 0, "t": 0, "r": 0}, hovermode="closest")
    return fig


@callback(
    Output("show-related-images", "figure"),
    Input("trajectories-scatter", "clickData"),
    Input("raw-pca-data", "data"),
)
def update_related_images(clickData, plot_json):
    if clickData is None:
        return _no_matchin_data_message()

    plot_df = pd.read_json(StringIO(plot_json))

    # in the dataframe we prepend the filename with "file_" to prevent
    # converting to float here we remove those first 5 characters again
    plot_df[filecolumn] = plot_df[filecolumn].apply(lambda x: x[5:])

    click_dict = clickData["points"][0]

    distances = np.zeros_like(plot_df["xcol"].values)
    for axis in ["x", "y", "z"]:
        distances += (plot_df[f"{axis}col"].values - click_dict[axis]) ** 2

    rown_nr_sorted_by_distance = np.argsort(distances)

    rows, cols = 3, 3
    fig = make_subplots(
        rows=rows, cols=cols, horizontal_spacing=0.1, vertical_spacing=0.01
    )
    for i in range(min(rows * cols, len(rown_nr_sorted_by_distance) - 1)):
        plot_row = i // cols + 1
        plot_col = i % cols + 1
        df_index = rown_nr_sorted_by_distance[i + 1]
        filename = plot_df.iloc[df_index].loc[filecolumn]
        distance = distances[df_index]

        fig.add_trace(show_hdf5_image(filename).data[0], row=plot_row, col=plot_col)

    fig.update_layout(
        {
            ax: {"visible": False, "matches": None}
            for ax in fig.to_dict()["layout"]
            if "axis" in ax
        }
    )
    fig.update_layout(margin={"l": 0, "b": 0, "t": 50, "r": 0})
    return fig


@callback(
    Output("raw-pca-data", "data"),
    Output("fitted-data", "data"),
    Input("like-button", "n_clicks"),
)
def update_raw_pca_data(n_clicks):
    edf = pd.read_csv(embeddings_file, sep=";", index_col=0)
    df = edf.copy()

    embeddings_columns = [col for col in df.columns if "emb_dim" in col]

    pca_decomposer = PCA()
    pca_vectors = pca_decomposer.fit_transform(df.loc[:, embeddings_columns].values)

    xvalues = df.loc[:, embeddings_columns]
    xvalues = pca_vectors[:, :3]

    filecolumn = "filename"
    # plot_df = df[[filecolumn]].copy()
    plot_df = df[[filecolumn, "class"]].copy()

    plot_df[filecolumn] = plot_df[filecolumn].apply(lambda x: f"file_{x}")
    plot_df["xcol"] = xvalues[:, 0]
    plot_df["ycol"] = xvalues[:, 1]
    plot_df["zcol"] = xvalues[:, 2]
    plot_df["ms"] = 10

    plot_df["class"] = plot_df["class"].apply(
        lambda x: {"regular": "normal", "outlier": "outlier"}[x]
    )

    df[filecolumn] = df[filecolumn].apply(lambda x: f"file_{x}")

    return plot_df.to_json(), df.to_json()


@callback(
    Output("x-values", "data"),
    Output("y-labeled", "data", allow_duplicate=True),
    Output("y-predicted", "data"),
    Input("fitted-data", "data"),
    prevent_initial_call="initial_duplicate",
)
def set_initial_xy_values(dataf):
    df = pd.read_json(StringIO(dataf))

    embeddings_columns = [col for col in df.columns if "emb_dim" in col]

    x_values = df.loc[
        :,
        [
            filecolumn,
        ]
        + embeddings_columns,
    ]

    y_labeled = df.loc[:, [filecolumn]].copy()
    y_labeled["label"] = -1

    # y_labeled = np.full(x_values.shape[0], fill_value=-1)
    # print(y_labeled[0:5])
    # y_labeled_bytestring = y_labeled.tobytes().decode(encoding=ENCODING)

    y_prediction = df.loc[:, [filecolumn]].copy()
    y_prediction["class"] = 0

    return x_values.to_json(), y_labeled.to_json(), y_prediction.to_json()


@callback(
    Output("label-selector", "value"),
    Output("label-selector", "style"),
    Input("trajectories-scatter", "clickData"),
    Input("y-labeled", "data"),
)
def display_label_container(click_data, labels):
    if click_data is None:
        return -1, {"visibility": "hidden"}

    trajectory_id = f"file_{click_data["points"][0]["customdata"][0]}"

    labels = pd.read_json(StringIO(labels))

    label = labels[labels[filecolumn] == trajectory_id]["label"].iloc[0]
    return label, {"visibility": "visible"}


@callback(
    Output("y-labeled", "data", allow_duplicate=True),
    Input("y-labeled", "data"),
    Input("trajectories-scatter", "clickData"),
    Input("label-selector", "value"),
    prevent_initial_call="initial_duplicate",
)
def update_label(all_labels, click_data, label):
    if click_data is None:
        return no_update

    trajectory_id = f"file_{click_data["points"][0]["customdata"][0]}"
    labels = pd.read_json(StringIO(all_labels))
    labels.loc[labels[filecolumn] == trajectory_id, "label"] = label

    return labels.to_json()


@callback(Input("y-labeled", "data"))
def print_labels(labels):
    ldf = pd.read_json(StringIO(labels))
    # ldf = np.frombuffer(labels.encode(encoding=ENCODING))
    # print(ldf[0:10])


if __name__ == "__main__":
    app.run(debug=True)
#     parser = argparse.ArgumentParser()
#     parser.add_argument("-e", "--embeddings", help="Embeddings file", required=True)
#     parser.add_argument(
#         "-a", "--image-archive", help="Trajectory image archive (hdf5)", required=True
#     )
#
#     args = parser.parse_args()
#
#     main(args.embeddings, args.image_archive)
