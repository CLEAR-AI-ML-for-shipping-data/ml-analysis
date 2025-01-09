import argparse
import json
import pickle
import sys
from io import StringIO
from time import time
from typing import Dict

import h5py
import numpy as np
import pandas as pd
import plotly.express as px
from dash import Dash, Input, Output, callback, callback_context, dcc, html, no_update
from dash_bootstrap_components import Popover, PopoverBody
from loguru import logger
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


def _no_trajectory_selected_message():
    return {
        "layout": {
            "xaxis": {"visible": False},
            "yaxis": {"visible": False},
            "annotations": [
                {
                    "text": "No point selected",
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
                        ),
                        html.Button("Least certain trajectory?", id="query-model"),
                        html.Div(
                            [
                                html.H6("Gamma", id="svm-gamma-param-title"),
                                dcc.Input(
                                    id="svm-gamma-param",
                                    type="number",
                                    value=0.03,
                                    min=0,
                                ),
                                Popover(
                                    [
                                        "Gamma defines the reach of a single training example. High gamma --> far reaching influence. (see ",
                                        html.A(
                                            "the SKLearn docs",
                                            href="https://scikit-learn.org/stable/auto_examples/svm/plot_rbf_parameters.html",
                                            target="_blank",
                                        ),
                                        ")",
                                    ],
                                    target="svm-gamma-param-title",
                                    id="gamma-popover",
                                    trigger="click",
                                    hide_arrow=False,
                                    placement="top",
                                ),
                            ],
                            style={
                                "display": "inline-block",
                                "text-align": "center",
                                "padding": "5pt",
                            },
                        ),
                        html.Div(
                            [
                                html.H6("C", id="svm-C-param-title"),
                                dcc.Input(
                                    id="svm-C-param", type="number", value=1.0, min=0
                                ),
                                Popover(
                                    [
                                        "C regularizes SVM decision function. Higher C --> more complex decision function. (see ",
                                        html.A(
                                            "the SKLearn docs",
                                            href="https://scikit-learn.org/stable/auto_examples/svm/plot_rbf_parameters.html",
                                            target="_blank",
                                        ),
                                        ")",
                                    ],
                                    target="svm-C-param-title",
                                    id="C-popover",
                                    trigger="click",
                                    hide_arrow=False,
                                    placement="top",
                                ),
                            ],
                            style={"display": "inline-block", "text-align": "center"},
                        ),
                    ],
                    id="label-prediction-container",
                    style={"inline": "true"},
                ),
                dcc.Graph(
                    id="show-related-images",
                    style={
                        "height": "60%",
                    },
                ),
                html.Button("Download Excel", id="btn-download-excel"),
                html.Button("Download CSV", id="btn-download-csv"),
                dcc.Download(id="download-data"),
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
                dcc.Store(id="selected-data-point"),
                dcc.Store(id="queried-data-point"),
                dcc.Store(id="svc-model"),
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
    # Input("trajectories-scatter", "clickData"),
    Input("selected-data-point", "data"),
    prevent_initial_call=True,
)
def render_trajectory_image(hoverData: Dict, clickData):
    if clickData is not None:
        filename = clickData["points"][0]["customdata"][0]
    elif hoverData is not None:
        filename = hoverData["points"][0]["customdata"][0]
    else:
        return _no_trajectory_selected_message()
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

    if predicted_labels is not None:
        plabels = pd.read_json(StringIO(predicted_labels))
        plot_df = plot_df.drop(columns="class")

        plot_df = pd.merge(plot_df, plabels, on=filecolumn)

    plot_df[filecolumn] = plot_df[filecolumn].apply(lambda x: x[5:])
    color_column = "class"

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
    Output("y-predicted", "data", allow_duplicate=True),
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

    y_prediction = df.loc[:, [filecolumn]].copy()
    y_prediction["class"] = "Regular"

    return x_values.to_json(), y_labeled.to_json(), y_prediction.to_json()


@callback(
    Output("label-selector", "value"),
    Output("label-selector", "style"),
    # Input("trajectories-scatter", "clickData"),
    Input("selected-data-point", "data"),
    Input("y-labeled", "data"),
)
def display_label_container(click_data, labels):
    if click_data is None:
        return -1, {"visibility": "hidden"}

    trajectory_id = f"file_{click_data["points"][0]["customdata"][0]}"
    logger.debug({f"Selected trajectory {trajectory_id}"})

    labels = pd.read_json(StringIO(labels))
    label = labels[labels[filecolumn] == trajectory_id]["label"].iloc[0]

    logger.debug(f"Current label: {label}")
    return label, {"visibility": "visible"}


@callback(
    Output("y-labeled", "data", allow_duplicate=True),
    Input("y-labeled", "data"),
    # Input("trajectories-scatter", "clickData"),
    Input("selected-data-point", "data"),
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
    ldf = pd.read_json(StringIO(labels))["label"].values
    logger.debug(f"Currently labeled values: {ldf[ldf >= 0]}")


@callback(
    Output("queried-data-point", "data", allow_duplicate=True),
    Output("svc-model", "data"),
    Output("y-predicted", "data", allow_duplicate=True),
    Input("query-model", "n_clicks"),
    Input("x-values", "data"),
    Input("y-labeled", "data"),
    Input("raw-pca-data", "data"),
    Input("svc-model", "data"),
    Input("svm-C-param", "value"),
    Input("svm-gamma-param", "value"),
    prevent_initial_call="initial_duplicate",
)
def query_model(button_click, x_values, y_labels, pca_data, model, svm_C, svm_gamma):
    if callback_context.triggered_id != "query-model":
        return no_update, no_update, no_update

    if model is not None:
        clf: SklearnClassifier = pickle.loads(bytes.fromhex(model))
    else:
        clf = SklearnClassifier(
            SVC(probability=True, kernel="rbf"), classes=[0, 1], missing_label=-1
        )
    logger.debug(f"Using model of type {type(clf)}")

    clf.estimator.set_params(C=svm_C, gamma=svm_gamma)

    x_values = pd.read_json(StringIO(x_values))
    files = x_values[[filecolumn]].copy()

    x_values = x_values.drop(columns=filecolumn)
    y_values = pd.read_json(StringIO(y_labels))["label"].values
    clf.fit(x_values, y_values)

    qs = UncertaintySampling(
        method="least_confident", random_state=42, missing_label=-1
    )
    query_idx = qs.query(x_values, y_values, clf)[0]
    file_id = files.loc[query_idx, filecolumn][5:]

    pcas = pd.read_json(StringIO(pca_data)).loc[query_idx, ["xcol", "ycol", "zcol"]]

    pca_dict = {}
    for col_index, axis in enumerate(["x", "y", "z"]):
        pca_dict.update({axis: pcas.values[col_index]})
    pca_dict.update(
        {
            "customdata": [
                file_id,
            ]
        }
    )

    clickdata = {"points": [pca_dict]}

    files["class"] = clf.predict(x_values)
    files["class"] = files["class"].apply(lambda x: ["Regular", "Outlier"][x])
    out_clf = pickle.dumps(clf).hex()
    return clickdata, out_clf, files.to_json()


@callback(Input("selected-data-point", "data"))
def show_click_data(clickData):
    logger.debug(f"Updating selected data point to: {clickData}")


@callback(
    Output("selected-data-point", "data"),
    Input("trajectories-scatter", "clickData"),
    Input("queried-data-point", "data"),
)
def update_selection(clickData, queryData):
    if "trajectories-scatter.clickData" in callback_context.triggered_prop_ids.keys():
        return clickData
    else:
        return queryData


@callback(
    Output("download-data", "data"),
    Input("btn-download-excel", "n_clicks"),
    Input("btn-download-csv", "n_clicks"),
    Input("x-values", "data"),
    Input("y-labeled", "data"),
    Input("y-predicted", "data"),
    prevent_initial_call=True,
)
def download_excel(n_clicks_excel, n_clicks_csv, x_values, y_labeled, y_predicted):
    if callback_context.triggered_id not in ["btn-download-excel", "btn-download-csv"]:
        return no_update
    x_values = pd.read_json(StringIO(x_values))
    y_labeled = pd.read_json(StringIO(y_labeled))
    y_predicted = pd.read_json(StringIO(y_predicted))
    out_df = pd.merge(x_values, y_labeled, on=filecolumn)
    out_df = pd.merge(out_df, y_predicted, on=filecolumn)

    if callback_context.triggered_id == "btn-download-excel":
        return dcc.send_data_frame(out_df.to_excel, "predictions.xlsx")
    else:
        return dcc.send_data_frame(out_df.to_csv, "predictions.csv")


if __name__ == "__main__":
    debug = True
    if debug is not True:
        logger.remove()
        logger.add(sys.stderr, level="INFO")
    app.run(debug=debug)
#     parser = argparse.ArgumentParser()
#     parser.add_argument("-e", "--embeddings", help="Embeddings file", required=True)
#     parser.add_argument(
#         "-a", "--image-archive", help="Trajectory image archive (hdf5)", required=True
#     )
#
#     args = parser.parse_args()
#
#     main(args.embeddings, args.image_archive)
