import argparse

import h5py
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dash import Dash, Input, Output, callback, dcc, html
from sklearn.decomposition import PCA


# def main(embeddings_file: str, arrays_file: str):

parser = argparse.ArgumentParser()
parser.add_argument("-e", "--embeddings", help="Embeddings file", required=True)
parser.add_argument(
    "-a", "--image-archive", help="Trajectory image archive (hdf5)", required=True
)

args = parser.parse_args()
embeddings_file = args.embeddings
arrays_file = args.image_archive

edf = pd.read_csv(embeddings_file, sep=";", index_col=0)
df = edf.copy()

embeddings_columns = [col for col in df.columns if "emb_dim" in col]

pca_decomposer = PCA()
pca_vectors = pca_decomposer.fit_transform(df.loc[:, embeddings_columns].values)

xvalues = df.loc[:, embeddings_columns]
xvalues = pca_vectors[:, :3]

plot_df = df[["filename"]].copy()
plot_df["xcol"] = xvalues[:, 0]
plot_df["ycol"] = xvalues[:, 1]
plot_df["zcol"] = xvalues[:, 2]
plot_df["ms"] = 10


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
                    # hoverData={"points": [{"customdata": "Japan"}]},
                    style={
                        
                "width": "100vh", 
                "height": "100vh",
                    }
                )
            ],
            style={
                "width": "60%", 
                # "height": "74%",
                # "width": "100vh", 
                # "height": "100vh",
                "display": "inline-block", 
                "padding": "0 20"
            },
            id="scatter-plot-container"
        ),
        html.Div(
            [
                dcc.Graph(id="show-trajectory"),
            ],
            style={
                "display": "inline-block", 
                "vertical-align": "top",
                "width": "23%"
            },
        ),
        html.Div([
            html.Button("Like!", id="like-button"),
        ])
        
    ]
)

@callback(Output("show-trajectory", "figure"),
          Input("trajectories-scatter", "hoverData"), prevent_initial_call=True)
def render_trajectory_image(hoverData: str):
    # print(hoverData)
    filename = hoverData["points"][0]["customdata"]
    with h5py.File(arrays_file) as file:
        farray = file[filename][()]
    farray = np.transpose(farray, (1, 2, 0))
    # if farray.shape[-1] == 2:
    while farray.shape[-1] < 3:
        farray = np.append(farray, np.zeros_like(farray)[:, :, 0:1], axis=-1)
    fig = px.imshow(farray)
    return fig


@callback(Output("trajectories-scatter", "figure"),
          Input("like-button", "n_clicks"))
def plot_trajectory_points(n_clicks):
    fig = px.scatter_3d(
        # x=plot_df["xcol"],
        # y=plot_df["ycol"],
        # z=plot_df["zcol"],
        # hover_name=plot_df["filename"],
        data_frame=plot_df,
        x="xcol",
        y="ycol",
        z="zcol",
        hover_name="filename",
        size="ms",
        opacity=0.5
        # mode="markers",
        # marker_symbol="circle",
    )
    fig.update_traces(customdata=plot_df["filename"])
    fig.update_layout(margin={'l': 40, 'b': 40, 't': 10, 'r': 0}, hovermode='closest')
    return fig


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
