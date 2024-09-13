import argparse
import datetime as dt
import os
import pickle
from typing import List, Optional

import geopandas as gpd
import numpy as np
import pandas as pd
from rasterio.features import rasterize
from rasterio.transform import from_bounds
from shapely.geometry import LineString, Point, box
from tqdm import tqdm


def voyage_array_from_points(
    data: gpd.GeoDataFrame,
    coastlines: Optional[gpd.GeoDataFrame] = None,
    resolution: int = 256,
    dtype: np.dtype = np.float32,
    filename: str = None,
):
    minx, miny, maxx, maxy = data.total_bounds
    center = box(minx, miny, maxx, maxy).centroid

    diagonal = max(
        (maxy - miny) / 2,  # divide by 2 to go from diameter to radius
        (maxx - minx) / 2,  # idem
        5.0 * 90 / 10_000.0,
    )
    square_box = Point(center).buffer(diagonal, cap_style=3)
    square_box = gpd.GeoDataFrame([{"geometry": square_box}], crs="EPSG:4326")

    transform = from_bounds(*square_box.total_bounds, resolution, resolution)

    travel_line = gpd.GeoDataFrame(
        [{"geometry": LineString(data.geometry)}], crs="EPSG:4326"
    )

    image = rasterize(
        travel_line.geometry,
        out_shape=(resolution, resolution),
        # out=image[0],
        transform=transform,
        dtype=dtype,
    )
    if coastlines is not None:
        try:
            coastline = coastlines.overlay(square_box, how="intersection")

            coast_raster = rasterize(
                coastline.geometry,
                out_shape=(resolution, resolution),
                transform=transform,
                dtype=dtype,
            )
            image = np.concat(
                [np.expand_dims(image, axis=0), np.expand_dims(coast_raster, axis=0)],
                axis=0,
            )
        except ValueError:
            image = np.concat(
                [np.expand_dims(image, axis=0), np.zeros((1, resolution, resolution))],
                axis=0,
            )
    assert image.shape == (2, resolution, resolution)
    if filename is not None:
        with open(f"{filename}.npy", "wb") as file:
            pickle.dump(image, file=file)
    return image


def time_windowing(
    dataf: gpd.GeoDataFrame,
    window_size: str = "4h",
    step_size: str = "2h",
    coastlines: Optional[gpd.GeoDataFrame] = None,
    prefix: str = None,
    export_dir: str = ".",
):
    df_index = dataf.index

    # convert to pandas Timedelta
    window_size = pd.Timedelta(window_size)
    step_size = pd.Timedelta(step_size)
    timesteps = pd.date_range(start=df_index[0], end=df_index[-1], freq=step_size)

    images: List[np.ndarray] = []
    for start_time in timesteps:
        image = None
        data = dataf.loc[start_time : start_time + window_size]

        start_string = start_time.strftime("%Y%m%d_%H%M%S")
        end_string = (start_time + window_size).strftime("%Y%m%d_%H%M%S")

        if data.shape[0] > 2:
            image = voyage_array_from_points(
                data,
                coastlines=coastlines,
                filename=f"{export_dir}/{prefix}_{start_string}_{end_string}",
            )

        if image is not None:
            images.append(image)

    return images


def convert_dataframe(
    dataf: gpd.GeoDataFrame,
    coastlines: Optional[gpd.GeoDataFrame] = None,
    window_size: str = "4h",
    step_size: str = "2h",
    timestamp: Optional[str] = None,
):
    images = []

    export_dir = f"./data/processed/processed_{timestamp}"
    os.mkdir(export_dir)

    for xlabel in tqdm(dataf.droplevel(-1).index.unique()):
        images += time_windowing(
            dataf.loc[xlabel],
            coastlines=coastlines,
            prefix=xlabel[0],
            window_size=window_size,
            step_size=step_size,
            export_dir=export_dir,
        )

    return images


def main(
    trajectory_file: str,
    coastline_file: str,
    window: str = "4h",
    step: str = "2h",
    timestamp: Optional[str] = None,
):
    """Run the data preparation pipeline.

    It uses trajectories as calculated in the FAIRSEA project,
    and coastline data from www.ngdc.noaa.gov/mgg/shorelines.

    Args:
        trajectory_file: file with trajectory information
        coastline_file: shapefile with coastline information
        window: time window size of trajectory snapshots
        step: offset between two consecutive snapshots
        timestamp: logging timestamp to track pipeline run
    """
    # Open voyages file
    with open(trajectory_file, "rb") as file:
        df = pickle.load(file=file)

    # Remove unnecessary indexing
    # Afterwards, index-levels are ship ID, voyage number, and datetime
    df = df.droplevel([2, 3])
    df = df.set_index("Timestamp_datetime", append=True)

    # Get the total bounding box of all voyages, and apply to coastlines data
    minx, miny, maxx, maxy = df.geometry.total_bounds
    margin_degrees = 1.0
    df_box = box(
        minx=minx - margin_degrees,
        miny=miny - margin_degrees,
        maxx=maxx + margin_degrees,
        maxy=maxy + margin_degrees,
    )
    df_box = gpd.GeoDataFrame([{"geometry": df_box}], crs="EPSG:4326")

    coastlines = gpd.read_file(coastline_file)
    coastlines = coastlines.overlay(df_box, how="intersection")

    convert_dataframe(df, coastlines=coastlines, timestamp=timestamp)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="CLEAR data preparation", description=None, epilog=None
    )

    parser.add_argument(
        "-d", "--datafile", help="Specify a trajectory file", required=True
    )
    parser.add_argument(
        "-c",
        "--coastlines",
        help="Specify a coastlines file",
        default="data/external/GSHHS_f_L1.shp",
    )
    parser.add_argument("-w", "--window", help="Specify a time window", default="4h")
    parser.add_argument("-s", "--step", help="Specify a time step", default="2h")

    starttime = dt.datetime.now().strftime("%Y%m%d_%H%M")

    args = parser.parse_args()

    main(
        trajectory_file=args.datafile,
        coastline_file=args.coastlines,
        window=args.window,
        step=args.step,
        timestamp=starttime,
    )
