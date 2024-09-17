import argparse
import datetime as dt
import os
import pickle
import pprint
from typing import List, Optional, Union

import geopandas as gpd
import numpy as np
import pandas as pd
from loguru import logger
from rasterio.features import rasterize
from rasterio.transform import from_bounds
from shapely.geometry import LineString, Point, box
from tqdm import tqdm


def voyage_array_from_points(
    data: gpd.GeoDataFrame,
    coastlines: List[gpd.GeoDataFrame] = [],
    resolution: int = 256,
    dtype: np.dtype = np.float32,
    filename: str = None,
):
    """Create a rasterized voyage array from a collection of coordinates.

    Additionally, it is possible to add layers with e.g. coastline information.

    Args:
        data: data of one voyage, has all the coordinates in the geometry column
        coastlines: coastlines metadata
        resolution: size of the output image
        dtype: datatype of the output array
        filename: file to save the array to

    Returns:
        a NumPy ndarray containing the rasterized voyage with possible coastline data
    """
    # Find a box (center, dx, and dy) that envelopes the trajectory
    minx, miny, maxx, maxy = data.total_bounds
    center = box(minx, miny, maxx, maxy).centroid

    # Create a square box of at least radius 5km from this envelope
    diagonal = max(
        (maxy - miny) / 2,  # divide by 2 to go from diameter to radius
        (maxx - minx) / 2,  # idem
        5.0 * 90 / 10_000.0,
    )
    square_box = Point(center).buffer(diagonal, cap_style=3)
    square_box = gpd.GeoDataFrame([{"geometry": square_box}], crs="EPSG:4326")

    transform = from_bounds(*square_box.total_bounds, resolution, resolution)

    # Create a line from the coordinates, and rasterize the line
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
    # Add coastline geometries to the trajectory
    if len(coastlines) > 0:
        image = np.expand_dims(image, axis=0)

    for coastline in coastlines:
        try:
            coastline = coastline.overlay(square_box, how="intersection")

            coast_raster = rasterize(
                coastline.geometry,
                out_shape=(resolution, resolution),
                transform=transform,
                dtype=dtype,
            )
            image = np.concat(
                [image, np.expand_dims(coast_raster, axis=0)],
                axis=0,
            )
        # If there is no overlap, GeoPandas raises a ValueError.
        # In that case, add a layer of zeros
        except ValueError:
            image = np.concat(
                [image, np.zeros((1, resolution, resolution))],
                axis=0,
            )
    # Export each individual array
    if len(coastlines) < 1:
        assert image.shape == (resolution, resolution)
    else:
        assert image.shape == (len(coastlines) + 1, resolution, resolution)
    if filename is not None:
        with open(f"{filename}.npy", "wb") as file:
            pickle.dump(image, file=file)
    return image


def time_windowing(
    dataf: gpd.GeoDataFrame,
    window_size: str = "4h",
    step_size: str = "2h",
    coastlines: List[gpd.GeoDataFrame] = [],
    prefix: str = None,
    export_dir: str = ".",
):
    """Create time-windowed snapshots of the voyage, and rasterize the snapshots.

    Args:
        dataf: voyage coordinates
        window_size: voyage duration in one snapshot
        step_size: time difference between each snapshot
        coastlines: coastline metadata
        prefix: voyage identifier used in output filenames
        export_dir: folder to save output files

    Returns:
        rasterized voyage snapshots
    """
    df_index = dataf.index

    # convert window and step to pandas Timedelta
    window_size = pd.Timedelta(window_size)
    step_size = pd.Timedelta(step_size)
    timesteps = pd.date_range(start=df_index[0], end=df_index[-1], freq=step_size)

    # Rasterize each snapshot of the voyage
    images: List[np.ndarray] = []
    for start_time in timesteps:
        image = None
        data = dataf.loc[start_time : start_time + window_size]

        start_string = start_time.strftime("%Y%m%d_%H%M%S")
        end_string = (start_time + window_size).strftime("%Y%m%d_%H%M%S")

        # We cannot say anything about a trajectory that is just 2 points
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
    coastlines: Union[List[gpd.GeoDataFrame], gpd.GeoDataFrame] = [],
    window_size: str = "4h",
    step_size: str = "2h",
    timestamp: Optional[str] = None,
):
    """Convert a dataframe full of voyages to rasterized voyage snapshots.

    These snapshots can optionally contain data about coastlines.
    In the future we would add functionality for more layers, e.g. EEZ information.

    Args:
        dataf: voyage data
        coastlines: coastline data
        window_size: voyage duration in each snapshot
        step_size: time difference between each snapshot
        timestamp: label used to identify the preparation run

    Returns:
        a list of voyage snapshots
    """
    if isinstance(coastlines, gpd.GeoDataFrame):
        coastlines = [coastlines]

    images = []

    export_dir = f"./data/processed/processed_{timestamp}"

    logger.info(f"Creating folder {export_dir}")
    os.mkdir(export_dir)
    logger.info(f"Data will be exported to folder {export_dir}")

    # Iterate over combination of (ship identifier, voyage number)
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


def load_external_geo_data(file: str, bounding_box: Optional[gpd.GeoDataFrame] = None):
    gdf = gpd.read_file(file)
    if bounding_box is not None:
        gdf = gdf.overlay(bounding_box, how="intersection")
    return gdf


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
    bbox_dict = {
        "x_min": float(minx),
        "x_max": float(maxx),
        "y_min": float(miny),
        "y_max": float(maxy),
    }
    logger.info(f"Bounding box: {bbox_dict}")
    margin_degrees = 1.0
    df_box = box(
        minx=minx - margin_degrees,
        miny=miny - margin_degrees,
        maxx=maxx + margin_degrees,
        maxy=maxy + margin_degrees,
    )
    df_box = gpd.GeoDataFrame([{"geometry": df_box}], crs="EPSG:4326")

    # coastlines = gpd.read_file(coastline_file)
    # coastlines = coastlines.overlay(df_box, how="intersection")
    coastlines = []
    for file in coastline_file:
        coastlines.append(load_external_geo_data(file, bounding_box=df_box))

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
        default="data/external/shorelines/GSHHS_f_L1.shp",
    )
    parser.add_argument(
        "-e",
        "--ecozones",
        help="Specify a territorial waters file",
        default="data/external/eco_zones/eez_territorial_fused.gpkg",
    )
    parser.add_argument("-w", "--window", help="Specify a time window", default="4h")
    parser.add_argument("-s", "--step", help="Specify a time step", default="2h")

    starttime = dt.datetime.now().strftime("%Y%m%d_%H%M")
    logfile = f"logs/preparation_{starttime}.log"
    logger.add(logfile)

    logger.info(f"Starting logfile at {logfile}")

    args = parser.parse_args()

    logger.info(
        "Starting data preparation with settings:\n{}", pprint.pformat(vars(args))
    )

    main(
        trajectory_file=args.datafile,
        coastline_file=[args.coastlines, args.ecozones],
        window=args.window,
        step=args.step,
        timestamp=starttime,
    )
