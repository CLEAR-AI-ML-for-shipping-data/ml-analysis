import argparse
import math
from time import perf_counter
from typing import Dict
from typing import List
from typing import Optional

import geopandas as gpd
import pandas as pd
from loguru import logger
from shapely import points
from shapely.geometry import box
from sqlalchemy import create_engine
from sqlalchemy import text
from tqdm import tqdm
from tqdm import trange

from helpers import data_preparation as dataprep

POSTGRES_DB = "gis"
POSTGRES_USER = "clear"
POSTGRES_PASSWORD = "clear"
POSTGRES_PORT = 5432
POSTGRES_HOST = "localhost"

logger.remove()
logger.add(lambda msg: tqdm.write(msg, end=""), colorize=True, level="INFO")


def process_gdf_chunk(
    df: gpd.GeoDataFrame,
    hdf5_filename: str,
    ship_id_col: str = "ship_id",
    t_start_col: str = "start_dt",
    t_end_col: str = "end_dt",
    geom_col: str = "coordinates",
    timestamp_col: str = "timestamps",
    external_geoms: List[gpd.GeoDataFrame] = [],
    conv_kernel_size: int = 1,
    heading_col: Optional[str] = None,
    cog_col: Optional[str] = None,
    scalar_value_cols: Optional[List[str]] = None,
):
    for row_nr in trange(df.shape[0], leave=False):
        # image = voyage_array_from_points(
        #     df.iloc[[row_nr], :], convert_from_points=False, coastlines=external_geoms
        # )
        voyage_data: gpd.GeoSeries = df.iloc[row_nr]
        timestamps = pd.DatetimeIndex(voyage_data[timestamp_col])
        positions = voyage_data[geom_col].coords

        data = gpd.GeoDataFrame(
            data={
                ship_id_col: voyage_data[ship_id_col],
                geom_col: points(positions).tolist(),
            },
            index=timestamps,
            geometry=geom_col,
            crs="EPSG:4326",
        )

        drift_col = None
        if heading_col is not None and cog_col is not None:
            drift_col = ["drift"]
            data[heading_col] = voyage_data[heading_col]
            data[cog_col] = voyage_data[cog_col]
            data[drift_col] = (
                (data[cog_col] - data[heading_col] + 180) % 360
            ) - 180

        if scalar_value_cols is not None:
            for column in scalar_value_cols:
                data[column] = voyage_data[column]

        all_value_cols = (scalar_value_cols or []) + (drift_col or [])

        dataprep.time_windowing(
            dataf=data,
            coastlines=external_geoms,
            zipfile=hdf5_filename,
            prefix=voyage_data[ship_id_col],
            export_dir=None,
            value_cols=all_value_cols,
        )

        # image[0] = convolve_image(image[0], kernel_size=conv_kernel_size)
        #
        # start_time = df.loc[row_nr, t_start_col].isoformat()
        # end_time = df.loc[row_nr, t_end_col].isoformat()
        #
        # filename = f"{df.loc[row_nr, ship_id_col]}_{start_time}_{end_time}.npy"
        #
        # with h5py.File(hdf5_filename, "a") as archive:
        #     archive.create_dataset(filename, data=image)


def make_bounding_box(df: gpd.GeoDataFrame, margin_degrees: float = 1.0):
    minx, miny, maxx, maxy = df.geometry.total_bounds
    df_box = box(
        minx=minx - margin_degrees,
        miny=miny - margin_degrees,
        maxx=maxx + margin_degrees,
        maxy=maxy + margin_degrees,
    )
    df_box = gpd.GeoDataFrame([{"geometry": df_box}], crs="EPSG:4326")
    return df_box


def main(
    database_url: str,
    geometries: List[str] = [],
    geometry_tables: Dict[str, str] = {},
):
    """

    Args:
        database_url:
        geometries:
        geometry_tables: dictionary of table and columns to be selected from geometry
                         tables
    """
    engine = create_engine(database_url)
    # Need the following attributes from voyage_segments
    # ship_id, start_dt, end_dt, ais_data

    external_dfs = []
    t0 = perf_counter()
    for geometry_file in geometries:
        logger.info(f"Loading {geometry_file}")
        external_dfs.append(dataprep.load_external_geo_data(geometry_file))
    t1 = perf_counter()

    logger.debug(f"Loaded external geometries in {int((t1 - t0) * 1_000):d}ms")
    logger.info(f"Connecting to {database_url}")

    chunksize = 100
    hdf5_file = "db_test.hdf5"

    # Here we build a the query looking like this:
    # SELECT boxed.* [, <geom_table1>.geometry [, ...]]
    # FROM (
    #   SELECT ship_id, start_dt, end_dt, ais_data,
    #       ST_Envelope(ais_data) AS voyage_envelope
    #   FROM <segment_table> ) AS boxed
    # [
    #   LEFT JOIN <geom_table1>
    #   ON ST_Crosses(boxed.voyage_envelope, <geom_table1>.geometry)
    #   [...]
    # ]

    segment_table = "trajectories_2023_02"

    ship_id_col = "mmsi"
    coord_col = "coordinates"
    scalar_value_cols = [
        "u10",
    ]  # , "v10", "mwd", "mwp", "swh"]
    svc_string = ", " + \
        svc if len(svc := ", ".join(scalar_value_cols)) > 0 else ""

    select_boxed_segments = (
        f"SELECT {ship_id_col}, start_dt, end_dt, timestamps, heading, "
        f"course_over_ground, {coord_col} {svc_string}, "
        f"ST_Envelope({coord_col}) AS voyage_envelope "
        f"FROM {segment_table}"
    )

    select_string = "SELECT boxed.*"

    geom_columns_list = []

    for table, columns in geometry_tables.items():
        geom_column = []
        if isinstance(columns, list):
            for col in columns:
                select_string += f", {table}.{col} AS {table}_{col}"
                geom_column.append(f"{table}_{col}")
        else:
            select_string += f", {table}.{columns} AS {table}_{columns}"
            geom_column.append(f"{table}_{columns}")

        geom_columns_list.append(geom_column)

    from_string = f"FROM ({select_boxed_segments}) AS boxed"

    for table in geometry_tables.keys():
        from_string += (
            f" LEFT JOIN {table} ON ST_Crosses(boxed.voyage_envelope, {table}.geometry)"
        )

    from_string += " INNER JOIN ships on boxed.mmsi = ships.mmsi"

    where_string = "where ships.type_of_ship = 18"

    query_string = f"{select_string} {from_string} {where_string} "

    with engine.connect() as conn:

        total_rows = conn.execute(
            text(f"select count(*) {from_string} {where_string}"),
        ).scalar()
        total_chunks = math.ceil(total_rows / chunksize)

        logger.info(query_string)
        gdf_iterator = gpd.GeoDataFrame.from_postgis(
            query_string,
            conn,
            geom_col="coordinates",
            chunksize=chunksize,
        )
        logger.info("Received dataframe iterator")

        for i, df in enumerate(
            tqdm(
                gdf_iterator,
                total=total_chunks,
                unit="chunk",
                dynamic_ncols=True,
            ),
        ):
            t0 = perf_counter()
            if external_dfs:
                # Trim external geo datasets for faster overlapping with trajectories
                # logger.info("Cutting external data to size")
                bbox_df = make_bounding_box(df)
                cut_external_dfs = [
                    tgdf.overlay(bbox_df, how="intersection") for tgdf in external_dfs
                ]
                t1 = perf_counter()
                # logger.debug(f"Cut external geometries in {int((t1-t0)*1_000):d}ms")
            else:
                cut_external_dfs = []
            # logger.info(f"Processing chunk {i} ({df.shape[0]} rows)")
            # logger.info(f"{df.columns}")
            tdf = df[
                [
                    ship_id_col,
                    "start_dt",
                    "end_dt",
                    coord_col,
                    "timestamps",
                    "heading",
                    "course_over_ground",
                ] + scalar_value_cols
            ]
            external_db_dfs = [
                df[geom_columns].set_geometry(geom_columns[0], crs="EPSG:4326")
                for geom_columns in geom_columns_list
            ]

            process_gdf_chunk(
                tdf,
                hdf5_file,
                external_geoms=cut_external_dfs + external_db_dfs,
                conv_kernel_size=5,
                ship_id_col=ship_id_col,
                geom_col=coord_col,
                heading_col="heading",
                cog_col="course_over_ground",
                scalar_value_cols=scalar_value_cols,
            )


if __name__ == "__main__":
    database_url = (
        f"postgresql+psycopg2://{POSTGRES_USER}:{POSTGRES_PASSWORD}@{POSTGRES_HOST}:"
        f"{POSTGRES_PORT}/{POSTGRES_DB}"
    )
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--db_url",
        type=str,
        default=database_url,
        help="Postgres database url",
    )
    parser.add_argument(
        "-g",
        "--geometries",
        help="Specify one or more geometry files for extra information",
        nargs="*",
        default=[],
    )

    args = parser.parse_args()

    geometry_tables: Dict[str, str] = {
        # "coastlines_fullres": "geometry",
        # "coastlines_highres": "geometry",
    }
    main(
        database_url=args.db_url,
        geometries=args.geometries,
        geometry_tables=geometry_tables,
    )
