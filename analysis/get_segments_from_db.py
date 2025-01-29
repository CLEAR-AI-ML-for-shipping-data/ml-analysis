import argparse
from time import perf_counter
from typing import List

import geopandas as gpd
import h5py
from loguru import logger
from prepare_2layer_data import load_external_geo_data, voyage_array_from_points
from sqlalchemy import create_engine
from tqdm import tqdm

POSTGRES_DB = "gis"
POSTGRES_USER = "clear"
POSTGRES_PASSWORD = "clear"
POSTGRES_PORT = 5432
POSTGRES_HOST = "localhost"

logger.remove()
logger.add(lambda msg: tqdm.write(msg, end=""), colorize=True)


def process_gdf_chunk(
    df: gpd.GeoDataFrame,
    hdf5_filename: str,
    ship_id_col: str = "ship_id",
    t_start_col: str = "start_dt",
    t_end_col: str = "end_dt",
    geom_col: str = "ais_data",
    external_geoms: List[gpd.GeoDataFrame] = [],
):
    for row_nr in tqdm(range(df.shape[0])):
        image = voyage_array_from_points(
            df.iloc[[row_nr], :], convert_from_points=False
        )
        start_time = df.loc[row_nr, t_start_col].isoformat()
        end_time = df.loc[row_nr, t_end_col].isoformat()

        filename = f"{df.loc[row_nr, ship_id_col]}_{start_time}_{end_time}.npy"

        with h5py.File(hdf5_filename, "a") as archive:
            logger.info(f"Writing file {filename}")
            archive.create_dataset(filename, data=image)


def main(database_url: str, geometries: List[str]):
    engine = create_engine(database_url)
    # Need the following attributes from voyage_segments
    # ship_id, start_dt, end_dt, ais_data

    external_dfs = []
    t0 = perf_counter()
    for geometry_file in geometries:
        logger.info(f"Loading {geometry_file}")
        external_dfs.append(load_external_geo_data(geometry_file))
    t1 = perf_counter()

    logger.info(f"Loaded external geometries in {int((t1-t0)*1_000):d}ms")

    logger.info(f"Connecting to {database_url}")

    chunksize = 10
    conn = engine.connect()
    gdf_iterator = gpd.GeoDataFrame.from_postgis(
        "SELECT * FROM voyage_segments",
        conn,
        geom_col="ais_data",
        chunksize=chunksize,
    )

    hdf5_file = "db_test.hdf5"

    for i, df in enumerate(gdf_iterator):
        logger.info(f"Processing chunk {i} ({df.shape[0]} rows)")
        df = df[["ship_id", "start_dt", "end_dt", "ais_data"]]
        process_gdf_chunk(df, hdf5_file)

    conn.close()


if __name__ == "__main__":
    database_url = f"postgresql+psycopg2://{POSTGRES_USER}:{POSTGRES_PASSWORD}@{POSTGRES_HOST}:{POSTGRES_PORT}/{POSTGRES_DB}"

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--db_url", type=str, default=database_url, help="Postgres database url"
    )
    parser.add_argument(
        "-g",
        "--geometries",
        help="Specify one or more geometry files for extra information",
        nargs="*",
        default=[],
    )

    args = parser.parse_args()

    main(database_url=args.db_url, geometries=args.geometries)
