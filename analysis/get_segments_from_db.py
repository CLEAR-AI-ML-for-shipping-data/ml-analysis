import argparse

import geopandas as gpd
from sqlalchemy import create_engine

POSTGRES_DB = "gis"
POSTGRES_USER = "clear"
POSTGRES_PASSWORD = "clear"
POSTGRES_PORT = 5432
POSTGRES_HOST = "localhost"
database_url = f"postgresql+psycopg2://{POSTGRES_USER}:{POSTGRES_PASSWORD}@{POSTGRES_HOST}:{POSTGRES_PORT}/{POSTGRES_DB}"

print(database_url)
parser = argparse.ArgumentParser()
parser.add_argument(
    "--db_url", type=str, default=database_url, help="Postgres database url"
)


engine = create_engine(database_url)
# Need the following attributes from voyage_segments
# ship_id, start_dt, end_dt, ais_data
with engine.connect() as conn:
    gdf = gpd.GeoDataFrame.from_postgis(
        "SELECT * FROM voyage_segments", conn, geom_col="ais_data"
    )
    print(gdf.info())
    print(gdf[["ship_id", "start_dt", "end_dt", "ais_data"]])
