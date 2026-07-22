import geopandas as gpd
import pandas as pd


shp_path = "../building_height_info/building_height.shp"
out_csv = "building_ids.csv"
id_column = "NF_ID"


gdf = gpd.read_file(shp_path)

if id_column not in gdf.columns:
    raise KeyError(f"'{id_column}' column not found. Available columns: {list(gdf.columns)}")

building_ids = (
    gdf[id_column]
    .dropna()
    .astype(str)
    .drop_duplicates()
    .reset_index(drop=True)
)

out_df = pd.DataFrame({"building_id": building_ids})
out_df.to_csv(out_csv, index=False, encoding="utf-8-sig")

print(f"saved: {out_csv}")
print(f"building id count: {len(out_df)}")
