import geopandas as gpd


shp_path = "../building_height_info/building_height.shp"
target_building_id = "BLD010000003AE8CL"
id_column = "NF_ID"


gdf = gpd.read_file(shp_path)

if id_column not in gdf.columns:
    raise KeyError(f"'{id_column}' column not found. Available columns: {list(gdf.columns)}")

matched = gdf[gdf[id_column].astype(str) == target_building_id].copy()

if matched.empty:
    raise ValueError(f"building id not found: {target_building_id}")

matched["geometry_wkt"] = matched.geometry.to_wkt()
out_df = matched.drop(columns="geometry")
out_df.to_csv("info_"+target_building_id, index=False, encoding="utf-8-sig")

print(f"saved: {target_building_id}")
print(f"matched count: {len(out_df)}")
