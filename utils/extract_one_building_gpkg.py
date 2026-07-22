import geopandas as gpd

# 입력 SHP
in_shp = "../building_height_info/building_height.shp"

target_id = "BLD010000003AE81A"

# 로드
gdf = gpd.read_file(in_shp)

# NF_ID 기준 필터링
one = gdf[gdf["NF_ID"] == target_id].copy()

print("추출 건물 수:", len(one))
print(one[["NF_ID", "BULD_NM", "BATC_NM", "BLDH_MN", "BLDH_BV"]])

# GPKG로만 저장
out_gpkg = f"{target_id}.gpkg"

one.to_file(
    out_gpkg,
    layer="building_one",
    driver="GPKG",
    encoding="utf-8"
)