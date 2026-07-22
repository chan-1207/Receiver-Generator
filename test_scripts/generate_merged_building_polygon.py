import geopandas as gpd
from shapely.validation import make_valid

# =========================
# 설정
# =========================
input_path = "../building_height_info/building_height.shp"
output_path = "outputs/building_merge_receivers.gpkg"

merge_distance_m = 2.0   # 가까운 건물을 묶는 기준
min_area_m2 = 10.0       # 너무 작은 결과 제거

# =========================
# 데이터 로드
# =========================
gdf = gpd.read_file(input_path)

print("CRS:", gdf.crs)
print("원본 건물 수:", len(gdf))

# =========================
# geometry 정리
# =========================
gdf = gdf[gdf.geometry.notnull()].copy()
gdf["geometry"] = gdf.geometry.apply(make_valid)
gdf = gdf[~gdf.geometry.is_empty].copy()

# 너무 작은 원본 건물 제거
gdf = gdf[gdf.geometry.area >= min_area_m2].copy()

# =========================
# 1. 건물 확장
# =========================
gdf_buffered = gdf.copy()
gdf_buffered["geometry"] = gdf_buffered.geometry.buffer(
    merge_distance_m,
    join_style="mitre"
)

# =========================
# 2. 전체 union
# =========================
merged_geom = gdf_buffered.geometry.union_all()

# =========================
# 3. 다시 원래 크기 방향으로 축소
# =========================
clustered_geom = merged_geom.buffer(
    -merge_distance_m,
    join_style="mitre"
)

clustered_geom = make_valid(clustered_geom)

# =========================
# 4. Polygon / MultiPolygon 분해
# =========================
cluster_gdf = gpd.GeoDataFrame(
    geometry=[clustered_geom],
    crs=gdf.crs
).explode(index_parts=False).reset_index(drop=True)

cluster_gdf = cluster_gdf[cluster_gdf.geometry.notnull()].copy()
cluster_gdf = cluster_gdf[~cluster_gdf.geometry.is_empty].copy()
cluster_gdf = cluster_gdf[cluster_gdf.geometry.area >= min_area_m2].copy()

cluster_gdf["cluster_id"] = range(1, len(cluster_gdf) + 1)
cluster_gdf["area_m2"] = cluster_gdf.geometry.area

# =========================
# 저장
# =========================
cluster_gdf.to_file(output_path, driver="GPKG")

print("병합 후 건물군 수:", len(cluster_gdf))
print("저장 완료:", output_path)