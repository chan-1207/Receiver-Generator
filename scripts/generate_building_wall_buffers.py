import os
import geopandas as gpd
import pandas as pd
from shapely.geometry import Polygon, MultiPolygon
from shapely.validation import make_valid

# =========================
# 설정값
# =========================
input_building_height_path = "../data/building_height/building_cropped_height.gpkg"
output_gpkg_path = "../receivers/building/building_cropped_buffer_10m_offset_1m.gpkg"

input_layer_name = "TN_BULD"      # None = 첫 번째 레이어 자동 선택
output_layer_name = "building_buffer"

# 버퍼 결과에 남길 최소 필드
# 수음점 생성까지 고려해서 높이 필드는 같이 유지
keep_cols = [
    "NF_ID",
    "BLDH_MN",
    "BLDH_BV",
    "Shape_Area",
]

dilation = 10.0             # 큰 버퍼 후 되돌리기 거리 [m]
simplify_m = 1.0            # 단순화 허용 오차 [m]
receiver_offset_m = 1.0     # 최종 수음점용 외곽 버퍼 [m]

area_col = "Shape_Area"
min_area_m2 = 25.0          # 모폴로지 연산 전 건물 면적 필터 기준 [m2]


# =========================
# 유틸 함수
# =========================
def count_vertices(geom):
    """Polygon/MultiPolygon 꼭짓점 수 계산"""
    if geom is None or geom.is_empty:
        return 0

    if isinstance(geom, Polygon):
        count = len(geom.exterior.coords)
        for interior in geom.interiors:
            count += len(interior.coords)
        return count

    if isinstance(geom, MultiPolygon):
        return sum(count_vertices(g) for g in geom.geoms)

    return 0


def to_multipolygon(geom):
    """GPKG 저장 시 geometry type을 MultiPolygon으로 통일"""
    if geom is None or geom.is_empty:
        return None

    if isinstance(geom, Polygon):
        return MultiPolygon([geom])

    if isinstance(geom, MultiPolygon):
        return geom

    return None


def simplify_building_polygon(geom):
    """
    건물 폴리곤 단순화 및 수음점용 외곽 버퍼 생성

    처리 순서:
    1) make_valid
    2) morphological closing: buffer(+dilation) -> buffer(-dilation)
    3) simplify
    4) receiver_offset_m 만큼 외곽 버퍼
    5) MultiPolygon으로 통일
    """
    if geom is None or geom.is_empty:
        return None

    geom = make_valid(geom)

    if geom is None or geom.is_empty:
        return None

    # 작은 홈, 계단형 요철 제거
    geom = geom.buffer(dilation, join_style="mitre").buffer(
        -dilation,
        join_style="mitre"
    )

    if geom is None or geom.is_empty:
        return None

    # 꼭짓점 단순화
    geom = geom.simplify(simplify_m, preserve_topology=True)

    if geom is None or geom.is_empty:
        return None

    # 최종 수음점 배치용 외곽 버퍼
    geom = geom.buffer(receiver_offset_m, join_style="mitre")

    if geom is None or geom.is_empty:
        return None

    geom = make_valid(geom)

    if geom is None or geom.is_empty:
        return None

    return to_multipolygon(geom)


# =========================
# 실행
# =========================
os.makedirs(os.path.dirname(output_gpkg_path), exist_ok=True)

if input_layer_name is None:
    gdf = gpd.read_file(input_building_height_path)
else:
    gdf = gpd.read_file(input_building_height_path, layer=input_layer_name)

print("[1] 입력 로드")
print(" - input:", input_building_height_path)
print(" - layer:", input_layer_name)
print(" - rows:", len(gdf))
print(" - CRS:", gdf.crs)
print(" - columns:", list(gdf.columns))

# EPSG:5179 같은 meter 단위 좌표계여야 함
if gdf.crs is None:
    raise ValueError("CRS가 없습니다. EPSG:5179 등 meter 기반 좌표계를 먼저 지정해야 합니다.")

# 필수 컬럼 확인
missing_cols = [c for c in keep_cols if c not in gdf.columns]
if missing_cols:
    raise ValueError(f"필수 컬럼이 없습니다: {missing_cols}")

# 필요한 필드만 유지
gdf = gdf[keep_cols + ["geometry"]].copy()

# 건물 데이터에 명시된 면적을 기준으로 작은 건물을 먼저 제거한다.
gdf[area_col] = pd.to_numeric(gdf[area_col], errors="coerce")
input_row_count = len(gdf)
gdf = gdf[
    gdf[area_col].notna()
    & (gdf[area_col] >= min_area_m2)
].copy()

print("[2] 건물 면적 사전 필터")
print(" - area column:", area_col)
print(" - minimum area [m2]:", min_area_m2)
print(" - removed rows:", input_row_count - len(gdf))
print(" - remaining rows:", len(gdf))

# 원본 통계
gdf["orig_vertex_count"] = gdf.geometry.apply(count_vertices)
gdf["orig_area_m2"] = gdf.geometry.area

print("[3] 버퍼/단순화 처리 시작")
gdf["geometry"] = gdf.geometry.apply(simplify_building_polygon)

# 유효 geometry만 유지
gdf = gdf[gdf.geometry.notnull()].copy()
gdf = gdf[~gdf.geometry.is_empty].copy()

# 결과 통계
gdf["simp_vertex_count"] = gdf.geometry.apply(count_vertices)
gdf["simp_area_m2"] = gdf.geometry.area

# 0 나누기 방지
gdf["vertex_reduction_ratio"] = 0.0
valid_vertex = gdf["orig_vertex_count"] > 0
gdf.loc[valid_vertex, "vertex_reduction_ratio"] = (
    1.0
    - gdf.loc[valid_vertex, "simp_vertex_count"]
    / gdf.loc[valid_vertex, "orig_vertex_count"]
).round(3)

gdf["area_change_ratio"] = 0.0
valid_area = gdf["orig_area_m2"] > 0
gdf.loc[valid_area, "area_change_ratio"] = (
    (
        gdf.loc[valid_area, "simp_area_m2"]
        - gdf.loc[valid_area, "orig_area_m2"]
    )
    / gdf.loc[valid_area, "orig_area_m2"]
).round(3)

# 저장
gdf.to_file(
    output_gpkg_path,
    driver="GPKG",
    layer=output_layer_name
)

print("[4] 저장 완료")
print(" - output:", output_gpkg_path)
print(" - layer:", output_layer_name)
print(" - rows:", len(gdf))

print(gdf[[
    "NF_ID",
    "BLDH_MN",
    "BLDH_BV",
    "orig_vertex_count",
    "simp_vertex_count",
    "vertex_reduction_ratio",
    "area_change_ratio"
]].head())
