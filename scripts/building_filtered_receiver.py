import os
import geopandas as gpd
import pandas as pd
import numpy as np
from shapely.geometry import LineString, Point
from shapely.validation import make_valid
from pyproj import Transformer
import math

# =========================
# 설정값
# =========================
input_polygon_gpkg_path = "../receivers/building/building_cropped_buffer_10m_offset_1m.gpkg"
output_cvs_path = "../receivers/building/building_cropped_filtered_receivers.csv"

input_layer_name = None  # None = 첫 번째 레이어 자동 선택

id_col = "NF_ID"
base_col = "BLDH_MN"   # 건물 지반 절대고도
top_col = "BLDH_BV"    # 건물 기본/지붕 절대고도

wall_spacing_m = 10.0
vertical_spacing_m = 10.0
start_height_m = 1.5
min_building_height_m = 2.0

z_tolerance_m = 0.05

save_debug_all_receivers = False
debug_csv = "outputs/BLD010000003AE81A_receivers_debug_all.csv"

# =========================
# 출력 폴더 생성
# =========================
os.makedirs(os.path.dirname(output_cvs_path), exist_ok=True)

if save_debug_all_receivers:
    os.makedirs(os.path.dirname(debug_csv), exist_ok=True)

# =========================
# 보조 함수
# =========================
def clean_geom(geom):
    if geom is None or geom.is_empty:
        return None

    geom = make_valid(geom)

    if geom is None or geom.is_empty:
        return None

    return geom


def polygon_parts(geom):
    if geom is None or geom.is_empty:
        return []

    if geom.geom_type == "Polygon":
        return [geom]

    if geom.geom_type == "MultiPolygon":
        return list(geom.geoms)

    if geom.geom_type == "GeometryCollection":
        parts = []
        for g in geom.geoms:
            parts.extend(polygon_parts(g))
        return parts

    return []


def get_exterior_segments(geom):
    segments = []

    for poly in polygon_parts(geom):
        if poly.is_empty:
            continue

        coords = list(poly.exterior.coords)

        for i in range(len(coords) - 1):
            seg = LineString([coords[i], coords[i + 1]])

            if seg.length > 0:
                segments.append(seg)

    return segments


def interpolate_points_on_line(line: LineString, spacing: float):
    length = line.length

    if length <= 0:
        return []

    n = max(1, math.ceil(length / spacing))

    points = []
    for i in range(n):
        d = (i + 0.5) * length / n
        points.append(line.interpolate(d))

    return points


def make_vertical_heights(building_h):
    if building_h <= 0:
        return np.array([])

    heights = np.arange(start_height_m, building_h, vertical_spacing_m)

    if len(heights) == 0:
        heights = np.array([building_h / 2.0])

    return heights


def is_blocked_by_other_building(receiver_row, conflict_gdf, sindex):
    """
    삭제 조건:
    1. 수음점 XY가 다른 건물의 버퍼 폴리곤 안/경계에 있음
    2. 수음점 alt가 상대 건물 top_alt 이하임

    높은 건물의 상부 수음점은 유지됨.
    """
    p = receiver_row.geometry
    my_id = receiver_row["reference"]
    alt = receiver_row["alt"]

    candidate_idx = list(sindex.query(p, predicate="intersects"))

    if len(candidate_idx) == 0:
        return False

    candidates = conflict_gdf.iloc[candidate_idx]

    for _, other in candidates.iterrows():
        other_id = other[id_col]

        # 자기 건물 제외
        if other_id == my_id:
            continue

        other_top = other[top_col]

        if pd.isna(other_top):
            continue

        if not other.geometry.intersects(p):
            continue

        # 상대 건물 지붕 이하이면 막힌 수음점으로 판단
        if alt <= float(other_top) + z_tolerance_m:
            return True

    return False


# =========================
# 1. 버퍼 폴리곤 로드
# =========================
if input_layer_name:
    buf = gpd.read_file(input_polygon_gpkg_path, layer=input_layer_name)
else:
    buf = gpd.read_file(input_polygon_gpkg_path)

print("CRS:", buf.crs)
print("buffer feature count:", len(buf))

if buf.crs is None:
    raise ValueError("버퍼 GPKG의 CRS가 없습니다.")

# 필수 필드 확인
required_cols = [id_col, base_col, top_col]
missing = [c for c in required_cols if c not in buf.columns]

if missing:
    raise ValueError(f"버퍼 레이어에 필수 필드가 없습니다: {missing}")

# =========================
# 2. 데이터 정리
# =========================
buf = buf[buf.geometry.notnull()].copy()
buf["geometry"] = buf.geometry.apply(clean_geom)
buf = buf[buf.geometry.notnull()].copy()
buf = buf[~buf.geometry.is_empty].copy()

buf[base_col] = pd.to_numeric(buf[base_col], errors="coerce")
buf[top_col] = pd.to_numeric(buf[top_col], errors="coerce")

buf = buf[
    buf[id_col].notna() &
    buf[base_col].notna() &
    buf[top_col].notna()
].copy()

buf["building_height"] = buf[top_col] - buf[base_col]

buf = buf[
    buf["building_height"] >= min_building_height_m
].copy()

buf = buf.reset_index(drop=True)

print("valid buffer count:", len(buf))

# =========================
# 3. 좌표 변환기
# =========================
transformer = Transformer.from_crs(buf.crs, "EPSG:4326", always_xy=True)

# =========================
# 4. 수음점 후보 생성
# =========================
records = []
receiver_count = 0

for idx, row in buf.iterrows():
    geom = row.geometry

    if geom is None or geom.is_empty:
        continue

    reference = row[id_col]

    base = float(row[base_col])
    top = float(row[top_col])
    building_h = float(row["building_height"])

    heights = make_vertical_heights(building_h)

    if len(heights) == 0:
        continue

    # 이미 만들어둔 버퍼 폴리곤 외곽선 기준
    segments = get_exterior_segments(geom)

    for seg_idx, seg in enumerate(segments):
        wall_points = interpolate_points_on_line(seg, wall_spacing_m)

        for pt_idx, pt in enumerate(wall_points):
            x, y = pt.x, pt.y
            lon, lat = transformer.transform(x, y)

            for h in heights:
                alt = base + float(h)

                receiver_count += 1

                records.append({
                    "reference": reference,
                    "x_epsg5179": x,
                    "y_epsg5179": y,
                    "lat": lat,
                    "lon": lon,
                    "alt": alt,
                    "geometry": Point(x, y)
                })

receivers = gpd.GeoDataFrame(records, geometry="geometry", crs=buf.crs)

print("candidate receivers count:", len(receivers))

if len(receivers) == 0:
    raise ValueError("생성된 수음점이 없습니다. 버퍼 폴리곤과 높이 필드를 확인하세요.")

# =========================
# 5. 3D 높이 기반 겹침 필터링
# =========================
# 같은 버퍼 레이어를 충돌 판단용으로 사용
conflict_gdf = buf[[id_col, top_col, "geometry"]].copy()
conflict_gdf = gpd.GeoDataFrame(conflict_gdf, geometry="geometry", crs=buf.crs)

sindex = conflict_gdf.sindex

receivers["is_blocked"] = receivers.apply(
    lambda r: is_blocked_by_other_building(r, conflict_gdf, sindex),
    axis=1
)

filtered = receivers[~receivers["is_blocked"]].copy()

print("blocked receivers count:", int(receivers["is_blocked"].sum()))
print("final receivers count:", len(filtered))

# =========================
# CSV 저장
# =========================
if save_debug_all_receivers:
    debug_df = receivers.drop(columns="geometry").copy()
    debug_df.to_csv(debug_csv, index=False, encoding="utf-8-sig")
    print(f"debug saved: {debug_csv}")

output_cols = [
    "reference",
    "x_epsg5179",
    "y_epsg5179",
    "lat",
    "lon",
    "alt",
]

out_df = filtered[output_cols].copy()
out_df.to_csv(output_cvs_path, index=False, encoding="utf-8-sig")

print(f"saved: {output_cvs_path}")
print(f"receivers count: {len(out_df)}")
print(out_df.head())
