import geopandas as gpd
import pandas as pd
import numpy as np
from shapely.geometry import LineString
from pyproj import Transformer
import math

# =========================
# 설정값
# =========================
shp_path = "../test/BLD010000003AE81A_buffer_1m.gpkg"
out_csv = "outputs/BLD010000003AE81A_buffer.csv"

wall_spacing_m = 10.0      # 외벽 수평 간격
vertical_spacing_m = 10.0  # 높이 방향 간격
start_height_m = 1.5      # 1층 수음점 높이
min_building_height_m = 2.0

# =========================
# 데이터 로드
# =========================
gdf = gpd.read_file(shp_path)

# 좌표계 확인
print("CRS:", gdf.crs)

# EPSG:5179 → EPSG:4326 변환기
# 만약 shp의 CRS가 EPSG:5179가 아니라면 gdf.crs를 확인해서 수정 필요
transformer = Transformer.from_crs(gdf.crs, "EPSG:4326", always_xy=True)

records = []
receiver_count = 0

# =========================
# 보조 함수
# =========================
def interpolate_points_on_line(line: LineString, spacing: float):
    """
    LineString 위에 spacing 간격으로 점 생성.
    짧은 선분도 최소 1개 생성.
    """
    length = line.length

    if length <= 0:
        return []

    n = max(1, math.ceil(length / spacing))

    points = []
    for i in range(n):
        # 양 끝점에 너무 붙지 않도록 중앙 배치
        d = (i + 0.5) * length / n
        points.append(line.interpolate(d))

    return points


def get_exterior_segments(polygon):
    """
    polygon exterior ring을 선분 리스트로 변환.
    MultiPolygon은 면적이 가장 큰 polygon만 우선 사용.
    """
    if polygon.geom_type == "MultiPolygon":
        polygon = max(polygon.geoms, key=lambda p: p.area)

    if polygon.geom_type != "Polygon":
        return []

    coords = list(polygon.exterior.coords)
    segments = []

    for i in range(len(coords) - 1):
        p1 = coords[i]
        p2 = coords[i + 1]
        seg = LineString([p1, p2])

        if seg.length > 0:
            segments.append(seg)

    return segments


# =========================
# 수음점 생성
# =========================
for idx, row in gdf.iterrows():
    geom = row.geometry

    if geom is None or geom.is_empty:
        continue

    building_id = row.get("NF_ID", f"building_{idx}")

    base = row.get("BLDH_MN", None)
    top = row.get("BLDH_BV", None)

    # noinspection PyTypeChecker
    if pd.isna(base) or pd.isna(top):
        continue

    # noinspection PyTypeChecker
    building_h = float(top) - float(base)

    if building_h < min_building_height_m:
        continue

    # 높이 방향 수음점
    heights = np.arange(start_height_m, building_h, vertical_spacing_m)

    if len(heights) == 0:
        heights = np.array([min(start_height_m, building_h / 2)])

    # 외벽 선분 추출
    segments = get_exterior_segments(geom)

    for seg in segments:
        wall_points = interpolate_points_on_line(seg, wall_spacing_m)

        for pt in wall_points:
            x, y = pt.x, pt.y
            lon, lat = transformer.transform(x, y)

            for h in heights:
                alt = float(base) + float(h)

                receiver_count += 1

                records.append({
                    "receiver_id": f"R{receiver_count:08d}",
                    "building_id": building_id,
                    "lat": lat,
                    "lon": lon,
                    "alt": alt
                })

# =========================
# 저장
# =========================
out_df = pd.DataFrame(records)
out_df.to_csv(out_csv, index=False, encoding="utf-8-sig")

print(f"saved: {out_csv}")
print(f"receiver count: {len(out_df)}")