import math
import os

import geopandas as gpd
import numpy as np
import pandas as pd
from pyproj import Transformer
from shapely.affinity import rotate
from shapely.geometry import LineString, Point
from shapely.validation import make_valid


# =========================
# 설정값
# =========================
input_building_gpkg_path = "../data/building_height/building_cropped_height.gpkg"
input_layer_name = "TN_BULD"
output_csv_path = "../receivers/building/building_cropped_roof_receivers.csv"

id_col = "NF_ID"
base_col = "BLDH_MN"
top_col = "BLDH_BV"
area_col = "Shape_Area"

min_building_area_m2 = 25.0
min_building_height_m = 2.0

roof_inset_m = 5.0
roof_grid_min_area_m2 = 100.0
roof_spacing_m = 10.0
roof_height_offset_m = 1.0


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

        for part in geom.geoms:
            parts.extend(polygon_parts(part))

        return parts

    return []


def make_oriented_grid_points(poly, spacing):
    if poly is None or poly.is_empty:
        return []

    if spacing <= 0:
        raise ValueError("지붕 격자 간격은 0보다 커야 합니다.")

    # 최소 회전 사각형의 긴 변을 기준으로 건물의 주 방향을 구한다.
    rectangle = poly.minimum_rotated_rectangle
    rectangle_coords = list(rectangle.exterior.coords)
    edges = [
        (rectangle_coords[i], rectangle_coords[i + 1])
        for i in range(len(rectangle_coords) - 1)
    ]
    start, end = max(edges, key=lambda edge: LineString(edge).length)
    angle = math.atan2(end[1] - start[1], end[0] - start[0])

    # 건물의 주 방향이 좌표축과 나란해지도록 회전한다.
    origin = (poly.centroid.x, poly.centroid.y)
    local_poly = rotate(
        poly,
        -angle,
        origin=origin,
        use_radians=True
    )
    min_x, min_y, max_x, max_y = local_poly.bounds
    width = max_x - min_x
    height = max_y - min_y

    if width <= 0 or height <= 0:
        return [poly.representative_point()]

    # 가로와 세로를 기준 간격에 따라 균등 분할하고 양 끝점을 포함한다.
    x_section_count = max(1, math.floor(width / spacing))
    y_section_count = max(1, math.floor(height / spacing))
    x_values = np.linspace(min_x, max_x, x_section_count + 1)
    y_values = np.linspace(min_y, max_y, y_section_count + 1)

    points = []
    # 회전 과정에서 발생하는 미세한 좌표 오차를 허용한다.
    local_poly_with_tolerance = local_poly.buffer(1e-8)

    for x in x_values:
        for y in y_values:
            local_point = Point(float(x), float(y))

            if not local_poly_with_tolerance.covers(local_point):
                continue

            points.append(
                rotate(
                    local_point,
                    angle,
                    origin=origin,
                    use_radians=True
                )
            )

    # 비정형 폴리곤에서 모든 격자점이 외부로 빠지면 대표점 하나를 사용한다.
    if len(points) == 0:
        return [poly.representative_point()]

    return points


def make_roof_receiver_points(geom):
    if geom is None or geom.is_empty:
        return [], "empty"

    # 외벽 수음점과 거리를 두기 위해 원본 건물 폴리곤을 5m 안쪽으로 줄인다.
    roof_geom = clean_geom(
        geom.buffer(-roof_inset_m, join_style="mitre")
    )

    # 안쪽 버퍼가 사라지면 원본 건물 내부의 대표점 하나를 사용한다.
    if roof_geom is None or roof_geom.is_empty:
        return [geom.representative_point()], "representative"

    # 안쪽 버퍼 처리 후 면적이 100㎡ 이하이면 대표점 하나만 배치한다.
    if roof_geom.area <= roof_grid_min_area_m2:
        return [roof_geom.representative_point()], "representative"

    points = []
    seen_xy = set()

    for poly in polygon_parts(roof_geom):
        for point in make_oriented_grid_points(poly, roof_spacing_m):
            xy = (round(point.x, 8), round(point.y, 8))

            if xy in seen_xy:
                continue

            seen_xy.add(xy)
            points.append(point)

    if len(points) == 0:
        return [roof_geom.representative_point()], "representative"

    return points, "grid"


def load_buildings():
    if input_layer_name:
        buildings = gpd.read_file(
            input_building_gpkg_path,
            layer=input_layer_name
        )
    else:
        buildings = gpd.read_file(input_building_gpkg_path)

    if buildings.crs is None:
        raise ValueError("건물 GPKG에 CRS가 없습니다.")

    if buildings.crs.is_geographic:
        raise ValueError("미터 단위의 투영 좌표계를 사용해야 합니다.")

    required_cols = [id_col, base_col, top_col, area_col]
    missing_cols = [
        col for col in required_cols
        if col not in buildings.columns
    ]

    if missing_cols:
        raise ValueError(
            f"건물 레이어에 필수 필드가 없습니다: {missing_cols}"
        )

    buildings = buildings[buildings.geometry.notnull()].copy()
    buildings["geometry"] = buildings.geometry.apply(clean_geom)
    buildings = buildings[buildings.geometry.notnull()].copy()
    buildings = buildings[~buildings.geometry.is_empty].copy()

    buildings[base_col] = pd.to_numeric(
        buildings[base_col],
        errors="coerce"
    )
    buildings[top_col] = pd.to_numeric(
        buildings[top_col],
        errors="coerce"
    )
    buildings[area_col] = pd.to_numeric(
        buildings[area_col],
        errors="coerce"
    )

    buildings = buildings[
        buildings[id_col].notna()
        & buildings[base_col].notna()
        & buildings[top_col].notna()
        & buildings[area_col].notna()
        & (buildings[area_col] >= min_building_area_m2)
    ].copy()

    buildings["building_height"] = (
        buildings[top_col] - buildings[base_col]
    )
    buildings = buildings[
        buildings["building_height"] >= min_building_height_m
    ].reset_index(drop=True)

    return buildings


def main():
    os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)

    buildings = load_buildings()
    transformer = Transformer.from_crs(
        buildings.crs,
        "EPSG:4326",
        always_xy=True
    )

    records = []
    representative_building_count = 0
    grid_building_count = 0

    for _, row in buildings.iterrows():
        points, placement_type = make_roof_receiver_points(row.geometry)

        if placement_type == "representative":
            representative_building_count += 1
        elif placement_type == "grid":
            grid_building_count += 1

        reference = row[id_col]
        roof_alt = float(row[top_col]) + roof_height_offset_m

        for point in points:
            x, y = point.x, point.y
            lon, lat = transformer.transform(x, y)

            records.append({
                "reference": reference,
                "x_epsg5179": x,
                "y_epsg5179": y,
                "lat": lat,
                "lon": lon,
                "alt": roof_alt,
            })

    if len(records) == 0:
        raise ValueError("생성된 지붕 수음점이 없습니다.")

    output_cols = [
        "reference",
        "x_epsg5179",
        "y_epsg5179",
        "lat",
        "lon",
        "alt",
    ]
    output_df = pd.DataFrame(records, columns=output_cols)
    output_df.to_csv(
        output_csv_path,
        index=False,
        encoding="utf-8-sig",
        float_format="%.7f"
    )

    print("valid building count:", len(buildings))
    print(
        "representative-point building count:",
        representative_building_count
    )
    print("grid building count:", grid_building_count)
    print("roof receiver count:", len(output_df))
    print("saved:", output_csv_path)


if __name__ == "__main__":
    main()
