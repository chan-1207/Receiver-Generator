import math
import os
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from pyproj import Transformer
from shapely.affinity import rotate
from shapely.geometry import LineString, Point
from shapely.ops import split
from shapely.validation import make_valid


# =========================
# 설정값
# =========================
project_dir = Path(__file__).resolve().parents[1]

input_building_buffer_gpkg_path = Path(os.environ.get(
    "RECEIVER_BUFFER_INPUT_GPKG",
    project_dir / "receivers/building/cropped_building_buffers_10m.gpkg",
))
input_building_metadata_gpkg_path = Path(os.environ.get(
    "BUILDING_METADATA_INPUT_GPKG",
    project_dir / "metadata/building/building_cropped_metadata.gpkg",
))
output_csv_path = Path(os.environ.get(
    "ROOF_RECEIVER_OUTPUT_CSV",
    project_dir / "receivers/building/cropped_building_roof_receivers.csv",
))

input_buffer_layer_name = "building_buffer"
input_metadata_layer_name = "building_metadata"

id_col = "NF_ID"
top_col = "BLDH_BV"
area_col = "Shape_Area"

min_building_area_m2 = 25.0

roof_resolution_m = 10.0
roof_inset_m = roof_resolution_m / 2
roof_height_offset_m = 1.0

rectangularity_threshold = 0.8
max_split_depth = 5
min_piece_area_m2 = 15.0


# =========================
# 보조 함수
# =========================
def clean_geom(geom):
    """도형 유효성 보정 및 빈 도형 제거"""
    if geom is None or geom.is_empty:
        return None

    geom = make_valid(geom)

    if geom is None or geom.is_empty:
        return None

    return geom


def polygon_parts(geom):
    """Polygon 조각 추출"""
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


def get_rectangle_edges(poly):
    """최소 회전 사각형의 변 반환"""
    rectangle = poly.minimum_rotated_rectangle
    rectangle_coords = list(rectangle.exterior.coords)

    return [
        (rectangle_coords[i], rectangle_coords[i + 1])
        for i in range(len(rectangle_coords) - 1)
    ]


def get_rectangularity(poly):
    """폴리곤과 최소 회전 사각형의 면적 비율 계산"""
    rectangle = poly.minimum_rotated_rectangle

    if rectangle.is_empty or rectangle.area <= 0:
        return 0.0

    return min(1.0, poly.area / rectangle.area)


def get_concave_vertices(poly):
    """폴리곤의 오목 꼭짓점 탐색"""
    coords = list(poly.exterior.coords)[:-1]

    if len(coords) < 4:
        return []

    is_ccw = poly.exterior.is_ccw
    concave_vertices = []

    for i, current in enumerate(coords):
        previous = coords[i - 1]
        following = coords[(i + 1) % len(coords)]
        incoming_x = current[0] - previous[0]
        incoming_y = current[1] - previous[1]
        outgoing_x = following[0] - current[0]
        outgoing_y = following[1] - current[1]
        cross = incoming_x * outgoing_y - incoming_y * outgoing_x

        if (is_ccw and cross < -1e-9) or (not is_ccw and cross > 1e-9):
            concave_vertices.append(current)

    return concave_vertices


def split_polygon_once(poly, min_piece_area_m2):
    """직사각형도를 개선하는 1회 분할 탐색"""
    rectangle = poly.minimum_rotated_rectangle
    edges = get_rectangle_edges(poly)
    start, end = max(edges, key=lambda edge: LineString(edge).length)
    angle = math.atan2(end[1] - start[1], end[0] - start[0])
    origin = (rectangle.centroid.x, rectangle.centroid.y)

    # 분할선을 건물의 주 방향과 나란하게 만들기 위해 로컬 좌표로 회전한다.
    local_poly = rotate(
        poly,
        -angle,
        origin=origin,
        use_radians=True
    )
    concave_vertices = get_concave_vertices(local_poly)

    if len(concave_vertices) == 0:
        return None

    min_x, min_y, max_x, max_y = local_poly.bounds
    margin = max(max_x - min_x, max_y - min_y) + 1.0
    cut_offset_m = 0.01
    current_score = get_rectangularity(local_poly)
    best_score = current_score
    best_parts = None
    candidate_keys = set()

    for vertex_x, vertex_y in concave_vertices:
        for offset in (-cut_offset_m, cut_offset_m):
            candidate_lines = [
                LineString([
                    (vertex_x + offset, min_y - margin),
                    (vertex_x + offset, max_y + margin),
                ]),
                LineString([
                    (min_x - margin, vertex_y + offset),
                    (max_x + margin, vertex_y + offset),
                ]),
            ]

            for cut_line in candidate_lines:
                key = tuple(
                    round(value, 4)
                    for coord in cut_line.coords
                    for value in coord
                )

                if key in candidate_keys:
                    continue

                candidate_keys.add(key)

                try:
                    result = split(local_poly, cut_line)
                except ValueError:
                    continue

                parts = [
                    part
                    for part in polygon_parts(result)
                    if not part.is_empty and part.area > 0
                ]

                if len(parts) < 2:
                    continue

                if any(
                    part.area < min_piece_area_m2
                    for part in parts
                ):
                    continue

                total_area = sum(part.area for part in parts)
                weighted_score = sum(
                    part.area * get_rectangularity(part)
                    for part in parts
                ) / total_area
                # 불필요하게 많은 조각으로 나뉘는 후보에는 작은 감점을 준다.
                weighted_score -= max(0, len(parts) - 2) * 0.01

                if weighted_score <= best_score + 1e-4:
                    continue

                best_score = weighted_score
                best_parts = parts

    if best_parts is None:
        return None

    return [
        rotate(
            part,
            angle,
            origin=origin,
            use_radians=True
        )
        for part in best_parts
    ]


def decompose_polygon(
    poly,
    rectangularity_threshold,
    max_split_depth,
    min_piece_area_m2,
    current_depth=0
):
    """직사각형도와 최대 깊이 기준 재귀 분할"""
    if (
        current_depth >= max_split_depth
        or get_rectangularity(poly) >= rectangularity_threshold
    ):
        return [poly]

    split_parts = split_polygon_once(poly, min_piece_area_m2)

    if split_parts is None:
        return [poly]

    result = []

    for part in split_parts:
        result.extend(
            decompose_polygon(
                part,
                rectangularity_threshold,
                max_split_depth,
                min_piece_area_m2,
                current_depth + 1
            )
        )

    return result


def decompose_geometry(geom):
    """도형의 Polygon 조각 분할"""
    original_parts = polygon_parts(geom)
    pieces = []

    for poly in original_parts:
        pieces.extend(
            decompose_polygon(
                poly,
                rectangularity_threshold,
                max_split_depth,
                min_piece_area_m2
            )
        )

    return pieces


def get_geometric_center(geom):
    """도형 내부 중심점 반환"""
    # 기하학적 중심이 폴리곤 내부에 있으면 그대로 사용한다.
    center = geom.centroid

    if geom.covers(center):
        return center

    # 오목한 형상이나 MultiPolygon의 중심이 외부에 있으면
    # 가장 넓은 폴리곤 조각의 중심을 사용한다.
    parts = polygon_parts(geom)

    if len(parts) == 0:
        return geom.representative_point()

    largest_part = max(parts, key=lambda poly: poly.area)
    center = largest_part.centroid

    if largest_part.covers(center):
        return center

    return largest_part.representative_point()


def get_centered_axis_values(min_value, max_value, resolution):
    """범위 중심 기준 등간격 좌표 생성"""
    center = (min_value + max_value) / 2.0
    half_length = (max_value - min_value) / 2.0
    step_count = math.floor(half_length / resolution)

    # 사각형 중심을 기준으로 양쪽에 정확한 간격으로 후보점을 배치한다.
    offsets = np.arange(-step_count, step_count + 1) * resolution

    return center + offsets


def make_oriented_grid_candidates(poly, resolution):
    """폴리곤 주 방향 격자 후보점 생성"""
    if poly is None or poly.is_empty:
        return []

    if resolution <= 0:
        raise ValueError("지붕 격자 해상도는 0보다 커야 합니다.")

    # 원본 건물을 감싸는 최소 면적 회전 사각형을 만든다.
    rectangle = poly.minimum_rotated_rectangle
    edges = get_rectangle_edges(poly)
    start, end = max(edges, key=lambda edge: LineString(edge).length)
    angle = math.atan2(end[1] - start[1], end[0] - start[0])

    # 회전 사각형의 긴 변이 좌표축과 나란해지도록 회전한다.
    origin = (rectangle.centroid.x, rectangle.centroid.y)
    local_rectangle = rotate(
        rectangle,
        -angle,
        origin=origin,
        use_radians=True
    )
    min_x, min_y, max_x, max_y = local_rectangle.bounds
    width = max_x - min_x
    height = max_y - min_y

    if width <= 0 or height <= 0:
        return [get_geometric_center(poly)]

    # 사각형 중심을 기준으로 정확히 10m 간격의 격자 후보를 만든다.
    x_values = get_centered_axis_values(min_x, max_x, resolution)
    y_values = get_centered_axis_values(min_y, max_y, resolution)

    candidates = []

    for x in x_values:
        for y in y_values:
            local_point = Point(float(x), float(y))
            candidates.append(
                rotate(
                    local_point,
                    angle,
                    origin=origin,
                    use_radians=True
                )
            )

    return candidates


def make_roof_receiver_points(geom, original_geom=None):
    """지붕 안전영역 내 수음점 생성"""
    if geom is None or geom.is_empty:
        return [], "empty", 0

    if original_geom is None:
        original_geom = geom

    containment_geom = clean_geom(geom.intersection(original_geom))

    if containment_geom is None or containment_geom.is_empty:
        return [], "empty", 0

    pieces = decompose_geometry(geom)

    # 외벽 수음점과의 이격 확보를 위한 외측 버퍼 기준 내측 축소
    roof_geom = clean_geom(
        geom
        .buffer(-roof_inset_m, join_style="mitre")
        .intersection(original_geom)
    )

    # 지붕 안전영역 소멸 시 버퍼·원본 교집합의 중심점 배치
    if roof_geom is None or roof_geom.is_empty:
        return [get_geometric_center(containment_geom)], "center", len(pieces)

    points = []
    seen_xy = set()

    # 각 분할 조각마다 안전 영역과 격자를 따로 계산한다.
    for poly in pieces:
        safe_piece = clean_geom(poly.intersection(roof_geom))

        if safe_piece is None or safe_piece.is_empty:
            continue

        safe_piece_with_tolerance = safe_piece.buffer(1e-8)
        candidates = make_oriented_grid_candidates(
            safe_piece,
            roof_resolution_m
        )
        piece_points = []

        # 현재 조각의 축소 안전영역 내부 후보점 선별
        for point in candidates:
            if not safe_piece_with_tolerance.covers(point):
                continue

            piece_points.append(point)

        # 후보가 하나 이하인 조각의 기하학적 중심점 배치
        if len(piece_points) <= 1:
            piece_points = [get_geometric_center(safe_piece)]

        for point in piece_points:
            xy = (round(point.x, 8), round(point.y, 8))

            if xy in seen_xy:
                continue

            seen_xy.add(xy)
            points.append(point)

    # 모든 조각의 안전 영역이 사라진 경우에만 건물 전체 중심점을 사용한다.
    if len(points) == 0:
        return [get_geometric_center(roof_geom)], "center", len(pieces)

    # 최종 수음점이 하나일 때 버퍼·원본 교집합의 중심점 재배치
    if len(points) == 1:
        return [get_geometric_center(containment_geom)], "center", len(pieces)

    placement_type = "grid" if len(points) > 1 else "center"

    return points, placement_type, len(pieces)


def load_buildings():
    """입력 건물 데이터 로드 및 필터링"""
    if input_buffer_layer_name:
        buildings = gpd.read_file(
            input_building_buffer_gpkg_path,
            layer=input_buffer_layer_name
        )
    else:
        buildings = gpd.read_file(input_building_buffer_gpkg_path)

    if buildings.crs is None:
        raise ValueError("건물 GPKG에 CRS가 없습니다.")

    if buildings.crs.is_geographic:
        raise ValueError("미터 단위의 투영 좌표계를 사용해야 합니다.")

    required_cols = [id_col, top_col, area_col]
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
        & buildings[top_col].notna()
        & buildings[area_col].notna()
        & (buildings[area_col] >= min_building_area_m2)
    ].copy()

    duplicate_building_id_mask = buildings[id_col].astype(str).duplicated(
        keep=False
    )

    if duplicate_building_id_mask.any():
        duplicate_ids = buildings.loc[
            duplicate_building_id_mask,
            id_col,
        ].head(5).tolist()
        raise ValueError(f"건물 버퍼에 중복 ID가 있습니다: {duplicate_ids}")

    original_buildings = gpd.read_file(
        input_building_metadata_gpkg_path,
        layer=input_metadata_layer_name
    )

    if original_buildings.crs is None:
        raise ValueError("건물 메타데이터 GPKG의 CRS가 없습니다.")

    if id_col not in original_buildings.columns:
        raise ValueError(
            f"건물 메타데이터 레이어에 ID 필드가 없습니다: {id_col}"
        )

    if original_buildings.crs != buildings.crs:
        original_buildings = original_buildings.to_crs(buildings.crs)

    original_buildings = original_buildings[
        original_buildings.geometry.notnull()
    ].copy()
    original_buildings["geometry"] = (
        original_buildings.geometry.apply(clean_geom)
    )
    original_buildings = original_buildings[
        original_buildings.geometry.notnull()
    ].copy()
    original_buildings = original_buildings[
        ~original_buildings.geometry.is_empty
    ].copy()
    original_buildings["_building_id_key"] = (
        original_buildings[id_col].astype(str)
    )
    duplicate_building_id_mask = original_buildings[
        "_building_id_key"
    ].duplicated(keep=False)

    if duplicate_building_id_mask.any():
        duplicate_ids = original_buildings.loc[
            duplicate_building_id_mask,
            id_col,
        ].head(5).tolist()
        raise ValueError(f"건물 메타데이터에 중복 ID가 있습니다: {duplicate_ids}")

    original_geometry_by_id = original_buildings.set_index(
        "_building_id_key"
    ).geometry

    buildings["_building_id_key"] = buildings[id_col].astype(str)
    buildings["original_geometry"] = buildings[
        "_building_id_key"
    ].map(original_geometry_by_id)

    missing_original_count = int(
        buildings["original_geometry"].isna().sum()
    )

    if missing_original_count > 0:
        raise ValueError(
            "메타데이터 원본 형상을 찾지 못한 버퍼 건물이 있습니다: "
            f"{missing_original_count}개"
        )

    buildings = buildings.drop(columns="_building_id_key")

    buildings = buildings.reset_index(drop=True)

    return buildings


def generate_receivers(buildings, transformer):
    """전체 지붕 수음점 생성 및 CSV 저장"""
    records = []
    center_building_count = 0
    grid_building_count = 0
    split_building_count = 0
    total_piece_count = 0

    for _, row in buildings.iterrows():
        original_part_count = len(polygon_parts(row.geometry))
        points, placement_type, piece_count = make_roof_receiver_points(
            row.geometry,
            row["original_geometry"]
        )
        total_piece_count += piece_count

        if piece_count > original_part_count:
            split_building_count += 1

        if placement_type == "center":
            center_building_count += 1
        elif placement_type == "grid":
            grid_building_count += 1

        building_id = row[id_col]
        roof_alt = float(row[top_col]) + roof_height_offset_m

        for point in points:
            x, y = point.x, point.y
            lon, lat = transformer.transform(x, y)

            records.append({
                "building_id": building_id,
                "x_epsg5179": x,
                "y_epsg5179": y,
                "lat": lat,
                "lon": lon,
                "alt": roof_alt,
            })

    if len(records) == 0:
        raise ValueError("생성된 지붕 수음점이 없습니다.")

    output_cols = [
        "building_id",
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
        "geometric-center building count:",
        center_building_count
    )
    print("grid building count:", grid_building_count)
    print("split building count:", split_building_count)
    print("total polygon piece count:", total_piece_count)
    print("roof receiver count:", len(output_df))
    print("saved:", output_csv_path)

def main():
    output_csv_path.parent.mkdir(parents=True, exist_ok=True)

    buildings = load_buildings()
    transformer = Transformer.from_crs(
        buildings.crs,
        "EPSG:4326",
        always_xy=True
    )
    generate_receivers(buildings, transformer)


if __name__ == "__main__":
    main()
