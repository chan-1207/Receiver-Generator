import math
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely
from pyproj import Transformer
from shapely.affinity import rotate
from shapely.geometry import LineString, Point, box
from shapely.ops import split, unary_union
from shapely.validation import make_valid

try:
    from scripts.pipeline_common import (
        get_env_float,
        get_env_int,
        get_env_path,
        process_map_ordered,
        validate_bounds,
        validate_input_paths,
        validate_positive,
        write_csv_atomically,
    )
except ModuleNotFoundError:
    from pipeline_common import (
        get_env_float,
        get_env_int,
        get_env_path,
        process_map_ordered,
        validate_bounds,
        validate_input_paths,
        validate_positive,
        write_csv_atomically,
    )


# =========================
# 설정값
# =========================
project_dir = Path(__file__).resolve().parents[1]

input_building_metadata_gpkg_path = get_env_path(
    "BUILDING_METADATA_INPUT_GPKG",
    project_dir / "metadata/building/building_metadata.gpkg",
)
output_csv_path = get_env_path(
    "ROOF_RECEIVER_OUTPUT_CSV",
    project_dir / "receivers/building/cropped_building_roof_receivers.csv",
)

input_building_layer_name = "building_simplified"

id_col = "NF_ID"
top_col = "BLDH_BV"

roof_resolution_m = get_env_float("RECEIVER_RESOLUTION_M", 10.0)
roof_height_offset_m = 1.0

min_x = get_env_float("RECEIVER_MIN_X", 1163000)
max_x = get_env_float("RECEIVER_MAX_X", 1164000)
min_y = get_env_float("RECEIVER_MIN_Y", 1732000)
max_y = get_env_float("RECEIVER_MAX_Y", 1733000)

rectangularity_threshold = 0.95
max_split_depth = 5
min_piece_area_m2 = 15.0
process_workers = get_env_int("PROCESS_WORKERS", 8)
building_chunk_size = get_env_int("BUILDING_CHUNK_SIZE", 250)


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


def get_section_count(length, resolution):
    """외벽 수음점과 같은 나머지 길이 기준으로 구간 수 결정"""
    if resolution <= 0:
        raise ValueError("지붕 격자 해상도는 0보다 커야 합니다.")

    full_section_count, remainder = divmod(length, resolution)
    section_count = int(full_section_count)

    if remainder > resolution / 2:
        section_count += 1

    return max(1, section_count)


def get_piece_frame(piece):
    """분할 조각을 MRR 축에 맞춘 로컬 좌표계로 변환"""
    rectangle = piece.minimum_rotated_rectangle
    edges = get_rectangle_edges(piece)
    start, end = max(edges, key=lambda edge: LineString(edge).length)
    angle = math.atan2(end[1] - start[1], end[0] - start[0])
    origin = (rectangle.centroid.x, rectangle.centroid.y)
    local_piece = rotate(piece, -angle, origin=origin, use_radians=True)
    local_rectangle = rotate(
        rectangle,
        -angle,
        origin=origin,
        use_radians=True,
    )
    return rectangle, local_piece, local_rectangle, angle, origin


def get_mrr_metrics(piece):
    """MRR 장변·단변·장축 방향 반환"""
    edge_values = []

    for start, end in get_rectangle_edges(piece):
        dx = end[0] - start[0]
        dy = end[1] - start[1]
        edge_values.append((math.hypot(dx, dy), math.atan2(dy, dx)))

    long_side, angle = max(edge_values, key=lambda item: item[0])
    short_side = min(length for length, _ in edge_values)
    return long_side, short_side, angle


def angle_difference(first, second):
    """180도 대칭을 고려한 두 MRR 축의 각도 차이"""
    difference = abs(first - second) % math.pi
    return min(difference, math.pi - difference)


def make_piece_cell_records(piece, piece_no, resolution):
    """MRR을 외벽 구간 규칙으로 분할하고 유효 셀마다 후보 생성"""
    rectangle, local_piece, local_rectangle, angle, origin = get_piece_frame(piece)
    min_local_x, min_local_y, max_local_x, max_local_y = local_rectangle.bounds
    width = max_local_x - min_local_x
    height = max_local_y - min_local_y
    column_count = get_section_count(width, resolution)
    row_count = get_section_count(height, resolution)
    cell_width = width / column_count
    cell_height = height / row_count
    records = []

    for row_no in range(row_count):
        bottom = min_local_y + row_no * cell_height
        top = min_local_y + (row_no + 1) * cell_height

        for column_no in range(column_count):
            left = min_local_x + column_no * cell_width
            right = min_local_x + (column_no + 1) * cell_width
            local_cell = box(left, bottom, right, top)
            local_overlap = clean_geom(local_cell.intersection(local_piece))

            if (
                local_overlap is None
                or local_overlap.is_empty
                or local_overlap.area <= 1e-6
            ):
                continue

            local_center = Point(
                (left + right) / 2.0,
                (bottom + top) / 2.0,
            )

            if local_piece.buffer(1e-8).covers(local_center):
                local_point = local_center
            else:
                local_point = local_overlap.representative_point()

            records.append({
                "piece_no": piece_no,
                "point": rotate(
                    local_point,
                    angle,
                    origin=origin,
                    use_radians=True,
                ),
                "cell": rotate(
                    local_cell,
                    angle,
                    origin=origin,
                    use_radians=True,
                ),
                "overlap_area_m2": float(local_overlap.area),
                "cell_width_m": cell_width,
                "cell_height_m": cell_height,
            })

    return records


def cell_size_penalty(piece, resolution):
    """10m에서 벗어난 셀 변 길이의 합"""
    return sum(
        abs(record["cell_width_m"] - resolution)
        + abs(record["cell_height_m"] - resolution)
        for record in make_piece_cell_records(piece, 1, resolution)
    )


def merge_piece_groups(first, second):
    return {
        "geometry": clean_geom(
            unary_union([first["geometry"], second["geometry"]])
        ),
        "source_piece_nos": sorted(
            first["source_piece_nos"] + second["source_piece_nos"]
        ),
    }


def merge_undersized_piece_groups(pieces, resolution):
    """좁은 자투리와 같은 방향으로 이어지는 조각을 MRR 단위로 병합"""
    groups = [
        {"geometry": piece, "source_piece_nos": [index]}
        for index, piece in enumerate(pieces, start=1)
    ]
    adjacency_tolerance_m = 0.05
    min_short_side_m = resolution * 0.75

    while True:
        undersized = [
            (index, get_mrr_metrics(group["geometry"])[1])
            for index, group in enumerate(groups)
            if get_mrr_metrics(group["geometry"])[1] < min_short_side_m
        ]

        if not undersized:
            break

        source_index, _ = min(undersized, key=lambda item: item[1])
        source = groups[source_index]
        source_penalty = cell_size_penalty(source["geometry"], resolution)
        candidates = []

        for target_index, target in enumerate(groups):
            if target_index == source_index:
                continue

            distance = source["geometry"].distance(target["geometry"])

            if distance > adjacency_tolerance_m:
                continue

            merged = merge_piece_groups(source, target)
            improvement = (
                source_penalty
                + cell_size_penalty(target["geometry"], resolution)
                - cell_size_penalty(merged["geometry"], resolution)
            )
            candidates.append((improvement, -distance, target_index, merged))

        if not candidates:
            break

        _, _, target_index, merged = max(
            candidates,
            key=lambda item: item[:2],
        )

        for index in sorted([source_index, target_index], reverse=True):
            groups.pop(index)

        groups.append(merged)

    while True:
        candidates = []

        for first_index, first in enumerate(groups):
            _, _, first_angle = get_mrr_metrics(first["geometry"])

            for second_index in range(first_index + 1, len(groups)):
                second = groups[second_index]

                if (
                    first["geometry"].distance(second["geometry"])
                    > adjacency_tolerance_m
                ):
                    continue

                _, _, second_angle = get_mrr_metrics(second["geometry"])

                if angle_difference(first_angle, second_angle) > math.radians(10):
                    continue

                merged = merge_piece_groups(first, second)
                merged_geom = merged["geometry"]

                if get_rectangularity(merged_geom) < 0.8:
                    continue

                improvement = (
                    cell_size_penalty(first["geometry"], resolution)
                    + cell_size_penalty(second["geometry"], resolution)
                    - cell_size_penalty(merged_geom, resolution)
                )

                if improvement <= 1e-6:
                    continue

                merged_long, merged_short, _ = get_mrr_metrics(merged_geom)
                candidates.append((
                    improvement,
                    merged_long / max(merged_short, 1e-9),
                    first_index,
                    second_index,
                    merged,
                ))

        if not candidates:
            break

        _, _, first_index, second_index, merged = max(
            candidates,
            key=lambda item: item[:2],
        )

        for index in sorted([first_index, second_index], reverse=True):
            groups.pop(index)

        groups.append(merged)

    groups.sort(key=lambda group: min(group["source_piece_nos"]))
    return groups


def deduplicate_cell_records(records):
    unique = []
    seen = set()

    for record in records:
        point = record["point"]
        key = (round(point.x, 8), round(point.y, 8))

        if key in seen:
            continue

        seen.add(key)
        unique.append(record)

    return unique


def resolve_overlapping_candidates(records, resolution, coverage_geom):
    """다른 MRR의 근접·중첩 후보 중 지붕 유효면적이 작은 후보 제거"""
    numbered_records = [
        dict(record, candidate_no=index)
        for index, record in enumerate(records, start=1)
    ]
    conflict_distance_m = resolution / 2.0

    def conflicts(first, second):
        if first["piece_no"] == second["piece_no"]:
            return False

        if first["point"].distance(second["point"]) >= conflict_distance_m:
            return False

        return first["cell"].intersection(second["cell"]).area > 1e-6

    accepted = []
    rejected = []

    for record in sorted(
        numbered_records,
        key=lambda item: (-item["overlap_area_m2"], item["candidate_no"]),
    ):
        blockers = [other for other in accepted if conflicts(record, other)]

        if blockers:
            rejected.append(record)
        else:
            accepted.append(record)

    while rejected:
        covered = unary_union([
            record["point"].buffer(resolution)
            for record in accepted
        ])
        uncovered = clean_geom(coverage_geom.difference(covered))

        if uncovered is None or uncovered.is_empty or uncovered.area <= 1e-6:
            break

        coverage_gains = [
            (
                uncovered.intersection(
                    record["point"].buffer(resolution)
                ).area,
                index,
            )
            for index, record in enumerate(rejected)
        ]
        gain, restore_index = max(coverage_gains)

        if gain <= 1e-6:
            break

        accepted.append(rejected.pop(restore_index))

    accepted.sort(key=lambda item: item["candidate_no"])
    return accepted


def get_centered_axis_values(
    min_value,
    max_value,
    resolution,
    center_value=None,
):
    """기준 중심에 맞추고 범위 안에 유지되는 등간격 좌표 생성"""
    span = max_value - min_value

    if span <= 0:
        return np.array([(min_value + max_value) / 2.0])

    if center_value is None:
        center_value = (min_value + max_value) / 2.0

    center_value = min(max(center_value, min_value), max_value)

    # 기존 중심 +/- n * 해상도 방식은 항상 홀수 개만 만들기 때문에
    # 해상도 이상 2배 미만인 범위에도 점을 하나만 배치했다.
    # 범위에 들어가는 최대 점 개수를 먼저 구해 짝수 개 배치도 허용한다.
    point_count = math.floor((span + 1e-9) / resolution) + 1
    used_span = (point_count - 1) * resolution
    start = center_value - used_span / 2.0

    # MRR 중심이 안전영역 경계 상자의 중앙과 다르더라도 모든 점을 범위 안에 둔다.
    start = max(min_value, min(start, max_value - used_span))

    return start + np.arange(point_count) * resolution


def make_oriented_grid_candidates(poly, resolution, orientation_poly=None):
    """기준 폴리곤의 MRR 방향에 맞춘 격자 후보점 생성"""
    if poly is None or poly.is_empty:
        return []

    if resolution <= 0:
        raise ValueError("지붕 격자 해상도는 0보다 커야 합니다.")

    if orientation_poly is None or orientation_poly.is_empty:
        orientation_poly = poly

    # 축소 전 분할 조각의 최소 면적 회전 사각형으로 격자 방향을 고정한다.
    rectangle = orientation_poly.minimum_rotated_rectangle
    edges = get_rectangle_edges(orientation_poly)
    start, end = max(edges, key=lambda edge: LineString(edge).length)
    angle = math.atan2(end[1] - start[1], end[0] - start[0])

    # 회전 사각형의 긴 변이 좌표축과 나란해지도록 회전한다.
    origin = (rectangle.centroid.x, rectangle.centroid.y)
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
        return [get_geometric_center(poly)]

    # 분할 조각의 MRR 중심축에 맞춰 정확한 해상도 간격의 후보를 만든다.
    # 폭이 해상도보다 좁으면 MRR 장축을 따르는 1차원 배치가 된다.
    x_values = get_centered_axis_values(
        min_x,
        max_x,
        resolution,
        center_value=origin[0],
    )
    y_values = get_centered_axis_values(
        min_y,
        max_y,
        resolution,
        center_value=origin[1],
    )

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
    """단순화 건물의 분할 MRR 셀을 기준으로 지붕 수음점 생성"""
    if geom is None or geom.is_empty:
        return [], "empty", 0

    simplified_geom = clean_geom(geom)

    if simplified_geom is None or simplified_geom.is_empty:
        return [], "empty", 0

    split_pieces = decompose_geometry(simplified_geom)
    piece_groups = merge_undersized_piece_groups(
        split_pieces,
        roof_resolution_m,
    )
    records = []

    for piece_no, group in enumerate(piece_groups, start=1):
        records.extend(
            make_piece_cell_records(
                group["geometry"],
                piece_no,
                roof_resolution_m,
            )
        )

    records = deduplicate_cell_records(records)
    records = resolve_overlapping_candidates(
        records,
        roof_resolution_m,
        simplified_geom,
    )

    if len(records) == 0:
        return [get_geometric_center(simplified_geom)], "center", len(piece_groups)

    points = [record["point"] for record in records]
    placement_type = "grid" if len(points) > 1 else "center"
    return points, placement_type, len(piece_groups)


def load_buildings():
    """단순화 건물 데이터 로드 및 필터링"""
    buildings = gpd.read_file(
        input_building_metadata_gpkg_path,
        layer=input_building_layer_name,
    )

    if buildings.crs is None:
        raise ValueError("단순화 건물 GPKG에 CRS가 없습니다.")

    if buildings.crs.is_geographic:
        raise ValueError("미터 단위의 투영 좌표계를 사용해야 합니다.")

    required_cols = [id_col, top_col]
    missing_cols = [
        col for col in required_cols
        if col not in buildings.columns
    ]

    if missing_cols:
        raise ValueError(
            f"건물 레이어에 필수 필드가 없습니다: {missing_cols}"
        )

    buildings = buildings[buildings.geometry.notnull()].copy()
    buildings["geometry"] = shapely.make_valid(
        buildings.geometry.to_numpy()
    )
    buildings = buildings[buildings.geometry.notnull()].copy()
    buildings = buildings[~buildings.geometry.is_empty].copy()

    buildings[top_col] = pd.to_numeric(
        buildings[top_col],
        errors="coerce"
    )
    buildings = buildings[
        buildings[id_col].notna()
        & buildings[top_col].notna()
    ].copy()

    duplicate_building_id_mask = buildings[id_col].astype(str).duplicated(
        keep=False
    )

    if duplicate_building_id_mask.any():
        duplicate_ids = buildings.loc[
            duplicate_building_id_mask,
            id_col,
        ].head(5).tolist()
        raise ValueError(f"단순화 건물에 중복 ID가 있습니다: {duplicate_ids}")

    buildings = buildings.reset_index(drop=True)

    return buildings


def make_roof_receiver_records(building_data):
    """단일 건물의 지붕 수음점과 통계 반환"""
    building_id, roof_top, geom = building_data
    records = []
    original_part_count = len(polygon_parts(geom))
    points, placement_type, piece_count = make_roof_receiver_points(geom)
    center_count = int(placement_type == "center")
    grid_count = int(placement_type == "grid")
    split_count = int(piece_count > original_part_count)
    roof_alt = float(roof_top) + roof_height_offset_m

    for point in points:
        x, y = point.x, point.y
        if not (min_x <= x <= max_x and min_y <= y <= max_y):
            continue

        records.append((building_id, x, y, roof_alt))

    return records, center_count, grid_count, split_count, piece_count


def generate_receivers(buildings, transformer):
    """전체 지붕 수음점 생성 및 CSV 저장"""
    print("process workers:", process_workers)
    print("maximum building chunk size:", building_chunk_size)
    building_values = buildings[
        [id_col, top_col, "geometry"]
    ].itertuples(index=False, name=None)
    building_results = process_map_ordered(
        make_roof_receiver_records,
        building_values,
        process_workers,
        building_chunk_size,
    )
    records = [
        record
        for building_records, _, _, _, _ in building_results
        for record in building_records
    ]
    center_building_count = sum(result[1] for result in building_results)
    grid_building_count = sum(result[2] for result in building_results)
    split_building_count = sum(result[3] for result in building_results)
    total_piece_count = sum(result[4] for result in building_results)

    if len(records) == 0:
        raise ValueError("생성된 지붕 수음점이 없습니다.")

    projected_cols = [
        "building_id",
        "x_epsg5179",
        "y_epsg5179",
        "alt",
    ]
    output_df = pd.DataFrame(records, columns=projected_cols)
    lon, lat = transformer.transform(
        output_df["x_epsg5179"].to_numpy(dtype=float),
        output_df["y_epsg5179"].to_numpy(dtype=float),
    )
    output_df["lat"] = lat
    output_df["lon"] = lon
    output_cols = [
        "building_id",
        "x_epsg5179",
        "y_epsg5179",
        "lat",
        "lon",
        "alt",
    ]
    output_df = output_df[output_cols]
    write_csv_atomically(
        output_df,
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
    validate_bounds(min_x, max_x, min_y, max_y)
    validate_positive(roof_resolution_m, "지붕 수음점 해상도")
    validate_positive(process_workers, "프로세스 작업 수")
    validate_positive(building_chunk_size, "건물 묶음 크기")
    validate_input_paths([
        input_building_metadata_gpkg_path,
    ])
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
