from pathlib import Path
import math

import geopandas as gpd
import pandas as pd
import numpy as np
import shapely
from shapely.geometry import LineString
from shapely.strtree import STRtree
from pyproj import Transformer

try:
    from scripts.pipeline_common import (
        get_env_float,
        get_env_int,
        get_env_path,
        parallel_map_ordered,
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
        parallel_map_ordered,
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

input_polygon_gpkg_path = get_env_path(
    "RECEIVER_BUFFER_INPUT_GPKG",
    project_dir / "receivers/building/cropped_building_buffers_10m.gpkg",
)
output_csv_path = get_env_path(
    "WALL_RECEIVER_OUTPUT_CSV",
    project_dir / "receivers/building/cropped_building_receivers.csv",
)

input_layer_name = "building_buffer"

id_col = "NF_ID"
base_col = "BLDH_MN"   # 건물 지반 절대고도
top_col = "BLDH_BV"    # 건물 기본/지붕 절대고도

wall_resolution_m = get_env_float("RECEIVER_RESOLUTION_M", 10.0)
vertical_resolution_m = get_env_float("RECEIVER_RESOLUTION_M", 10.0)
start_height_m = 1.5
min_building_height_m = 1.5

min_x = get_env_float("RECEIVER_MIN_X", 1163000)
max_x = get_env_float("RECEIVER_MAX_X", 1164000)
min_y = get_env_float("RECEIVER_MIN_Y", 1732000)
max_y = get_env_float("RECEIVER_MAX_Y", 1733000)

z_tolerance_m = 0.05
process_workers = get_env_int("PROCESS_WORKERS", 8)
thread_workers = get_env_int("THREAD_WORKERS", 8)
building_chunk_size = get_env_int("BUILDING_CHUNK_SIZE", 250)

# =========================
# 보조 함수
# =========================
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


def get_section_count(length: float, resolution: float):
    # 남는 길이가 단위 간격의 절반 이하면 버리고, 초과하면 구간을 하나 추가
    if resolution <= 0:
        raise ValueError("resolution must be greater than 0")

    full_section_count, remainder = divmod(length, resolution)
    section_count = int(full_section_count)

    if remainder > resolution / 2:
        section_count += 1

    return max(1, section_count)


def interpolate_points_on_line(line: LineString, resolution: float):
    length = line.length

    if length <= 0:
        return []

    if resolution <= 0:
        raise ValueError("wall resolution must be greater than 0")

    # 버퍼 폴리곤의 각 벽면을 단위 간격을 기준으로 균등하게 나누어 수음점 배치
    # 폴리곤의 꼭지점에 수음점 배치
    # 남는 길이가 단위 간격의 절반 이하면 버리고, 초과하면 구간을 하나 추가
    section_count = get_section_count(length, resolution)
    section_length = length / section_count

    return [
        line.interpolate(i * section_length)
        for i in range(section_count)
    ]


def get_exterior_receiver_points(geom, resolution: float):
    points = []
    seen_xy = set()

    for edge_no, segment in enumerate(get_exterior_segments(geom), start=1):
        for point in interpolate_points_on_line(segment, resolution):
            xy = (point.x, point.y)

            if xy in seen_xy:
                continue

            seen_xy.add(xy)
            points.append((point, edge_no))

    return points


def make_vertical_heights(building_h):
    if building_h <= 0:
        return np.array([])

    if vertical_resolution_m <= 0:
        raise ValueError("vertical resolution must be greater than 0")

    # 건물 높이가 최하단 설정 높이 이하이면 최상단에만 수음점을 배치
    if building_h <= start_height_m:
        return np.array([building_h])

    # 최하단 설정 높이부터 건물 최상단까지의 높이 차를 기준 간격으로 나눈다.
    # 나눈 구간의 길이가 모두 같도록 배치하며, 최하단과 최상단을 모두 포함
    vertical_length = building_h - start_height_m
    section_count = get_section_count(vertical_length, vertical_resolution_m)

    return np.linspace(
        start_height_m,
        building_h,
        section_count + 1
    )


def find_blocked_receivers(receivers, conflict_gdf):
    """공유 공간 인덱스를 스레드로 조회해 다른 건물 버퍼와 겹치는 점을 찾는다."""
    receiver_geometries = shapely.points(
        receivers["x_epsg5179"].to_numpy(dtype=float),
        receivers["y_epsg5179"].to_numpy(dtype=float),
    )
    conflict_geometries = conflict_gdf.geometry.to_numpy()
    conflict_tree = STRtree(conflict_geometries)
    receiver_building_ids = receivers["building_id"].to_numpy()
    receiver_altitudes = receivers["alt"].to_numpy(dtype=float)
    conflict_building_ids = conflict_gdf[id_col].to_numpy()
    conflict_top_altitudes = conflict_gdf[top_col].to_numpy(dtype=float)
    point_count = len(receivers)
    effective_chunk_size = max(
        10_000,
        min(
            100_000,
            math.ceil(point_count / max(1, thread_workers * 4)),
        ),
    )
    chunk_ranges = [
        (start, min(start + effective_chunk_size, point_count))
        for start in range(0, point_count, effective_chunk_size)
    ]

    def calculate(chunk_range):
        start, end = chunk_range
        chunk_blocked = np.zeros(end - start, dtype=bool)
        candidate_pairs = conflict_tree.query(
            receiver_geometries[start:end],
            predicate="intersects",
        )
        if candidate_pairs.shape[1] == 0:
            return start, chunk_blocked

        local_receiver_ids = candidate_pairs[0]
        conflict_ids = candidate_pairs[1]
        receiver_ids = receiver_building_ids[start:end][local_receiver_ids]
        receiver_z = receiver_altitudes[start:end][local_receiver_ids]
        other_building = receiver_ids != conflict_building_ids[conflict_ids]
        below_other_roof = (
            receiver_z
            <= conflict_top_altitudes[conflict_ids] + z_tolerance_m
        )
        chunk_blocked[local_receiver_ids[other_building & below_other_roof]] = True
        return start, chunk_blocked

    print("overlap query thread workers:", thread_workers)
    print("overlap query chunk size:", effective_chunk_size)
    blocked = np.zeros(point_count, dtype=bool)
    for start, chunk_blocked in parallel_map_ordered(
        calculate,
        chunk_ranges,
        thread_workers,
        1,
    ):
        blocked[start:start + len(chunk_blocked)] = chunk_blocked
    return blocked


def make_wall_receiver_records(building_data):
    """단일 건물의 벽면 수음점 후보 반환"""
    building_id, base, building_h, geom = building_data
    records = []
    outside_xy_count = 0

    if geom is None or geom.is_empty:
        return records, outside_xy_count

    heights = make_vertical_heights(float(building_h))
    if len(heights) == 0:
        return records, outside_xy_count

    wall_points = get_exterior_receiver_points(geom, wall_resolution_m)
    for point, edge_no in wall_points:
        x, y = point.x, point.y
        if not (min_x <= x <= max_x and min_y <= y <= max_y):
            outside_xy_count += 1
            continue

        for height in heights:
            records.append((
                building_id,
                edge_no,
                x,
                y,
                float(base) + float(height),
            ))

    return records, outside_xy_count


def main():
    validate_bounds(min_x, max_x, min_y, max_y)
    validate_positive(wall_resolution_m, "벽면 수음점 수평 해상도")
    validate_positive(vertical_resolution_m, "벽면 수음점 수직 해상도")
    validate_positive(process_workers, "프로세스 작업 수")
    validate_positive(thread_workers, "스레드 작업 수")
    validate_positive(building_chunk_size, "건물 묶음 크기")
    validate_input_paths([input_polygon_gpkg_path])
    output_csv_path.parent.mkdir(parents=True, exist_ok=True)

    # =========================
    # 버퍼 폴리곤 로드
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
    # 데이터 정리
    # =========================
    buf = buf[buf.geometry.notnull()].copy()
    buf["geometry"] = shapely.make_valid(buf.geometry.to_numpy())
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
    # 좌표 변환기
    # =========================
    # =========================
    # 수음점 후보 생성
    # =========================
    print("process workers:", process_workers)
    print("maximum building chunk size:", building_chunk_size)
    building_values = buf[
        [id_col, base_col, "building_height", "geometry"]
    ].itertuples(index=False, name=None)
    building_results = process_map_ordered(
        make_wall_receiver_records,
        building_values,
        process_workers,
        building_chunk_size,
    )
    records = [
        record
        for building_records, _ in building_results
        for record in building_records
    ]
    outside_xy_count = sum(
        outside_count
        for _, outside_count in building_results
    )

    receivers = pd.DataFrame(
        records,
        columns=[
            "building_id",
            "edge_no",
            "x_epsg5179",
            "y_epsg5179",
            "alt",
        ],
    )

    print("candidate receivers count:", len(receivers))
    print("outside XY candidate count:", outside_xy_count)

    if len(receivers) == 0:
        raise ValueError("생성된 수음점이 없습니다. 버퍼 폴리곤과 높이 필드를 확인하세요.")

    # =========================
    # 3D 높이 기반 겹침 필터링
    # =========================
    # 같은 버퍼 레이어를 충돌 판단용으로 사용
    conflict_gdf = buf[[id_col, top_col, "geometry"]].copy()
    conflict_gdf = gpd.GeoDataFrame(conflict_gdf, geometry="geometry", crs=buf.crs)

    receivers["is_blocked"] = find_blocked_receivers(
        receivers,
        conflict_gdf,
    )

    filtered = receivers[~receivers["is_blocked"]].copy()

    print("blocked receivers count:", int(receivers["is_blocked"].sum()))
    print("final receivers count:", len(filtered))

    # =========================
    # CSV 저장
    # =========================
    transformer = Transformer.from_crs(buf.crs, "EPSG:4326", always_xy=True)
    lon, lat = transformer.transform(
        filtered["x_epsg5179"].to_numpy(dtype=float),
        filtered["y_epsg5179"].to_numpy(dtype=float),
    )
    filtered["lat"] = lat
    filtered["lon"] = lon

    output_cols = [
        "building_id",
        "edge_no",
        "x_epsg5179",
        "y_epsg5179",
        "lat",
        "lon",
        "alt",
    ]

    out_df = filtered[output_cols].copy()
    write_csv_atomically(
        out_df,
        output_csv_path,
        index=False,
        encoding="utf-8-sig",
    )

    print(f"saved: {output_csv_path}")
    print(f"receivers count: {len(out_df)}")
    print(out_df.head())


if __name__ == "__main__":
    main()
