from pathlib import Path
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait

import geopandas as gpd
import pandas as pd
import numpy as np
import pyogrio
import shapely

from pyproj import CRS, Transformer
from shapely.geometry import box
from shapely.strtree import STRtree

try:
    from scripts.pipeline_common import (
        get_env_float,
        get_env_int,
        get_env_path,
        validate_bounds,
        validate_input_paths,
        validate_positive,
        validate_spatial_file_coverage,
    )
except ModuleNotFoundError:
    from pipeline_common import (
        get_env_float,
        get_env_int,
        get_env_path,
        validate_bounds,
        validate_input_paths,
        validate_positive,
        validate_spatial_file_coverage,
    )


# =========================
# 설정값
# =========================
project_dir = Path(__file__).resolve().parents[1]

input_contour_path = get_env_path(
    "TERRAIN_CONTOUR_INPUT_SHP",
    project_dir / "data/terrain/terrain.shp",
)
input_land_cover_path = get_env_path(
    "LAND_COVER_INPUT_GPKG",
    project_dir / "data/land_cover_map/land_cover_map.gpkg",
)
ground_factor_mapping_path = get_env_path(
    "GROUND_FACTOR_MAPPING_CSV",
    project_dir / "config/land_cover_ground_factor.csv",
)
output_csv_path = get_env_path(
    "TERRAIN_RECEIVER_OUTPUT_CSV",
    project_dir / "receivers/terrain/cropped_terrain_receivers_center.csv",
)

land_cover_layer_name = "land_cover_map"
land_cover_code_field = "L2_CODE"
elevation_field = "CONT"
receiver_crs = "EPSG:5179"
coverage_tolerance = 1.0e-4
spatial_epsilon = 1.0e-6

grid_m = get_env_float("RECEIVER_RESOLUTION_M", 10.0)

min_x = get_env_float("RECEIVER_MIN_X", 1163000)
max_x = get_env_float("RECEIVER_MAX_X", 1164000)
min_y = get_env_float("RECEIVER_MIN_Y", 1732000)
max_y = get_env_float("RECEIVER_MAX_Y", 1733000)

receiver_height_m = 1.5

spatial_chunk_size = 100_000
idw_search_radius_m = get_env_float(
    "TERRAIN_IDW_SEARCH_RADIUS_M",
    800.0,
)
idw_max_search_radius_m = get_env_float(
    "TERRAIN_IDW_MAX_SEARCH_RADIUS_M",
    2000.0,
)
idw_min_contours = get_env_int("TERRAIN_IDW_MIN_CONTOURS", 4)
idw_max_contours = get_env_int("TERRAIN_IDW_MAX_CONTOURS", 8)
idw_min_elevation_levels = get_env_int(
    "TERRAIN_IDW_MIN_ELEVATION_LEVELS",
    2,
)
idw_workers = get_env_int("TERRAIN_IDW_WORKERS", 8)
contour_distance_chunk_size = get_env_int(
    "TERRAIN_IDW_CHUNK_SIZE",
    2_000,
)
idw_zero_distance_tolerance_m = 1.0e-9
IDW_POWER = 2.0
contour_simplify_tolerance_m = min(2.0, grid_m / 5.0)

save_debug_grid = False
debug_grid_csv_path = project_dir / "receivers/terrain/debug_grid_10m.csv"


# =========================
# 격자 생성 함수
# =========================
def make_grid():
    transformer = Transformer.from_crs(
        "EPSG:5179",
        "EPSG:4326",
        always_xy=True
    )

    x_cell_count = (max_x - min_x) / grid_m
    y_cell_count = (max_y - min_y) / grid_m
    if not np.isclose(x_cell_count, round(x_cell_count)):
        raise ValueError("X 범위는 격자 크기의 정수배여야 합니다.")
    if not np.isclose(y_cell_count, round(y_cell_count)):
        raise ValueError("Y 범위는 격자 크기의 정수배여야 합니다.")

    # 셀 중앙 수음점 배치
    xs = min_x + (np.arange(int(round(x_cell_count))) + 0.5) * grid_m
    ys = min_y + (np.arange(int(round(y_cell_count))) + 0.5) * grid_m

    # 행 우선 배열 기반 수음점 좌표 생성
    grid_x, grid_y = np.meshgrid(xs, ys, indexing="xy")
    flat_x = grid_x.ravel(order="C")
    flat_y = grid_y.ravel(order="C")
    lon, lat = transformer.transform(flat_x, flat_y)

    return pd.DataFrame({
        "x_epsg5179": flat_x,
        "y_epsg5179": flat_y,
        "lat": lat,
        "lon": lon,
    })


# =========================
# 지면계수 매핑 함수
# =========================
def read_ground_factor_mapping(mapping_path):
    mapping_df = pd.read_csv(
        mapping_path,
        dtype={"L2_CODE": "string"},
    )
    required_cols = ["L2_CODE", "ground_factor"]
    missing_cols = [
        col for col in required_cols
        if col not in mapping_df.columns
    ]
    if missing_cols:
        raise ValueError(f"지면계수 매핑 필드가 없습니다: {missing_cols}")

    mapping_df["L2_CODE"] = mapping_df["L2_CODE"].str.strip()
    mapping_df["ground_factor"] = pd.to_numeric(
        mapping_df["ground_factor"],
        errors="raise",
    )

    duplicate_codes = mapping_df.loc[
        mapping_df["L2_CODE"].duplicated(keep=False),
        "L2_CODE",
    ].tolist()
    if duplicate_codes:
        raise ValueError(f"중복된 토지피복 코드가 있습니다: {duplicate_codes}")

    invalid_factor_mask = ~mapping_df["ground_factor"].between(0.0, 1.0)
    if invalid_factor_mask.any():
        invalid_codes = mapping_df.loc[
            invalid_factor_mask,
            "L2_CODE",
        ].tolist()
        raise ValueError(f"지면계수 범위를 벗어난 코드가 있습니다: {invalid_codes}")

    return mapping_df.set_index("L2_CODE")["ground_factor"]


# =========================
# 셀 지면계수 계산 함수
# =========================
def assign_ground_factor_legacy(grid_df):
    half_grid_m = grid_m / 2.0
    cell_geometry = [
        box(
            x - half_grid_m,
            y - half_grid_m,
            x + half_grid_m,
            y + half_grid_m,
        )
        for x, y in zip(
            grid_df["x_epsg5179"],
            grid_df["y_epsg5179"],
        )
    ]
    cells = gpd.GeoDataFrame(
        {"_cell_id": np.arange(len(grid_df), dtype=np.int64)},
        geometry=cell_geometry,
        crs=receiver_crs,
    )

    land_cover_info = pyogrio.read_info(
        input_land_cover_path,
        layer=land_cover_layer_name,
    )
    land_cover_crs = land_cover_info.get("crs")
    if land_cover_crs is None:
        raise ValueError("토지피복도에 CRS가 없습니다.")

    # 셀 범위에 한정된 토지피복 폴리곤 로드
    cell_bounds = gpd.GeoSeries(
        [box(min_x, min_y, max_x, max_y)],
        crs=receiver_crs,
    ).to_crs(land_cover_crs).total_bounds
    land_cover = gpd.read_file(
        input_land_cover_path,
        layer=land_cover_layer_name,
        bbox=tuple(cell_bounds),
    )
    if land_cover_code_field not in land_cover.columns:
        raise ValueError(
            f"토지피복 코드 필드가 없습니다: {land_cover_code_field}"
        )
    if land_cover.empty:
        raise ValueError("수음점 셀과 겹치는 토지피복 폴리곤이 없습니다.")

    land_cover = land_cover[
        land_cover.geometry.notna()
        & ~land_cover.geometry.is_empty
    ].copy()
    land_cover[land_cover_code_field] = (
        land_cover[land_cover_code_field].astype("string").str.strip()
    )
    if land_cover[land_cover_code_field].isna().any():
        raise ValueError("토지피복 코드가 비어 있는 폴리곤이 있습니다.")

    ground_factor_mapping = read_ground_factor_mapping(
        ground_factor_mapping_path
    )
    unknown_codes = sorted(
        set(land_cover[land_cover_code_field])
        - set(ground_factor_mapping.index)
    )
    if unknown_codes:
        raise ValueError(f"지면계수가 정의되지 않은 토지피복 코드: {unknown_codes}")

    land_cover["_surface_factor"] = land_cover[
        land_cover_code_field
    ].map(ground_factor_mapping)
    land_cover = land_cover.to_crs(receiver_crs)

    intersections = gpd.overlay(
        cells,
        land_cover[["_surface_factor", "geometry"]],
        how="intersection",
        keep_geom_type=False,
    )
    intersections["_intersection_area"] = intersections.geometry.area
    intersections = intersections[
        intersections["_intersection_area"] > 0.0
    ].copy()
    intersections["_weighted_factor"] = (
        intersections["_intersection_area"]
        * intersections["_surface_factor"]
    )

    summary = intersections.groupby("_cell_id").agg(
        covered_area=("_intersection_area", "sum"),
        weighted_factor=("_weighted_factor", "sum"),
    ).reindex(cells["_cell_id"])

    cell_area = grid_m * grid_m
    coverage = summary["covered_area"] / cell_area
    invalid_coverage = (
        coverage.isna()
        | (coverage < 1.0 - coverage_tolerance)
        | (coverage > 1.0 + coverage_tolerance)
    )
    if invalid_coverage.any():
        invalid_ids = summary.index[invalid_coverage].tolist()
        example_lines = []
        for cell_id in invalid_ids[:10]:
            coverage_value = coverage.loc[cell_id]
            status = (
                "미피복"
                if pd.isna(coverage_value) or coverage_value < 1.0
                else "중복"
            )
            row = grid_df.iloc[cell_id]
            coverage_text = (
                "없음"
                if pd.isna(coverage_value)
                else f"{coverage_value:.6f}"
            )
            example_lines.append(
                f" - cell={cell_id}, "
                f"x={row['x_epsg5179']:.3f}, "
                f"y={row['y_epsg5179']:.3f}, "
                f"상태={status}, 피복률={coverage_text}"
            )
        raise ValueError(
            "토지피복도가 계산 격자를 완전히 덮지 못하거나 중복됩니다.\n"
            f"오류 셀 수: {len(invalid_ids)}\n"
            + "\n".join(example_lines)
        )

    result = grid_df.copy()
    result["ground_factor"] = (
        summary["weighted_factor"] / summary["covered_area"]
    ).clip(0.0, 1.0).to_numpy()

    print("[지면계수]")
    print(" - land cover:", input_land_cover_path)
    print(" - mapped cells:", len(result))
    print(" - min:", result["ground_factor"].min())
    print(" - max:", result["ground_factor"].max())

    return result


def assign_ground_factor(grid_df):
    half_grid_m = grid_m / 2.0
    cell_area = grid_m * grid_m
    result_factors = np.full(len(grid_df), np.nan, dtype=float)

    land_cover_info = pyogrio.read_info(
        input_land_cover_path,
        layer=land_cover_layer_name,
    )
    land_cover_crs = land_cover_info.get("crs")
    if land_cover_crs is None:
        raise ValueError("토지피복도에 CRS가 없습니다.")

    cell_bounds = gpd.GeoSeries(
        [box(min_x, min_y, max_x, max_y)],
        crs=receiver_crs,
    ).to_crs(land_cover_crs).total_bounds
    land_cover = gpd.read_file(
        input_land_cover_path,
        layer=land_cover_layer_name,
        bbox=tuple(cell_bounds),
    )
    if land_cover_code_field not in land_cover.columns:
        raise ValueError(
            f"토지피복 코드 필드가 없습니다: {land_cover_code_field}"
        )
    if land_cover.empty:
        raise ValueError("수음점과 겹치는 토지피복 레이어가 없습니다.")

    land_cover = land_cover[
        land_cover.geometry.notna()
        & ~land_cover.geometry.is_empty
    ].copy()
    land_cover[land_cover_code_field] = (
        land_cover[land_cover_code_field].astype("string").str.strip()
    )
    if land_cover[land_cover_code_field].isna().any():
        raise ValueError("토지피복 코드가 비어 있는 레이어가 있습니다.")

    ground_factor_mapping = read_ground_factor_mapping(
        ground_factor_mapping_path
    )
    unknown_codes = sorted(
        set(land_cover[land_cover_code_field])
        - set(ground_factor_mapping.index)
    )
    if unknown_codes:
        raise ValueError(
            f"지면계수가 정의되지 않은 토지피복 코드: {unknown_codes}"
        )

    land_cover["_surface_factor"] = land_cover[
        land_cover_code_field
    ].map(ground_factor_mapping)
    land_cover = land_cover.to_crs(receiver_crs)
    land_geometries = land_cover.geometry.to_numpy()
    land_factors = land_cover["_surface_factor"].to_numpy(dtype=float)
    shapely.prepare(land_geometries)
    land_tree = STRtree(land_geometries)
    land_boundaries = shapely.boundary(land_geometries)
    boundary_search_distance = half_grid_m * np.sqrt(2.0)
    grid_x = grid_df["x_epsg5179"].to_numpy(dtype=float)
    grid_y = grid_df["y_epsg5179"].to_numpy(dtype=float)
    chunk_count = int(np.ceil(len(grid_df) / spatial_chunk_size))
    direct_cell_count = 0
    exact_cell_count = 0

    print("[지면계수 공간계산]")
    print(" - cell count:", len(grid_df))
    print(" - land cover polygons:", len(land_cover))
    print(" - chunk size:", spatial_chunk_size)

    for chunk_no, start in enumerate(
        range(0, len(grid_df), spatial_chunk_size),
        start=1,
    ):
        end = min(start + spatial_chunk_size, len(grid_df))
        chunk_length = end - start
        points = shapely.points(
            grid_x[start:end],
            grid_y[start:end],
        )
        center_bbox_pairs = land_tree.query(points)
        center_bbox_cell_ids = center_bbox_pairs[0]
        center_bbox_land_ids = center_bbox_pairs[1]
        center_inside = shapely.contains(
            land_geometries[center_bbox_land_ids],
            points[center_bbox_cell_ids],
        )
        center_cell_ids = center_bbox_cell_ids[center_inside]
        center_land_ids = center_bbox_land_ids[center_inside]
        center_count = np.bincount(
            center_cell_ids,
            minlength=chunk_length,
        )
        exact_required = center_count != 1

        center_single_mask = center_count[center_cell_ids] == 1
        center_single_cell_ids = center_cell_ids[center_single_mask]
        center_single_land_ids = center_land_ids[center_single_mask]
        center_boundary_distance = shapely.distance(
            points[center_single_cell_ids],
            land_boundaries[center_single_land_ids],
        )
        near_boundary = (
            center_boundary_distance
            <= boundary_search_distance + spatial_epsilon
        )
        exact_required[
            center_single_cell_ids[near_boundary]
        ] = True
        direct_mask = ~near_boundary
        direct_ids = center_single_cell_ids[direct_mask]
        direct_land_ids = center_single_land_ids[direct_mask]
        result_factors[start + direct_ids] = land_factors[direct_land_ids]
        direct_cell_count += len(direct_ids)

        boundary_ids = np.flatnonzero(exact_required)
        boundary_cells = shapely.box(
            grid_x[start + boundary_ids] - half_grid_m,
            grid_y[start + boundary_ids] - half_grid_m,
            grid_x[start + boundary_ids] + half_grid_m,
            grid_y[start + boundary_ids] + half_grid_m,
        )
        candidate_bbox_pairs = land_tree.query(boundary_cells)
        candidate_bbox_boundary_ids = candidate_bbox_pairs[0]
        candidate_bbox_land_ids = candidate_bbox_pairs[1]
        candidate_intersects = shapely.intersects(
            land_geometries[candidate_bbox_land_ids],
            boundary_cells[candidate_bbox_boundary_ids],
        )
        boundary_pair_ids = candidate_bbox_boundary_ids[
            candidate_intersects
        ]
        candidate_land_ids = candidate_bbox_land_ids[
            candidate_intersects
        ]
        candidate_count = np.bincount(
            boundary_pair_ids,
            minlength=len(boundary_ids),
        )
        candidate_factor_min = np.full(
            len(boundary_ids),
            np.inf,
            dtype=float,
        )
        candidate_factor_max = np.full(
            len(boundary_ids),
            -np.inf,
            dtype=float,
        )
        np.minimum.at(
            candidate_factor_min,
            boundary_pair_ids,
            land_factors[candidate_land_ids],
        )
        np.maximum.at(
            candidate_factor_max,
            boundary_pair_ids,
            land_factors[candidate_land_ids],
        )
        resolved_boundary = np.zeros(len(boundary_ids), dtype=bool)

        single_pair_mask = candidate_count[boundary_pair_ids] == 1
        single_boundary_pair_ids = boundary_pair_ids[single_pair_mask]
        single_land_ids = candidate_land_ids[single_pair_mask]
        if len(single_boundary_pair_ids) > 0:
            fully_covered = shapely.covers(
                land_geometries[single_land_ids],
                boundary_cells[single_boundary_pair_ids],
            )
            direct_boundary_ids = single_boundary_pair_ids[fully_covered]
            direct_land_ids = single_land_ids[fully_covered]
            direct_chunk_ids = boundary_ids[direct_boundary_ids]
            result_factors[start + direct_chunk_ids] = land_factors[direct_land_ids]
            resolved_boundary[direct_boundary_ids] = True
            direct_cell_count += len(direct_boundary_ids)

        same_factor_boundary_ids = np.flatnonzero(
            (candidate_count > 1)
            & (candidate_factor_min == candidate_factor_max)
        )
        same_factor_chunk_ids = boundary_ids[same_factor_boundary_ids]
        result_factors[start + same_factor_chunk_ids] = candidate_factor_min[
            same_factor_boundary_ids
        ]
        resolved_boundary[same_factor_boundary_ids] = True
        direct_cell_count += len(same_factor_boundary_ids)

        intersection_required = ~resolved_boundary
        exact_pair_mask = intersection_required[boundary_pair_ids]
        exact_pair_cell_ids = boundary_ids[
            boundary_pair_ids[exact_pair_mask]
        ]
        exact_pair_land_ids = candidate_land_ids[exact_pair_mask]
        exact_ids = boundary_ids[intersection_required]
        exact_cell_count += len(exact_ids)

        covered_area = np.zeros(chunk_length, dtype=float)
        weighted_factor = np.zeros(chunk_length, dtype=float)
        if len(exact_pair_cell_ids) > 0:
            intersection_area = shapely.area(
                shapely.intersection(
                    shapely.box(
                        grid_x[start + exact_pair_cell_ids] - half_grid_m,
                        grid_y[start + exact_pair_cell_ids] - half_grid_m,
                        grid_x[start + exact_pair_cell_ids] + half_grid_m,
                        grid_y[start + exact_pair_cell_ids] + half_grid_m,
                    ),
                    land_geometries[exact_pair_land_ids],
                )
            )
            positive_area = intersection_area > 0.0
            area_cell_ids = exact_pair_cell_ids[positive_area]
            area_land_ids = exact_pair_land_ids[positive_area]
            area_values = intersection_area[positive_area]
            covered_area += np.bincount(
                area_cell_ids,
                weights=area_values,
                minlength=chunk_length,
            )
            weighted_factor += np.bincount(
                area_cell_ids,
                weights=area_values * land_factors[area_land_ids],
                minlength=chunk_length,
            )

        exact_coverage = covered_area[exact_ids] / cell_area
        invalid_exact = (
            (exact_coverage < 1.0 - coverage_tolerance)
            | (exact_coverage > 1.0 + coverage_tolerance)
        )
        if invalid_exact.any():
            invalid_local_ids = exact_ids[invalid_exact]
            invalid_global_ids = start + invalid_local_ids
            example_lines = []
            for local_id, global_id in zip(
                invalid_local_ids[:10],
                invalid_global_ids[:10],
            ):
                coverage_value = covered_area[local_id] / cell_area
                status = "미피복" if coverage_value < 1.0 else "중복"
                row = grid_df.iloc[global_id]
                example_lines.append(
                    f" - cell={global_id}, "
                    f"x={row['x_epsg5179']:.3f}, "
                    f"y={row['y_epsg5179']:.3f}, "
                    f"상태={status}, 피복률={coverage_value:.6f}"
                )
            raise ValueError(
                "토지피복도가 계산 격자를 완전히 채우지 못하거나 "
                "중복됩니다.\n"
                f"현재 묶음 오류 셀 수: {len(invalid_local_ids)}\n"
                + "\n".join(example_lines)
            )

        result_factors[start + exact_ids] = (
            weighted_factor[exact_ids] / covered_area[exact_ids]
        )
        print(
            f" - chunk {chunk_no}/{chunk_count}: "
            f"{end:,}/{len(grid_df):,}"
        )

    if not np.isfinite(result_factors).all():
        invalid_ids = np.flatnonzero(~np.isfinite(result_factors))[:10]
        raise ValueError(
            "지면계수를 계산하지 못한 셀이 있습니다: "
            f"{invalid_ids.tolist()}"
        )

    result = grid_df.copy()
    result["ground_factor"] = np.clip(result_factors, 0.0, 1.0)

    print("[지면계수]")
    print(" - land cover:", input_land_cover_path)
    print(" - mapped cells:", len(result))
    print(" - direct cells:", direct_cell_count)
    print(" - exact intersection cells:", exact_cell_count)
    print(" - min:", result["ground_factor"].min())
    print(" - max:", result["ground_factor"].max())

    return result


# =========================
# 등고선 객체 로드 함수
# =========================
def read_contours(contour_path):
    """IDW 입력 등고선 객체 로드"""
    search_bounds = (
        min_x - idw_max_search_radius_m,
        min_y - idw_max_search_radius_m,
        max_x + idw_max_search_radius_m,
        max_y + idw_max_search_radius_m,
    )
    contour_info = pyogrio.read_info(contour_path)
    contour_crs_value = contour_info.get("crs")
    if contour_crs_value is None:
        raise ValueError(f"등고선 CRS가 없습니다: {contour_path}")
    contour_crs = CRS.from_user_input(contour_crs_value)
    source_bounds = search_bounds
    if contour_crs.to_epsg() != 5179:
        bounds_transformer = Transformer.from_crs(
            receiver_crs,
            contour_crs,
            always_xy=True,
        )
        source_bounds = bounds_transformer.transform_bounds(
            *search_bounds,
            densify_pts=21,
        )

    contour_gdf = gpd.read_file(contour_path, bbox=source_bounds)
    if elevation_field not in contour_gdf.columns:
        raise ValueError(
            f"등고선 고도 필드가 없습니다: {elevation_field}"
        )
    if contour_gdf.crs.to_epsg() != 5179:
        contour_gdf = contour_gdf.to_crs(receiver_crs)

    contour_gdf[elevation_field] = pd.to_numeric(
        contour_gdf[elevation_field],
        errors="coerce",
    )
    valid_mask = (
        contour_gdf.geometry.notna()
        & ~contour_gdf.geometry.is_empty
        & np.isfinite(contour_gdf[elevation_field])
    )
    contour_gdf = contour_gdf.loc[valid_mask].copy()
    if contour_gdf.empty:
        raise ValueError("IDW 계산에 사용할 등고선 객체가 없습니다.")

    simplified_geometries = shapely.simplify(
        contour_gdf.geometry.to_numpy(),
        contour_simplify_tolerance_m,
        preserve_topology=False,
    )
    simplified_valid_mask = ~shapely.is_empty(simplified_geometries)
    contours = simplified_geometries[simplified_valid_mask]
    elevations = contour_gdf.loc[
        simplified_valid_mask,
        elevation_field,
    ].to_numpy(dtype=float)

    print("[IDW 등고선 입력]")
    print(" - path:", contour_path)
    print(" - contour features:", len(contours))
    print(" - simplification tolerance:", contour_simplify_tolerance_m, "m")
    print(" - elevation min:", elevations.min())
    print(" - elevation max:", elevations.max())
    return contours, elevations


# =========================
# 등고선별 최근접 거리 계산 함수
# =========================
def query_nearest_contours(points, contour_tree, contours, elevations, radius):
    """반경 내 등고선별 최근접 거리 행렬 반환"""
    point_count = len(points)
    distance_matrix = np.full(
        (point_count, idw_max_contours),
        np.inf,
        dtype=float,
    )
    elevation_matrix = np.full(
        (point_count, idw_max_contours),
        np.nan,
        dtype=float,
    )
    candidate_pairs = contour_tree.query(
        points,
        predicate="dwithin",
        distance=radius,
    )
    if candidate_pairs.shape[1] == 0:
        return distance_matrix, elevation_matrix

    point_ids = candidate_pairs[0]
    contour_ids = candidate_pairs[1]
    distances = shapely.distance(
        points[point_ids],
        contours[contour_ids],
    )
    within_mask = distances <= radius + spatial_epsilon
    point_ids = point_ids[within_mask]
    contour_ids = contour_ids[within_mask]
    distances = distances[within_mask]
    order = np.lexsort((contour_ids, distances, point_ids))
    point_ids = point_ids[order]
    contour_ids = contour_ids[order]
    distances = distances[order]
    group_starts = np.flatnonzero(
        np.r_[True, point_ids[1:] != point_ids[:-1]]
    )
    ranks = np.arange(len(point_ids)) - np.repeat(
        group_starts,
        np.diff(np.r_[group_starts, len(point_ids)]),
    )
    selected_mask = ranks < idw_max_contours
    selected_point_ids = point_ids[selected_mask]
    selected_contour_ids = contour_ids[selected_mask]
    selected_ranks = ranks[selected_mask]
    distance_matrix[selected_point_ids, selected_ranks] = distances[
        selected_mask
    ]
    elevation_matrix[selected_point_ids, selected_ranks] = elevations[
        selected_contour_ids
    ]
    return distance_matrix, elevation_matrix


def count_elevation_levels(elevation_matrix):
    """수음점별 서로 다른 등고선 표고 수 반환"""
    valid_mask = np.isfinite(elevation_matrix)
    sorted_elevations = np.sort(
        np.where(valid_mask, elevation_matrix, np.inf),
        axis=1,
    )
    lower_levels = sorted_elevations[:, :-1]
    upper_levels = sorted_elevations[:, 1:]
    finite_pairs = np.isfinite(lower_levels) & np.isfinite(upper_levels)
    elevation_differences = np.zeros_like(lower_levels)
    np.subtract(
        upper_levels,
        lower_levels,
        out=elevation_differences,
        where=finite_pairs,
    )
    level_changes = finite_pairs & (np.abs(elevation_differences) > 1.0e-6)
    return valid_mask.any(axis=1).astype(np.int64) + level_changes.sum(axis=1)


def calculate_neighbor_idw(distance_matrix, elevation_matrix, receiver_xy):
    """등고선 최근접 거리 기반 IDW 고도 반환"""
    valid_mask = np.isfinite(distance_matrix) & np.isfinite(elevation_matrix)
    exact_mask = valid_mask & (
        distance_matrix <= idw_zero_distance_tolerance_m
    )
    exact_rows = exact_mask.any(axis=1)
    for row_id in np.flatnonzero(exact_rows):
        exact_elevations = elevation_matrix[row_id, exact_mask[row_id]]
        if exact_elevations.max() - exact_elevations.min() > 1.0e-6:
            raise ValueError(
                "수음점과 교차하는 등고선의 표고가 서로 다릅니다.\n"
                f"좌표: x={receiver_xy[row_id, 0]:.3f}, "
                f"y={receiver_xy[row_id, 1]:.3f}\n"
                f"표고: {exact_elevations.tolist()}"
            )

    weights = np.zeros_like(distance_matrix, dtype=float)
    weighted_mask = valid_mask & ~exact_rows[:, None]
    weights[weighted_mask] = (
        1.0 / distance_matrix[weighted_mask] ** IDW_POWER
    )
    ground_z = np.zeros(len(distance_matrix), dtype=float)
    weighted_rows = ~exact_rows
    ground_z[weighted_rows] = (
        (weights * np.nan_to_num(elevation_matrix)).sum(axis=1)[weighted_rows]
        / weights.sum(axis=1)[weighted_rows]
    )
    if exact_rows.any():
        exact_columns = exact_mask.argmax(axis=1)
        ground_z[exact_rows] = elevation_matrix[
            np.arange(len(elevation_matrix))[exact_rows],
            exact_columns[exact_rows],
        ]
    return ground_z, exact_rows


# =========================
# 등고선 IDW 묶음 계산 함수
# =========================
def calculate_contour_idw_chunk(
    chunk_no,
    start,
    end,
    recv_x,
    recv_y,
    contour_tree,
    contours,
    elevations,
):
    """단일 수음점 묶음의 IDW 계산 결과 반환"""
    receiver_coordinates = np.column_stack([
        recv_x[start:end],
        recv_y[start:end],
    ])
    receiver_points = shapely.points(
        receiver_coordinates[:, 0],
        receiver_coordinates[:, 1],
    )
    distance_matrix, elevation_matrix = query_nearest_contours(
        receiver_points,
        contour_tree,
        contours,
        elevations,
        idw_search_radius_m,
    )
    valid_mask = np.isfinite(distance_matrix)
    contour_counts = valid_mask.sum(axis=1)
    elevation_level_counts = count_elevation_levels(elevation_matrix)
    exact_mask = valid_mask & (
        distance_matrix <= idw_zero_distance_tolerance_m
    )
    exact_rows = exact_mask.any(axis=1)
    fallback_rows = (~exact_rows) & (
        (contour_counts < idw_min_contours)
        | (elevation_level_counts < idw_min_elevation_levels)
    )

    if fallback_rows.any():
        fallback_distances, fallback_elevations = query_nearest_contours(
            receiver_points[fallback_rows],
            contour_tree,
            contours,
            elevations,
            idw_max_search_radius_m,
        )
        distance_matrix[fallback_rows] = fallback_distances
        elevation_matrix[fallback_rows] = fallback_elevations
        valid_mask = np.isfinite(distance_matrix)
        contour_counts = valid_mask.sum(axis=1)
        elevation_level_counts = count_elevation_levels(elevation_matrix)

    insufficient_mask = (~exact_rows) & (
        (contour_counts < idw_min_contours)
        | (elevation_level_counts < idw_min_elevation_levels)
    )
    if insufficient_mask.any():
        example_lines = []
        for local_id in np.flatnonzero(insufficient_mask)[:10]:
            global_id = start + local_id
            example_lines.append(
                f" - cell={global_id}, "
                f"x={recv_x[global_id]:.3f}, "
                f"y={recv_y[global_id]:.3f}, "
                f"등고선={contour_counts[local_id]}, "
                f"표고 단계={elevation_level_counts[local_id]}"
            )
        raise ValueError(
            "IDW 최대 반경 안에서 최소 등고선 조건을 "
            "충족하지 못했습니다.\n"
            f"최대 반경: {idw_max_search_radius_m:.1f}m\n"
            f"최소 등고선 개수: {idw_min_contours}\n"
            f"최소 표고 단계 수: {idw_min_elevation_levels}\n"
            f"실패 수음점 수: {int(insufficient_mask.sum()):,}\n"
            + "\n".join(example_lines)
        )

    chunk_ground_z, exact_rows = calculate_neighbor_idw(
        distance_matrix,
        elevation_matrix,
        receiver_coordinates,
    )
    effective_contour_counts = contour_counts.copy()
    effective_level_counts = elevation_level_counts.copy()
    effective_contour_counts[exact_rows] = 1
    effective_level_counts[exact_rows] = 1
    used_mask = valid_mask & ~exact_rows[:, None]
    used_distances = distance_matrix[used_mask]
    maximum_used_distance = (
        float(used_distances.max())
        if len(used_distances) > 0
        else 0.0
    )
    return {
        "chunk_no": chunk_no,
        "start": start,
        "end": end,
        "ground_z": chunk_ground_z,
        "fallback_count": int(fallback_rows.sum()),
        "exact_count": int(exact_rows.sum()),
        "selected_total": int(effective_contour_counts.sum()),
        "selected_minimum": int(effective_contour_counts.min()),
        "selected_maximum": int(effective_contour_counts.max()),
        "level_total": int(effective_level_counts.sum()),
        "level_minimum": int(effective_level_counts.min()),
        "level_maximum": int(effective_level_counts.max()),
        "maximum_used_distance": maximum_used_distance,
    }


# =========================
# 병렬 IDW 묶음 실행 함수
# =========================
def iterate_contour_idw_chunks(
    chunk_ranges,
    recv_x,
    recv_y,
    contour_tree,
    contours,
    elevations,
):
    """설정된 작업 수에 따른 IDW 묶음 결과 반환"""
    def calculate(chunk_range):
        chunk_no, start, end = chunk_range
        return calculate_contour_idw_chunk(
            chunk_no,
            start,
            end,
            recv_x,
            recv_y,
            contour_tree,
            contours,
            elevations,
        )

    if idw_workers == 1:
        for chunk_range in chunk_ranges:
            yield calculate(chunk_range)
        return

    chunk_iterator = iter(chunk_ranges)
    with ThreadPoolExecutor(max_workers=idw_workers) as executor:
        pending = {}
        for _ in range(min(idw_workers, len(chunk_ranges))):
            chunk_range = next(chunk_iterator)
            pending[executor.submit(calculate, chunk_range)] = chunk_range

        while pending:
            completed, _ = wait(pending, return_when=FIRST_COMPLETED)
            for future in completed:
                pending.pop(future)
                try:
                    yield future.result()
                except Exception:
                    for pending_future in pending:
                        pending_future.cancel()
                    raise

                next_chunk_range = next(chunk_iterator, None)
                if next_chunk_range is not None:
                    pending[
                        executor.submit(calculate, next_chunk_range)
                    ] = next_chunk_range


# =========================
# 적응형 등고선 IDW 고도 계산 함수
# =========================
def calculate_contour_idw_ground_z(contours, elevations, grid_df):
    """기준 반경과 최소 등고선 조건 기반 IDW 고도 계산"""
    recv_x = grid_df["x_epsg5179"].to_numpy(dtype=float)
    recv_y = grid_df["y_epsg5179"].to_numpy(dtype=float)
    contour_tree = STRtree(contours)
    ground_z = np.full(len(grid_df), np.nan, dtype=float)
    chunk_ranges = [
        (
            chunk_no,
            start,
            min(start + contour_distance_chunk_size, len(grid_df)),
        )
        for chunk_no, start in enumerate(
            range(0, len(grid_df), contour_distance_chunk_size),
            start=1,
        )
    ]
    chunk_count = len(chunk_ranges)
    fallback_receiver_count = 0
    exact_receiver_count = 0
    selected_contour_total = 0
    selected_contour_minimum = idw_max_contours
    selected_contour_maximum = 0
    elevation_level_total = 0
    elevation_level_minimum = idw_max_contours
    elevation_level_maximum = 0
    maximum_used_distance = 0.0
    completed_receiver_count = 0

    print("[적응형 등고선 IDW 지면고도]")
    print(" - receivers:", len(grid_df))
    print(" - reference radius:", idw_search_radius_m, "m")
    print(" - maximum radius:", idw_max_search_radius_m, "m")
    print(" - minimum contours:", idw_min_contours)
    print(" - maximum contours:", idw_max_contours)
    print(" - minimum elevation levels:", idw_min_elevation_levels)
    print(" - power:", IDW_POWER)
    print(" - workers:", idw_workers)
    print(" - chunk size:", contour_distance_chunk_size)

    chunk_results = iterate_contour_idw_chunks(
        chunk_ranges,
        recv_x,
        recv_y,
        contour_tree,
        contours,
        elevations,
    )
    for completed_chunk_count, result in enumerate(chunk_results, start=1):
        start = result["start"]
        end = result["end"]
        ground_z[start:end] = result["ground_z"]
        fallback_receiver_count += result["fallback_count"]
        exact_receiver_count += result["exact_count"]
        selected_contour_total += result["selected_total"]
        selected_contour_minimum = min(
            selected_contour_minimum,
            result["selected_minimum"],
        )
        selected_contour_maximum = max(
            selected_contour_maximum,
            result["selected_maximum"],
        )
        elevation_level_total += result["level_total"]
        elevation_level_minimum = min(
            elevation_level_minimum,
            result["level_minimum"],
        )
        elevation_level_maximum = max(
            elevation_level_maximum,
            result["level_maximum"],
        )
        maximum_used_distance = max(
            maximum_used_distance,
            result["maximum_used_distance"],
        )
        completed_receiver_count += end - start
        print(
            f" - completed chunks {completed_chunk_count}/{chunk_count}: "
            f"{completed_receiver_count:,}/{len(grid_df):,}"
        )

    if not np.isfinite(ground_z).all():
        invalid_ids = np.flatnonzero(~np.isfinite(ground_z))[:10]
        raise ValueError(
            "IDW 고도를 계산하지 못한 수음점이 있습니다: "
            f"{invalid_ids.tolist()}"
        )

    print("[IDW 계산 결과]")
    print(" - exact contour points:", exact_receiver_count)
    print(" - expanded-radius receivers:", fallback_receiver_count)
    print(
        " - selected contours min/mean/max:",
        selected_contour_minimum,
        f"{selected_contour_total / len(grid_df):.1f}",
        selected_contour_maximum,
    )
    print(
        " - elevation levels min/mean/max:",
        elevation_level_minimum,
        f"{elevation_level_total / len(grid_df):.1f}",
        elevation_level_maximum,
    )
    print(" - maximum used distance:", f"{maximum_used_distance:.1f}m")
    print(" - elevation min:", ground_z.min())
    print(" - elevation max:", ground_z.max())
    return ground_z


# =========================
# 수음점 생성 함수
# =========================
def make_terrain_receivers():
    grid_df = assign_ground_factor(make_grid())
    contours, elevations = read_contours(input_contour_path)
    ground_z = calculate_contour_idw_ground_z(
        contours,
        elevations,
        grid_df,
    )

    receivers = grid_df.copy()
    receivers["ground_alt"] = ground_z
    receivers["alt"] = receivers["ground_alt"] + receiver_height_m

    return receivers


def main():
    validate_bounds(min_x, max_x, min_y, max_y)
    validate_positive(grid_m, "지면 수음점 해상도")
    validate_positive(idw_search_radius_m, "IDW 기준 반경")
    validate_positive(idw_max_search_radius_m, "IDW 최대 반경")
    validate_positive(idw_min_contours, "IDW 최소 등고선 개수")
    validate_positive(idw_max_contours, "IDW 최대 등고선 개수")
    validate_positive(
        idw_min_elevation_levels,
        "IDW 최소 표고 단계 수",
    )
    validate_positive(idw_workers, "IDW 병렬 작업 수")
    validate_positive(contour_distance_chunk_size, "IDW 묶음 크기")
    if idw_search_radius_m > idw_max_search_radius_m:
        raise ValueError("IDW 기준 반경은 최대 반경보다 클 수 없습니다.")
    if idw_min_contours > idw_max_contours:
        raise ValueError(
            "IDW 최소 등고선 개수는 최대 등고선 개수보다 클 수 없습니다."
        )
    if idw_min_elevation_levels > idw_max_contours:
        raise ValueError(
            "IDW 최소 표고 단계 수는 최대 등고선 개수보다 클 수 없습니다."
        )
    validate_input_paths([
        input_contour_path,
        input_land_cover_path,
        ground_factor_mapping_path,
    ])
    area_bounds = (min_x, min_y, max_x, max_y)
    validate_spatial_file_coverage(
        path=input_land_cover_path,
        label="토지피복도",
        required_bounds=area_bounds,
        layer=land_cover_layer_name,
    )
    contour_bounds = (
        min_x - idw_max_search_radius_m,
        min_y - idw_max_search_radius_m,
        max_x + idw_max_search_radius_m,
        max_y + idw_max_search_radius_m,
    )
    validate_spatial_file_coverage(
        path=input_contour_path,
        label="등고선",
        required_bounds=contour_bounds,
    )
    output_csv_path.parent.mkdir(parents=True, exist_ok=True)

    # =========================
    # 실행
    # =========================
    receivers = make_terrain_receivers()

    print("[수음점 생성 결과]")
    print(" - receivers:", len(receivers))
    print(" - output:", output_csv_path)

    if save_debug_grid:
        receivers.to_csv(
            debug_grid_csv_path,
            index=False,
            encoding="utf-8-sig"
        )
        print(" - debug saved:", debug_grid_csv_path)

    # =========================
    # CSV 저장
    # =========================
    output_cols = [
        "ground_factor",
        "x_epsg5179",
        "y_epsg5179",
        "lat",
        "lon",
        "alt",
    ]

    out_df = receivers[output_cols].copy()

    out_df.to_csv(
        output_csv_path,
        index=False,
        encoding="utf-8-sig",
        float_format="%.7f"
    )

    print("[저장 완료]")
    print(" - csv:", output_csv_path)
    print(" - rows:", len(out_df))
    print(out_df.head())


if __name__ == "__main__":
    main()
