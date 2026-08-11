from pathlib import Path

import geopandas as gpd
import pandas as pd
import numpy as np
import pyogrio
import rasterio
import shapely

from pyproj import Transformer
from shapely.geometry import box
from shapely.strtree import STRtree

try:
    from scripts.pipeline_common import (
        get_env_float,
        get_env_path,
        validate_bounds,
        validate_input_paths,
        validate_positive,
        validate_spatial_file_coverage,
    )
except ModuleNotFoundError:
    from pipeline_common import (
        get_env_float,
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

input_dem_path = get_env_path(
    "TERRAIN_DEM_INPUT_TIF",
    project_dir / "metadata/terrain/terrain_dem.tif",
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
# DEM 고도 추출 함수
# =========================
def read_dem_ground_z(dem_path, grid_df):
    recv_x = grid_df["x_epsg5179"].to_numpy(dtype=float)
    recv_y = grid_df["y_epsg5179"].to_numpy(dtype=float)

    with rasterio.open(dem_path) as dataset:
        if dataset.crs is None:
            raise ValueError(f"DEM CRS가 없습니다: {dem_path}")
        if dataset.crs.to_epsg() != 5179:
            raise ValueError(
                "DEM 좌표계는 EPSG:5179여야 합니다: "
                f"{dataset.crs}"
            )
        if not np.isclose(dataset.transform.b, 0.0) or not np.isclose(
            dataset.transform.d,
            0.0,
        ):
            raise ValueError("회전된 DEM 격자는 지원하지 않습니다.")

        pixel_width = abs(dataset.transform.a)
        pixel_height = abs(dataset.transform.e)
        if not np.isclose(pixel_width, grid_m) or not np.isclose(
            pixel_height,
            grid_m,
        ):
            raise ValueError(
                "DEM 해상도와 수음점 해상도가 다릅니다: "
                f"DEM={pixel_width}x{pixel_height}m, "
                f"수음점={grid_m}m"
            )

        bounds = dataset.bounds
        tolerance = max(pixel_width, pixel_height) * 1.0e-6
        if (
            bounds.left > min_x + tolerance
            or bounds.right < max_x - tolerance
            or bounds.bottom > min_y + tolerance
            or bounds.top < max_y - tolerance
        ):
            raise ValueError(
                "DEM이 계산 영역 전체를 포함하지 않습니다.\n"
                f"DEM: X={bounds.left:.1f}~{bounds.right:.1f}, "
                f"Y={bounds.bottom:.1f}~{bounds.top:.1f}\n"
                f"영역: X={min_x:.1f}~{max_x:.1f}, "
                f"Y={min_y:.1f}~{max_y:.1f}"
            )

        columns = np.floor(
            (recv_x - dataset.transform.c) / pixel_width
        ).astype(np.int64)
        rows = np.floor(
            (dataset.transform.f - recv_y) / pixel_height
        ).astype(np.int64)
        inside = (
            (rows >= 0)
            & (rows < dataset.height)
            & (columns >= 0)
            & (columns < dataset.width)
        )
        if not inside.all():
            raise ValueError(
                "DEM 바깥에 위치한 지면 수음점이 있습니다: "
                f"{int((~inside).sum()):,}개"
            )

        dem = dataset.read(1, masked=True)
        sampled = dem[rows, columns]
        ground_z = np.asarray(sampled.filled(np.nan), dtype=float)

        print("[DEM 지면고도]")
        print(" - path:", dem_path)
        print(" - crs:", dataset.crs)
        print(" - resolution:", pixel_width, "m")
        print(" - receivers:", len(ground_z))

    missing_mask = ~np.isfinite(ground_z)
    if missing_mask.any():
        missing_grid = grid_df.loc[
            missing_mask,
            ["x_epsg5179", "y_epsg5179"],
        ].head(10)
        raise ValueError(
            "DEM 고도가 없는 지면 수음점이 있습니다.\n"
            f"누락 수음점 수: {int(missing_mask.sum()):,}\n"
            f"예시 좌표:\n{missing_grid.to_string(index=False)}"
        )

    return ground_z


# =========================
# 수음점 생성 함수
# =========================
def make_terrain_receivers():
    grid_df = assign_ground_factor(make_grid())
    ground_z = read_dem_ground_z(input_dem_path, grid_df)

    receivers = grid_df.copy()
    receivers["ground_alt"] = ground_z
    receivers["alt"] = receivers["ground_alt"] + receiver_height_m

    return receivers


def main():
    validate_bounds(min_x, max_x, min_y, max_y)
    validate_positive(grid_m, "지면 수음점 해상도")
    validate_input_paths([
        input_dem_path,
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
