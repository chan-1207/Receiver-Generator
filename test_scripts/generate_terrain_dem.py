import time
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.transform import from_origin
from scipy.spatial import Delaunay, QhullError
from shapely.geometry import LineString, MultiLineString

try:
    from scripts.pipeline_common import (
        TERRAIN_DEM_SOURCE_PADDING_M,
        get_env_float,
        get_env_path,
        validate_bounds,
        validate_input_paths,
        validate_positive,
    )
except ModuleNotFoundError:
    from pipeline_common import (
        TERRAIN_DEM_SOURCE_PADDING_M,
        get_env_float,
        get_env_path,
        validate_bounds,
        validate_input_paths,
        validate_positive,
    )


# 경로 및 좌표계 설정
project_dir = Path(__file__).resolve().parents[1]
input_contour_path = get_env_path(
    "TERRAIN_CONTOUR_INPUT_SHP",
    project_dir / "data/terrain/terrain.shp",
)
output_dem_path = get_env_path(
    "TERRAIN_DEM_OUTPUT_TIF",
    project_dir / "metadata/terrain/terrain_dem.tif",
)
receiver_crs = "EPSG:5179"
elevation_field = "CONT"


# DEM 영역 및 해상도 설정
min_x = get_env_float("RECEIVER_MIN_X", 1163000)
max_x = get_env_float("RECEIVER_MAX_X", 1164000)
min_y = get_env_float("RECEIVER_MIN_Y", 1732000)
max_y = get_env_float("RECEIVER_MAX_Y", 1733000)
dem_resolution_m = get_env_float(
    "TERRAIN_DEM_RESOLUTION_M",
    get_env_float("RECEIVER_RESOLUTION_M", 10.0),
)
source_padding_m = get_env_float(
    "TERRAIN_DEM_SOURCE_PADDING_M",
    TERRAIN_DEM_SOURCE_PADDING_M,
)
contour_simplify_tolerance_m = min(2.0, dem_resolution_m / 5.0)
output_chunk_rows = 100
nodata_value = -9999.0


def validate_dem_grid():
    """DEM 영역과 해상도 검증"""
    validate_bounds(min_x, max_x, min_y, max_y)
    validate_positive(dem_resolution_m, "DEM 해상도")
    validate_positive(source_padding_m, "DEM 원본 여유 범위")

    x_cell_count = (max_x - min_x) / dem_resolution_m
    y_cell_count = (max_y - min_y) / dem_resolution_m
    if not np.isclose(x_cell_count, round(x_cell_count)):
        raise ValueError(
            "X 영역 길이는 DEM 해상도의 정수배여야 합니다: "
            f"{max_x - min_x} / {dem_resolution_m}"
        )
    if not np.isclose(y_cell_count, round(y_cell_count)):
        raise ValueError(
            "Y 영역 길이는 DEM 해상도의 정수배여야 합니다: "
            f"{max_y - min_y} / {dem_resolution_m}"
        )


def iter_lines(geometry):
    """선형 지오메트리 구성요소 반복"""
    if isinstance(geometry, LineString):
        yield geometry
    elif isinstance(geometry, MultiLineString):
        yield from geometry.geoms


def read_tin_vertices():
    """등고선 기반 TIN 꼭짓점 배열 생성"""
    source_bounds = (
        min_x - source_padding_m,
        min_y - source_padding_m,
        max_x + source_padding_m,
        max_y + source_padding_m,
    )
    contour_gdf = gpd.read_file(input_contour_path, bbox=source_bounds)
    if contour_gdf.crs is None:
        raise ValueError(f"등고선 CRS가 없습니다: {input_contour_path}")
    if contour_gdf.crs.to_epsg() != 5179:
        contour_gdf = contour_gdf.to_crs(receiver_crs)
    if elevation_field not in contour_gdf.columns:
        raise ValueError(
            f"등고선 고도 필드가 없습니다: {elevation_field}"
        )

    coordinate_parts = []
    elevation_parts = []
    simplified_geometries = contour_gdf.geometry.simplify(
        contour_simplify_tolerance_m,
        preserve_topology=False,
    )
    for geometry, elevation_value in zip(
        simplified_geometries,
        contour_gdf[elevation_field],
    ):
        if geometry is None or geometry.is_empty:
            continue
        try:
            elevation = float(elevation_value)
        except (TypeError, ValueError):
            continue
        if not np.isfinite(elevation):
            continue

        for line in iter_lines(geometry):
            coordinates = np.asarray(line.coords, dtype=float)
            if len(coordinates) < 2:
                continue
            coordinate_parts.append(coordinates[:, :2])
            elevation_parts.append(
                np.full(len(coordinates), elevation, dtype=float)
            )

    if not coordinate_parts:
        raise ValueError("TIN 생성에 사용할 등고선 좌표가 없습니다.")

    coordinates = np.concatenate(coordinate_parts)
    elevations = np.concatenate(elevation_parts)
    unique_coordinates, inverse = np.unique(
        coordinates,
        axis=0,
        return_inverse=True,
    )
    minimum_elevations = np.full(len(unique_coordinates), np.inf)
    maximum_elevations = np.full(len(unique_coordinates), -np.inf)
    np.minimum.at(minimum_elevations, inverse, elevations)
    np.maximum.at(maximum_elevations, inverse, elevations)

    conflict_mask = (
        maximum_elevations - minimum_elevations
    ) > 1.0e-6
    if conflict_mask.any():
        examples = unique_coordinates[conflict_mask][:10]
        raise ValueError(
            "동일한 좌표에 서로 다른 등고선 고도가 있습니다.\n"
            f"예시 좌표:\n{examples}"
        )

    print("[등고선 TIN 입력]")
    print(" - contour features:", len(contour_gdf))
    print(" - simplify tolerance:", contour_simplify_tolerance_m, "m")
    print(" - unique vertices:", len(unique_coordinates))
    print(" - elevation min:", minimum_elevations.min())
    print(" - elevation max:", minimum_elevations.max())
    return unique_coordinates, minimum_elevations


def interpolate_dem(tin, elevations):
    """TIN 평면 기반 DEM 배열 생성"""
    width = int(round((max_x - min_x) / dem_resolution_m))
    height = int(round((max_y - min_y) / dem_resolution_m))
    x_centers = min_x + (
        np.arange(width, dtype=float) + 0.5
    ) * dem_resolution_m
    dem = np.full((height, width), nodata_value, dtype=np.float32)
    missing_examples = []
    missing_count = 0
    chunk_count = int(np.ceil(height / output_chunk_rows))

    for chunk_no, row_start in enumerate(
        range(0, height, output_chunk_rows),
        start=1,
    ):
        row_end = min(row_start + output_chunk_rows, height)
        y_centers = max_y - (
            np.arange(row_start, row_end, dtype=float) + 0.5
        ) * dem_resolution_m
        points = np.column_stack([
            np.tile(x_centers, len(y_centers)),
            np.repeat(y_centers, width),
        ])
        simplex_ids = tin.find_simplex(points)
        valid_mask = simplex_ids >= 0
        if not valid_mask.all():
            missing_points = points[~valid_mask]
            missing_count += len(missing_points)
            if len(missing_examples) < 10:
                missing_examples.extend(
                    missing_points[:10 - len(missing_examples)].tolist()
                )

        valid_simplex_ids = simplex_ids[valid_mask]
        offsets = (
            points[valid_mask]
            - tin.transform[valid_simplex_ids, 2]
        )
        first_weights = np.einsum(
            "nij,nj->ni",
            tin.transform[valid_simplex_ids, :2],
            offsets,
        )
        weights = np.column_stack([
            first_weights,
            1.0 - first_weights.sum(axis=1),
        ])
        interpolated = np.einsum(
            "ni,ni->n",
            elevations[tin.simplices[valid_simplex_ids]],
            weights,
        )
        block = np.full(len(points), nodata_value, dtype=np.float32)
        block[valid_mask] = interpolated.astype(np.float32)
        dem[row_start:row_end] = block.reshape(row_end - row_start, width)
        print(
            f" - DEM chunk {chunk_no}/{chunk_count}: "
            f"{row_end:,}/{height:,} rows"
        )

    if missing_count:
        raise ValueError(
            "등고선 TIN이 DEM 생성 영역 전체를 덮지 못합니다.\n"
            f"누락 셀 수: {missing_count:,}\n"
            f"예시 좌표: {missing_examples}"
        )

    return dem


def write_dem(dem):
    """좌표계 포함 GeoTIFF DEM 저장"""
    output_dem_path.parent.mkdir(parents=True, exist_ok=True)
    transform = from_origin(
        min_x,
        max_y,
        dem_resolution_m,
        dem_resolution_m,
    )
    with rasterio.open(
        output_dem_path,
        "w",
        driver="GTiff",
        width=dem.shape[1],
        height=dem.shape[0],
        count=1,
        dtype="float32",
        crs=receiver_crs,
        transform=transform,
        nodata=nodata_value,
        compress="deflate",
        predictor=3,
        tiled=True,
        blockxsize=256,
        blockysize=256,
    ) as dataset:
        dataset.write(dem, 1)
        dataset.update_tags(
            source=str(input_contour_path),
            elevation_field=elevation_field,
            interpolation="Delaunay TIN linear",
            contour_simplify_tolerance_m=(
                contour_simplify_tolerance_m
            ),
        )


def main():
    """등고선 기반 DEM 생성 실행"""
    started_at = time.perf_counter()
    validate_dem_grid()
    validate_input_paths([input_contour_path])
    vertices, elevations = read_tin_vertices()

    print("[TIN 생성]")
    try:
        tin = Delaunay(vertices)
    except QhullError as error:
        raise ValueError("등고선 TIN 생성에 실패했습니다.") from error
    print(" - triangles:", len(tin.simplices))

    print("[DEM 보간]")
    dem = interpolate_dem(tin, elevations)
    write_dem(dem)

    elapsed_seconds = time.perf_counter() - started_at
    print("[DEM 생성 완료]")
    print(" - output:", output_dem_path)
    print(" - shape:", dem.shape)
    print(" - resolution:", dem_resolution_m, "m")
    print(" - elevation min:", float(dem.min()))
    print(" - elevation max:", float(dem.max()))
    print(" - elapsed:", f"{elapsed_seconds:.1f}s")


if __name__ == "__main__":
    main()
