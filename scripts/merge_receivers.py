from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd

from shapely.geometry import Point

try:
    from scripts.pipeline_common import (
        get_env_bool,
        get_env_float,
        get_env_path,
        validate_bounds,
        validate_input_paths,
        validate_positive,
    )
except ModuleNotFoundError:
    from pipeline_common import (
        get_env_bool,
        get_env_float,
        get_env_path,
        validate_bounds,
        validate_input_paths,
        validate_positive,
    )


# =========================
# 설정값
# =========================
project_dir = Path(__file__).resolve().parents[1]

terrain_receiver_csv_path = get_env_path(
    "MERGE_TERRAIN_INPUT_CSV",
    project_dir / "receivers/terrain/cropped_terrain_receivers_center.csv",
)
building_receiver_csv_path = get_env_path(
    "MERGE_WALL_INPUT_CSV",
    project_dir / "receivers/building/cropped_building_receivers.csv",
)
roof_receiver_csv_path = get_env_path(
    "MERGE_ROOF_INPUT_CSV",
    project_dir / "receivers/building/cropped_building_roof_receivers.csv",
)
building_buffer_gpkg_path = get_env_path(
    "MERGE_BUFFER_INPUT_GPKG",
    project_dir / "receivers/building/cropped_building_buffers_10m.gpkg",
)

output_csv_path = get_env_path(
    "MERGED_RECEIVER_OUTPUT_CSV",
    project_dir / "receivers/cropped_merged_receivers.csv",
)

buffer_layer_name = "building_buffer"

crs_epsg = "EPSG:5179"

grid_size_m = get_env_float("GRID_SIZE_M", 100.0)
cell_size_m = get_env_float("RECEIVER_RESOLUTION_M", 10.0)
boundary_tolerance_m = 1.0e-6

min_x = get_env_float("RECEIVER_MIN_X", 1163000)
max_x = get_env_float("RECEIVER_MAX_X", 1164000)
min_y = get_env_float("RECEIVER_MIN_Y", 1732000)
max_y = get_env_float("RECEIVER_MAX_Y", 1733000)
has_buildings = get_env_bool("RECEIVER_HAS_BUILDINGS", True)


# =========================
# CSV 로드 함수
# =========================
def read_receiver_csv(csv_path, receiver_type):
    df = pd.read_csv(csv_path)

    required_cols = [
        "x_epsg5179",
        "y_epsg5179",
        "lat",
        "lon",
        "alt",
    ]
    if receiver_type == "terrain":
        required_cols.append("ground_factor")
    else:
        required_cols.append("building_id")
    if receiver_type == "building":
        required_cols.append("edge_no")

    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"{csv_path} 필수 필드 없음: {missing}")

    if "building_id" not in df.columns:
        df["building_id"] = None
    if "ground_factor" not in df.columns:
        df["ground_factor"] = np.nan
    if "edge_no" not in df.columns:
        df["edge_no"] = pd.NA

    df["type"] = receiver_type

    df["x_epsg5179"] = pd.to_numeric(df["x_epsg5179"], errors="coerce")
    df["y_epsg5179"] = pd.to_numeric(df["y_epsg5179"], errors="coerce")
    df["lat"] = pd.to_numeric(df["lat"], errors="coerce")
    df["lon"] = pd.to_numeric(df["lon"], errors="coerce")
    df["alt"] = pd.to_numeric(df["alt"], errors="coerce")
    df["ground_factor"] = pd.to_numeric(
        df["ground_factor"],
        errors="coerce",
    )
    df["edge_no"] = pd.to_numeric(
        df["edge_no"],
        errors="coerce",
    ).astype("Int32")

    df = df.dropna(subset=[
        "x_epsg5179",
        "y_epsg5179",
        "lat",
        "lon",
        "alt",
    ]).copy()

    if receiver_type == "terrain":
        missing_ground_factor = df["ground_factor"].isna()
        invalid_ground_factor = ~df["ground_factor"].between(0.0, 1.0)
        invalid_count = int(
            (missing_ground_factor | invalid_ground_factor).sum()
        )
        if invalid_count > 0:
            raise ValueError(
                f"{csv_path} 지면계수 누락 또는 범위 오류: {invalid_count}개"
            )
    else:
        missing_building_id = (
            df["building_id"].isna()
            | df["building_id"].astype("string").str.strip().eq("")
        )
        if missing_building_id.any():
            raise ValueError(
                f"{csv_path} 건물 ID 누락: {int(missing_building_id.sum())}개"
            )

    if receiver_type == "building":
        invalid_edge_no = df["edge_no"].isna() | (df["edge_no"] < 1)
        if invalid_edge_no.any():
            raise ValueError(
                f"{csv_path} edge_no 누락 또는 범위 오류: "
                f"{int(invalid_edge_no.sum())}개"
            )

    return df


def validate_receiver_bounds(df, receiver_type):
    outside = (
        (df["x_epsg5179"] < min_x)
        | (df["x_epsg5179"] > max_x)
        | (df["y_epsg5179"] < min_y)
        | (df["y_epsg5179"] > max_y)
    )
    outside_count = int(outside.sum())
    if outside_count == 0:
        return

    examples = df.loc[
        outside,
        ["x_epsg5179", "y_epsg5179"],
    ].head(5)
    raise ValueError(
        f"{receiver_type} 대상지역 외부 수음점: {outside_count}개\n"
        f"{examples.to_string(index=False)}"
    )


# =========================
# GeoDataFrame 변환 함수
# =========================
def to_point_gdf(df):
    geom = [
        Point(x, y)
        for x, y in zip(df["x_epsg5179"], df["y_epsg5179"])
    ]

    return gpd.GeoDataFrame(
        df.copy(),
        geometry=geom,
        crs=crs_epsg
    )


# =========================
# 지면 수음점 필터링 함수
# =========================
def remove_terrain_points_inside_building_buffer(terrain_df):
    terrain_gdf = to_point_gdf(terrain_df)

    if buffer_layer_name is None:
        buffer_gdf = gpd.read_file(building_buffer_gpkg_path)
    else:
        buffer_gdf = gpd.read_file(
            building_buffer_gpkg_path,
            layer=buffer_layer_name
        )

    if buffer_gdf.crs is None:
        raise ValueError("버퍼 GPKG에 CRS가 없습니다.")

    if buffer_gdf.crs.to_string() != crs_epsg:
        buffer_gdf = buffer_gdf.to_crs(crs_epsg)

    buffer_gdf = buffer_gdf[buffer_gdf.geometry.notnull()].copy()
    buffer_gdf = buffer_gdf[~buffer_gdf.geometry.is_empty].copy()

    # 버퍼 내부·경계 수음점 제거
    joined = gpd.sjoin(
        terrain_gdf,
        buffer_gdf[["geometry"]],
        how="left",
        predicate="intersects"
    )

    filtered = joined[joined["index_right"].isna()].copy()
    filtered = filtered.drop(columns=["index_right", "geometry"], errors="ignore")

    print("[지면 수음점 필터링]")
    print(" - before:", len(terrain_df))
    print(" - removed:", len(terrain_df) - len(filtered))
    print(" - after:", len(filtered))

    return pd.DataFrame(filtered)


# =========================
# 그리드 ID 부여 함수
# =========================
def add_grid_id(df):
    df = df.copy()

    grid_column_count_value = (max_x - min_x) / grid_size_m
    grid_row_count_value = (max_y - min_y) / grid_size_m
    if not np.isclose(
        grid_column_count_value,
        round(grid_column_count_value),
    ):
        raise ValueError("X 범위는 그리드 크기의 정수배여야 합니다.")
    if not np.isclose(
        grid_row_count_value,
        round(grid_row_count_value),
    ):
        raise ValueError("Y 범위는 그리드 크기의 정수배여야 합니다.")

    grid_column_count = int(round(grid_column_count_value))
    grid_row_count = int(round(grid_row_count_value))

    grid_i = np.floor(
        (
            df["x_epsg5179"]
            - min_x
            + boundary_tolerance_m
        )
        / grid_size_m
    ).astype(int)
    grid_row = np.floor(
        (
            max_y
            - df["y_epsg5179"]
            + boundary_tolerance_m
        )
        / grid_size_m
    ).astype(int)

    # 최외곽 최대 인덱스 제한
    df["grid_i"] = np.minimum(
        grid_i,
        grid_column_count - 1,
    )
    df["grid_row"] = np.minimum(
        grid_row,
        grid_row_count - 1,
    )
    df["grid_j"] = (
        grid_row_count - 1 - df["grid_row"]
    )

    # 그리드 행별 스네이크 순회 순번 생성
    df["grid_column_count"] = grid_column_count
    df["grid_row_count"] = grid_row_count
    grid_x_order = np.where(
        df["grid_row"] % 2 == 0,
        df["grid_i"],
        grid_column_count - 1 - df["grid_i"],
    )
    df["grid_order"] = (
        df["grid_row"] * grid_column_count
        + grid_x_order
    )

    df["grid_id"] = (
        df["grid_i"].astype(str).str.zfill(4)
        + "_"
        + df["grid_j"].astype(str).str.zfill(4)
    )

    return df


# =========================
# receiver_id 및 스네이크 배열 인덱스 부여
# =========================
def assign_receiver_id(df):
    df = df.copy()

    if grid_size_m <= 0 or cell_size_m <= 0:
        raise ValueError("그리드와 셀 크기는 0보다 커야 합니다.")

    cells_per_grid = grid_size_m / cell_size_m
    if not np.isclose(cells_per_grid, round(cells_per_grid)):
        raise ValueError("그리드 크기는 셀 크기의 정수배여야 합니다.")

    # 동일 좌표 안정 정렬용 mm 단위 키 생성
    df["x_key"] = df["x_epsg5179"].round(3)
    df["y_key"] = df["y_epsg5179"].round(3)

    grid_top_y = max_y - df["grid_row"] * grid_size_m
    local_cell_row = np.floor(
        (
            grid_top_y
            - df["y_epsg5179"]
            + boundary_tolerance_m
        )
        / cell_size_m
    ).astype(int)
    df["local_cell_row"] = np.minimum(
        local_cell_row,
        int(round(cells_per_grid)) - 1,
    )

    # 그리드 진행 방향을 반영한 내부 셀 스네이크 방향
    local_row_moves_left_to_right = (
        (df["local_cell_row"] + df["grid_row"]) % 2 == 0
    )
    df["snake_x_key"] = np.where(
        local_row_moves_left_to_right,
        df["x_key"],
        -df["x_key"],
    )

    # 그리드 및 내부 셀 스네이크 순서 정렬
    df = df.sort_values(
        by=[
            "grid_order",
            "local_cell_row",
            "snake_x_key",
            "y_key",
            "alt",
        ],
        ascending=[
            True,
            True,
            True,
            True,
            False,
        ],
    ).reset_index(drop=True)

    # 정렬 결과 기반 연속 수음점 ID 생성
    df["receiver_id"] = np.arange(1, len(df) + 1, dtype=np.int64)

    return df.drop(
        columns=["x_key", "y_key", "snake_x_key"],
        errors="ignore",
    )


def main():
    validate_bounds(min_x, max_x, min_y, max_y)
    validate_positive(grid_size_m, "격자 크기")
    validate_positive(cell_size_m, "수음점 해상도")
    required_input_paths = [terrain_receiver_csv_path]
    if has_buildings:
        required_input_paths.extend([
            building_receiver_csv_path,
            roof_receiver_csv_path,
            building_buffer_gpkg_path,
        ])
    validate_input_paths(required_input_paths)
    output_csv_path.parent.mkdir(parents=True, exist_ok=True)

    # =========================
    # 실행
    # =========================
    terrain_df = read_receiver_csv(
        terrain_receiver_csv_path,
        receiver_type="terrain"
    )

    validate_receiver_bounds(terrain_df, "terrain")

    if has_buildings:
        building_df = read_receiver_csv(
            building_receiver_csv_path,
            receiver_type="building"
        )
        roof_df = read_receiver_csv(
            roof_receiver_csv_path,
            receiver_type="roof"
        )
        validate_receiver_bounds(building_df, "building")
        validate_receiver_bounds(roof_df, "roof")
        terrain_filtered = remove_terrain_points_inside_building_buffer(
            terrain_df
        )
        merged = pd.concat(
            [terrain_filtered, building_df, roof_df],
            ignore_index=True
        )
    else:
        print("[건물 수음점 병합]")
        print(" - 계산 영역 내 건물 없음")
        print(" - 벽면·지붕 수음점 및 건물 버퍼 처리 생략")
        print("[지면 수음점 필터링]")
        print(" - 건물 버퍼 필터링 생략")
        print(" - before:", len(terrain_df))
        print(" - removed: 0")
        print(" - after:", len(terrain_df))
        merged = terrain_df.copy()

    merged = add_grid_id(merged)
    merged = assign_receiver_id(merged)


    # =========================
    # CSV 저장
    # =========================
    output_cols = [
        "receiver_id",
        "building_id",
        "edge_no",
        "ground_factor",
        "type",
        "grid_id",
        "grid_order",
        "x_epsg5179",
        "y_epsg5179",
        "lat",
        "lon",
        "alt",
    ]

    out_df = merged[output_cols].copy()

    out_df.to_csv(
        output_csv_path,
        index=False,
        encoding="utf-8-sig",
        float_format="%.7f"
    )

    print("[저장 완료]")
    print(" - output:", output_csv_path)
    print(" - rows:", len(out_df))
    print(" - terrain:", (out_df["type"] == "terrain").sum())
    print(" - building:", (out_df["type"] == "building").sum())
    print(" - roof:", (out_df["type"] == "roof").sum())
    print(out_df.head())


if __name__ == "__main__":
    main()
