import numpy as np
import pandas as pd
import geopandas as gpd

from shapely.geometry import Point


# =========================
# 설정값
# =========================
terrain_receiver_csv_path = "../receivers/terrain/cropped_terrain_receivers_center.csv"
building_receiver_csv_path = "../receivers/building/cropped_building_receivers_new.csv"
roof_receiver_csv_path = "../receivers/building/cropped_building_roof_receivers.csv"
building_buffer_gpkg_path = "../receivers/building/cropped_building_buffers_10m.gpkg"

output_csv_path = "../receivers/cropped_merged_receivers_center.csv"

buffer_layer_name = None

crs_epsg = "EPSG:5179"

big_grid_m = 100
receiver_grid_m = 10
grid_boundary_tolerance_m = 1.0e-6

min_x = 1163000
max_x = 1164000
min_y = 1732000
max_y = 1733000


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

    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"{csv_path} 필수 필드 없음: {missing}")

    if "reference" not in df.columns:
        df["reference"] = None

    df["type"] = receiver_type

    df["x_epsg5179"] = pd.to_numeric(df["x_epsg5179"], errors="coerce")
    df["y_epsg5179"] = pd.to_numeric(df["y_epsg5179"], errors="coerce")
    df["lat"] = pd.to_numeric(df["lat"], errors="coerce")
    df["lon"] = pd.to_numeric(df["lon"], errors="coerce")
    df["alt"] = pd.to_numeric(df["alt"], errors="coerce")

    df = df.dropna(subset=[
        "x_epsg5179",
        "y_epsg5179",
        "lat",
        "lon",
        "alt",
    ]).copy()

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
# 대격자 ID 부여 함수
# =========================
def add_big_grid_id(df):
    df = df.copy()

    big_grid_column_count_value = (max_x - min_x) / big_grid_m
    big_grid_row_count_value = (max_y - min_y) / big_grid_m
    if not np.isclose(
        big_grid_column_count_value,
        round(big_grid_column_count_value),
    ):
        raise ValueError("X 범위는 대격자 크기의 정수배여야 합니다.")
    if not np.isclose(
        big_grid_row_count_value,
        round(big_grid_row_count_value),
    ):
        raise ValueError("Y 범위는 대격자 크기의 정수배여야 합니다.")

    big_grid_column_count = int(round(big_grid_column_count_value))
    big_grid_row_count = int(round(big_grid_row_count_value))

    big_grid_i = np.floor(
        (
            df["x_epsg5179"]
            - min_x
            + grid_boundary_tolerance_m
        )
        / big_grid_m
    ).astype(int)
    big_grid_row = np.floor(
        (
            max_y
            - df["y_epsg5179"]
            + grid_boundary_tolerance_m
        )
        / big_grid_m
    ).astype(int)

    # 최외곽 최대 인덱스 제한
    df["big_grid_i"] = np.minimum(
        big_grid_i,
        big_grid_column_count - 1,
    )
    df["big_grid_row"] = np.minimum(
        big_grid_row,
        big_grid_row_count - 1,
    )
    df["big_grid_j"] = (
        big_grid_row_count - 1 - df["big_grid_row"]
    )

    # 좌표 기반 조회용 행 우선(row-major) 1차원 인덱스 생성
    df["big_grid_column_count"] = big_grid_column_count
    df["big_grid_row_count"] = big_grid_row_count
    df["big_grid_index"] = (
        df["big_grid_row"] * big_grid_column_count
        + df["big_grid_i"]
    )

    df["big_grid_id"] = (
        df["big_grid_i"].astype(str).str.zfill(4)
        + "_"
        + df["big_grid_j"].astype(str).str.zfill(4)
    )

    return df


# =========================
# receiver_id 및 스네이크 배열 인덱스 부여
# =========================
def assign_receiver_id(df):
    df = df.copy()

    if big_grid_m <= 0 or receiver_grid_m <= 0:
        raise ValueError("대격자와 수음점 격자 크기는 0보다 커야 합니다.")

    cells_per_big_grid = big_grid_m / receiver_grid_m
    if not np.isclose(cells_per_big_grid, round(cells_per_big_grid)):
        raise ValueError("대격자 크기는 수음점 격자 크기의 정수배여야 합니다.")

    # 동일 좌표 안정 정렬용 mm 단위 키 생성
    df["x_key"] = df["x_epsg5179"].round(3)
    df["y_key"] = df["y_epsg5179"].round(3)

    grid_top_y = max_y - df["big_grid_row"] * big_grid_m
    local_grid_row = np.floor(
        (
            grid_top_y
            - df["y_epsg5179"]
            + grid_boundary_tolerance_m
        )
        / receiver_grid_m
    ).astype(int)
    df["local_grid_row"] = np.minimum(
        local_grid_row,
        int(round(cells_per_big_grid)) - 1,
    )

    # 상단 0행 기준 짝수 행 좌→우·홀수 행 우→좌 진행
    df["snake_x_key"] = np.where(
        df["local_grid_row"] % 2 == 0,
        df["x_key"],
        -df["x_key"],
    )

    # 행 우선 대격자 및 내부 스네이크 순서 정렬
    df = df.sort_values(
        by=[
            "big_grid_index",
            "local_grid_row",
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


# =========================
# 실행
# =========================
terrain_df = read_receiver_csv(
    terrain_receiver_csv_path,
    receiver_type="terrain"
)

building_df = read_receiver_csv(
    building_receiver_csv_path,
    receiver_type="building"
)

roof_df = read_receiver_csv(
    roof_receiver_csv_path,
    receiver_type="roof"
)

validate_receiver_bounds(terrain_df, "terrain")
validate_receiver_bounds(building_df, "building")
validate_receiver_bounds(roof_df, "roof")

terrain_filtered = remove_terrain_points_inside_building_buffer(terrain_df)

merged = pd.concat(
    [
        terrain_filtered,
        building_df,
        roof_df,
    ],
    ignore_index=True
)

merged = add_big_grid_id(merged)
merged = assign_receiver_id(merged)


# =========================
# CSV 저장
# =========================
output_cols = [
    "receiver_id",
    "reference",
    "type",
    "big_grid_id",
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
