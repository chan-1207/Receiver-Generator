import pandas as pd
import geopandas as gpd

from shapely.geometry import Point


# =========================
# 설정값
# =========================
terrain_receiver_csv_path = "../receivers/terrain/terrain_receivers.csv"
building_receiver_csv_path = "../receivers/building/building_cropped_filtered_receiver_.csv"
building_buffer_gpkg_path = "../receivers/building/building_cropped_buffer_10m_offset_1m.gpkg"

output_csv_path = "../receivers/merged_receivers.csv"

buffer_layer_name = None

crs_epsg = "EPSG:5179"

big_grid_m = 100

min_x = 1163000
min_y = 1732000


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

    # intersects 사용:
    # 점이 버퍼 내부 또는 경계에 있으면 제거
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

    df["big_grid_i"] = ((df["x_epsg5179"] - min_x) // big_grid_m).astype(int)
    df["big_grid_j"] = ((df["y_epsg5179"] - min_y) // big_grid_m).astype(int)

    df["big_grid_id"] = (
        "G"
        + df["big_grid_i"].astype(str).str.zfill(4)
        + "_"
        + df["big_grid_j"].astype(str).str.zfill(4)
    )

    return df


# =========================
# receiver_id 부여 함수
# =========================
def assign_receiver_id(df):
    df = df.copy()

    # 같은 좌표 판정을 위한 안정화 필드
    # EPSG:5179 좌표는 mm 수준까지 필요 없으므로 0.001m 단위 반올림
    df["x_key"] = df["x_epsg5179"].round(3)
    df["y_key"] = df["y_epsg5179"].round(3)

    # 정렬 기준:
    # 대격자 -> lat -> lon -> 동일 좌표면 alt 높은 순
    df = df.sort_values(
        by=[
            "big_grid_j",
            "big_grid_i",
            "lat",
            "lon",
            "x_key",
            "y_key",
            "alt",
        ],
        ascending=[
            True,
            True,
            True,
            True,
            True,
            True,
            False,
        ]
    ).reset_index(drop=True)

    # 1부터 시작하는 정수 ID
    df["receiver_id"] = range(1, len(df) + 1)

    df = df.drop(columns=["x_key", "y_key"], errors="ignore")

    return df


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

terrain_filtered = remove_terrain_points_inside_building_buffer(terrain_df)

merged = pd.concat(
    [
        terrain_filtered,
        building_df,
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
print(out_df.head())