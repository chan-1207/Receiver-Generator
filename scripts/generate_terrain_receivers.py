import os
import geopandas as gpd
import pandas as pd
import numpy as np

from pyproj import Transformer
from scipy.spatial import cKDTree
from shapely.geometry import LineString, MultiLineString


# =========================
# 설정값
# =========================
input_contour_path = "../data/terrain/ulsan.shp"
output_csv_path = "../receivers/terrain/cropped_terrain_receivers_center.csv"

elev_field = "CONT"

grid_m = 10

min_x = 1163000
max_x = 1164000
min_y = 1732000
max_y = 1733000

receiver_height_m = 1.5

idw_power = 2.0
idw_radius_m = 800.0
idw_k = 32
eps = 1e-6

save_debug_grid = False
debug_grid_csv_path = "../receivers/terrain/debug_grid_10m.csv"


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
        "reference": "",
        "x_epsg5179": flat_x,
        "y_epsg5179": flat_y,
        "lat": lat,
        "lon": lon,
    })


# =========================
# 등고선 읽기 함수
# =========================
def read_contour_vertices(contour_path):
    gdf = gpd.read_file(contour_path)

    print("[입력 등고선]")
    print(" - path:", contour_path)
    print(" - rows:", len(gdf))
    print(" - crs:", gdf.crs)
    print(" - columns:", list(gdf.columns))

    if gdf.crs is None:
        raise ValueError("등고선 SHP에 CRS가 없습니다.")

    if gdf.crs.to_epsg() != 5179:
        gdf = gdf.to_crs(5179)

    if elev_field not in gdf.columns:
        raise ValueError(f"고도 필드가 없습니다: {elev_field}")

    xs_list = []
    ys_list = []
    zs_list = []

    for _, row in gdf.iterrows():
        geom = row.geometry

        if geom is None or geom.is_empty:
            continue

        try:
            z = float(row[elev_field])
        except Exception:
            continue

        if isinstance(geom, LineString):
            lines = [geom]
        elif isinstance(geom, MultiLineString):
            lines = list(geom.geoms)
        else:
            continue

        for line in lines:
            coords = np.asarray(line.coords)

            if len(coords) < 2:
                continue

            xs_list.append(coords[:, 0])
            ys_list.append(coords[:, 1])
            zs_list.append(np.full(len(coords), z, dtype=float))

    if len(xs_list) == 0:
        raise ValueError("등고선 vertex를 읽지 못했습니다.")

    terrain_x = np.concatenate(xs_list)
    terrain_y = np.concatenate(ys_list)
    terrain_z = np.concatenate(zs_list)

    print("[등고선 vertex]")
    print(" - count:", len(terrain_x))
    print(" - z min:", np.nanmin(terrain_z))
    print(" - z max:", np.nanmax(terrain_z))

    return terrain_x, terrain_y, terrain_z


# =========================
# IDW 보간 함수
# =========================
def calc_ground_z(terrain_x, terrain_y, terrain_z, recv_x, recv_y):
    tree = cKDTree(np.column_stack([terrain_x, terrain_y]))
    recv_xy = np.column_stack([recv_x, recv_y])

    try:
        dist, idx = tree.query(
            recv_xy,
            k=idw_k,
            distance_upper_bound=idw_radius_m,
            workers=-1
        )
    except TypeError:
        dist, idx = tree.query(
            recv_xy,
            k=idw_k,
            distance_upper_bound=idw_radius_m
        )

    if dist.ndim == 1:
        dist = dist[:, None]
        idx = idx[:, None]

    valid = np.isfinite(dist) & (idx < len(terrain_z))

    dist_safe = np.maximum(dist, eps)

    weight = np.zeros_like(dist_safe, dtype=float)
    weight[valid] = 1.0 / (dist_safe[valid] ** idw_power)

    weight_sum = np.sum(weight, axis=1)

    idx_safe = np.clip(idx, 0, len(terrain_z) - 1)
    z_sum = np.sum(weight * terrain_z[idx_safe], axis=1)

    ground_z = np.full(len(recv_x), np.nan, dtype=float)

    mask = weight_sum > 0
    ground_z[mask] = z_sum[mask] / weight_sum[mask]

    return ground_z


# =========================
# 수음점 생성 함수
# =========================
def make_terrain_receivers():
    grid_df = make_grid()

    recv_x = grid_df["x_epsg5179"].to_numpy(dtype=float)
    recv_y = grid_df["y_epsg5179"].to_numpy(dtype=float)

    terrain_x, terrain_y, terrain_z = read_contour_vertices(input_contour_path)

    ground_z = calc_ground_z(
        terrain_x=terrain_x,
        terrain_y=terrain_y,
        terrain_z=terrain_z,
        recv_x=recv_x,
        recv_y=recv_y
    )

    receivers = grid_df.copy()
    receivers["ground_alt"] = ground_z
    receivers["alt"] = receivers["ground_alt"] + receiver_height_m

    receivers = receivers[np.isfinite(receivers["alt"])].copy()

    return receivers


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
    "reference",
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
