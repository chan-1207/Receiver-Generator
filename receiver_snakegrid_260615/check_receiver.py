#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from shapely.geometry import Point, box


# =========================================================
# 설정
# =========================================================
TERRAIN_SHP = Path("./in_terrain/ulsan.shp")

GRID_CSV = Path("./out_receivers/grid_100m.csv")
RECEIVER_CSV = Path("./out_receivers/receiver_index_100m.csv")

OUT_DIR = Path("./out_check")
OUT_LOG = OUT_DIR / "check_log.txt"
OUT_PNG = OUT_DIR / "check_area.png"

GRID_M = 100

REQ_MIN_X = 1140450
REQ_MAX_X = 1177650
REQ_MIN_Y = 1720300
REQ_MAX_Y = 1737900


# =========================================================
# 좌표 파일 읽기
# =========================================================
def load_point_table():
    if RECEIVER_CSV.exists():
        df = pd.read_csv(RECEIVER_CSV)

        if "x" in df.columns and "y" in df.columns:
            return df, "x", "y", RECEIVER_CSV

        if "X_EPSG5179" in df.columns and "Y_EPSG5179" in df.columns:
            return df, "X_EPSG5179", "Y_EPSG5179", RECEIVER_CSV

        raise ValueError("receiver csv에 x, y 또는 X_EPSG5179, Y_EPSG5179 컬럼이 없습니다.")

    if GRID_CSV.exists():
        df = pd.read_csv(GRID_CSV)

        if "X_EPSG5179" not in df.columns or "Y_EPSG5179" not in df.columns:
            raise ValueError("grid csv에 X_EPSG5179, Y_EPSG5179 컬럼이 없습니다.")

        return df, "X_EPSG5179", "Y_EPSG5179", GRID_CSV

    raise FileNotFoundError("receiver_index_100m.csv 또는 grid_100m.csv가 없습니다.")


# =========================================================
# shp 읽기
# =========================================================
def load_terrain():
    gdf = gpd.read_file(TERRAIN_SHP)

    if gdf.crs is None:
        raise ValueError("terrain shp에 CRS가 없습니다.")

    if gdf.crs.to_epsg() != 5179:
        gdf = gdf.to_crs(5179)

    return gdf


# =========================================================
# polygon 여부 확인
# =========================================================
def has_polygon(gdf):
    geom_types = set(gdf.geometry.geom_type.dropna().unique())
    return bool(geom_types & {"Polygon", "MultiPolygon"})


# =========================================================
# 로그 저장
# =========================================================
def save_log(lines):
    with open(OUT_LOG, "w", encoding="utf-8") as f:
        for line in lines:
            f.write(line + "\n")


# =========================================================
# 그림 저장
# =========================================================
def save_plot(terrain_gdf, receiver_edge_bounds, terrain_bounds, outside_gdf):
    fig, ax = plt.subplots(figsize=(12, 7))

    terrain_gdf.plot(
        ax=ax,
        color="lightgray",
        edgecolor="lightgray",
        linewidth=0.3,
        alpha=0.8,
    )

    rx_min, ry_min, rx_max, ry_max = receiver_edge_bounds
    receiver_box = box(rx_min, ry_min, rx_max, ry_max)

    gpd.GeoSeries([receiver_box], crs="EPSG:5179").boundary.plot(
        ax=ax,
        color="red",
        linewidth=2.2,
    )

    tx_min, ty_min, tx_max, ty_max = terrain_bounds
    terrain_box = box(tx_min, ty_min, tx_max, ty_max)

    gpd.GeoSeries([terrain_box], crs="EPSG:5179").boundary.plot(
        ax=ax,
        color="blue",
        linewidth=1.8,
        linestyle="--",
    )

    if len(outside_gdf) > 0:
        outside_gdf.plot(
            ax=ax,
            color="black",
            markersize=5,
        )

    legend_items = [
        Patch(facecolor="lightgray", edgecolor="lightgray", label="terrain shp"),
        Line2D([0], [0], color="red", linewidth=2.2, label="receiver grid area"),
        Line2D([0], [0], color="blue", linewidth=1.8, linestyle="--", label="terrain bbox"),
        Line2D([0], [0], color="black", marker="o", linestyle="", markersize=5, label="outside points"),
    ]

    ax.legend(handles=legend_items, loc="lower center", bbox_to_anchor=(0.5, -0.22), ncol=2)

    ax.set_title("receiver area check")
    ax.set_xlabel("X EPSG:5179 m")
    ax.set_ylabel("Y EPSG:5179 m")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, linewidth=0.3, alpha=0.4)

    plt.tight_layout()
    plt.savefig(OUT_PNG, dpi=220, bbox_inches="tight")
    plt.close()


# =========================================================
# 실행
# =========================================================
def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    df, x_col, y_col, point_file = load_point_table()
    terrain_gdf = load_terrain()

    x = df[x_col].to_numpy(dtype=float)
    y = df[y_col].to_numpy(dtype=float)

    min_x = float(np.min(x))
    max_x = float(np.max(x))
    min_y = float(np.min(y))
    max_y = float(np.max(y))

    nx = len(np.unique(x))
    ny = len(np.unique(y))

    center_width_km = (max_x - min_x) / 1000.0
    center_height_km = (max_y - min_y) / 1000.0

    edge_min_x = min_x - GRID_M / 2
    edge_max_x = max_x + GRID_M / 2
    edge_min_y = min_y - GRID_M / 2
    edge_max_y = max_y + GRID_M / 2

    edge_width_km = (edge_max_x - edge_min_x) / 1000.0
    edge_height_km = (edge_max_y - edge_min_y) / 1000.0

    req_match = (
        min_x == REQ_MIN_X and
        max_x == REQ_MAX_X and
        min_y == REQ_MIN_Y and
        max_y == REQ_MAX_Y
    )

    t_min_x, t_min_y, t_max_x, t_max_y = terrain_gdf.total_bounds

    outside_bbox_mask = (
        (x < t_min_x) |
        (x > t_max_x) |
        (y < t_min_y) |
        (y > t_max_y)
    )

    outside_bbox_count = int(np.sum(outside_bbox_mask))

    point_gdf = gpd.GeoDataFrame(
        df.copy(),
        geometry=[Point(px, py) for px, py in zip(x, y)],
        crs="EPSG:5179",
    )

    if has_polygon(terrain_gdf):
        polygon_gdf = terrain_gdf[
            terrain_gdf.geometry.geom_type.isin(["Polygon", "MultiPolygon"])
        ].copy()

        try:
            terrain_union = polygon_gdf.geometry.union_all()
        except AttributeError:
            terrain_union = polygon_gdf.geometry.unary_union

        inside_polygon = point_gdf.geometry.apply(lambda p: terrain_union.covers(p))
        outside_polygon_count = int((~inside_polygon).sum())
        outside_polygon_text = f"{outside_polygon_count:,}"
    else:
        outside_polygon_text = "skipped, terrain shp is not polygon"

    outside_gdf = point_gdf[outside_bbox_mask].copy()

    receiver_edge_bounds = (edge_min_x, edge_min_y, edge_max_x, edge_max_y)
    terrain_bounds = (t_min_x, t_min_y, t_max_x, t_max_y)

    save_plot(
        terrain_gdf=terrain_gdf,
        receiver_edge_bounds=receiver_edge_bounds,
        terrain_bounds=terrain_bounds,
        outside_gdf=outside_gdf,
    )

    geom_types = sorted(list(set(terrain_gdf.geometry.geom_type.dropna().unique())))

    log_lines = [
        "[INPUT]",
        f"point file: {point_file}",
        f"terrain shp: {TERRAIN_SHP}",
        f"terrain crs: {terrain_gdf.crs}",
        f"terrain geometry types: {geom_types}",
        "",
        "[GRID CENTER]",
        f"x range: {int(min_x)} ~ {int(max_x)}",
        f"y range: {int(min_y)} ~ {int(max_y)}",
        f"grid count: {nx} x {ny}",
        f"total points: {len(df):,}",
        f"center size: {center_width_km:.1f} km x {center_height_km:.1f} km",
        f"matches requested range: {req_match}",
        "",
        "[GRID EDGE]",
        f"x edge range: {int(edge_min_x)} ~ {int(edge_max_x)}",
        f"y edge range: {int(edge_min_y)} ~ {int(edge_max_y)}",
        f"edge size: {edge_width_km:.1f} km x {edge_height_km:.1f} km",
        "",
        "[TERRAIN COVERAGE]",
        f"terrain x bounds: {int(t_min_x)} ~ {int(t_max_x)}",
        f"terrain y bounds: {int(t_min_y)} ~ {int(t_max_y)}",
        f"points outside terrain bbox: {outside_bbox_count:,}",
        f"points outside terrain polygon: {outside_polygon_text}",
        "",
        "[OUTPUT]",
        f"log: {OUT_LOG}",
        f"figure: {OUT_PNG}",
    ]

    save_log(log_lines)

    for line in log_lines:
        print(line)


if __name__ == "__main__":
    main()