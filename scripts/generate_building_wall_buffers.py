from pathlib import Path

import geopandas as gpd
from shapely.geometry import MultiPolygon, Polygon
from shapely.validation import make_valid

try:
    from scripts.pipeline_common import (
        get_env_int,
        get_env_path,
        parallel_map_ordered,
        validate_input_paths,
        validate_positive,
    )
except ModuleNotFoundError:
    from pipeline_common import (
        get_env_int,
        get_env_path,
        parallel_map_ordered,
        validate_input_paths,
        validate_positive,
    )

# =========================
# 설정값
# =========================
project_dir = Path(__file__).resolve().parents[1]

input_building_metadata_path = get_env_path(
    "BUILDING_METADATA_INPUT_GPKG",
    project_dir / "metadata/building/building_cropped_metadata.gpkg",
)
output_gpkg_path = get_env_path(
    "RECEIVER_BUFFER_OUTPUT_GPKG",
    project_dir / "receivers/building/cropped_building_buffers_10m.gpkg",
)

simplified_layer_name = "building_simplified"
output_layer_name = "building_buffer"

# 버퍼 결과에 남길 최소 필드
# 수음점 생성까지 고려해서 높이 필드는 같이 유지
keep_cols = [
    "NF_ID",
    "BLDH_MN",
    "BLDH_BV",
    "Shape_Area",
]

receiver_offset_m = 1.0     # 최종 수음점용 외곽 버퍼 [m]

parallel_workers = get_env_int("PARALLEL_WORKERS", 8)
building_chunk_size = get_env_int("BUILDING_CHUNK_SIZE", 250)


# =========================
# 유틸 함수
# =========================
def to_multipolygon(geom):
    """GPKG 저장 시 geometry type을 MultiPolygon으로 통일"""
    if geom is None or geom.is_empty:
        return None

    if isinstance(geom, Polygon):
        return MultiPolygon([geom])

    if isinstance(geom, MultiPolygon):
        return geom

    return None


def make_receiver_buffer(geom):
    """
    공통 단순화 폴리곤 기반 수음점용 외곽 버퍼
    """
    if geom is None or geom.is_empty:
        return None

    geom = make_valid(geom)

    if geom is None or geom.is_empty:
        return None

    # 최종 수음점 배치용 외곽 버퍼
    geom = geom.buffer(receiver_offset_m, join_style="mitre")

    if geom is None or geom.is_empty:
        return None

    geom = make_valid(geom)

    if geom is None or geom.is_empty:
        return None

    return to_multipolygon(geom)


def main():
    validate_positive(parallel_workers, "공통 병렬 작업 수")
    validate_positive(building_chunk_size, "건물 묶음 크기")
    validate_input_paths([input_building_metadata_path])

    # =========================
    # 실행
    # =========================
    output_gpkg_path.parent.mkdir(parents=True, exist_ok=True)

    simplified_gdf = gpd.read_file(
        input_building_metadata_path,
        layer=simplified_layer_name,
    )

    print("[1] 입력 로드")
    print(" - input:", input_building_metadata_path)
    print(" - geometry layer:", simplified_layer_name)
    print(" - rows:", len(simplified_gdf))
    print(" - CRS:", simplified_gdf.crs)

    # 미터 단위 투영 좌표계 검증
    if simplified_gdf.crs is None:
        raise ValueError("CRS가 없습니다. EPSG:5179 등 meter 기반 좌표계를 먼저 지정해야 합니다.")

    if simplified_gdf.crs.is_geographic:
        raise ValueError("수음점용 버퍼에는 미터 단위 투영 좌표계가 필요합니다.")

    # 필수 컬럼 확인
    missing_cols = [c for c in keep_cols if c not in simplified_gdf.columns]
    if missing_cols:
        raise ValueError(f"단순화 레이어 필수 컬럼이 없습니다: {missing_cols}")

    if simplified_gdf["NF_ID"].duplicated().any():
        raise ValueError("단순화 레이어에 중복 NF_ID가 있습니다.")

    gdf = simplified_gdf[keep_cols + ["geometry"]].copy()

    print("[2] 수음점용 외곽 버퍼 처리 시작")
    print(" - parallel workers:", parallel_workers)
    print(" - maximum chunk size:", building_chunk_size)
    buffer_gdf = gdf.copy()
    buffer_gdf["geometry"] = parallel_map_ordered(
        make_receiver_buffer,
        buffer_gdf.geometry,
        parallel_workers,
        building_chunk_size,
    )
    buffer_gdf = buffer_gdf[
        buffer_gdf.geometry.notnull()
    ].copy()
    buffer_gdf = buffer_gdf[
        ~buffer_gdf.geometry.is_empty
    ].copy()

    if output_gpkg_path.exists():
        output_gpkg_path.unlink()

    buffer_gdf.to_file(
        output_gpkg_path,
        driver="GPKG",
        layer=output_layer_name,
        index=False,
    )
    print("[3] 저장 완료")
    print(" - output:", output_gpkg_path)
    print(" - buffer layer:", output_layer_name)
    print(" - buffer rows:", len(buffer_gdf))


if __name__ == "__main__":
    main()
