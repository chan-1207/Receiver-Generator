from pathlib import Path

import geopandas as gpd
import pandas as pd
from shapely.affinity import translate
from shapely.geometry import MultiPolygon, Polygon
from shapely.validation import make_valid

try:
    from scripts.pipeline_common import (
        get_env_float,
        get_env_int,
        get_env_path,
        process_map_ordered,
        prepare_temporary_output,
        replace_temporary_output,
        validate_bounds,
        validate_input_paths,
        validate_positive,
        validate_spatial_file_coverage,
        write_csv_atomically,
    )
except ModuleNotFoundError:
    from pipeline_common import (
        get_env_float,
        get_env_int,
        get_env_path,
        process_map_ordered,
        prepare_temporary_output,
        replace_temporary_output,
        validate_bounds,
        validate_input_paths,
        validate_positive,
        validate_spatial_file_coverage,
        write_csv_atomically,
    )


# =========================================================
# 경로 설정
# =========================================================
project_dir = Path(__file__).resolve().parents[1]

input_building_height_gpkg_path = get_env_path(
    "BUILDING_HEIGHT_INPUT_GPKG",
    project_dir / "data/building_height/building_height.gpkg",
)
output_gpkg_path = get_env_path(
    "BUILDING_METADATA_OUTPUT_GPKG",
    project_dir / "metadata/building/building_cropped_metadata.gpkg",
)
output_csv_path = get_env_path(
    "BUILDING_METADATA_OUTPUT_CSV",
    project_dir / "metadata/building/building_cropped_metadata.csv",
)
input_building_height_layer = "TN_BULD"
simplified_layer = "building_simplified"
mrr_layer = "building_mrr"

min_x = get_env_float("RECEIVER_MIN_X", 1163000)
max_x = get_env_float("RECEIVER_MAX_X", 1164000)
min_y = get_env_float("RECEIVER_MIN_Y", 1732000)
max_y = get_env_float("RECEIVER_MAX_Y", 1733000)

closing_distance_m = 10.0
simplify_tolerance_m = 1.0
mrr_padding_m = 0.001
min_area_m2 = 25.0
process_workers = get_env_int("PROCESS_WORKERS", 8)
building_chunk_size = get_env_int("BUILDING_CHUNK_SIZE", 250)

# =========================================================
# 유틸 함수
# =========================================================
def clean_str(x):
    if pd.isna(x):
        return "NULL"

    s = str(x).strip()

    if s == "" or s.lower() in ["nan", "none", "null"]:
        return "NULL"

    return s


def polygon_parts(geom):
    """Polygon 조각 목록"""
    if geom is None or geom.is_empty:
        return []

    if geom.geom_type == "Polygon":
        return [geom]

    if geom.geom_type == "MultiPolygon":
        return list(geom.geoms)

    if geom.geom_type == "GeometryCollection":
        parts = []

        for part in geom.geoms:
            parts.extend(polygon_parts(part))

        return parts

    return []


def to_multipolygon(geom):
    """유효 MultiPolygon 변환"""
    if geom is None or geom.is_empty:
        return None

    valid_geom = make_valid(geom)
    parts = [
        part
        for part in polygon_parts(valid_geom)
        if not part.is_empty and part.area > 0
    ]

    if len(parts) == 0:
        return None

    return MultiPolygon(parts)


def simplify_building_polygon(geom):
    """공통 건물 폴리곤 단순화"""
    source_geom = to_multipolygon(geom)

    if source_geom is None:
        return None

    # 미세 요철과 좁은 홈 제거
    closed_geom = source_geom.buffer(
        closing_distance_m,
        join_style="mitre",
    ).buffer(
        -closing_distance_m,
        join_style="mitre",
    )
    closed_geom = to_multipolygon(closed_geom)

    # 모폴로지 연산 소멸 시 원본 형상 사용
    if closed_geom is None:
        closed_geom = source_geom

    # 위상 보존 꼭짓점 단순화
    simplified_geom = closed_geom.simplify(
        simplify_tolerance_m,
        preserve_topology=True,
    )
    simplified_geom = to_multipolygon(simplified_geom)

    if simplified_geom is None:
        return source_geom

    return simplified_geom


def make_minimum_rotated_rectangle(geom):
    """로컬 좌표 기반 최소면적 회전사각형"""
    if geom is None or geom.is_empty:
        return None

    center = geom.centroid
    local_geom = translate(
        geom,
        xoff=-center.x,
        yoff=-center.y,
    )
    local_rectangle = local_geom.minimum_rotated_rectangle

    if local_rectangle is None or local_rectangle.is_empty:
        return None

    # 후보군 누락 방지용 수치 안전 여유
    local_rectangle = local_rectangle.buffer(
        mrr_padding_m,
        join_style="mitre",
    )

    rectangle = translate(
        local_rectangle,
        xoff=center.x,
        yoff=center.y,
    )

    if not isinstance(rectangle, Polygon) or rectangle.area <= 0:
        return None

    return rectangle


def main():
    validate_bounds(min_x, max_x, min_y, max_y)
    validate_positive(process_workers, "프로세스 작업 수")
    validate_positive(building_chunk_size, "건물 묶음 크기")
    validate_input_paths([
        input_building_height_gpkg_path,
    ])
    validate_spatial_file_coverage(
        path=input_building_height_gpkg_path,
        label="건물 데이터",
        required_bounds=(min_x, min_y, max_x, max_y),
        layer=input_building_height_layer,
    )
    for output_path in [
        output_gpkg_path,
        output_csv_path,
    ]:
        output_path.parent.mkdir(parents=True, exist_ok=True)

    # =========================================================
    # 건물높이정보 로드
    # =========================================================
    print("[1] 건물높이정보 로드")

    bld = gpd.read_file(
        input_building_height_gpkg_path,
        layer=input_building_height_layer,
        bbox=(min_x, min_y, max_x, max_y),
    )

    if bld.empty:
        raise ValueError("계산 영역과 교차하는 건물이 없음")

    print(" - rows:", len(bld))
    print(" - crs:", bld.crs)
    print(" - columns:", list(bld.columns))

    bld_keep = [
        "NF_ID",
        "BPRP_SE",
        "BFLR_CO",
        "BLDH_MN",
        "BLDH_BV",
        "Shape_Area",
        "geometry",
    ]
    missing_bld = [col for col in bld_keep if col not in bld.columns]

    if missing_bld:
        raise ValueError(f"건물높이정보에 필수 필드가 없음: {missing_bld}")

    if bld.crs is None:
        raise ValueError("건물높이정보에 CRS가 없음")

    if bld.crs.is_geographic:
        raise ValueError("건물 단순화에는 미터 단위 투영 좌표계가 필요함")

    bld = bld[bld_keep].copy()

    # 건물 수음점과 지면 IDW 입력의 건물 구성을 일치시킨다.
    bld["Shape_Area"] = pd.to_numeric(bld["Shape_Area"], errors="coerce")
    input_row_count = len(bld)
    bld = bld[
        bld["Shape_Area"].notna()
        & (bld["Shape_Area"] >= min_area_m2)
    ].copy()

    print("[2] 건물 면적 사전 필터")
    print(" - area column: Shape_Area")
    print(" - minimum area [m2]:", min_area_m2)
    print(" - removed rows:", input_row_count - len(bld))
    print(" - remaining rows:", len(bld))

    if bld.empty:
        raise ValueError(
            f"계산 영역에 면적 {min_area_m2} m2 이상인 건물이 없음"
        )

    bld["NF_ID"] = bld["NF_ID"].apply(clean_str)

    invalid_id_mask = bld["NF_ID"] == "NULL"

    if invalid_id_mask.any():
        raise ValueError(
            "NF_ID가 없는 건물높이정보가 있음: "
            f"{int(invalid_id_mask.sum())}개"
        )

    duplicate_id_mask = bld["NF_ID"].duplicated(keep=False)

    if duplicate_id_mask.any():
        duplicate_ids = bld.loc[duplicate_id_mask, "NF_ID"].head(5).tolist()
        raise ValueError(f"중복 NF_ID가 있음: {duplicate_ids}")

    print(" - process workers:", process_workers)
    print(" - maximum chunk size:", building_chunk_size)
    bld["geometry"] = process_map_ordered(
        to_multipolygon,
        bld.geometry,
        process_workers,
        building_chunk_size,
    )
    invalid_geometry_mask = (
        bld.geometry.isna()
        | bld.geometry.is_empty
        | ~bld.geometry.is_valid
    )

    if invalid_geometry_mask.any():
        invalid_ids = bld.loc[invalid_geometry_mask, "NF_ID"].head(5).tolist()
        raise ValueError(f"유효한 건물 형상이 없는 NF_ID가 있음: {invalid_ids}")

    # =========================================================
    # 최종 필드 정리
    # =========================================================
    print("[3] 최종 필드 정리")

    final_cols = [
        "NF_ID",
        "BPRP_SE",
        "BFLR_CO",
        "BLDH_MN",
        "BLDH_BV",
        "Shape_Area",
        "geometry",
    ]

    missing_final_cols = [col for col in final_cols if col not in bld.columns]

    if missing_final_cols:
        raise ValueError(f"최종 필수 필드가 없음: {missing_final_cols}")

    meta_final = bld[final_cols].copy()

    # 공통 단순화 형상과 전파 계산 속성 생성
    simplified_gdf = meta_final.copy()
    simplified_gdf["geometry"] = process_map_ordered(
        simplify_building_polygon,
        simplified_gdf.geometry,
        process_workers,
        building_chunk_size,
    )
    simplified_invalid_mask = (
        simplified_gdf.geometry.isna()
        | simplified_gdf.geometry.is_empty
        | ~simplified_gdf.geometry.is_valid
    )

    if simplified_invalid_mask.any():
        invalid_ids = simplified_gdf.loc[
            simplified_invalid_mask,
            "NF_ID",
        ].head(5).tolist()
        raise ValueError(f"공통 단순화 형상 생성 실패 NF_ID: {invalid_ids}")

    # 최소면적 회전사각형 생성
    mrr_gdf = simplified_gdf[["NF_ID", "geometry"]].copy()
    mrr_gdf["geometry"] = process_map_ordered(
        make_minimum_rotated_rectangle,
        mrr_gdf.geometry,
        process_workers,
        building_chunk_size,
    )
    mrr_invalid_mask = (
        mrr_gdf.geometry.isna()
        | mrr_gdf.geometry.is_empty
        | ~mrr_gdf.geometry.is_valid
    )

    if mrr_invalid_mask.any():
        invalid_ids = mrr_gdf.loc[
            mrr_invalid_mask,
            "NF_ID",
        ].head(5).tolist()
        raise ValueError(f"최소면적 회전사각형 생성 실패 NF_ID: {invalid_ids}")

    mrr_covers_mask = mrr_gdf.geometry.covers(
        simplified_gdf.geometry,
        align=False,
    )

    if not mrr_covers_mask.all():
        uncovered_ids = mrr_gdf.loc[
            ~mrr_covers_mask,
            "NF_ID",
        ].head(5).tolist()
        raise ValueError(f"단순화 형상을 포함하지 않는 MRR NF_ID: {uncovered_ids}")

    if not (
        len(meta_final) == len(simplified_gdf) == len(mrr_gdf)
        and set(meta_final["NF_ID"]) == set(simplified_gdf["NF_ID"])
        and set(meta_final["NF_ID"]) == set(mrr_gdf["NF_ID"])
    ):
        raise ValueError("건물 레이어 간 NF_ID 구성이 일치하지 않음")


    # =========================================================
    # 저장
    # =========================================================
    print("[4] 저장")

    temporary_output_gpkg_path = prepare_temporary_output(output_gpkg_path)

    simplified_gdf.to_file(
        temporary_output_gpkg_path,
        layer=simplified_layer,
        driver="GPKG",
        index=False,
    )
    mrr_gdf.to_file(
        temporary_output_gpkg_path,
        layer=mrr_layer,
        driver="GPKG",
        mode="a",
        index=False,
    )

    written_layers = set(
        gpd.list_layers(temporary_output_gpkg_path)["name"]
    )
    expected_layers = {simplified_layer, mrr_layer}

    if written_layers != expected_layers:
        raise ValueError(
            "GeoPackage 레이어 저장 결과가 올바르지 않음: "
            f"{sorted(written_layers)}"
        )

    replace_temporary_output(temporary_output_gpkg_path, output_gpkg_path)

    write_csv_atomically(
        meta_final.drop(columns="geometry"),
        output_csv_path,
        index=False,
        encoding="utf-8-sig",
    )


    # =========================================================
    # 요약 출력
    # =========================================================
    print("\n[DONE]")
    print(" -", output_gpkg_path)
    print(" -", output_csv_path)
    print(" - layers:", [simplified_layer, mrr_layer])
    print(" - simplification closing [m]:", closing_distance_m)
    print(" - simplification tolerance [m]:", simplify_tolerance_m)
    print(" - MRR safety padding [m]:", mrr_padding_m)

    print("\n[BUILDING CHECK]")
    print("total buildings:", len(meta_final))


if __name__ == "__main__":
    main()
