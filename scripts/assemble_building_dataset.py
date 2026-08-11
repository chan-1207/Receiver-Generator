from pathlib import Path

import geopandas as gpd
import pandas as pd
from shapely.affinity import translate
from shapely.geometry import MultiPolygon, Polygon
from shapely.validation import make_valid

try:
    from scripts.pipeline_common import (
        get_env_float,
        get_env_path,
        validate_bounds,
        validate_input_paths,
        validate_spatial_file_coverage,
    )
except ModuleNotFoundError:
    from pipeline_common import (
        get_env_float,
        get_env_path,
        validate_bounds,
        validate_input_paths,
        validate_spatial_file_coverage,
    )


# =========================================================
# 경로 설정
# =========================================================
project_dir = Path(__file__).resolve().parents[1]

input_building_height_gpkg_path = get_env_path(
    "BUILDING_HEIGHT_INPUT_GPKG",
    project_dir / "data/building_height/building_height.gpkg",
)
input_building_register_csv_path = get_env_path(
    "BUILDING_REGISTER_INPUT_CSV",
    project_dir / "data/building_register/building_register.csv",
)
output_gpkg_path = get_env_path(
    "BUILDING_METADATA_OUTPUT_GPKG",
    project_dir / "metadata/building/building_cropped_metadata.gpkg",
)
output_csv_path = get_env_path(
    "BUILDING_METADATA_OUTPUT_CSV",
    project_dir / "metadata/building/building_cropped_metadata.csv",
)
output_raw_mapping_csv_path = get_env_path(
    "BUILDING_RAW_MAPPING_OUTPUT_CSV",
    project_dir / "config/building_cropped_raw_mapping.csv",
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

# =========================================================
# 유틸 함수
# =========================================================
def read_csv_auto(path):
    encodings = ["utf-8-sig", "cp949", "euc-kr", "utf-8"]
    last_error = None

    for encoding in encodings:
        try:
            return pd.read_csv(path, dtype=str, encoding=encoding)
        except Exception as error:
            last_error = error

    raise RuntimeError(
        f"CSV 읽기 실패: {path}\n마지막 오류: {last_error}"
    )


def clean_str(x):
    if pd.isna(x):
        return "NULL"

    s = str(x).strip()

    if s == "" or s.lower() in ["nan", "none", "null"]:
        return "NULL"

    return s


def mode_or_null(series):
    values = series.dropna().apply(clean_str)
    values = values[values != "NULL"]

    if len(values) == 0:
        return "NULL"

    return values.value_counts().idxmax()


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


# =========================================================
# 코드 매핑 함수
# =========================================================
def map_struct_code(raw):
    """
    STRUCT_CODE
    0: unknown
    1: concrete
    2: masonry
    3: steel
    4: wood_panel_other
    """
    v = clean_str(raw)

    concrete = {
        "철근콘크리트구조",
        "프리케스트콘크리트구조",
        "기타콘크리트구조",
        "보강콘크리트조",
        "콘크리트구조",
        "철골철근콘크리트구조",
        "철골콘크리트구조",
        "철골철근콘크리트합성구조",
        "기타철골철근콘크리트구조",
        "라멘조",
    }

    masonry = {
        "벽돌구조",
        "블록구조",
        "기타조적구조",
        "조적구조",
        "시멘트블럭조",
        "석구조",
        "흙벽돌조",
        "돌담 및 토담조",
    }

    steel = {
        "경량철골구조",
        "일반철골구조",
        "강파이프구조",
        "기타강구조",
        "공업화박판강구조(PEB)",
        "철골구조",
        "스틸하우스조",
        "철파이프조",
        "단일형강구조",
        "트러스구조",
    }

    wood_panel_other = {
        "일반목구조",
        "목구조",
        "통나무구조",
        "트러스목구조",
        "컨테이너조",
        "조립식판넬조",
        "막구조",
    }

    if v in concrete:
        return 1
    if v in masonry:
        return 2
    if v in steel:
        return 3
    if v in wood_panel_other:
        return 4

    return 0


def map_use_code(raw):
    """
    USE_CODE
    0: unknown
    1: residential
    2: commercial_service
    3: industrial_storage
    4: public_cultural_etc
    """
    v = clean_str(raw)

    residential = {
        "단독주택",
        "공동주택",
    }

    commercial_service = {
        "제1종근린생활시설",
        "제2종근린생활시설",
        "근린생활시설",
        "판매시설",
        "업무시설",
        "숙박시설",
        "위락시설",
        "관광휴게시설",
        "운수시설",
        "자동차관련시설",
        "부대시설",
        "성당",
        "어린이집",
    }

    industrial_storage = {
        "공장",
        "창고시설",
        "기타창고시설",
        "위험물저장및처리시설",
        "자원순환관련시설",
        "분뇨.쓰레기처리시설",
        "발전시설",
        "동물및식물관련시설",
    }

    public_cultural_etc = {
        "교육연구시설",
        "교육연구및복지시설",
        "노유자시설",
        "종교시설",
        "문화및집회시설",
        "의료시설",
        "운동시설",
        "교정및군사시설",
        "국방,군사시설",
        "방송통신시설",
        "묘지관련시설",
        "장례시설",
        "야영장시설",
        "수련시설",
        "공공용시설",
    }

    if v in residential:
        return 1
    if v in commercial_service:
        return 2
    if v in industrial_storage:
        return 3
    if v in public_cultural_etc:
        return 4

    return 0


def main():
    validate_bounds(min_x, max_x, min_y, max_y)
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
        output_raw_mapping_csv_path,
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
        "PNU_NO",
        "BPRP_SE",
        "BFLR_CO",
        "BLDH_MN",
        "BLDH_BV",
        "BLDFH_MX",
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

    bld["NF_ID"] = bld["NF_ID"].apply(clean_str)
    bld["PNU_NO"] = bld["PNU_NO"].apply(clean_str)

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

    bld["geometry"] = bld.geometry.apply(to_multipolygon)
    invalid_geometry_mask = (
        bld.geometry.isna()
        | bld.geometry.is_empty
        | ~bld.geometry.is_valid
    )

    if invalid_geometry_mask.any():
        invalid_ids = bld.loc[invalid_geometry_mask, "NF_ID"].head(5).tolist()
        raise ValueError(f"유효한 건물 형상이 없는 NF_ID가 있음: {invalid_ids}")

    # 건물높이정보 PNU 뒤 9자리
    bld["PNU_SUFFIX"] = bld["PNU_NO"].apply(lambda x: clean_str(x)[-9:])


    # =========================================================
    # 건축물대장 로드 및 PNU_SUFFIX 생성
    # =========================================================
    print("[2] 건축물대장 로드")

    if input_building_register_csv_path.exists():
        reg = read_csv_auto(input_building_register_csv_path)
    else:
        print(
            " - 건축물대장 없음, 용도 코드를 미상으로 처리: "
            f"{input_building_register_csv_path}"
        )
        reg = pd.DataFrame(columns=[
            "대지구분",
            "번",
            "지",
            "주구조",
            "주용도",
        ])

    print(" - rows:", len(reg))
    print(" - columns:", list(reg.columns))

    required_reg_cols = ["대지구분", "번", "지"]
    missing_reg = [c for c in required_reg_cols if c not in reg.columns]

    if missing_reg:
        raise ValueError(f"건축물대장에 PNU_SUFFIX 생성 필드가 부족함: {missing_reg}")

    if reg.empty:
        reg["PNU_SUFFIX"] = pd.Series(dtype="string")
    else:
        reg["PNU_SUFFIX"] = reg.apply(make_pnu_suffix, axis=1)

    reg_keep = [
        "PNU_SUFFIX",
        "PK",
        "시도",
        "시군구",
        "법정동",
        "번",
        "지",
        "대지구분",
        "건물명",
        "동명",
        "주구조",
        "주용도",
    ]

    reg_keep = [c for c in reg_keep if c in reg.columns]
    reg = reg[reg_keep].copy()

    for col in ["주구조", "주용도"]:
        if col not in reg.columns:
            reg[col] = "NULL"
        reg[col] = reg[col].apply(clean_str)

    # PNU_SUFFIX 중복 처리: 첫 행이 아니라 최빈값으로 대표 속성 생성
    reg_before = len(reg)

    reg_grouped = (
        reg.groupby("PNU_SUFFIX", as_index=False)
        .agg({
            "주구조": mode_or_null,
            "주용도": mode_or_null,
        })
    )

    print(f" - register grouped by PNU_SUFFIX: {reg_before} -> {len(reg_grouped)}")


    # =========================================================
    # 건물높이정보 + 건축물대장 조인
    # =========================================================
    print("[3] PNU_SUFFIX 기준 조인")

    meta = bld.merge(reg_grouped, on="PNU_SUFFIX", how="left")

    # 조인 완료 후 식별 필드 제거
    meta = meta.drop(columns=["PNU_NO", "PNU_SUFFIX"])

    for col in ["주구조", "주용도"]:
        meta[col] = meta[col].apply(clean_str)

    matched = (meta["주구조"] != "NULL").sum()

    print(" - building rows:", len(meta))
    print(" - matched register rows:", matched)
    print(" - unmatched rows:", len(meta) - matched)


    # =========================================================
    # 코드 필드 생성
    # =========================================================
    print("[4] 코드 필드 생성")

    meta["USE_CODE"] = meta["주용도"].apply(map_use_code).astype("int16")

    # =========================================================
    # 최종 필드 정리
    # =========================================================
    print("[5] 최종 필드 정리")

    final_cols = [
        "NF_ID",
        "BPRP_SE",
        "BFLR_CO",
        "BLDH_MN",
        "BLDH_BV",
        "BLDFH_MX",
        "Shape_Area",
        "USE_CODE",
        "geometry",
    ]

    missing_final_cols = [col for col in final_cols if col not in meta.columns]

    if missing_final_cols:
        raise ValueError(f"최종 필수 필드가 없음: {missing_final_cols}")

    meta_final = meta[final_cols].copy()

    # 공통 단순화 형상과 전파 계산 속성 생성
    simplified_gdf = meta_final.copy()
    simplified_gdf["geometry"] = simplified_gdf.geometry.apply(
        simplify_building_polygon
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
    mrr_gdf["geometry"] = mrr_gdf.geometry.apply(
        make_minimum_rotated_rectangle
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
    # raw_value_mapping.csv 생성
    # =========================================================
    print("[6] raw_value_mapping.csv 생성")

    raw_mapping_rows = []

    for raw_value in sorted(meta["주구조"].dropna().unique()):
        raw_mapping_rows.append({
            "field_name": "STRUCT_CODE",
            "raw_field": "주구조",
            "raw_value": raw_value,
            "code": map_struct_code(raw_value),
        })

    for raw_value in sorted(meta["주용도"].dropna().unique()):
        raw_mapping_rows.append({
            "field_name": "USE_CODE",
            "raw_field": "주용도",
            "raw_value": raw_value,
            "code": map_use_code(raw_value),
        })

    raw_mapping = pd.DataFrame(raw_mapping_rows)

    code_label_map = {
        ("STRUCT_CODE", 0): ("미상", "unknown"),
        ("STRUCT_CODE", 1): ("콘크리트계 구조", "concrete"),
        ("STRUCT_CODE", 2): ("조적계 구조", "masonry"),
        ("STRUCT_CODE", 3): ("철골계 구조", "steel"),
        ("STRUCT_CODE", 4): ("목구조·패널·기타", "wood_panel_other"),

        ("USE_CODE", 0): ("미상", "unknown"),
        ("USE_CODE", 1): ("주거시설", "residential"),
        ("USE_CODE", 2): ("상업·서비스시설", "commercial_service"),
        ("USE_CODE", 3): ("산업·창고시설", "industrial_storage"),
        ("USE_CODE", 4): ("공공·문화·기타시설", "public_cultural_etc"),

    }

    raw_mapping["code_label_ko"] = raw_mapping.apply(
        lambda r: code_label_map.get(
            (r["field_name"], int(r["code"])),
            ("미상", "unknown")
        )[0],
        axis=1,
    )

    raw_mapping["code_label_en"] = raw_mapping.apply(
        lambda r: code_label_map.get(
            (r["field_name"], int(r["code"])),
            ("미상", "unknown")
        )[1],
        axis=1,
    )


    # =========================================================
    # 저장
    # =========================================================
    print("[7] 저장")

    temporary_output_gpkg_path = output_gpkg_path.with_name(
        f"{output_gpkg_path.stem}.tmp{output_gpkg_path.suffix}"
    )

    if temporary_output_gpkg_path.exists():
        temporary_output_gpkg_path.unlink()

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

    temporary_output_gpkg_path.replace(output_gpkg_path)

    meta_final.drop(columns="geometry").to_csv(output_csv_path, index=False, encoding="utf-8-sig")
    raw_mapping.to_csv(output_raw_mapping_csv_path, index=False, encoding="utf-8-sig")


    # =========================================================
    # 요약 출력
    # =========================================================
    print("\n[DONE]")
    print(" -", output_gpkg_path)
    print(" -", output_csv_path)
    print(" -", output_raw_mapping_csv_path)
    print(" - layers:", [simplified_layer, mrr_layer])
    print(" - simplification closing [m]:", closing_distance_m)
    print(" - simplification tolerance [m]:", simplify_tolerance_m)
    print(" - MRR safety padding [m]:", mrr_padding_m)

    print("\n[CODE COUNTS]")
    for col in [
        "USE_CODE",
    ]:
        print(f"\n{col}")
        print(meta_final[col].value_counts(dropna=False).sort_index())

    print("\n[JOIN CHECK]")
    print("total buildings:", len(meta_final))
    print("register matched:", (meta["주구조"] != "NULL").sum())
    print("register unmatched:", (meta["주구조"] == "NULL").sum())

    print("\n[PNU_SUFFIX CHECK]")
    print("unique building PNU_SUFFIX:", bld["PNU_SUFFIX"].nunique())
    print("unique register PNU_SUFFIX:", reg_grouped["PNU_SUFFIX"].nunique())


if __name__ == "__main__":
    main()
