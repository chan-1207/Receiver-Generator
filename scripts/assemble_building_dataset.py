import os
import pandas as pd
import geopandas as gpd


# =========================================================
# 경로 설정
# =========================================================
input_building_height_gpkg_path = "../data/building_height/building_cropped_height.gpkg"
input_building_register_csv_path = "../data/building_register/building_register.csv"

output_gpkg_path = "../metadata/building_metadata/building_crop_metadata.gpkg"
output_csv_path = "../metadata/building_metadata/building_crop_metadata.csv"
output_raw_mapping_cvs_path = "../config/building_cropped_raw_mapping.csv"

output_layer = "building_metadata"

# 출력 폴더 생성
for path in [output_gpkg_path, output_csv_path, output_raw_mapping_cvs_path]:
    os.makedirs(os.path.dirname(path), exist_ok=True)


# =========================================================
# 유틸 함수
# =========================================================
def read_csv_auto(path):
    encodings = ["utf-8-sig", "cp949", "euc-kr", "utf-8"]
    last_error = None

    for enc in encodings:
        try:
            return pd.read_csv(path, dtype=str, encoding=enc)
        except Exception as e:
            last_error = e

    raise RuntimeError(f"CSV 읽기 실패: {path}\n마지막 오류: {last_error}")


def clean_str(x):
    if pd.isna(x):
        return "NULL"

    s = str(x).strip()

    if s == "" or s.lower() in ["nan", "none", "null"]:
        return "NULL"

    return s


def normalize_number_text(x):
    """
    번, 지 값 정리.
    예:
        5      -> 5
        5.0    -> 5
        0005   -> 5
        NULL   -> 0
    """
    s = clean_str(x)

    if s == "NULL":
        return "0"

    try:
        return str(int(float(s)))
    except Exception:
        return s


def make_pnu_suffix(row):
    """
    건축물대장 필드로 PNU 뒤 9자리 생성.

    PNU_SUFFIX = 대지구분코드 1자리 + 번 4자리 + 지 4자리

    예:
        대지 / 5 / 0 -> 100050000
        산 / 12 / 3 -> 200120003
    """
    land_raw = clean_str(row.get("대지구분", "대지"))

    if land_raw in ["대지", "일반", "1"]:
        land_type = "1"
    elif land_raw in ["산", "2"]:
        land_type = "2"
    else:
        land_type = "1"

    bun = normalize_number_text(row.get("번", "0")).zfill(4)
    ji = normalize_number_text(row.get("지", "0")).zfill(4)

    return land_type + bun + ji


def mode_or_null(series):
    """
    같은 PNU_SUFFIX에 여러 건축물대장 레코드가 있을 때
    가장 많이 등장한 값을 대표값으로 사용.
    """
    values = series.dropna().apply(clean_str)
    values = values[values != "NULL"]

    if len(values) == 0:
        return "NULL"

    return values.value_counts().idxmax()


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


def map_roof_code(raw):
    """
    ROOF_CODE
    0: unknown
    1: concrete_roof
    2: slate_roof
    3: tile_roof
    4: other_roof
    """
    v = clean_str(raw)

    if v == "(철근)콘크리트":
        return 1
    if v == "슬레이트":
        return 2
    if v == "기와":
        return 3
    if v == "기타지붕":
        return 4

    return 0


def estimate_wall_mat_code(struct_code, use_code):
    """
    WALL_MAT_CODE
    0: unknown
    1: concrete_wall
    2: masonry_wall
    3: metal_panel_wall
    4: wood_panel_other_wall
    """
    if struct_code == 1:
        return 1
    if struct_code == 2:
        return 2
    if struct_code == 3:
        return 3
    if struct_code == 4:
        return 4

    if use_code == 3:
        return 3

    return 0


# =========================================================
# 건물높이정보 로드
# =========================================================
print("[1] 건물높이정보 로드")

bld = gpd.read_file(input_building_height_gpkg_path)

print(" - rows:", len(bld))
print(" - crs:", bld.crs)
print(" - columns:", list(bld.columns))

required_bld_cols = ["NF_ID", "PNU_NO", "geometry"]
missing_bld = [c for c in required_bld_cols if c not in bld.columns]

if missing_bld:
    raise ValueError(f"건물높이정보에 필수 필드가 없음: {missing_bld}")

bld_keep = [
    "NF_ID",
    "PNU_NO",
    "BPRP_SE",
    "BFLR_CO",
    "BLDH_MN",
    "BLDH_BV",
    "BLDH_MX",
    "BLDFH_MX",
    "geometry",
]

bld_keep = [c for c in bld_keep if c in bld.columns]
bld = bld[bld_keep].copy()

bld["NF_ID"] = bld["NF_ID"].apply(clean_str)
bld["PNU_NO"] = bld["PNU_NO"].apply(clean_str)

# 건물높이정보 PNU 뒤 9자리
bld["PNU_SUFFIX"] = bld["PNU_NO"].apply(lambda x: clean_str(x)[-9:])


# =========================================================
# 건축물대장 로드 및 PNU_SUFFIX 생성
# =========================================================
print("[2] 건축물대장 로드")

reg = read_csv_auto(input_building_register_csv_path)

print(" - rows:", len(reg))
print(" - columns:", list(reg.columns))

required_reg_cols = ["대지구분", "번", "지"]
missing_reg = [c for c in required_reg_cols if c not in reg.columns]

if missing_reg:
    raise ValueError(f"건축물대장에 PNU_SUFFIX 생성 필드가 부족함: {missing_reg}")

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
    "주지붕",
]

reg_keep = [c for c in reg_keep if c in reg.columns]
reg = reg[reg_keep].copy()

for col in ["주구조", "주용도", "주지붕"]:
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
        "주지붕": mode_or_null,
    })
)

print(f" - register grouped by PNU_SUFFIX: {reg_before} -> {len(reg_grouped)}")


# =========================================================
# 건물높이정보 + 건축물대장 조인
# =========================================================
print("[3] PNU_SUFFIX 기준 조인")

meta = bld.merge(reg_grouped, on="PNU_SUFFIX", how="left")

for col in ["주구조", "주용도", "주지붕"]:
    meta[col] = meta[col].apply(clean_str)

matched = (meta["주구조"] != "NULL").sum()

print(" - building_metadata rows:", len(meta))
print(" - matched register rows:", matched)
print(" - unmatched rows:", len(meta) - matched)


# =========================================================
# 코드 필드 생성
# =========================================================
print("[4] 코드 필드 생성")

meta["STRUCT_CODE"] = meta["주구조"].apply(map_struct_code).astype("int16")
meta["USE_CODE"] = meta["주용도"].apply(map_use_code).astype("int16")
meta["ROOF_CODE"] = meta["주지붕"].apply(map_roof_code).astype("int16")

meta["WALL_MAT_CODE"] = meta.apply(
    lambda r: estimate_wall_mat_code(r["STRUCT_CODE"], r["USE_CODE"]),
    axis=1,
).astype("int16")

# =========================================================
# 최종 필드 정리
# =========================================================
print("[5] 최종 필드 정리")

final_cols = [
    "NF_ID",
    "PNU_NO",
    "PNU_SUFFIX",
    "BPRP_SE",
    "BFLR_CO",
    "BLDH_MN",
    "BLDH_BV",
    "BLDH_MX",
    "BLDFH_MX",
    "STRUCT_CODE",
    "USE_CODE",
    "ROOF_CODE",
    "WALL_MAT_CODE",
    "geometry",
]

final_cols = [c for c in final_cols if c in meta.columns]
meta_final = meta[final_cols].copy()


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

for raw_value in sorted(meta["주지붕"].dropna().unique()):
    raw_mapping_rows.append({
        "field_name": "ROOF_CODE",
        "raw_field": "주지붕",
        "raw_value": raw_value,
        "code": map_roof_code(raw_value),
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

    ("ROOF_CODE", 0): ("미상", "unknown"),
    ("ROOF_CODE", 1): ("콘크리트 지붕", "concrete_roof"),
    ("ROOF_CODE", 2): ("슬레이트 지붕", "slate_roof"),
    ("ROOF_CODE", 3): ("기와 지붕", "tile_roof"),
    ("ROOF_CODE", 4): ("기타 지붕", "other_roof"),
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

if os.path.exists(output_gpkg_path):
    os.remove(output_gpkg_path)

meta_final.to_file(output_gpkg_path, layer=output_layer, driver="GPKG")

meta_final.drop(columns="geometry").to_csv(output_csv_path, index=False, encoding="utf-8-sig")
raw_mapping.to_csv(output_raw_mapping_cvs_path, index=False, encoding="utf-8-sig")


# =========================================================
# 요약 출력
# =========================================================
print("\n[DONE]")
print(" -", output_gpkg_path)
print(" -", output_csv_path)
print(" -", output_raw_mapping_cvs_path)

print("\n[CODE COUNTS]")
for col in ["STRUCT_CODE", "USE_CODE", "ROOF_CODE", "WALL_MAT_CODE"]:
    print(f"\n{col}")
    print(meta_final[col].value_counts(dropna=False).sort_index())

print("\n[JOIN CHECK]")
print("total buildings:", len(meta_final))
print("register matched:", (meta["주구조"] != "NULL").sum())
print("register unmatched:", (meta["주구조"] == "NULL").sum())

print("\n[PNU_SUFFIX CHECK]")
print("unique building PNU_SUFFIX:", bld["PNU_SUFFIX"].nunique())
print("unique register PNU_SUFFIX:", reg_grouped["PNU_SUFFIX"].nunique())
