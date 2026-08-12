import json
import math
import os
from pathlib import Path

import pyogrio
from pyproj import Transformer


NUMERIC_SETTING_KEYS = [
    "min_x",
    "max_x",
    "min_y",
    "max_y",
    "resolution_m",
    "grid_size_m",
]

INPUT_FILE_KEYS = [
    "building_height_gpkg",
    "terrain_contour_shp",
    "land_cover_gpkg",
    "ground_factor_mapping_csv",
]

RECEIVER_CRS = "EPSG:5179"
TERRAIN_DEM_SOURCE_PADDING_M = 2000.0
DEFAULT_TERRAIN_IDW_SETTINGS = {
    "search_radius_m": 800.0,
    "max_search_radius_m": 2000.0,
    "min_contours": 4,
    "max_contours": 8,
    "min_elevation_levels": 2,
    "workers": 8,
    "chunk_size": 2000,
}
INVALID_FILENAME_CHARACTERS = '<>:"/\\|?*'


def get_env_path(name, default):
    """환경변수 기반 경로 반환"""
    return Path(os.environ.get(name, default))


def get_env_float(name, default):
    """환경변수 기반 실수 반환"""
    try:
        return float(os.environ.get(name, default))
    except (TypeError, ValueError) as error:
        raise ValueError(f"환경변수 {name}의 값이 숫자가 아닙니다.") from error


def get_env_int(name, default):
    """환경변수 기반 정수 반환"""
    raw_value = os.environ.get(name, default)
    try:
        value = int(raw_value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"환경변수 {name}의 값이 정수가 아닙니다.") from error
    return value


def get_env_bool(name, default):
    """환경변수 기반 불리언 반환"""
    raw_value = os.environ.get(name)
    if raw_value is None:
        return default

    normalized_value = raw_value.strip().lower()
    if normalized_value in {"1", "true", "yes", "on"}:
        return True
    if normalized_value in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"환경변수 {name}의 값이 불리언이 아닙니다.")


def resolve_input_path(path_value, project_dir):
    """프로젝트 루트 기준 입력 경로 변환"""
    input_path = Path(path_value).expanduser()
    if input_path.is_absolute():
        return input_path

    return (project_dir / input_path).resolve()


def validate_bounds(min_x, max_x, min_y, max_y):
    """계산 영역 검증"""
    if min_x >= max_x or min_y >= max_y:
        raise ValueError("계산 영역의 최솟값은 최댓값보다 작아야 합니다.")


def validate_positive(value, name):
    """양수 설정값 검증"""
    if value <= 0:
        raise ValueError(f"{name}은 0보다 커야 합니다.")


def validate_filename_affix(value, name):
    """파일명 접두사 및 접미사 검증"""
    if not isinstance(value, str):
        raise ValueError(f"{name}은 문자열이어야 합니다.")
    if any(character in INVALID_FILENAME_CHARACTERS for character in value):
        raise ValueError(
            f"{name}에 사용할 수 없는 문자가 포함되어 있습니다: "
            f"{INVALID_FILENAME_CHARACTERS}"
        )
    if any(ord(character) < 32 for character in value):
        raise ValueError(f"{name}에 제어 문자를 사용할 수 없습니다.")
    return value


def validate_grid_settings(settings):
    """영역 및 격자 설정 검증"""
    min_x = settings["min_x"]
    max_x = settings["max_x"]
    min_y = settings["min_y"]
    max_y = settings["max_y"]
    resolution_m = settings["resolution_m"]
    grid_size_m = settings["grid_size_m"]

    validate_bounds(min_x, max_x, min_y, max_y)
    validate_positive(resolution_m, "해상도")
    validate_positive(grid_size_m, "격자 크기")

    for axis_name, length in [
        ("X", max_x - min_x),
        ("Y", max_y - min_y),
    ]:
        cell_count = length / resolution_m
        if not math.isclose(cell_count, round(cell_count)):
            raise ValueError(
                f"{axis_name} 영역 길이는 해상도의 정수배여야 합니다: "
                f"{length} / {resolution_m}"
            )

        grid_count = length / grid_size_m
        if not math.isclose(grid_count, round(grid_count)):
            raise ValueError(
                f"{axis_name} 영역 길이는 격자 크기의 정수배여야 합니다: "
                f"{length} / {grid_size_m}"
            )

    cells_per_grid = grid_size_m / resolution_m
    if not math.isclose(cells_per_grid, round(cells_per_grid)):
        raise ValueError(
            "격자 크기는 해상도의 정수배여야 합니다: "
            f"{grid_size_m} / {resolution_m}"
        )


def load_terrain_idw_settings(raw_settings):
    """IDW 설정 로드 및 검증"""
    raw_idw = raw_settings.get("terrain_idw", {})
    if not isinstance(raw_idw, dict):
        raise ValueError("설정 파일의 terrain_idw 항목은 객체여야 합니다.")

    idw_settings = DEFAULT_TERRAIN_IDW_SETTINGS.copy()
    idw_settings.update(raw_idw)
    try:
        for name in ("search_radius_m", "max_search_radius_m"):
            idw_settings[name] = float(idw_settings[name])
        for name in (
            "min_contours",
            "max_contours",
            "min_elevation_levels",
            "workers",
            "chunk_size",
        ):
            raw_value = idw_settings[name]
            numeric_value = float(raw_value)
            if not numeric_value.is_integer():
                raise ValueError
            idw_settings[name] = int(numeric_value)
    except (TypeError, ValueError) as error:
        raise ValueError(
            "terrain_idw의 반경·등고선 개수·병렬 설정은 "
            "숫자여야 합니다."
        ) from error

    validate_positive(idw_settings["search_radius_m"], "IDW 기준 반경")
    validate_positive(
        idw_settings["max_search_radius_m"],
        "IDW 최대 반경",
    )
    validate_positive(idw_settings["min_contours"], "IDW 최소 등고선 개수")
    validate_positive(idw_settings["max_contours"], "IDW 최대 등고선 개수")
    validate_positive(
        idw_settings["min_elevation_levels"],
        "IDW 최소 표고 단계 수",
    )
    validate_positive(idw_settings["workers"], "IDW 병렬 작업 수")
    validate_positive(idw_settings["chunk_size"], "IDW 묶음 크기")
    if (
        idw_settings["search_radius_m"]
        > idw_settings["max_search_radius_m"]
    ):
        raise ValueError("IDW 기준 반경은 최대 반경보다 클 수 없습니다.")
    if idw_settings["min_contours"] > idw_settings["max_contours"]:
        raise ValueError(
            "IDW 최소 등고선 개수는 최대 등고선 개수보다 클 수 없습니다."
        )
    if (
        idw_settings["min_elevation_levels"]
        > idw_settings["max_contours"]
    ):
        raise ValueError(
            "IDW 최소 표고 단계 수는 최대 등고선 개수보다 클 수 없습니다."
        )
    return idw_settings


def validate_input_paths(paths):
    """입력 파일 존재 여부 검증"""
    missing_paths = [Path(path) for path in paths if not Path(path).exists()]
    if missing_paths:
        missing_text = "\n".join(f" - {path}" for path in missing_paths)
        raise FileNotFoundError(f"입력 파일이 없습니다:\n{missing_text}")


def spatial_file_has_features(path, layer, bbox):
    """지정 영역과 교차하는 공간 객체 존재 여부 반환"""
    features = pyogrio.read_dataframe(
        path,
        layer=layer,
        bbox=bbox,
        columns=[],
        read_geometry=False,
        fid_as_index=True,
        max_features=1,
    )
    return len(features) > 0


def read_spatial_bounds(path, layer=None, target_crs=RECEIVER_CRS):
    """대상 좌표계 기준 공간 데이터 범위 반환"""
    info = pyogrio.read_info(path, layer=layer)
    source_crs = info.get("crs")
    source_bounds = info.get("total_bounds")

    if source_crs is None:
        raise ValueError(f"공간 입력 파일에 CRS가 없습니다: {path}")
    if source_bounds is None:
        raise ValueError(f"공간 입력 파일의 범위를 읽을 수 없습니다: {path}")

    transformer = Transformer.from_crs(
        source_crs,
        target_crs,
        always_xy=True,
    )
    return transformer.transform_bounds(
        *source_bounds,
        densify_pts=21,
    )


def get_coverage_gaps(data_bounds, required_bounds, tolerance_m=0.01):
    """필수 영역 대비 방향별 부족 거리 반환"""
    data_min_x, data_min_y, data_max_x, data_max_y = data_bounds
    req_min_x, req_min_y, req_max_x, req_max_y = required_bounds
    gaps = {
        "서쪽": max(0.0, data_min_x - req_min_x),
        "동쪽": max(0.0, req_max_x - data_max_x),
        "남쪽": max(0.0, data_min_y - req_min_y),
        "북쪽": max(0.0, req_max_y - data_max_y),
    }
    return {
        direction: distance
        for direction, distance in gaps.items()
        if distance > tolerance_m
    }


def format_bounds(bounds):
    """공간 범위 출력 문자열 반환"""
    min_x, min_y, max_x, max_y = bounds
    return (
        f"X={min_x:.1f}~{max_x:.1f}, "
        f"Y={min_y:.1f}~{max_y:.1f}"
    )


def get_spatial_coverage_problem(
    path,
    label,
    required_bounds,
    layer=None,
):
    """공간 범위 부족 설명 반환"""
    data_bounds = read_spatial_bounds(path, layer=layer)
    gaps = get_coverage_gaps(data_bounds, required_bounds)
    if not gaps:
        return None

    gap_text = ", ".join(
        f"{direction} {distance:.1f}m"
        for direction, distance in gaps.items()
    )
    return (
        f"[{label}] {path}\n"
        f" - 데이터 범위: {format_bounds(data_bounds)}\n"
        f" - 필요 범위: {format_bounds(required_bounds)}\n"
        f" - 부족 범위: {gap_text}"
    )


def validate_spatial_file_coverage(
    path,
    label,
    required_bounds,
    layer=None,
):
    """단일 공간 입력 범위 검증"""
    problem = get_spatial_coverage_problem(
        path=path,
        label=label,
        required_bounds=required_bounds,
        layer=layer,
    )
    if problem:
        raise ValueError(f"공간 입력 범위가 부족합니다:\n{problem}")


def validate_pipeline_spatial_coverage(settings):
    """전체 공간 입력 범위 검증"""
    area_bounds = (
        settings["min_x"],
        settings["min_y"],
        settings["max_x"],
        settings["max_y"],
    )
    terrain_padding_m = settings["terrain_idw"]["max_search_radius_m"]
    terrain_bounds = (
        settings["min_x"] - terrain_padding_m,
        settings["min_y"] - terrain_padding_m,
        settings["max_x"] + terrain_padding_m,
        settings["max_y"] + terrain_padding_m,
    )
    input_files = settings["input_files"]
    specifications = [
        (
            input_files["building_height_gpkg"],
            "건물 데이터",
            area_bounds,
            "TN_BULD",
        ),
        (
            input_files["land_cover_gpkg"],
            "토지피복도",
            area_bounds,
            "land_cover_map",
        ),
        (
            input_files["terrain_contour_shp"],
            "등고선",
            terrain_bounds,
            None,
        ),
    ]

    problems = []
    for path, label, required_bounds, layer in specifications:
        problem = get_spatial_coverage_problem(
            path=path,
            label=label,
            required_bounds=required_bounds,
            layer=layer,
        )
        if problem:
            problems.append(problem)

    if problems:
        raise ValueError(
            "공간 입력 데이터가 계산 영역을 포함하지 않습니다.\n\n"
            + "\n\n".join(problems)
        )


def load_pipeline_settings(config_path, project_dir):
    """JSON 파이프라인 설정 로드"""
    config_path = Path(config_path)
    project_dir = Path(project_dir)
    validate_input_paths([config_path])

    with config_path.open("r", encoding="utf-8") as config_file:
        raw_settings = json.load(config_file)

    missing_keys = [
        key for key in NUMERIC_SETTING_KEYS
        if key not in raw_settings
    ]
    if missing_keys:
        raise ValueError(f"설정 파일에 필수 항목이 없습니다: {missing_keys}")

    input_files = raw_settings.get("input_files")
    if not isinstance(input_files, dict):
        raise ValueError("설정 파일의 input_files 항목은 객체여야 합니다.")

    missing_input_keys = [
        key for key in INPUT_FILE_KEYS
        if key not in input_files
    ]
    if missing_input_keys:
        raise ValueError(
            "input_files에 필수 항목이 없습니다: "
            f"{missing_input_keys}"
        )

    try:
        settings = {
            key: float(raw_settings[key])
            for key in NUMERIC_SETTING_KEYS
        }
    except (TypeError, ValueError) as error:
        raise ValueError("영역, 해상도, 격자 크기는 숫자여야 합니다.") from error

    settings["input_files"] = {
        key: resolve_input_path(input_files[key], project_dir)
        for key in INPUT_FILE_KEYS
    }
    settings["terrain_idw"] = load_terrain_idw_settings(raw_settings)

    output_filename = raw_settings.get("output_filename", {})
    if not isinstance(output_filename, dict):
        raise ValueError("설정 파일의 output_filename 항목은 객체여야 합니다.")
    settings["output_filename"] = {
        name: validate_filename_affix(
            output_filename.get(name, ""),
            f"output_filename.{name}",
        )
        for name in ("prefix", "suffix")
    }

    validate_grid_settings(settings)
    validate_input_paths(settings["input_files"].values())
    validate_pipeline_spatial_coverage(settings)

    return settings
