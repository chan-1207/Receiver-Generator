import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

from scripts.pipeline_common import (
    load_pipeline_settings,
    spatial_file_has_features,
    validate_filename_affix,
)


# =========================================================
# 프로젝트 경로
# =========================================================
project_dir = Path(__file__).resolve().parent
config_path = project_dir / "config/receiver_generation_full_range.json"


# =========================================================
# 기본 파일명
# =========================================================
DEFAULT_OUTPUT_FILES = {
    "building_metadata_gpkg": "metadata/building/building_metadata.gpkg",
    "building_metadata_csv": "metadata/building/building_metadata.csv",
    "building_buffer": "receivers/building/building_buffers_10m.gpkg",
    "wall_receiver": "receivers/building/building_receivers.csv",
    "roof_receiver": "receivers/building/building_roof_receivers.csv",
    "terrain_receiver": "receivers/terrain/terrain_receivers_center.csv",
    "merged_receiver": "receivers/merged_receivers.csv",
}
BUILDING_INPUT_LAYER = "TN_BULD"


# =========================================================
# 실행 단계
# =========================================================
pipeline = [
    ("건물 데이터셋 생성", "scripts/assemble_building_dataset.py", True),
    ("건물 벽면 버퍼 생성", "scripts/generate_building_wall_buffers.py", True),
    ("건물 벽면 수음점 생성", "scripts/generate_building_wall_receivers.py", True),
    ("건물 지붕 수음점 생성", "scripts/generate_building_roof_receivers.py", True),
    ("지면 수음점 생성", "scripts/generate_terrain_receivers.py", False),
    ("수음점 병합", "scripts/merge_receivers.py", False),
]


def load_settings():
    """JSON 설정 로드"""
    return load_pipeline_settings(config_path, project_dir)


def parse_filename_affix(value):
    """명령행 파일명 표식 검증"""
    try:
        return validate_filename_affix(value, "파일명 표식")
    except ValueError as error:
        raise argparse.ArgumentTypeError(str(error)) from error


def parse_args():
    """실행 인자 구성"""
    parser = argparse.ArgumentParser(
        description="수음점 생성 파이프라인 실행",
    )
    parser.add_argument(
        "--prefix",
        default=None,
        type=parse_filename_affix,
        help="설정 파일의 접두사를 임시로 덮어쓸 문자열",
    )
    parser.add_argument(
        "--suffix",
        default=None,
        type=parse_filename_affix,
        help="설정 파일의 접미사를 임시로 덮어쓸 문자열",
    )
    return parser.parse_args()


def make_output_paths(prefix="", suffix=""):
    """케이스 표식을 적용한 산출물 경로 구성"""
    output_paths = {}
    for output_name, relative_path_text in DEFAULT_OUTPUT_FILES.items():
        relative_path = Path(relative_path_text)
        filename = f"{prefix}{relative_path.stem}{suffix}{relative_path.suffix}"
        output_paths[output_name] = project_dir / relative_path.with_name(filename)
    return output_paths


def make_environment(settings, output_paths):
    """하위 스크립트 공통 환경변수 구성"""
    input_files = settings["input_files"]
    parallel_processing = settings["parallel_processing"]
    terrain_idw = settings["terrain_idw"]
    env = os.environ.copy()
    env.update({
        "PYTHONUTF8": "1",
        "RECEIVER_MIN_X": str(settings["min_x"]),
        "RECEIVER_MAX_X": str(settings["max_x"]),
        "RECEIVER_MIN_Y": str(settings["min_y"]),
        "RECEIVER_MAX_Y": str(settings["max_y"]),
        "RECEIVER_RESOLUTION_M": str(settings["resolution_m"]),
        "GRID_SIZE_M": str(settings["grid_size_m"]),
        "PARALLEL_WORKERS": str(parallel_processing["workers"]),
        "BUILDING_CHUNK_SIZE": str(
            parallel_processing["building_chunk_size"]
        ),
        "GROUND_FACTOR_CHUNK_SIZE": str(
            parallel_processing["ground_factor_chunk_size"]
        ),
        "TERRAIN_IDW_CHUNK_SIZE": str(
            parallel_processing["terrain_chunk_size"]
        ),
        "BUILDING_HEIGHT_INPUT_GPKG": str(input_files["building_height_gpkg"]),
        "BUILDING_ORIGINAL_INPUT_GPKG": str(input_files["building_height_gpkg"]),
        "TERRAIN_CONTOUR_INPUT_SHP": str(input_files["terrain_contour_shp"]),
        "TERRAIN_IDW_SEARCH_RADIUS_M": str(
            terrain_idw["search_radius_m"]
        ),
        "TERRAIN_IDW_MAX_SEARCH_RADIUS_M": str(
            terrain_idw["max_search_radius_m"]
        ),
        "TERRAIN_IDW_MIN_CONTOURS": str(terrain_idw["min_contours"]),
        "TERRAIN_IDW_MAX_CONTOURS": str(terrain_idw["max_contours"]),
        "TERRAIN_IDW_MIN_ELEVATION_LEVELS": str(
            terrain_idw["min_elevation_levels"]
        ),
        "TERRAIN_IDW_CONTOUR_SIMPLIFY_TOLERANCE_M": str(
            terrain_idw["contour_simplify_tolerance_m"]
        ),
        "LAND_COVER_INPUT_GPKG": str(input_files["land_cover_gpkg"]),
        "GROUND_FACTOR_MAPPING_CSV": str(
            input_files["ground_factor_mapping_csv"]
        ),
        "BUILDING_METADATA_OUTPUT_GPKG": str(
            output_paths["building_metadata_gpkg"]
        ),
        "BUILDING_METADATA_OUTPUT_CSV": str(
            output_paths["building_metadata_csv"]
        ),
        "BUILDING_METADATA_INPUT_GPKG": str(
            output_paths["building_metadata_gpkg"]
        ),
        "RECEIVER_BUFFER_OUTPUT_GPKG": str(output_paths["building_buffer"]),
        "RECEIVER_BUFFER_INPUT_GPKG": str(output_paths["building_buffer"]),
        "WALL_RECEIVER_OUTPUT_CSV": str(output_paths["wall_receiver"]),
        "ROOF_RECEIVER_OUTPUT_CSV": str(output_paths["roof_receiver"]),
        "TERRAIN_RECEIVER_OUTPUT_CSV": str(output_paths["terrain_receiver"]),
        "MERGE_TERRAIN_INPUT_CSV": str(output_paths["terrain_receiver"]),
        "MERGE_WALL_INPUT_CSV": str(output_paths["wall_receiver"]),
        "MERGE_ROOF_INPUT_CSV": str(output_paths["roof_receiver"]),
        "MERGE_BUFFER_INPUT_GPKG": str(output_paths["building_buffer"]),
        "MERGED_RECEIVER_OUTPUT_CSV": str(output_paths["merged_receiver"]),
    })
    return env


def run_pipeline(prefix=None, suffix=None):
    """전체 수음점 생성 파이프라인 실행"""
    settings = load_settings()
    filename_settings = settings["output_filename"]
    if prefix is None:
        prefix = filename_settings["prefix"]
    if suffix is None:
        suffix = filename_settings["suffix"]
    output_paths = make_output_paths(prefix, suffix)
    env = make_environment(settings, output_paths)
    area_bounds = (
        settings["min_x"],
        settings["min_y"],
        settings["max_x"],
        settings["max_y"],
    )
    has_buildings = spatial_file_has_features(
        settings["input_files"]["building_height_gpkg"],
        layer=BUILDING_INPUT_LAYER,
        bbox=area_bounds,
    )
    env["RECEIVER_HAS_BUILDINGS"] = "1" if has_buildings else "0"
    started_at = time.perf_counter()

    print("=" * 72)
    print("수음점 생성 시작")
    print(
        f"영역: X {settings['min_x']} ~ {settings['max_x']}, "
        f"Y {settings['min_y']} ~ {settings['max_y']}"
    )
    print(f"해상도: {settings['resolution_m']} m")
    print(f"격자 크기: {settings['grid_size_m']} m")
    print(f"파일명 형식: {prefix}<기본 파일명>{suffix}")
    if has_buildings:
        print("건물 데이터: 계산 영역 내 건물 있음")
    else:
        print("건물 데이터: 계산 영역 내 건물 없음, 건물 단계 생략")
    print("입력 파일:")
    for input_name, input_path in settings["input_files"].items():
        print(f" - {input_name}: {input_path}")
    print("=" * 72)

    for step_no, (
        step_name,
        relative_script_path,
        requires_buildings,
    ) in enumerate(
        pipeline,
        start=1,
    ):
        if requires_buildings and not has_buildings:
            print(
                f"\n[{step_no}/{len(pipeline)}] {step_name} "
                "- 건물 없음, 건너뜀",
                flush=True,
            )
            continue

        script_path = project_dir / relative_script_path
        print(f"\n[{step_no}/{len(pipeline)}] {step_name}", flush=True)
        subprocess.run(
            [sys.executable, str(script_path)],
            cwd=project_dir,
            env=env,
            check=True,
        )

    elapsed_seconds = time.perf_counter() - started_at
    print("\n" + "=" * 72)
    print("수음점 생성 완료")
    print(f"병합 결과: {output_paths['merged_receiver']}")
    print(f"총 실행 시간: {elapsed_seconds:.1f}초")
    print("=" * 72)


if __name__ == "__main__":
    arguments = parse_args()
    run_pipeline(arguments.prefix, arguments.suffix)
