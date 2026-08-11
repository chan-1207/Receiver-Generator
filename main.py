import os
import subprocess
import sys
import time
from pathlib import Path

from scripts.pipeline_common import load_pipeline_settings


# =========================================================
# 프로젝트 경로
# =========================================================
project_dir = Path(__file__).resolve().parent
config_path = project_dir / "config/receiver_generation.json"


# =========================================================
# 중간 산출물 및 최종 결과 경로
# =========================================================
building_metadata_gpkg_path = (
    project_dir / "metadata/building/building_metadata.gpkg"
)
building_metadata_csv_path = (
    project_dir / "metadata/building/building_metadata.csv"
)
building_buffer_path = (
    project_dir / "receivers/building/building_buffers_10m.gpkg"
)
wall_receiver_path = (
    project_dir / "receivers/building/building_receivers.csv"
)
roof_receiver_path = (
    project_dir / "receivers/building/building_roof_receivers.csv"
)
terrain_receiver_path = (
    project_dir / "receivers/terrain/terrain_receivers_center.csv"
)
terrain_dem_path = project_dir / "metadata/terrain/terrain_dem.tif"
merged_receiver_path = project_dir / "receivers/merged_receivers.csv"


# =========================================================
# 실행 단계
# =========================================================
pipeline = [
    ("건물 데이터셋 생성", "scripts/assemble_building_dataset.py"),
    ("건물 벽면 버퍼 생성", "scripts/generate_building_wall_buffers.py"),
    ("건물 벽면 수음점 생성", "scripts/generate_building_wall_receivers.py"),
    ("건물 지붕 수음점 생성", "scripts/generate_building_roof_receivers.py"),
    ("지형 DEM 생성", "scripts/generate_terrain_dem.py"),
    ("지면 수음점 생성", "scripts/generate_terrain_receivers.py"),
    ("수음점 병합", "scripts/merge_receivers.py"),
]


def load_settings():
    """JSON 설정 로드"""
    return load_pipeline_settings(config_path, project_dir)


def make_environment(settings):
    """하위 스크립트 공통 환경변수 구성"""
    input_files = settings["input_files"]
    env = os.environ.copy()
    env.update({
        "PYTHONUTF8": "1",
        "RECEIVER_MIN_X": str(settings["min_x"]),
        "RECEIVER_MAX_X": str(settings["max_x"]),
        "RECEIVER_MIN_Y": str(settings["min_y"]),
        "RECEIVER_MAX_Y": str(settings["max_y"]),
        "RECEIVER_RESOLUTION_M": str(settings["resolution_m"]),
        "GRID_SIZE_M": str(settings["grid_size_m"]),
        "BUILDING_HEIGHT_INPUT_GPKG": str(input_files["building_height_gpkg"]),
        "BUILDING_ORIGINAL_INPUT_GPKG": str(input_files["building_height_gpkg"]),
        "TERRAIN_CONTOUR_INPUT_SHP": str(input_files["terrain_contour_shp"]),
        "TERRAIN_DEM_OUTPUT_TIF": str(terrain_dem_path),
        "TERRAIN_DEM_INPUT_TIF": str(terrain_dem_path),
        "TERRAIN_DEM_RESOLUTION_M": str(settings["resolution_m"]),
        "LAND_COVER_INPUT_GPKG": str(input_files["land_cover_gpkg"]),
        "GROUND_FACTOR_MAPPING_CSV": str(
            input_files["ground_factor_mapping_csv"]
        ),
        "BUILDING_METADATA_OUTPUT_GPKG": str(building_metadata_gpkg_path),
        "BUILDING_METADATA_OUTPUT_CSV": str(building_metadata_csv_path),
        "BUILDING_METADATA_INPUT_GPKG": str(building_metadata_gpkg_path),
        "RECEIVER_BUFFER_OUTPUT_GPKG": str(building_buffer_path),
        "RECEIVER_BUFFER_INPUT_GPKG": str(building_buffer_path),
        "WALL_RECEIVER_OUTPUT_CSV": str(wall_receiver_path),
        "ROOF_RECEIVER_OUTPUT_CSV": str(roof_receiver_path),
        "TERRAIN_RECEIVER_OUTPUT_CSV": str(terrain_receiver_path),
        "MERGE_TERRAIN_INPUT_CSV": str(terrain_receiver_path),
        "MERGE_WALL_INPUT_CSV": str(wall_receiver_path),
        "MERGE_ROOF_INPUT_CSV": str(roof_receiver_path),
        "MERGE_BUFFER_INPUT_GPKG": str(building_buffer_path),
        "MERGED_RECEIVER_OUTPUT_CSV": str(merged_receiver_path),
    })
    return env


def run_pipeline():
    """전체 수음점 생성 파이프라인 실행"""
    settings = load_settings()
    env = make_environment(settings)
    started_at = time.perf_counter()

    print("=" * 72)
    print("수음점 생성 시작")
    print(
        f"영역: X {settings['min_x']} ~ {settings['max_x']}, "
        f"Y {settings['min_y']} ~ {settings['max_y']}"
    )
    print(f"해상도: {settings['resolution_m']} m")
    print(f"격자 크기: {settings['grid_size_m']} m")
    print("입력 파일:")
    for input_name, input_path in settings["input_files"].items():
        print(f" - {input_name}: {input_path}")
    print("=" * 72)

    for step_no, (step_name, relative_script_path) in enumerate(
        pipeline,
        start=1,
    ):
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
    print(f"병합 결과: {merged_receiver_path}")
    print(f"총 실행 시간: {elapsed_seconds:.1f}초")
    print("=" * 72)


if __name__ == "__main__":
    run_pipeline()
