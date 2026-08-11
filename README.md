# Receiver Generator for noise map

## 기준 건물 데이터

`scripts/assemble_building_dataset.py`가
`metadata/building/building_cropped_metadata.gpkg`를 생성합니다.

- `building_simplified`: 전파 계산 속성을 포함한 공통 단순화 건물 형상
- `building_mrr`: 소음 전파 모델의 후보군 필터링용 최소면적 회전사각형

전파용 GeoPackage에는 원본 건물 형상을 중복 저장하지 않습니다. 원본 형상이
필요한 수음점 전처리는 `data/building_height/cropped_building_height.gpkg`를
직접 사용합니다.

`building_simplified`는 모폴로지 닫힘 10m와 위상 보존 단순화 1m를
적용한 형상입니다. 수음점용 1m 외곽 버퍼는 포함하지 않습니다.
`building_mrr`는 이 단순화 형상 전체를 감싸는 `NF_ID`당 하나의
사각형입니다. 후보군 누락 방지를 위해 외곽에 1mm의 수치 안전 여유를
적용합니다.

## 수음점 생성 순서

전체 과정을 한 번에 실행하려면 먼저
`config/receiver_generation.json`에서 계산 범위와 해상도를 설정합니다.

```json
{
  "min_x": 1163000,
  "max_x": 1164000,
  "min_y": 1732000,
  "max_y": 1733000,
  "resolution_m": 10.0,
  "grid_size_m": 100.0,
  "input_files": {
    "building_height_gpkg": "data/building_height/building_height.gpkg",
    "terrain_contour_shp": "data/terrain/terrain.shp",
    "land_cover_gpkg": "data/land_cover_map/land_cover_map.gpkg",
    "ground_factor_mapping_csv": "config/land_cover_ground_factor.csv"
  }
}
```

프로젝트 루트에서 다음 명령을 실행합니다.

```powershell
.\.venv\Scripts\python.exe main.py
```

`resolution_m`은 지형 DEM과 지면 격자, 건물 벽면·지붕 수음점에 공통으로
적용됩니다. `grid_size_m`은 `resolution_m`의 정수배여야 하며,
X/Y 영역 길이는 두 크기의 정수배여야 합니다.

`input_files`의 경로는 프로젝트 루트 기준 상대경로 또는 절대경로로
설정할 수 있습니다. `main.py`가 건물 메타데이터, 건물 버퍼, 벽면·지붕·지면
수음점을 순서대로 새로 생성하므로 기존 중간 산출물은 필요하지 않습니다.

파이프라인 시작 전 모든 공간 입력을 EPSG:5179로 변환해 범위를 검사합니다.
건물과 토지피복도는 계산 영역 전체, 등고선은 TIN 생성을 위한 계산 영역 외곽 2km까지
포함해야 합니다. 범위가 부족하면 데이터 범위, 필요 범위, 방향별 부족 거리를
출력하고 계산을 시작하지 않습니다. 등고선으로 설정 해상도의 GeoTIFF DEM을
생성하고, TIN이 영역 전체를 덮는지도 검사합니다. 토지피복도는 지면 격자별
실제 피복률을 검사하며, DEM 고도가 하나라도 없으면 지면 수음점 저장을 중단합니다.

개별 단계는 `scripts` 디렉터리에서 다음 순서로 실행할 수도 있습니다.

```powershell
..\.venv\Scripts\python.exe assemble_building_dataset.py
..\.venv\Scripts\python.exe generate_building_wall_buffers.py
..\.venv\Scripts\python.exe generate_building_wall_receivers.py
..\.venv\Scripts\python.exe generate_building_roof_receivers.py
..\.venv\Scripts\python.exe generate_terrain_dem.py
..\.venv\Scripts\python.exe generate_terrain_receivers.py
..\.venv\Scripts\python.exe merge_receivers.py
```

## 코드 구조

- `main.py`: 설정 로드, 단계별 환경 구성, 전체 파이프라인 실행
- `scripts/pipeline_common.py`: JSON 설정, 환경변수, 영역, 해상도, 입력 경로 검증
- `scripts/assemble_building_dataset.py`: 건물 메타데이터 생성
- `scripts/generate_building_wall_buffers.py`: 벽면 수음점용 건물 버퍼 생성
- `scripts/generate_building_wall_receivers.py`: 벽면 수음점 생성
- `scripts/generate_building_roof_receivers.py`: 지붕 수음점 생성
- `scripts/generate_terrain_dem.py`: 등고선 TIN 기반 GeoTIFF DEM 생성
- `scripts/generate_terrain_receivers.py`: DEM 고도 기반 지면 수음점 및 지면계수 생성
- `scripts/merge_receivers.py`: 수음점 필터링, 정렬, ID 부여 및 병합

각 단계 스크립트의 실행부는 `main()`에 있으며, 파일을 모듈로 가져올 때는
계산이나 파일 저장이 시작되지 않습니다. 별도 클래스 없이 변환 단계별 함수와
실행 진입점을 분리한 구조입니다.

벽면 수음점용 `building_buffer`는 기준 `building_simplified`에 1m 외곽
버퍼만 적용한 파생 레이어입니다. `main.py` 실행 시 지붕 수음점 전처리는
이 버퍼와 `input_files.building_height_gpkg`의 원본 형상을 함께 사용합니다.
