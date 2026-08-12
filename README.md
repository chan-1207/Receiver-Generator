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
  "terrain_idw": {
    "search_radius_m": 800.0,
    "max_search_radius_m": 2000.0,
    "min_contours": 4,
    "max_contours": 8,
    "min_elevation_levels": 2,
    "power": 2.0
  },
  "output_filename": {
    "prefix": "case_a_",
    "suffix": "_10m"
  },
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

케이스별 산출물을 구분하려면 `output_filename.prefix`와
`output_filename.suffix`를 설정합니다. 위 예시의 최종 파일명은
`case_a_merged_receivers_10m.csv`입니다. 접두사와 접미사는 모든 중간
산출물과 최종 병합 파일에 동일하게 적용됩니다.

명령행 옵션은 설정 파일의 값을 이번 실행에만 덮어쓸 때 사용할 수 있습니다.

```powershell
# 접두사 임시 지정 예시
.\.venv\Scripts\python.exe main.py --prefix case_a_

# 접미사 임시 지정 예시
.\.venv\Scripts\python.exe main.py --suffix _case_a

# 동시 임시 지정 예시
.\.venv\Scripts\python.exe main.py --prefix test_ --suffix _case_a
```

명령행 옵션 없이 실행하면 설정 파일의 값을 사용합니다. `output_filename` 항목을
생략하면 접두사와 접미사가 없는 기존 기본 파일명을 사용합니다.

`resolution_m`은 지면 격자와 건물 벽면·지붕 수음점에 공통으로 적용됩니다.
`grid_size_m`은 `resolution_m`의 정수배여야 하며, X/Y 영역 길이는 두
크기의 정수배여야 합니다.

지면고도는 등고선 객체별 최근접 거리의 적응형 IDW 보간으로 계산합니다. 같은
등고선의 여러 정점은 하나의 고도 정보로 처리합니다. 먼저 `search_radius_m` 안에서
가까운 등고선을 최대 `max_contours`개 사용합니다. `min_contours`개 또는 서로 다른
표고 `min_elevation_levels`단계를 확보하지 못하면 `max_search_radius_m`까지 탐색을
확대합니다. 최대 반경에서도 조건을 충족하지 못하면 실패 좌표와 확보 정보를
출력하고 중단합니다. 수음점이 등고선과 직접 교차하면 해당 표고를 사용합니다.
거리 계산용 등고선은 수음점 해상도의 1/5과 2m 중 작은 값으로 단순화하며,
`power`는 거리 가중 지수입니다.

`input_files`의 경로는 프로젝트 루트 기준 상대경로 또는 절대경로로
설정할 수 있습니다. `main.py`가 건물 메타데이터, 건물 버퍼, 벽면·지붕·지면
수음점을 순서대로 새로 생성하므로 기존 중간 산출물은 필요하지 않습니다.

파이프라인 시작 전 모든 공간 입력을 EPSG:5179로 변환해 범위를 검사합니다.
건물과 토지피복도는 계산 영역 전체, 등고선은 계산 영역 외곽
`max_search_radius_m`까지 포함해야 합니다. 범위가 부족하면 데이터 범위, 필요 범위,
방향별 부족 거리를 출력하고 계산을 시작하지 않습니다. 토지피복도는 지면 격자별
실제 피복률을 검사하며, IDW 조건을 만족하지 못한 수음점이 하나라도 있으면 저장을
중단합니다.

건물 입력 파일은 존재하지만 계산 영역과 교차하는 건물이 없으면 오류로 중단하지
않습니다. 건물 데이터셋·버퍼·벽면·지붕 수음점 단계를 로그와 함께 건너뛰고,
지면 수음점을 생성한 뒤 지면 수음점만 최종 파일로 병합합니다. 이 경우
이전 실행에서 생성된 건물 산출물은 읽지 않습니다.

개별 단계는 `scripts` 디렉터리에서 다음 순서로 실행할 수도 있습니다.

```powershell
..\.venv\Scripts\python.exe assemble_building_dataset.py
..\.venv\Scripts\python.exe generate_building_wall_buffers.py
..\.venv\Scripts\python.exe generate_building_wall_receivers.py
..\.venv\Scripts\python.exe generate_building_roof_receivers.py
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
- `scripts/generate_terrain_receivers.py`: 등고선 IDW 고도 기반 지면 수음점 및 지면계수 생성
- `scripts/merge_receivers.py`: 수음점 필터링, 정렬, ID 부여 및 병합

각 단계 스크립트의 실행부는 `main()`에 있으며, 파일을 모듈로 가져올 때는
계산이나 파일 저장이 시작되지 않습니다. 별도 클래스 없이 변환 단계별 함수와
실행 진입점을 분리한 구조입니다.

벽면 수음점용 `building_buffer`는 기준 `building_simplified`에 1m 외곽
버퍼만 적용한 파생 레이어입니다. `main.py` 실행 시 지붕 수음점 전처리는
이 버퍼와 `input_files.building_height_gpkg`의 원본 형상을 함께 사용합니다.
