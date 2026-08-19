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
  "area": {
    "min_x": 1163000,
    "max_x": 1164000,
    "min_y": 1732000,
    "max_y": 1733000
  },
  "grid": {
    "resolution_m": 10.0,
    "grid_size_m": 100.0
  },
  "parallel_processing": {
    "workers": 8,
    "building_chunk_size": 250,
    "ground_factor_chunk_size": 20000,
    "terrain_chunk_size": 2000
  },
  "terrain_idw": {
    "search_radius_m": 800.0,
    "max_search_radius_m": 2000.0,
    "min_contours": 4,
    "max_contours": 8,
    "min_elevation_levels": 2,
    "contour_simplify_tolerance_m": 2.0
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

### 설정 항목

| 1단계 그룹 | 2단계 항목 | 의미 | 기본값 |
|---|---|---|---:|
| **`area`** | `min_x` | EPSG:5179 X 최솟값 | 필수 |
|  | `max_x` | EPSG:5179 X 최댓값 | 필수 |
|  | `min_y` | EPSG:5179 Y 최솟값 | 필수 |
|  | `max_y` | EPSG:5179 Y 최댓값 | 필수 |
| **`grid`** | `resolution_m` | 지면 및 건물 수음점 간격 | 10m |
|  | `grid_size_m` | 병합 결과의 격자 ID 크기 | 100m |
| **`parallel_processing`** | `workers` | 모든 병렬 계산의 공통 작업 수 | 8 |
|  | `building_chunk_size` | 작업당 최대 건물 수 | 250 |
|  | `ground_factor_chunk_size` | 작업당 최대 지면 셀 수 | 20,000 |
|  | `terrain_chunk_size` | 작업당 최대 IDW 보간 수음점 수 | 2,000 |
| **`terrain_idw`** | `search_radius_m` | 우선 탐색할 등고선 반경 | 800m |
|  | `max_search_radius_m` | 조건 부족 시 확장할 최대 반경 | 2,000m |
|  | `min_contours` | 수음점별 최소 등고선 수 | 4 |
|  | `max_contours` | 수음점별 최대 등고선 수 | 8 |
|  | `min_elevation_levels` | 최소 서로 다른 표고 단계 수 | 2 |
|  | `contour_simplify_tolerance_m` | 거리 계산용 등고선 단순화 허용오차 | 2m |
| **`output_filename`** | `prefix` | 모든 산출물의 파일명 접두사 | 빈 문자열 |
|  | `suffix` | 모든 산출물의 파일명 접미사 | 빈 문자열 |
| **`input_files`** | `building_height_gpkg` | 건물 높이 GeoPackage | 필수 |
|  | `terrain_contour_shp` | 등고선 Shapefile | 필수 |
|  | `land_cover_gpkg` | 토지피복 GeoPackage | 필수 |
|  | `ground_factor_mapping_csv` | 토지피복별 지면계수 매핑 | 필수 |

`grid.resolution_m`은 지면 격자와 건물 벽면·지붕 수음점에 공통으로 적용됩니다.
`grid.grid_size_m`은 `grid.resolution_m`의 정수배여야 하며, X/Y 영역 길이는 두
크기의 정수배여야 합니다. 데이터가 작으면 모든 작업자가 사용될 수 있도록 실제
묶음 크기를 설정값보다 작게 자동 조정합니다. 작업 수를 `1`로 지정하면 모든
계산을 순차 실행합니다.

지면고도는 등고선 객체까지의 거리를 이용한 적응형 IDW로 계산합니다. 우선
`search_radius_m` 안에서 가까운 등고선을 최대 `max_contours`개 사용합니다.
최소 등고선 수 또는 서로 다른 표고 단계 수가 부족하면
`max_search_radius_m`까지 탐색을 확대합니다. 수음점이 등고선과 직접 겹치면 해당
등고선 표고를 사용합니다.

`main.py`는 지면 수음점 생성 전에 토지피복 코드 `720`(해양수)과 비해양
폴리곤의 공통 경계를 추출해 고도 0m 해안선을 생성합니다. 해양수 격자는 수음점에서
제외하고, 생성한 해안선은 등고선과 함께 IDW 입력으로 사용해 해안가 고도가 내륙
등고선만으로 과대 추정되는 현상을 줄입니다.

IDW 입력에는 등고선뿐 아니라 `building_simplified` 건물 폴리곤도 포함합니다.
각 건물 폴리곤까지의 최근접 거리와 해당 건물의 지반고 `BLDH_MN`을 사용하며,
최소 `min_contours`개의 등고선을 우선 확보한 뒤 남은 후보를 등고선과 건물 중
가까운 객체로 채워 지면고도를 계산합니다. 따라서 밀집 건물 지역에서도 등고선이
IDW 입력에서 완전히 제외되지 않습니다. 동시에
건물 폴리곤 내부 또는 경계에 놓인 지면 격자 중심점을 제거하고, 최종 병합 단계에서
기존 1m 건물 버퍼 내부의 지면 수음점을 추가로 제거합니다. 보간 후 건물에 맞춰
주변 지형고를 다시 자르는 사후 보정은 적용하지 않습니다.
건물 마스킹은 전체 지면 격자를 제한된 크기의 묶음으로 나누어
`parallel_processing.workers`만큼 병렬 처리하며, 위경도 변환은 건물과 해양수
격자를 제거한 최종 육지 수음점에만 수행합니다.

`input_files`의 경로는 프로젝트 루트 기준 상대경로 또는 절대경로로
설정할 수 있습니다. `main.py`가 건물 메타데이터, 건물 버퍼, 벽면·지붕·지면
수음점을 순서대로 새로 생성하므로 기존 중간 산출물은 필요하지 않습니다.
모든 GPKG와 CSV 산출물은 최종 파일과 같은 폴더의 `.tmp` 파일에 먼저 완전히
기록한 뒤 최종 경로로 교체합니다. 최종 파일이 QGIS나 Excel에 열려 있어 교체할 수
없으면 기존 파일과 완성된 임시 파일을 모두 보존하고 경로를 오류 메시지에 표시합니다.

파이프라인 시작 전 건물과 토지피복도를 EPSG:5179로 변환해 계산 영역 전체를
포함하는지 검사합니다. 등고선은 해안 밖 바다에는 존재하지 않으므로 사각형 외곽
`max_search_radius_m` 전체에 대한 BBOX 검사는 적용하지 않습니다. 대신 해양수
수음점을 제외한 뒤 각 육지 수음점에서 실제 IDW 입력 조건을 검사하며, 조건을
만족하지 못한 육지 수음점이 하나라도 있으면 저장을 중단합니다. 토지피복도는 지면
격자별 실제 피복률을 검사합니다.

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
..\.venv\Scripts\python.exe extract_coastline.py --overwrite
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
  ([상세 생성 로직](docs/roof-receiver-generation.md))
- `scripts/extract_coastline.py`: 토지피복도에서 고도 0m 해안선 생성
- `scripts/generate_terrain_receivers.py`: 등고선 IDW 및 건물 폴리곤 마스킹 기반 지면 수음점 및 지면계수 생성
- `scripts/merge_receivers.py`: 수음점 필터링, 정렬, ID 부여 및 병합

각 단계 스크립트의 실행부는 `main()`에 있으며, 파일을 모듈로 가져올 때는
계산이나 파일 저장이 시작되지 않습니다. 별도 클래스 없이 변환 단계별 함수와
실행 진입점을 분리한 구조입니다.

벽면 수음점용 `building_buffer`는 기준 `building_simplified`에 1m 외곽
버퍼만 적용한 파생 레이어입니다. 지붕 수음점은 버퍼가 아닌
`building_simplified`를 직사각형도 0.95까지 분할한 뒤, 좁은 자투리를 병합하고
각 조각의 MRR을 외벽 수음점과 같은 나머지 길이 규칙으로 격자화해 배치합니다.
서로 다른 MRR의 셀이 겹치고 후보점 간 거리가 해상도의 절반보다 짧으면 지붕
유효면적이 작은 후보를 제거하되, 10m 지붕 커버리지에 공백이 생기면 복원합니다.
