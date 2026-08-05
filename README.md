# Receiver Generator for noise map

## 기준 건물 데이터

`scripts/assemble_building_dataset.py`가
`metadata/building/building_cropped_metadata.gpkg`를 생성합니다.

- `building_metadata`: 원본 건물 형상과 속성
- `building_simplified`: 공통 단순화 건물 형상
- `building_mrr`: 소음 전파 모델의 후보군 필터링용 최소면적 회전사각형

`building_simplified`는 모폴로지 닫힘 10m와 위상 보존 단순화 1m를
적용한 형상입니다. 수음점용 1m 외곽 버퍼는 포함하지 않습니다.
`building_mrr`는 이 단순화 형상 전체를 감싸는 `NF_ID`당 하나의
사각형입니다. 후보군 누락 방지를 위해 외곽에 1mm의 수치 안전 여유를
적용하며, `MRR_PAD_M` 필드에 그 값을 기록합니다.

## 수음점 생성 순서

`scripts` 디렉터리에서 다음 순서로 실행합니다.

```powershell
..\.venv\Scripts\python.exe assemble_building_dataset.py
..\.venv\Scripts\python.exe generate_building_wall_buffers.py
..\.venv\Scripts\python.exe generate_building_wall_receivers.py
..\.venv\Scripts\python.exe generate_building_roof_receivers.py
..\.venv\Scripts\python.exe generate_terrain_receivers.py
..\.venv\Scripts\python.exe merge_receivers.py
```

벽면 수음점용 `building_buffer`는 기준 `building_simplified`에 1m 외곽
버퍼만 적용한 파생 레이어입니다. 지붕 수음점은 이 버퍼와
`building_metadata`의 원본 형상을 함께 사용합니다.
