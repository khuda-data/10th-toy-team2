# 전처리 파이프라인 (GPX → DEM 고도 매핑 → 구간화 → v_user 추정)

`data/raw/`의 GPX 원본에서 좌표를 뽑아 국토지리정보원 등고선 기반 5m DEM으로 고도를
매핑하고, 30~50m 구간으로 잘라 회귀분석에 쓸 학습 테이블을 만드는 스크립트 모음입니다.

## 전체 흐름

```
등고선 SHP(팀원별 지역) ──▶ dem_utils.py (build_dem_from_contours) ──▶ 5m DEM(GeoTIFF)
                                                                            │
GPX(팀원별 다수 파일) ──▶ gpx_parser.py ──▶ 좌표+시간 포인트 테이블          │
                                              │                            │
                                              ▼                            │
                                    dem_utils.py (sample_dem) ◀────────────┘
                                              │
                                              ▼
                                    segmentation.py (30~50m 구간화)
                                              │
                                              ▼
                              data/processed/segments.csv  (build_dataset.py)
                                              │
                                              ▼
                                    model_speed.py (speed_ratio, k_slope 회귀, v_user)
                                              │
                                              ▼
                              data/processed/v_user.csv
```

## 0. 사전 준비

### 0-1. 라이브러리 설치

```bash
pip install -r requirements.txt
```

### 0-2. DEM(수치표고모형) 준비 — 등고선 SHP로 직접 생성 (팀 확정 방식)

GPX의 `<ele>` 값은 스마트폰 GPS 특성상 부정확해서(같은 구간 반복 측정해도 값이 들쭉날쭉) 쓰지 않습니다.
국토지리정보원의 완성된 공개DEM(90m)도 검토했지만, 30m 단위 구간화에는 격자가 너무 성겨서
경사도가 계단식으로 튀는 문제가 있었습니다(실측 비교 결과 90m 대비 5m TIN DEM이 구간 경사도
분산을 약 83% 줄임). 그래서 **팀 표준은 국토정보플랫폼 등고선 SHP를 TIN(Delaunay) 선형보간해
직접 5m DEM을 만드는 방식**으로 확정했습니다.

1. **국토정보플랫폼**(https://map.ngii.go.kr) 회원가입 → 로그인
2. 상단 메뉴 **"국토정보맵" → "통합지도검색"** 클릭
3. 검색창 위 탭에서 **"지명검색"이 아니라 "통합검색"**을 선택해야 함 (지명검색 탭에는 영역 그리기 기능이 안 보임)
4. 검색창 아래 **"간편지도 검색 >"** 버튼을 클릭해야 그 밑에 **"인덱스 | 영역 | 반경 | 행정구역 | 간편선택"** 줄이 펼쳐짐 (기본 화면엔 접혀서 안 보임)
5. **"영역"** 클릭 → 지도 위에서 마우스로 드래그해 걸은 구간을 감싸는 사각형을 그림
6. 결과 목록에서 **"수치지형도"** (1:5,000, Ver2.0, SHP 형식) 선택 → 도엽을 골라 다운로드 후 압축 해제

> **주의 — 도엽(지도 격자) 단위로 나뉘어 있습니다.**
> 선택한 사각형이 걸치는 도엽만 검색 결과에 나옵니다. 즉 팀원들이 서로 멀리 떨어진 지역에서
> 측정했다면(예: 한 명은 용인, 한 명은 다른 시/군), **각자 자기 지역 등고선을 따로 받아야 합니다.**
> `data/raw/*.gpx`의 좌표 범위를 먼저 확인하고, 그 범위를 커버하는 도엽을 받으세요.

7. 압축을 풀면 SHP 레이어가 여러 개 나오는데(`N3A_*`=면, `N3L_*`=선, `N3P_*`=점), 그중
   **등고선 레이어는 `N3L_F0010000.shp`**입니다. 이 파일명과 고도 필드명(`등고수치`)은
   국토지리정보원의 전국 공통 표준 코드라 어느 지역을 받아도 동일합니다.
8. 아래 명령으로 자기 지역 5m DEM을 바로 생성합니다(프로젝트 루트에서):

```bash
python -m src.dem_utils \
  --contour-shp "<압축 푼 폴더>/N3L_F0010000.shp" \
  --out data/raw/dem/dem_5m_<자기지역코드>.tif
```

> DEM 결과 GeoTIFF는 팀원마다 지역이 달라 용량 문제도 있어 `.gitignore`(`data/raw/dem/`)에
> 등록되어 git에는 올라가지 않습니다. **팀원마다 각자 생성해서 로컬에 준비**해야 합니다.
> 원본 등고선 SHP(수십 MB)도 커밋하지 않고 로컬에만 둡니다.

## 1. 실행

프로젝트 루트에서:

```bash
# 1) GPX + DEM -> 구간별 테이블 (data/raw/dem/에 자기 지역 DEM을 넣어둔 상태)
python -m src.build_dataset --dem-dir data/raw/dem

# 2) speed_ratio / k_slope 회귀 / v_user(Shrinkage) 추정
python -m src.model_speed
```

`build_dataset.py` 옵션:

| 옵션 | 기본값 | 설명 |
|---|---|---|
| `--raw-dir` | `data/raw` | GPX 폴더 |
| `--dem-dir` | (필수) | DEM(.tif/.tiff/.img/.asc) 파일 폴더 |
| `--out` | `data/processed/segments.csv` | 출력 경로 |
| `--target-m` | `40.0` | 목표 구간 길이(m), 30~50m 권장 |

`model_speed.py` 옵션:

| 옵션 | 기본값 | 설명 |
|---|---|---|
| `--segments` | `data/processed/segments.csv` | build_dataset.py 출력 |
| `--out` | `data/processed/v_user.csv` | 출력 경로 |
| `--shrink-k` | 사람별 구간수 중앙값 | Shrinkage 강도(작을수록 개인차 더 반영) |

## 2. 모듈별 역할

- **`gpx_parser.py`** — GPX 파일명(`이름_카테고리_회차[회차].gpx`, "회차" 접미사는 선택)에서 메타데이터를 파싱하고, 트랙포인트(위도·경도·시간)를 표로 만듭니다. macOS 파일명 유니코드(NFC/NFD 혼재) 이슈를 정규화해서 처리합니다.
- **`dem_utils.py`** — 국토정보플랫폼 등고선 SHP에서 TIN(Delaunay) 선형보간으로 5m DEM(GeoTIFF)을 만들고(`build_dem_from_contours`, CLI: `python -m src.dem_utils`), WGS84 좌표를 DEM 좌표계로 변환해 픽셀 고도값을 샘플링합니다(`sample_dem`). 등고선 밀도가 낮은 구간의 보간 이상치를 찾는 `detect_sparse_artifacts`도 포함. 팀의 확정된 DEM 표준 방식입니다.
- **`segmentation.py`** — haversine 거리로 누적거리를 계산해 30~50m 단위로 트랙을 구간화하고, 구간별 거리·시간·속도·경사도(%)를 산출합니다. GPX 타임스탬프 간격이 불규칙해 생기는 이상치도 여기서 처리합니다: `remove_gps_jumps`가 순간속도 15km/h를 넘는 포인트(GPS 튐)를 원본 포인트 단계에서 제거하고, `build_segments`는 3초 이상 정지구간의 시간을 뺀 `moving_dt_s`(이동시간)로 `speed_mps`를 계산합니다.
- **`build_dataset.py`** — 위 세 모듈을 순서대로 실행하는 진입점. `segments.csv` 생성.
- **`model_speed.py`** — `segments.csv`를 읽어 `speed_ratio`(=구간속도÷개인 평지속도) 정규화 → pooled 2차 다항회귀로 `k_slope` 적합 → 잔차의 사람별 평균에 Shrinkage(`w_i = n_i/(n_i+k)`)를 적용해 `v_user`(개인별 평지 기준 보행속도)를 추정.

## 3. 출력 스키마

### `data/processed/segments.csv`

| 컬럼 | 설명 |
|---|---|
| `person`, `category`, `trial`, `source_file` | GPX 파일명에서 파싱한 메타데이터 |
| `segment_id` | 파일 내 구간 순번 |
| `dist_m`, `dt_s`, `moving_dt_s`, `speed_mps` | 구간 거리(m)·실제 경과시간(s)·정지구간 뺀 이동시간(s)·평균속도(m/s, `dist_m/moving_dt_s`) |
| `elev_start_m`, `elev_end_m`, `slope_pct` | 구간 시작/끝 DEM 고도, 경사도(%) |
| `start_lat/lon`, `end_lat/lon`, `start_time`, `end_time` | 구간 경계 좌표·시각 |

### `data/processed/v_user.csv`

| 컬럼 | 설명 |
|---|---|
| `person` | 이름 |
| `residual_mean`, `n_segments` | k_slope 회귀 잔차의 사람별 평균, 구간 표본수 |
| `shrink_k`, `shrink_w`, `shrunk_residual` | Shrinkage 파라미터 및 보정된 잔차 |
| `v_flat_raw` | 실측 평지속도(정규화 분모) |
| `v_user_mps` | 최종 개인별 평지 기준 보행속도(m/s) |

## 4. 설계 결정 (검토·조정 환영)

`v_user` 산출 방식은 README 기획서가 정확한 수식까지 규정하진 않아서 아래처럼 확정했습니다.
다른 방식이 더 낫다고 판단되면 `model_speed.py`의 `fit_k_slope` / `estimate_v_user` 함수만
수정하면 됩니다(다른 모듈과 독립적).

- "개인 평지속도"(정규화 분모)는 `category=='평지'` 구간들의 실측 속도 평균을 사용
- `k_slope` 회귀는 전원 데이터를 합친(pooled) `speed_ratio ~ slope_pct + slope_pct²` 2차 다항회귀
- `v_user`는 그 회귀의 잔차를 사람별로 평균 낸 값에 Shrinkage를 적용해 "전체 평균 ↔ 개인 고유값" 사이로 보간한 뒤, 평지속도(m/s) 스케일로 환산
