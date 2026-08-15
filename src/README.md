# 전처리 파이프라인 (GPX → DEM 고도 매핑 → 구간화 → v_user 추정)

`data/raw/`의 GPX 원본에서 좌표를 뽑아 국토지리정보원 DEM으로 고도를 매핑하고,
30~50m 구간으로 잘라 회귀분석에 쓸 학습 테이블을 만드는 스크립트 모음입니다.

## 전체 흐름

```
GPX(팀원별 다수 파일) ──▶ gpx_parser.py ──▶ 좌표+시간 포인트 테이블
                                              │
                          DEM(.tif/.img/.asc) ▼
                                        dem.py (고도 샘플링)
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

### 0-2. DEM(수치표고모형) 파일 준비

GPX의 `<ele>` 값은 스마트폰 GPS 특성상 부정확해서(같은 구간 반복 측정해도 값이 들쭉날쭉) 쓰지 않습니다.
대신 **국토지리정보원 국토정보플랫폼**(https://map.ngii.go.kr)의 정밀 DEM을 좌표에 매핑해 고도를 구합니다.

1. map.ngii.go.kr 회원가입 → 로그인
2. 상단 메뉴 **"공간정보받기"** 클릭
3. 좌측 도구에서 **"영역 > 사각형"**(또는 다각형) 선택 → 지도에서 걸은 구간을 감싸는 범위를 드래그
4. 결과 목록에서 **"공개DEM"** 선택 → 연도(최신 권장)와 도엽을 선택해 다운로드

> **주의 — 도엽(지도 격자) 단위로 나뉘어 있습니다.**
> 선택한 사각형이 걸치는 도엽만 검색 결과에 나옵니다. 즉 팀원들이 서로 멀리 떨어진 지역에서
> 측정했다면(예: 한 명은 용인, 한 명은 다른 시/군), **DEM도 지역별로 각각 따로 받아야 합니다.**
> `data/raw/*.gpx`의 좌표 범위를 먼저 확인하고, 그 범위를 커버하는 도엽을 전부 받으세요.
>
> 파일 형식은 `.tif` / `.img` / `.asc` 중 하나로 받아지며, 좌표계는 보통 GRS80 중부원점(EPSG:5186) 등
> GPX의 WGS84(EPSG:4326)와 다릅니다 — `dem.py`가 자동으로 좌표 변환하니 신경 쓰지 않아도 됩니다.
> `.prj`, `.tfw` 같은 부속 파일이 같이 왔다면 DEM 파일과 **같은 폴더**에 두세요.

받은 DEM 파일들은 한 폴더(예: `data/dem/`)에 모아두면 됩니다. 이 폴더는 `.gitignore`에 등록되어
있어 git에는 올라가지 않으니, **팀원마다 각자 다운로드해서 로컬에 준비**해야 합니다.

## 1. 실행

프로젝트 루트에서:

```bash
# 1) GPX + DEM -> 구간별 테이블
python -m src.build_dataset --dem-dir data/dem

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

- **`gpx_parser.py`** — GPX 파일명(`이름_카테고리_회차.gpx`)에서 메타데이터를 파싱하고, 트랙포인트(위도·경도·시간)를 표로 만듭니다. macOS 파일명 유니코드(NFC/NFD 혼재) 이슈를 정규화해서 처리합니다.
- **`dem.py`** — 여러 DEM 도엽을 하나로 모자이크한 뒤, WGS84 좌표를 DEM 좌표계로 변환해 픽셀 고도값을 샘플링합니다(`DEMSampler`).
- **`segmentation.py`** — haversine 거리로 누적거리를 계산해 30~50m 단위로 트랙을 구간화하고, 구간별 거리·시간·속도·경사도(%)를 산출합니다.
- **`build_dataset.py`** — 위 세 모듈을 순서대로 실행하는 진입점. `segments.csv` 생성.
- **`model_speed.py`** — `segments.csv`를 읽어 `speed_ratio`(=구간속도÷개인 평지속도) 정규화 → pooled 2차 다항회귀로 `k_slope` 적합 → 잔차의 사람별 평균에 Shrinkage(`w_i = n_i/(n_i+k)`)를 적용해 `v_user`(개인별 평지 기준 보행속도)를 추정.

## 3. 출력 스키마

### `data/processed/segments.csv`

| 컬럼 | 설명 |
|---|---|
| `person`, `category`, `trial`, `source_file` | GPX 파일명에서 파싱한 메타데이터 |
| `segment_id` | 파일 내 구간 순번 |
| `dist_m`, `dt_s`, `speed_mps` | 구간 거리(m)·경과시간(s)·평균속도(m/s) |
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
