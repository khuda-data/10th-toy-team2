"""검증 걷기 경로 ↔ 횡단보도 연결 — 신호대기 항을 ETA 에 넣기 위한 도구.

`영통구_횡단보도별_신호대기.csv` 에는 좌표와 관리번호가 다 있지만,
검증 걷기 데이터(GPX)에는 관리번호가 없다. 둘을 잇는 것이 이 모듈이다.

두 방향을 지원한다.

  1) 좌표 → 횡단보도 후보        `nearby()`      사람이 눈으로 확인
  2) GPX 트랙 → 건넌 횡단보도    `crossed()`     자동 판정

주소로는 못 잇는다 — `소재지도로명주소` 가 351개 중 147개(42%) 결측이다.
좌표가 유일하게 100% 채워진 키다.

통과 판정 규칙
──────────────
단순 반경(예: 40m)은 쓰면 안 된다. 교차로 옆을 스쳐 지나기만 해도 잡혀서
건너지도 않은 횡단보도의 대기가 붙는다. 실측 예: 274m 를 196초에 걸은
트랙(정지 2초)에 40m 반경은 신호 횡단보도 2개·110.6초를 붙였는데,
가장 가까운 것도 트랙에서 31.9m 떨어져 있었다(횡단거리 12.6m). 안 건넌 것이다.

세 가지를 쓴다.

  1) 임계값을 각 횡단보도의 **자기 길이**로 정한다
         임계값 = max(횡단거리 ÷ 2, 5m) + GPS 오차
     실제로 건넜다면 트랙이 그 중심 부근을 통과해야 하기 때문이다.

  2) 거리를 **점이 아니라 선분**으로 잰다
     구간이 40m 라 끝점만 보면 중간에 건넌 횡단보도를 놓친다.

  3) 한 교차로에서는 **가장 가까운 하나만** 센다
     교차로 중심을 지나면 사방 3~4개가 다 임계값 안에 들어온다.
     실제로 건너는 것은 보통 1개다.

⚠️ 3)은 휴리스틱이다. 대각선으로 건너면(2개 건넘) 과소계산한다.
   자동 판정은 어디까지나 보조 수단이고, 정답은 걸으면서 기록한
   스톱워치 실측이다. `--ids` 로 관리번호를 직접 넣는 쪽이 항상 정확하다.

실행
────
    python src/route_wait.py --gpx data/raw/박준서_평지_1회차.gpx
    python src/route_wait.py --at 37.25033 127.07931
    python src/route_wait.py --ids 수원-1965 수원-1951
    python src/route_wait.py --geojson          # 지도용 파일 생성
"""

from __future__ import annotations

import argparse
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

HERE = Path(__file__).resolve().parent
_PROC = HERE.parent / "data" / "processed"
#: 수원시 전체 + 신호등 보충본(signal_suwon.py)을 우선 쓴다. 없으면 영통구 구본.
_SUWON = _PROC / "수원시_횡단보도별_신호대기.csv"
DEFAULT_CSV = _SUWON if _SUWON.exists() else _PROC / "영통구_횡단보도별_신호대기.csv"
DEFAULT_GEOJSON = _PROC / "영통구_횡단보도.geojson"

#: 좌표 오차 여유(m). 이름은 GPS 지만 실제 지배 요인은 GPS 가 아니라
#: **표준데이터 좌표의 등록 오차**다. 검증 걷기에서 실제로 건넌 것이 확인된
#: 횡단보도까지 트랙 거리를 재면 3.9 / 19.3 / 22.8 / 30.5 / 44.2m 로,
#: 보도를 따라 걸은 트랙과 등록 좌표가 체계적으로 어긋나 있다.
#: 10m 로는 확정 9곳 중 2곳밖에 못 잡는다.
#: 16회차 실측 대기시간 기준 LOPO: 10m → MAE 97초, 30m → MAE 35.9초.
GPS_ERR_M = 30.0

#: 횡단거리가 비어 있을 때 쓸 기본값(m). 영통구 중앙값.
DEFAULT_LEN_M = 15.0

#: 교차로 군집 반경(m). 한 교차로의 횡단보도를 묶어 1개만 세기 위한 값.
#: 40m 는 좁다 — 큰 교차로는 마주보는 횡단보도가 54~71m 떨어져 있다.
#: 80m 는 실제로 다른 교차로인 수원-1944/1945(71m 간격)를 합쳐버린다.
#: 실측 확정 9곳 + 16회차 대기시간으로 다시 맞춘 결과 60m 가 최소 오차다.
#: (이전 80m 는 답사 추정치 "3개"에 맞춘 값이었고, 실제로 걸어보니 4개였다.)
INTERSECTION_M = 60.0

#: 임계값 상한(m). 이걸 안 걸면 넓은 도로에서 오검출이 난다.
#: 32.5m 짜리 횡단보도는 half=16.3m 라 임계값이 26m 가 되는데,
#: 그 도로 인도를 따라 걷기만 해도 중심에서 16m 라 걸려 버린다.
#: 다만 20m 는 등록 오차가 큰 지점(최대 44m)을 통째로 놓친다. 35m 가
#: 검출과 오검출의 균형점이다(16회차 LOPO 로 확인).
MAX_THRESHOLD_M = 35.0

#: 이론 대기(E[W]=R²/2C)를 실측에 맞추는 보정계수.
#: 6명 × 2경로 × 2회차 = 24회차, Σ실측 ÷ Σ이론 = 0.793.
#: 이론이 약 21% 과대추정한다. 경로별로 따로 추정해도 0.784~0.800으로
#: 거의 같고(반대 경로 λ를 교차 적용해도 MAE 0.5초 차이), 사람 1명을
#: 통째로 빼고 재추정해도(LOPO) 0.79~0.80대에서 안 움직인다 — 특정
#: 경로·사람에 과적합된 값이 아니라는 근거. 부트스트랩 95% CI = [0.709, 0.872].
#: ⚠️ 그래도 경로 2개·사람 6명(수원 영통 일대)에서 나온 값이다.
#: 다른 지역·다른 신호 체계에 그대로 쓰면 안 된다.
LAMBDA = 0.793


# ─────────────────────────────────────────────────────────────
# 기본
# ─────────────────────────────────────────────────────────────

def load(path: str | Path = DEFAULT_CSV) -> pd.DataFrame:
    """횡단보도별 신호대기 표를 읽는다."""
    return pd.read_csv(path, encoding="utf-8-sig")


def haversine(lat1, lon1, lat2, lon2) -> np.ndarray:
    """두 좌표 사이 거리(m). 배열 브로드캐스팅 가능."""
    R = 6371000.0
    p1, p2 = np.radians(lat1), np.radians(lat2)
    a = (np.sin((p2 - p1) / 2) ** 2
         + np.cos(p1) * np.cos(p2) * np.sin(np.radians(lon2 - lon1) / 2) ** 2)
    return 2 * R * np.arcsin(np.sqrt(a))


def read_gpx(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """GPX 트랙의 (위도, 경도) 배열. 네임스페이스에 의존하지 않는다."""
    root = ET.parse(path).getroot()
    pts = [(float(p.get("lat")), float(p.get("lon")))
           for p in root.iter()
           if p.tag.endswith("trkpt") or p.tag.endswith("wpt")]
    if not pts:
        raise ValueError(f"트랙 포인트가 없습니다: {path}")
    return (np.array([p[0] for p in pts]), np.array([p[1] for p in pts]))


def _threshold(cw: pd.DataFrame, gps_err: float = GPS_ERR_M) -> np.ndarray:
    """횡단보도별 통과 판정 임계값(m)."""
    half = cw["횡단거리_m"].fillna(DEFAULT_LEN_M).to_numpy() / 2.0
    return np.minimum(np.maximum(half, 5.0) + gps_err, MAX_THRESHOLD_M)


def _to_xy(lat, lon, lat0: float, lon0: float) -> tuple[np.ndarray, np.ndarray]:
    """위경도 → 기준점 중심 평면좌표(m). 수 km 범위에서 오차 무시할 수준."""
    k = 111320.0
    return ((np.asarray(lon, float) - lon0) * k * np.cos(np.radians(lat0)),
            (np.asarray(lat, float) - lat0) * k)


def dist_to_track(track_lat, track_lon, lat: float, lon: float) -> float:
    """한 점에서 트랙(선분들의 연결)까지 최단거리(m).

    점끼리만 재면 구간이 길 때 사이를 지나간 횡단보도를 놓친다.
    40m 구간의 끝점만 보면 중간의 횡단보도는 20m 밖으로 계산된다.
    """
    tx, ty = _to_xy(track_lat, track_lon, lat, lon)
    px = py = 0.0                                   # 기준점이 곧 그 횡단보도
    if len(tx) == 1:
        return float(np.hypot(tx[0] - px, ty[0] - py))

    ax, ay = tx[:-1], ty[:-1]
    bx, by = tx[1:], ty[1:]
    dx, dy = bx - ax, by - ay
    L2 = dx * dx + dy * dy
    # 선분 위 최근접점의 매개변수 t 를 [0,1] 로 자른다 (선분 밖이면 끝점)
    with np.errstate(invalid="ignore", divide="ignore"):
        t = np.where(L2 > 0, ((px - ax) * dx + (py - ay) * dy) / L2, 0.0)
    t = np.clip(t, 0.0, 1.0)
    return float(np.hypot(ax + t * dx - px, ay + t * dy - py).min())


def _cluster_intersections(cw: pd.DataFrame,
                           radius_m: float = INTERSECTION_M) -> np.ndarray:
    """횡단보도를 교차로 단위로 묶는다.

    한 교차로에 3~4개가 붙어 있는데 통과 시 보통 1개만 건넌다.
    묶어 두고 가장 가까운 하나만 채택하기 위한 전처리다.
    """
    lat = cw["위도"].to_numpy()
    lon = cw["경도"].to_numpy()
    n = len(cw)
    label = np.full(n, -1, dtype=int)
    nxt = 0
    for i in range(n):
        if label[i] >= 0:
            continue
        d = haversine(lat[i], lon[i], lat, lon)
        near = (d <= radius_m) & (label < 0)
        label[near] = nxt
        nxt += 1
    return label


# ─────────────────────────────────────────────────────────────
# 1. 좌표 → 횡단보도 후보
# ─────────────────────────────────────────────────────────────

def nearby(cw: pd.DataFrame, lat: float, lon: float,
           k: int = 5, radius_m: float = 80.0) -> pd.DataFrame:
    """한 지점 주변 횡단보도를 가까운 순으로.

    걸으면서 "여기서 건넜다"고 찍은 좌표에 관리번호를 붙일 때 쓴다.
    교차로에는 횡단보도가 3~4개 붙어 있으므로 후보를 여러 개 보여주고
    사람이 고르게 하는 것이 맞다. 자동으로 하나만 고르면 틀린다.
    """
    d = haversine(lat, lon, cw["위도"].to_numpy(), cw["경도"].to_numpy())
    out = cw.assign(거리_m=d.round(1))
    out = out[out["거리_m"] <= radius_m].nsmallest(k, "거리_m")
    return out[["횡단보도관리번호", "거리_m", "횡단거리_m", "차로수",
                "보행신호", "주기_s", "기대대기_s", "소재지도로명주소"]]


# ─────────────────────────────────────────────────────────────
# 2. GPX 트랙 → 건넌 횡단보도
# ─────────────────────────────────────────────────────────────

def crossed(cw: pd.DataFrame, track_lat: np.ndarray, track_lon: np.ndarray,
            gps_err: float = GPS_ERR_M, one_per_intersection: bool = True,
            intersection_m: float = INTERSECTION_M) -> pd.DataFrame:
    """트랙이 실제로 건넌 것으로 판정되는 횡단보도.

    각 횡단보도 중심에서 트랙(선분)까지의 최단거리를 재고, 그 횡단보도
    자신의 길이로 만든 임계값과 비교한다(모듈 docstring 참조).
    같은 교차로에서 여러 개가 걸리면 가장 가까운 하나만 남긴다.
    """
    track_lat = np.asarray(track_lat, float)
    track_lon = np.asarray(track_lon, float)
    d = np.array([dist_to_track(track_lat, track_lon, y, x)
                  for y, x in zip(cw["위도"], cw["경도"])])
    hit = d <= _threshold(cw, gps_err)
    out = cw[hit].assign(트랙거리_m=d[hit].round(1))
    if one_per_intersection and len(out) > 1:
        out = out.assign(_교차로=_cluster_intersections(out, intersection_m))
        out = (out.sort_values("트랙거리_m")
                  .drop_duplicates("_교차로", keep="first")
                  .drop(columns="_교차로"))
    return out.sort_values("트랙거리_m")


def wait_of(cw: pd.DataFrame, ids: list[str]) -> pd.DataFrame:
    """관리번호 목록 → 해당 행. 없는 번호는 경고만 하고 건너뛴다."""
    known = set(cw["횡단보도관리번호"])
    missing = [i for i in ids if i not in known]
    if missing:
        print(f"  ⚠️  표에 없는 관리번호 {len(missing)}개: {missing[:5]}")
    return cw[cw["횡단보도관리번호"].isin(ids)]


# ─────────────────────────────────────────────────────────────
# 3. 경로 합계
# ─────────────────────────────────────────────────────────────

def summarize(rows: pd.DataFrame) -> dict:
    """경로가 지나는 횡단보도들의 대기 합과 불확실성.

    기대값만 내면 ETA 가 실제보다 정확해 보인다. 신호대기는 표준편차가
    평균에 육박하므로(0초 아니면 100초) 같이 낸다.
    Var(W) = R³/(3C) − R⁴/(4C²), 횡단보도 간 독립 가정으로 합산.
    """
    sig = rows[rows["보행신호"].astype(bool)]
    R, C = sig["적색_R_s"].to_numpy(), sig["주기_s"].to_numpy()
    var = np.maximum(R ** 3 / (3 * C) - R ** 4 / (4 * C ** 2), 0.0).sum()
    return {
        "횡단보도": len(rows),
        "신호": len(sig),
        "기대대기_s": round(float(sig["기대대기_s"].sum()), 1),
        "표준편차_s": round(float(np.sqrt(var)), 1),
    }


# ─────────────────────────────────────────────────────────────
# 4. 지도용 내보내기
# ─────────────────────────────────────────────────────────────

def to_geojson(cw: pd.DataFrame, path: str | Path) -> Path:
    """횡단보도 351개를 GeoJSON 으로. GitHub 이 지도로 바로 렌더한다.

    검증 경로를 짤 때 "이 길에 횡단보도가 몇 개인가"를 눈으로 보려는 용도다.
    9절의 손익분기 밀도(2.4~5.3개/km) 위아래로 경로를 배치하려면 필요하다.
    """
    feats = []
    for _, r in cw.iterrows():
        sig = bool(r["보행신호"])
        feats.append({
            "type": "Feature",
            "geometry": {"type": "Point",
                         "coordinates": [float(r["경도"]), float(r["위도"])]},
            "properties": {
                "관리번호": r["횡단보도관리번호"],
                "기대대기_s": None if not sig else float(r["기대대기_s"]),
                "보행신호": "有" if sig else "無",
                "횡단거리_m": _opt_num(r["횡단거리_m"]),
                "차로수": _opt_num(r["차로수"]),
                "주기_s": _opt_num(r["주기_s"]),
                # marker-color 는 GitHub 지도 렌더가 읽는 속성이다
                "marker-color": "#d73027" if sig else "#4575b4",
                "marker-size": "small",
            },
        })
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    # ensure_ascii=False 로 두어야 한글 속성이 지도 팝업에 그대로 보인다
    import json
    p.write_text(json.dumps({"type": "FeatureCollection", "features": feats},
                            ensure_ascii=False, indent=1), encoding="utf-8")
    return p


def _opt_num(v):
    """NaN 을 None 으로. JSON 에 NaN 을 쓰면 표준 파서가 거부한다."""
    return None if pd.isna(v) else float(v)


# ─────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────

def _report(rows: pd.DataFrame, dist_col: str | None = None) -> None:
    if rows.empty:
        print("  건넌 것으로 판정된 횡단보도가 없습니다.")
        return
    cols = ["횡단보도관리번호"] + ([dist_col] if dist_col else []) + \
           ["횡단거리_m", "차로수", "보행신호", "주기_s", "기대대기_s"]
    print(rows[cols].to_string(index=False))
    s = summarize(rows)
    print(f"\n  횡단보도 {s['횡단보도']}개 (신호 {s['신호']}개)")
    print(f"  신호대기 합  {s['기대대기_s']:.1f}초  (± {s['표준편차_s']:.1f}초)")
    print(f"\n  ETA = Σ(구간거리 ÷ (v_user × k_slope)) + {s['기대대기_s']:.1f}초")


def main() -> int:
    ap = argparse.ArgumentParser(description="검증 경로 ↔ 횡단보도 연결")
    ap.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--gpx", type=Path, help="GPX 트랙 → 건넌 횡단보도 자동 판정")
    g.add_argument("--at", nargs=2, type=float, metavar=("위도", "경도"),
                   help="한 지점 주변 횡단보도 후보")
    g.add_argument("--ids", nargs="+", metavar="관리번호",
                   help="관리번호 목록 → 대기 합")
    g.add_argument("--geojson", nargs="?", type=Path, const=DEFAULT_GEOJSON,
                   help="지도용 GeoJSON 생성 (GitHub 에서 지도로 보임)")
    ap.add_argument("--gps-err", type=float, default=GPS_ERR_M,
                    help=f"GPS 오차 여유 m (기본 {GPS_ERR_M})")
    ap.add_argument("--intersection", type=float, default=INTERSECTION_M,
                    help=f"교차로 군집 반경 m (기본 {INTERSECTION_M})")
    ap.add_argument("--all-nearby", action="store_true",
                    help="교차로 묶기 없이 근처 전부 (진단용)")
    a = ap.parse_args()

    if not a.csv.exists():
        print(f"  ❌ 표가 없습니다: {a.csv}")
        return 1
    cw = load(a.csv)

    if a.geojson:
        p = to_geojson(cw, a.geojson)
        n_sig = int(cw["보행신호"].astype(bool).sum())
        print(f"\n  {p}  ({len(cw)}개 · 신호 {n_sig}개 빨강 / 무신호 파랑)")
        print(f"  GitHub 에서 이 파일을 열면 지도로 보입니다.")
    elif a.gpx:
        lat, lon = read_gpx(a.gpx)
        L = haversine(lat[:-1], lon[:-1], lat[1:], lon[1:]).sum()
        print(f"\n{a.gpx.name}  포인트 {len(lat)}개 · 경로 {L:.0f}m\n")
        _report(crossed(cw, lat, lon, a.gps_err,
                        one_per_intersection=not a.all_nearby,
                        intersection_m=a.intersection), "트랙거리_m")
    elif a.at:
        print(f"\n({a.at[0]}, {a.at[1]}) 주변 횡단보도\n")
        n = nearby(cw, a.at[0], a.at[1])
        print("  후보가 없습니다." if n.empty else n.to_string(index=False))
        print("\n  ※ 교차로에는 횡단보도가 3~4개 있습니다. 건넌 방향의 것을 고르세요.")
    else:
        print(f"\n관리번호 {len(a.ids)}개\n")
        _report(wait_of(cw, a.ids))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
