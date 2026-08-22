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

그래서 각 횡단보도의 **자기 길이**를 기준으로 잡는다.

    임계값 = max(횡단거리 ÷ 2, 5m) + GPS 오차

횡단보도를 실제로 건넜다면 트랙이 그 중심 부근을 통과해야 하기 때문이다.

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
DEFAULT_CSV = HERE.parent / "data" / "processed" / "영통구_횡단보도별_신호대기.csv"
DEFAULT_GEOJSON = HERE.parent / "data" / "processed" / "영통구_횡단보도.geojson"

#: 스마트폰 GPS 오차 여유(m). 도심 보행 기록의 통상 수준.
GPS_ERR_M = 10.0

#: 횡단거리가 비어 있을 때 쓸 기본값(m). 영통구 중앙값.
DEFAULT_LEN_M = 15.0


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
    return np.maximum(half, 5.0) + gps_err


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
            gps_err: float = GPS_ERR_M) -> pd.DataFrame:
    """트랙이 실제로 건넌 것으로 판정되는 횡단보도.

    각 횡단보도 중심에서 트랙까지의 최단거리를 재고, 그 횡단보도
    자신의 길이로 만든 임계값과 비교한다(모듈 docstring 참조).
    """
    d = np.array([haversine(track_lat, track_lon, y, x).min()
                  for y, x in zip(cw["위도"], cw["경도"])])
    hit = d <= _threshold(cw, gps_err)
    out = cw[hit].assign(트랙거리_m=d[hit].round(1))
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
        _report(crossed(cw, lat, lon, a.gps_err), "트랙거리_m")
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
