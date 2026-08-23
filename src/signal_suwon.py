"""수원시 전체 신호대기 테이블 — 영통구 파이프라인을 넓히고 결측을 신호등으로 메운다.

`signal_yeongtong.py` 는 영통구 351개만 다뤘다. 검증 걷기에서 두 가지가 드러났다.

  1) 확정 9곳 중 3곳이 횡단보도 표준데이터에 아예 없다.
     그런데 그 자리에 **신호등 행은 있다.** 주기만 있으면 E[W] 는 계산된다.
       A-1·A-2 → 신호등 599 (수원시 영통구 덕영대로 1689) C=180초
       B-1     → 신호등 1336 (영통동 995-5) C=140초
     → 횡단보도 행이 없는 교차로에 **가상 횡단보도**를 세운다.

  2) 영통구로 자른 것이 경계에서 손해다. 경로가 구 경계를 넘으면 그만큼 빈다.
     → 시군구 필터를 '수원시'로 넓힌다. 351개 → 2,064개.

⚠️ 용인시(경희대 국제캠퍼스 정문 방향)는 이 방법으로도 못 메운다.
   행안부 표준데이터에 용인시 행이 전국을 통틀어 0건이다(경기도 31개 시군 중
   15개가 미제출). 덕영대로는 1689 번지까지만 데이터가 있다.

실행:
    python -m src.signal_suwon
    python -m src.signal_suwon --crosswalk <전국횡단보도.csv> --signal <전국신호등.csv>
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

try:
    from . import signal_yeongtong as sy
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import signal_yeongtong as sy

HERE = Path(__file__).resolve().parent
PROJ = HERE.parent

#: 이 반경 안에 횡단보도 행이 하나도 없는 교차로만 가상 횡단보도로 세운다.
#: 60m — signal_yeongtong 의 횡단보도↔교차로 결합 중앙거리(31m)의 2배.
#: 이보다 좁히면 이미 매칭된 교차로에 유령이 하나 더 생긴다.
ORPHAN_M = 60.0

OUT_CSV = PROJ / "data" / "processed" / "수원시_횡단보도별_신호대기.csv"

KEEP = ["횡단보도관리번호", "소재지도로명주소", "위도", "경도", "차로수", "횡단거리_m",
        "보행신호", "주기_s", "주기출처", "기대대기_s", "출처"]


def _hav(lat1, lon1, lat2, lon2) -> np.ndarray:
    lat1, lon1, lat2, lon2 = (np.radians(np.asarray(v, dtype=float))
                              for v in (lat1, lon1, lat2, lon2))
    a = (np.sin((lat2 - lat1) / 2) ** 2
         + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2)
    return 2 * 6371000.0 * np.arcsin(np.sqrt(a))


def virtual_crossings(inter: pd.DataFrame, cw: pd.DataFrame,
                      orphan_m: float = ORPHAN_M) -> pd.DataFrame:
    """횡단보도 행이 없는 교차로 → 가상 횡단보도.

    횡단거리를 모르므로 신호 횡단보도 중앙값을 쓴다. 짧은 횡단보도에서
    과대추정하는 방향이다(A-2 실측 24.5초 vs 이론 66.8초).
    """
    clat, clon = cw["위도"].to_numpy(float), cw["경도"].to_numpy(float)
    d = np.array([_hav(clat, clon, la, lo).min()
                  for la, lo in zip(inter["위도"], inter["경도"])])
    orph = inter[d > orphan_m].copy()

    length = float(cw.loc[cw["보행신호"], "횡단거리_m"].median())
    C = orph["주기_s"].to_numpy(float)
    # apply_formula 와 같은 식을 쓴다 — 녹색 상한(주기의 MAX_GREEN_FRAC)까지 동일
    green = np.minimum(sy.sw.green_time(length), C * sy.sw.MAX_GREEN_FRAC)
    wait = np.array([sy.sw.wait_mean(c - g, c) for c, g in zip(C, green)])
    return pd.DataFrame({
        "횡단보도관리번호": ["가상-" + str(i) for i in orph.index],
        "소재지도로명주소": None,
        "위도": orph["위도"].to_numpy(),
        "경도": orph["경도"].to_numpy(),
        "차로수": np.nan,
        "횡단거리_m": length,
        "보행신호": True,
        "주기_s": C,
        "주기출처": "신호등보충",
        "기대대기_s": np.round(wait, 1),
        "출처": "신호등보충",
    })


def build(cw_path: Path, sg_path: Path, sgg: str = "수원") -> pd.DataFrame:
    sy.SGG_KEY = sgg
    cw = sy.load_region(cw_path)
    sg = sy.load_region(sg_path)

    _, inter = sy.signal_cycles(sg)
    d = sy.apply_formula(sy.fill_missing(sy.join_cycles(cw, inter, sy.MATCH_M))[0])
    # 원본 문자열 컬럼을 버리고 정제본만 남긴다 — 이름이 겹친다
    d = d.drop(columns=[c for c in ("차로수", "횡단보도연장") if c in d])
    d = d.rename(columns={"length_m": "횡단거리_m", "lanes": "차로수", "has_signal": "보행신호"})
    d["출처"] = "횡단보도행"

    virt = virtual_crossings(inter, d)
    out = pd.concat([d[KEEP], virt[KEEP]], ignore_index=True)
    print(f"\n  횡단보도행 {len(d)} + 신호등보충 {len(virt)} = {len(out)}개")
    print(f"  보행신호 있고 기대대기>0: {int((out['보행신호'] & (out['기대대기_s'] > 0)).sum())}개")
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="수원시 신호대기 테이블 (신호등 보충 포함)")
    ap.add_argument("--crosswalk", type=Path, default=sy.DEFAULT_CW)
    ap.add_argument("--signal", type=Path, default=sy.DEFAULT_SG)
    ap.add_argument("--sgg", default="수원")
    ap.add_argument("--out", type=Path, default=OUT_CSV)
    a = ap.parse_args()

    out = build(a.crosswalk, a.signal, a.sgg)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(a.out, index=False, encoding="utf-8-sig")
    print(f"  -> {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
