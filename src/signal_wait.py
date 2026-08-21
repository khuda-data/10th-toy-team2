"""신호대기 수식 코어 — 기획안 B절.

횡단보도 하나의 (녹색, 적색, 주기)로부터 기대 대기시간을 내는 순수 함수 모음.
데이터를 읽거나 쓰지 않는다. 실제 파이프라인은 `signal_yeongtong.py` 다.

    G = 진입 7초 + 횡단거리 ÷ 1.0 m/s     경찰청 「교통신호기 설치·관리 매뉴얼」
    R = C − G
    E[W] = R² / (2C)                      균등도착 가정 (기획안 B절)

주기 C 의 출처
──────────────
기본값은 차로수 기반 표준값(가정)이다. `load_region_params()` 로
`signal_yeongtong.py` 가 만든 지역 파라미터를 읽으면 실측 중앙값으로 바뀐다.
영통구는 신호등 표준데이터에서 실측 주기를 확보했으므로 그 경로를 쓴다.

    import signal_wait as sw
    sw.load_region_params('models/yeongtong_signal_params.json')
    sw.CYCLE_SOURCE      # '경기도 수원시 영통구 실측 중앙값 (...)'

경로 단위로 쓸 때
─────────────────
    wait_s, wait_sd = sw.route_wait(df_crossings, lam=1.0)

단, 영통구 경로라면 `data/processed/영통구_횡단보도별_신호대기.csv` 에서
해당 횡단보도의 `기대대기_s` 를 직접 합산하는 쪽이 정확하다.
route_wait 은 그 표에 없는 횡단보도를 위한 대비책이다.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

# ─────────────────────────────────────────────────────────────
# 상수 — 전부 근거를 명시한다. 임의값이 하나도 없어야 방어된다.
# ─────────────────────────────────────────────────────────────

#: 보행 진입시간(초). 경찰청 매뉴얼 표준값.
ENTRY_TIME_S = 7.0

#: 설계 보행속도(m/s). 도로교통공단 보행신호 산정 기준.
#: 우리 모델의 v_user(1.29~1.63)보다 느린데, 고령자·아동을 포함한 설계
#: 기준이라 그렇다. 신호 길이를 정하는 값이지 우리가 걷는 속도가 아니다.
DESIGN_WALK_MS = 1.0

#: 보행 녹색이 주기에서 차지할 수 있는 현실적 상한.
#: 짧은 주기 + 긴 횡단보도에서 G > C 가 되는 것을 막는다.
MAX_GREEN_FRAC = 0.60

#: 차로수 → 주기 기본값(초). 도로 위계의 대리변수로 차로수를 쓴다.
#: ⚠️ 가정이다. load_region_params() 로 실측치로 교체하는 것이 원칙.
CYCLE_BY_LANES = {
    "간선 (6차로 이상)": (6, 150.0),
    "보조간선 (4~5차로)": (4, 120.0),
    "집산·이면 (3차로 이하)": (0, 90.0),
}

#: 민감도 분석에서 훑을 주기 범위(초). 도시부 신호주기의 통상 범위.
CYCLE_GRID = (60.0, 90.0, 120.0, 150.0, 180.0)

#: 지금 CYCLE_BY_LANES 가 가정인지 실측인지. 리포트에 출처로 찍는다.
CYCLE_SOURCE = "도로 위계별 표준값 (가정)"


# ─────────────────────────────────────────────────────────────
# 1. 신호 파라미터
# ─────────────────────────────────────────────────────────────

def load_region_params(path: str | Path) -> dict:
    """지역 파라미터 JSON 으로 주기 기본값을 실측 중앙값으로 교체한다.

    signal_yeongtong.py 가 만든 `yeongtong_signal_params.json` 을 읽는다.
    부르지 않으면 CYCLE_BY_LANES 의 가정값이 그대로 쓰인다.
    교체된 항목만 담은 딕셔너리를 돌려준다.
    """
    global CYCLE_SOURCE
    obj = json.loads(Path(path).read_text(encoding="utf-8"))
    key = {"6차로 이상": "간선 (6차로 이상)",
           "4~5차로": "보조간선 (4~5차로)",
           "3차로 이하": "집산·이면 (3차로 이하)"}
    updated = {}
    for bin_name, cycle in (obj.get("cycle_by_lane_bin") or {}).items():
        k = key.get(bin_name)
        if k and k in CYCLE_BY_LANES and cycle:
            CYCLE_BY_LANES[k] = (CYCLE_BY_LANES[k][0], float(cycle))
            updated[k] = float(cycle)
    if updated:
        CYCLE_SOURCE = f"{obj.get('region', '?')} 실측 중앙값 ({obj.get('source', '')})"
    return updated


def cycle_for_lanes(lanes: int) -> float:
    """차로수 → 주기(초)."""
    for _, (min_lanes, cycle) in sorted(
            CYCLE_BY_LANES.items(), key=lambda kv: -kv[1][0]):
        if lanes >= min_lanes:
            return cycle
    return 90.0


def green_time(length_m: float) -> float:
    """보행 녹색시간(초) = 진입 7초 + 횡단거리 ÷ 1.0 m/s."""
    return ENTRY_TIME_S + float(length_m) / DESIGN_WALK_MS


def _opt_cycle(v) -> float | None:
    """빈칸·NaN·0 을 전부 '미지정'으로 정규화한다.

    CSV 빈 셀은 NaN 으로 읽히는데 `if nan` 은 참이라, 그냥 두면
    주기가 NaN 으로 전파되어 계산이 통째로 깨진다.
    """
    if v is None or not pd.notna(v):
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if f > 0 else None


def crossing_params(length_m: float, lanes: int,
                    cycle_s: float | None = None) -> tuple[float, float, float]:
    """(G, R, C) 를 초 단위로 반환. cycle_s 를 주면 그 값을, 없으면 차로수 기본값."""
    c = _opt_cycle(cycle_s)
    C = c if c is not None else cycle_for_lanes(lanes)
    G = min(green_time(length_m), C * MAX_GREEN_FRAC)
    return G, C - G, C


# ─────────────────────────────────────────────────────────────
# 2. 대기시간 분포
# ─────────────────────────────────────────────────────────────
#
# 균등도착 가정: 보행자가 주기 내 임의 시점에 도착한다.
#   녹색에 도착(확률 G/C) → 대기 0
#   적색에 도착(확률 R/C) → 대기 ~ Uniform(0, R)
#
#   E[W]   = R²/(2C)
#   Var(W) = R³/(3C) − R⁴/(4C²)
#
# 분산까지 내는 이유: 신호 대기는 표준편차가 평균에 육박한다.
# 점추정만 내면 ETA 가 실제보다 훨씬 정확한 것처럼 보인다.

def wait_mean(R: float, C: float) -> float:
    """기대 대기시간(초)."""
    return R ** 2 / (2.0 * C) if C > 0 else 0.0


def wait_var(R: float, C: float) -> float:
    """대기시간 분산(초²)."""
    if C <= 0:
        return 0.0
    return max(R ** 3 / (3.0 * C) - R ** 4 / (4.0 * C ** 2), 0.0)


def route_wait(crossings: pd.DataFrame, lam: float = 1.0) -> tuple[float, float]:
    """경로 전체의 (기대 대기 합, 표준편차) 초.

    필수 컬럼: length_m, lanes, has_signal      선택: cycle_s
    lam 은 검증 걷기 후 추정하는 보정계수(Σ실측/Σ이론). 기본 1.0.
    횡단보도 간 대기는 독립으로 보고 분산을 더한다.
    """
    if crossings is None or len(crossings) == 0:
        return 0.0, 0.0

    tot_mean, tot_var = 0.0, 0.0
    for _, r in crossings.iterrows():
        if not bool(r.get("has_signal", True)):
            continue                       # 무신호 횡단보도는 대기 0
        _, R, C = crossing_params(r["length_m"], int(r.get("lanes", 2)),
                                  r.get("cycle_s"))
        tot_mean += wait_mean(R, C)
        tot_var += wait_var(R, C)
    return lam * tot_mean, lam * float(np.sqrt(tot_var))
