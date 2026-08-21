"""
신호대기 계수 — 수학식 전용 (D5: 공공데이터 미사용 확정)

현장 관측도, 공공데이터 수집도 하지 않는다. 지도에서 잴 수 있는 두 값
(횡단거리, 차로수)만으로 기대 대기시간을 산출한다.

    이론 근거
    ─────────
    ① 보행 녹색시간      경찰청 「교통신호기 설치·관리 매뉴얼」
                        G = 진입시간 7초 + 횡단거리 ÷ 1.0 m/s
    ② 주기              도로 위계별 표준값 (차로수로 대리)
    ③ 기대 대기시간      균등도착 가정 하에
                        E[W] = R²/(2C),  R = C − G

    사용
    ────
    python signal_wait.py --template              # 입력 양식 생성
    python signal_wait.py --crossings <csv>       # 산출
    python signal_wait.py --demo                  # 예시로 동작 확인

    B·C 에서:
        from signal_wait import route_wait
        mean_s, sd_s = route_wait(df_crossings, lam=1.0)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

HERE = Path(__file__).resolve().parent
DEFAULT_OUT = HERE / "out"

# ─────────────────────────────────────────────────────────────
# 상수 — 전부 근거를 명시한다. 임의값이 하나도 없어야 방어된다.
# ─────────────────────────────────────────────────────────────

#: 보행 진입시간(초). 경찰청 매뉴얼 표준값.
ENTRY_TIME_S = 7.0

#: 설계 보행속도(m/s). 도로교통공단 보행신호 산정 기준.
#: 우리 모델의 v_user(1.29~1.63)보다 느린 값인데, 고령자·아동을 포함한
#: 설계 기준이라 그렇다. 신호 길이를 정하는 값이지 우리가 걷는 속도가 아니다.
DESIGN_WALK_MS = 1.0

#: 도로 위계별 주기 표준값(초). 차로수를 위계 대리변수로 쓴다.
#: ⚠️ 이 표가 이 모듈에서 가장 약한 가정이다. 반드시 민감도 분석을 함께 보고할 것.
CYCLE_BY_LANES = {
    "간선 (6차로 이상)": (6, 150.0),
    "보조간선 (4~5차로)": (4, 120.0),
    "집산·이면 (3차로 이하)": (0, 90.0),
}

#: 민감도 분석에서 훑을 주기 범위(초). 도시부 신호주기의 통상 범위.
CYCLE_GRID = (60.0, 90.0, 120.0, 150.0, 180.0)

#: 보행 녹색이 주기에서 차지할 수 있는 현실적 상한.
#: 계산상 G > C 가 되는 것을 막는다(짧은 주기 + 긴 횡단보도).
MAX_GREEN_FRAC = 0.60


# ─────────────────────────────────────────────────────────────
# 1. 신호 파라미터
# ─────────────────────────────────────────────────────────────

#: 표준값을 실측치로 바꿨을 때 그 출처를 남긴다(리포트에 찍기 위함).
CYCLE_SOURCE = "도로 위계별 표준값 (가정)"


def load_region_params(path: str | Path) -> dict:
    """crosswalk_data.py 가 만든 JSON 으로 주기 표준값을 실측 중앙값으로 교체.

    행안부 전국횡단보도표준데이터에서 뽑은 값이라, 이걸 부르고 나면
    주기는 더 이상 가정이 아니다. 부르지 않으면 기존 표준값 그대로다.
    """
    global CYCLE_BY_LANES, CYCLE_SOURCE
    obj = json.loads(Path(path).read_text(encoding="utf-8"))
    tbl = obj.get("cycle_by_lane_bin") or {}
    key = {"6차로 이상": "간선 (6차로 이상)",
           "4~5차로": "보조간선 (4~5차로)",
           "3차로 이하": "집산·이면 (3차로 이하)"}
    updated = {}
    for bin_name, cycle in tbl.items():
        k = key.get(bin_name)
        if k and k in CYCLE_BY_LANES and cycle:
            min_lanes = CYCLE_BY_LANES[k][0]
            CYCLE_BY_LANES[k] = (min_lanes, float(cycle))
            updated[k] = float(cycle)
    if updated:
        CYCLE_SOURCE = f"{obj.get('region','?')} 실측 중앙값 ({obj.get('source','')})"
    return updated


def cycle_for_lanes(lanes: int) -> float:
    """차로수 → 주기 표준값(초)."""
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

    CSV의 빈 셀은 NaN 으로 읽히는데 `if nan` 은 참이라 그냥 두면
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
    """(G, R, C) 를 초 단위로 반환.

    cycle_s 를 주면 그 값을 쓰고, 없으면 차로수 표준값을 쓴다.
    G 가 주기의 60%를 넘으면 상한으로 자른다 — 그런 신호는 현실에 없다.
    """
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
#   E[W²]  = R³/(3C)
#   Var(W) = R³/(3C) − R⁴/(4C²)
#
# 분산까지 내는 이유: 신호 대기는 평균보다 표준편차가 크다.
# 점추정만 내면 ETA가 실제보다 훨씬 정확한 것처럼 보인다.

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

    crossings 필수 컬럼: length_m, lanes, has_signal
    선택 컬럼: cycle_s (알고 있으면 표준값 대신 사용)

    lam 은 검증 걷기 후 추정하는 보정계수. 기본 1.0.
    횡단보도 간 대기는 독립으로 가정하고 분산을 더한다.
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


def annotate(crossings: pd.DataFrame) -> pd.DataFrame:
    """횡단보도별 G/R/C/기대대기/SD 를 붙인 표를 돌려준다."""
    rows = []
    for _, r in crossings.iterrows():
        sig = bool(r.get("has_signal", True))
        if sig:
            G, R, C = crossing_params(r["length_m"], int(r.get("lanes", 2)),
                                      r.get("cycle_s"))
            m, sd = wait_mean(R, C), float(np.sqrt(wait_var(R, C)))
        else:
            G = R = C = m = sd = 0.0
        rows.append({**r.to_dict(), "G_s": round(G, 1), "R_s": round(R, 1),
                     "C_s": round(C, 1), "wait_mean_s": round(m, 1),
                     "wait_sd_s": round(sd, 1),
                     "source": "규정식" if sig else "무신호"})
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────
# 3. 민감도 — 주기 가정이 결과를 얼마나 흔드는가
# ─────────────────────────────────────────────────────────────

def sensitivity(crossings: pd.DataFrame, lam: float = 1.0) -> pd.DataFrame:
    """주기 C 를 60~180초로 바꿔가며 경로 대기 합을 다시 계산한다.

    수학식만 쓰기로 한 이상 C 는 검증 불가능한 가정이다.
    이 표를 함께 내는 것이 그 약점을 정직하게 다루는 방법이다.
    """
    rows = []
    for C in CYCLE_GRID:
        sub = crossings.assign(cycle_s=C)
        m, sd = route_wait(sub, lam)
        rows.append({"cycle_s": C, "wait_mean_s": round(m, 1),
                     "wait_sd_s": round(sd, 1)})
    out = pd.DataFrame(rows)
    base = route_wait(crossings, lam)[0]
    out["기준대비"] = (out.wait_mean_s / base - 1).round(3) if base else np.nan
    return out


# ─────────────────────────────────────────────────────────────
# 4. 입력 양식
# ─────────────────────────────────────────────────────────────

TEMPLATE_COLS = ["route_id", "crossing_id", "lat", "lon",
                 "length_m", "lanes", "has_signal", "cycle_s", "note"]

TEMPLATE_HELP = """\
# 검증경로 횡단보도 입력 양식
#
# 지도(네이버·카카오)의 거리재기 도구로 5분이면 채울 수 있습니다.
#
#   route_id     경로 번호 (R1, R2, ...)
#   crossing_id  경로 내 순번 (1, 2, ...)
#   lat, lon     위경도 (선택. 나중에 찾기 쉬우라고)
#   length_m     횡단거리 (m) — 지도에서 횡단보도 양 끝을 재세요 ⭐
#   lanes        차로수 — 위성사진에서 세세요 ⭐
#   has_signal   신호등 유무 (TRUE / FALSE)
#   cycle_s      주기를 아는 경우에만. 비워두면 차로수 표준값 사용
#   note         비고 (지하보도, 육교 등)
#
# ⭐ 두 칸만 채우면 나머지는 자동 계산됩니다.
"""


def write_template(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    demo = pd.DataFrame([
        dict(route_id="R1", crossing_id=1, lat="", lon="", length_m=22, lanes=4,
             has_signal=True, cycle_s="", note="예시 — 지우고 채우세요"),
        dict(route_id="R1", crossing_id=2, lat="", lon="", length_m=9, lanes=2,
             has_signal=True, cycle_s="", note=""),
        dict(route_id="R2", crossing_id=1, lat="", lon="", length_m=14, lanes=3,
             has_signal=False, cycle_s="", note="무신호 횡단보도"),
    ], columns=TEMPLATE_COLS)
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        f.write(TEMPLATE_HELP)
        demo.to_csv(f, index=False)
    return path


def demo_frame() -> pd.DataFrame:
    """--demo 용. 전형적인 캠퍼스 주변 경로 2개."""
    return pd.DataFrame([
        dict(route_id="R1", crossing_id=1, length_m=22, lanes=4, has_signal=True),
        dict(route_id="R1", crossing_id=2, length_m=9, lanes=2, has_signal=True),
        dict(route_id="R1", crossing_id=3, length_m=30, lanes=6, has_signal=True),
        dict(route_id="R2", crossing_id=1, length_m=14, lanes=3, has_signal=False),
        dict(route_id="R2", crossing_id=2, length_m=18, lanes=4, has_signal=True),
    ])


# ─────────────────────────────────────────────────────────────
# 5. 실행
# ─────────────────────────────────────────────────────────────

def main() -> int:
    ap = argparse.ArgumentParser(description="신호대기 — 수학식 전용")
    ap.add_argument("--crossings", help="횡단보도 CSV")
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--lam", type=float, default=1.0, help="보정계수 (검증 후 갱신)")
    ap.add_argument("--template", action="store_true", help="입력 양식만 생성")
    ap.add_argument("--demo", action="store_true", help="예시로 동작 확인")
    a = ap.parse_args()

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    W = "=" * 74

    if a.template:
        p = write_template(out / "crossings_template.csv")
        print(f"입력 양식 생성: {p}")
        print(TEMPLATE_HELP)
        return 0

    if a.demo:
        df = demo_frame()
        print(f"{W}\n(--demo) 예시 데이터로 실행합니다. 실제 값이 아닙니다.\n{W}")
    elif a.crossings:
        df = pd.read_csv(a.crossings, encoding="utf-8-sig", comment="#")
    else:
        print("--crossings <csv> 또는 --demo 또는 --template 중 하나가 필요합니다.")
        print(f"\n먼저 양식부터:  python {Path(__file__).name} --template")
        return 1

    df["has_signal"] = df["has_signal"].astype(str).str.upper().isin(
        ["TRUE", "1", "Y", "YES", "T"])

    print(f"\n{W}\n[1] 파라미터 산출   G = {ENTRY_TIME_S:.0f}초 + 횡단거리 ÷ "
          f"{DESIGN_WALK_MS:.1f} m/s\n{W}")
    ann = annotate(df)
    cols = ["route_id", "crossing_id", "length_m", "lanes", "has_signal",
            "G_s", "R_s", "C_s", "wait_mean_s", "wait_sd_s", "source"]
    print(ann[cols].to_string(index=False))
    ann.to_csv(out / "crossings_annotated.csv", index=False, encoding="utf-8-sig")

    print(f"\n{W}\n[2] 경로별 대기시간   (보정계수 λ = {a.lam})\n{W}")
    rows = []
    for rid, g in df.groupby("route_id"):
        m, sd = route_wait(g, a.lam)
        n_sig = int(g["has_signal"].sum())
        rows.append({"route_id": rid, "n_crossings": len(g), "n_signal": n_sig,
                     "wait_mean_s": round(m, 1), "wait_sd_s": round(sd, 1),
                     "wait_mean_min": round(m / 60, 2)})
    routes = pd.DataFrame(rows)
    print(routes.to_string(index=False))
    routes.to_csv(out / "route_wait.csv", index=False, encoding="utf-8-sig")

    print(f"\n{W}\n[3] 민감도 — 주기 가정이 결과를 얼마나 흔드는가\n{W}")
    print("  수학식만 쓰기로 한 이상 주기 C 는 검증 불가능한 가정입니다.")
    print("  이 표를 발표에 함께 넣으면 그 약점을 정직하게 다룰 수 있습니다.\n")
    for rid, g in df.groupby("route_id"):
        s = sensitivity(g, a.lam)
        print(f"  [{rid}]")
        for _, r in s.iterrows():
            bar = "█" * int(r.wait_mean_s / 4)
            print(f"    C={r.cycle_s:>5.0f}초   {r.wait_mean_s:>6.1f}초 "
                  f"({r['기준대비']:+.0%})  {bar}")
        print()
    pd.concat([sensitivity(g, a.lam).assign(route_id=rid)
               for rid, g in df.groupby("route_id")]).to_csv(
        out / "wait_sensitivity.csv", index=False, encoding="utf-8-sig")

    print(f"{W}\n[4] 해석 시 주의\n{W}")
    tot = routes.wait_mean_s.sum()
    if tot:
        print(f"  · 대기의 표준편차가 평균과 비슷하거나 큽니다.")
        print(f"    → ETA를 점추정이 아니라 구간으로 제시하는 편이 정직합니다.")
    print(f"  · 균등도착 가정은 실제와 다릅니다. 보행자는 멀리서 신호를 보고")
    print(f"    속도를 조절하므로 실측이 이론보다 짧게 나오는 경향이 있습니다.")
    print(f"    → 검증 걷기에서 실측 대기를 기록해 λ = Σ실측 / Σ이론 을 추정하세요.")
    print(f"  · 주기 C 는 차로수 기반 표준값입니다. [3]의 민감도 폭이 이 가정의 비용입니다.")

    print(f"\n{W}\n[5] 산출물\n{W}")
    for f in ["crossings_annotated.csv", "route_wait.csv", "wait_sensitivity.csv"]:
        print(f"  {out / f}")
    print(f"\n  C 에서 쓰는 법:")
    print(f"    from signal_wait import route_wait")
    print(f"    wait_s, wait_sd = route_wait(df_crossings, lam=1.0)")
    print(f"    eta = t_walk + wait_s")
    print(W)
    return 0


if __name__ == "__main__":
    sys.exit(main())
