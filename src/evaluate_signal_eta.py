"""신호대기 모델 평가 — 자동 매핑이 실제로 되는가, 최종 ETA 에 얼마나 기여하는가.

`evaluate_validation.py`(권동하)와 **평가하는 대상이 다르다.**

  evaluate_validation.py : M3/M4 의 신호 항에 **스톱워치 실측을 그대로 더한다.**
                           `M3_신호 − M2_개인화 == 신호대기_실측` 이 16/16 성립한다.
                           신호를 완벽히 알 때의 상한(오라클)을 보는 설계다.
  이 스크립트           : 신호 항을 **GPX 좌표 → 표준데이터 매칭 → E[W]** 로
                           예측한다. 정답을 안 쓴다.

둘 다 필요하다. 전자는 "신호를 완벽히 알면 얼마나 되나", 후자는 "공개데이터만
으로 서비스하면 얼마나 되나"를 답한다. 발표에는 후자를 써야 한다.

실행:
    python -m src.evaluate_signal_eta
    python -m src.evaluate_signal_eta --ablation      # 무엇이 개선을 만들었는지
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
    from . import route_wait as rw
    from .eta import EtaEngine
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    import src.route_wait as rw
    from src.eta import EtaEngine

PROJ = Path(__file__).resolve().parent.parent

#: 실측으로 확정한 검증 경로의 신호 횡단보도 9곳.
#: 4명(박준서·최희수·홍민기·권동하)이 걸은 16개 GPX 에서 전원이 이 9곳을
#: 최대 6.8m 이내로 통과했고, 목록 밖에서 멈춘 기록은 58건 중 0건이다.
#: 자동 매핑의 정답지로만 쓴다 — 예측에는 쓰지 않는다.
GROUND_TRUTH = {
    "A": [("A-1", 37.24761, 127.07601), ("A-2", 37.24784, 127.07575),
          ("A-3", 37.24922, 127.07458), ("A-4", 37.25069, 127.07243)],
    "B": [("B-1", 37.25256, 127.07280), ("B-2", 37.25381, 127.07519),
          ("B-3", 37.25324, 127.07611), ("B-4", 37.25236, 127.07755),
          ("B-5", 37.25172, 127.07881)],
}

#: 지점별 스톱워치 실측(초). 지점 순서는 GROUND_TRUTH 와 같다.
#: 권동하는 기록지에 지점별 내역이 없어 회차 총합만 있다(validation_stopwatch.csv).
MEASURED = {
    ("박준서", "A", 1): [105, 19, 13, 8],   ("박준서", "A", 2): [81, 57, 16, 61],
    ("박준서", "B", 1): [6, 70, 57, 64, 23], ("박준서", "B", 2): [0, 36, 0, 67, 88],
    ("최희수", "A", 1): [115, 27, 25, 40],  ("최희수", "A", 2): [52, 29, 18, 35],
    ("최희수", "B", 1): [0, 39, 60, 73, 16], ("최희수", "B", 2): [0, 24, 57, 65, 7],
    ("홍민기", "A", 1): [92, 12, 0, 0],     ("홍민기", "A", 2): [42, 3, 56, 72],
    ("홍민기", "B", 1): [9, 27, 28, 20, 106], ("홍민기", "B", 2): [14, 29, 63, 28, 25],
}

ROUTE_OF = {"검증1": "A", "검증2": "B"}
BAR = "=" * 100


def _hav(lat1, lon1, lat2, lon2) -> np.ndarray:
    lat1, lon1, lat2, lon2 = (np.radians(np.asarray(v, dtype=float))
                              for v in (lat1, lon1, lat2, lon2))
    a = (np.sin((lat2 - lat1) / 2) ** 2
         + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2)
    return 2 * 6371000.0 * np.arcsin(np.sqrt(a))


def assign(hit: pd.DataFrame, route: str):
    """검출 후보 ↔ 확정지점 1:1 탐욕 배정(가까운 쌍부터). 반환: {지점: (행번호, 거리)}."""
    pts = GROUND_TRUTH[route]
    pairs = sorted((float(_hav([r["위도"]], [r["경도"]], gla, glo)[0]), hi, gi)
                   for hi, (_, r) in enumerate(hit.iterrows())
                   for gi, (_, gla, glo) in enumerate(pts))
    used_h, used_g, out = set(), set(), {}
    for d, hi, gi in pairs:
        if hi in used_h or gi in used_g:
            continue
        used_h.add(hi); used_g.add(gi); out[pts[gi][0]] = (hi, d)
    return out, [hi for hi in range(len(hit)) if hi not in used_h]


def evaluate(root: Path = PROJ):
    eng = EtaEngine.load(root)
    seg = pd.read_csv(root / "data/processed/validation_segments.csv")
    sw = pd.read_csv(root / "data/processed/validation_stopwatch.csv")
    total_wait = {(r.person, ROUTE_OF[r.route], r.trial): r.wait_s for r in sw.itertuples()}

    print(f"[cfg] 좌표여유 {rw.GPS_ERR_M}m · 임계상한 {rw.MAX_THRESHOLD_M}m · "
          f"교차로 {rw.INTERSECTION_M}m · λ {rw.LAMBDA}")

    detail, route_rows = [], []
    for (person, cat, trial), d in seg.groupby(["person", "category", "trial"]):
        route = ROUTE_OF[cat]
        lat, lon = eng._route_track(d)
        hit = rw.crossed(eng.signals, lat, lon).reset_index(drop=True)
        amap, extra = assign(hit, route)
        man = MEASURED.get((person, route, trial))

        for i, (pid, _, _) in enumerate(GROUND_TRUTH[route]):
            theo = 0.0; matched, dist = "-", np.nan
            if pid in amap:
                hi, dist = amap[pid]
                theo = float(hit.iloc[hi]["기대대기_s"])
                matched = hit.iloc[hi]["횡단보도관리번호"]
            detail.append(dict(person=person, 경로=route, 회차=trial, 지점=pid,
                               검출=pid in amap, 매칭=matched,
                               배정거리_m=round(dist) if dist == dist else np.nan,
                               이론_s=round(theo, 1), 보정_s=round(theo * rw.LAMBDA, 1),
                               실측_s=man[i] if man else np.nan))
        for hi in extra:
            r = hit.iloc[hi]
            detail.append(dict(person=person, 경로=route, 회차=trial, 지점="초과검출",
                               검출=True, 매칭=r["횡단보도관리번호"], 배정거리_m=np.nan,
                               이론_s=round(float(r["기대대기_s"]), 1),
                               보정_s=round(float(r["기대대기_s"]) * rw.LAMBDA, 1),
                               실측_s=np.nan))

        theo = float(hit["기대대기_s"].sum())
        v = eng.v_user.get(person, float(np.mean(list(eng.v_user.values()))))
        moving = float((d["dist_m"].to_numpy()
                        / (v * eng.predicted_ratio(d["slope_pct"].fillna(0.0)))).sum())
        actual = float(d["dt_s"].sum())
        mw = total_wait[(person, route, trial)]
        route_rows.append(dict(person=person, 경로=route, 회차=trial,
                               거리_m=round(d["dist_m"].sum()), 검출수=len(hit),
                               확정수=len(GROUND_TRUTH[route]),
                               신호_이론_s=round(theo), 신호_보정_s=round(theo * rw.LAMBDA),
                               신호_실측_s=mw,
                               도보_예측_s=round(moving), 도보_실측_s=round(actual - mw),
                               ETA_예측_s=round(moving + theo * rw.LAMBDA),
                               ETA_실측_s=round(actual),
                               정속4kmh_s=round(d["dist_m"].sum() / (4 / 3.6))))
    return pd.DataFrame(detail), pd.DataFrame(route_rows).sort_values(["person", "경로", "회차"])


def report(D: pd.DataFrame, R: pd.DataFrame) -> None:
    g = D[D["지점"] != "초과검출"]
    print(f"\n{BAR}\n[1] 자동 매핑 — 확정 9곳이 잡히는가\n{BAR}")
    print(R.pivot_table(index="person", columns="경로", values="검출수", aggfunc=list).to_string())
    print(f"\n  경로 A 확정 4곳 / 경로 B 확정 5곳")
    print(f"  1:1 배정 {int(g['검출'].sum())}/{len(g)} = {g['검출'].mean():.0%}"
          f"   초과검출 {int((D['지점'] == '초과검출').sum())}건")

    print(f"\n{BAR}\n[2] 지점별 이론 / 보정 / 실측 (초)\n{BAR}")
    t = g.groupby(["경로", "지점"]).agg(
        검출률=("검출", "mean"), 매칭=("매칭", lambda s: s.mode().iloc[0]),
        배정거리_m=("배정거리_m", "mean"), 이론=("이론_s", "mean"),
        보정=("보정_s", "mean"), 실측평균=("실측_s", "mean"))
    t["보정−실측"] = t["보정"] - t["실측평균"]
    print(t.round(1).to_string())

    print(f"\n{BAR}\n[3] 사람별 × 회차별 (초)\n{BAR}")
    print(R.to_string(index=False))

    print(f"\n{BAR}\n[4] 오차\n{BAR}")
    rows = []
    for name, pc, ac in [("신호대기 — 이론", "신호_이론_s", "신호_실측_s"),
                         (f"신호대기 — 보정(λ={rw.LAMBDA})", "신호_보정_s", "신호_실측_s"),
                         ("도보시간", "도보_예측_s", "도보_실측_s"),
                         ("최종 ETA", "ETA_예측_s", "ETA_실측_s"),
                         ("최종 ETA — 정속 4km/h", "정속4kmh_s", "ETA_실측_s")]:
        e = R[pc] - R[ac]
        rows.append(dict(항목=name, MAE=abs(e).mean(), 편향=e.mean(),
                         MAPE=(abs(e) / R[ac]).mean() * 100))
    print(pd.DataFrame(rows).round(1).to_string(index=False))

    e = R["도보_실측_s"] + R["신호_보정_s"] - R["ETA_실측_s"]
    print(f"\n  도보를 정답으로 고정 → 신호 항만: MAE {abs(e).mean():.1f}초 "
          f"(오라클 하한 35.8초)")
    e2 = R["도보_예측_s"] - R["ETA_실측_s"]
    print(f"  신호를 무시 → 도보 항만:        MAE {abs(e2).mean():.1f}초 "
          f"편향 {e2.mean():+.1f}초")


def compare_with_validation_results(R: pd.DataFrame, root: Path = PROJ) -> None:
    """evaluate_validation.py 산출물과 같은 축으로 대조."""
    p = root / "data/processed/validation_results.csv"
    if not p.exists():
        print("\n(validation_results.csv 없음 — 대조 생략)")
        return
    V = pd.read_csv(p)
    m = V.merge(R.assign(route=R["경로"].map({"A": "검증1", "B": "검증2"}))
                 [["person", "route", "회차", "신호_보정_s"]],
                left_on=["person", "route", "trial"], right_on=["person", "route", "회차"])

    leak = np.allclose(m["M3_신호"] - m["M2_개인화"], m["신호대기_실측"])
    print(f"\n{BAR}\n[5] evaluate_validation.py 와 대조\n{BAR}")
    print(f"  M3_신호 − M2_개인화 == 신호대기_실측 ?  {leak}")
    print("  → True 면 그쪽 M3/M4 의 신호 항은 예측이 아니라 그 회차의 정답이다.")
    share = m["신호대기_실측"].mean() / m["actual_s"].mean() * 100
    print(f"  정답(actual_s) 중 신호대기가 차지하는 몫: 평균 {share:.0f}%")

    m["M3_공공_기존"] = m["M2_개인화"] + m["신호대기"]        # 기존 영통구 검출기
    m["M3_공공_신규"] = m["M2_개인화"] + m["신호_보정_s"]      # 이 모델
    m["M4_공공_기존"] = m["M4_캘리브"] - m["신호대기_실측"] + m["신호대기"]
    m["M4_공공_신규"] = m["M4_캘리브"] - m["신호대기_실측"] + m["신호_보정_s"]

    for label, d in [("16회차 전체", m), ("정방향 12회 (권동하 주분석)", m[m["방향"] == "정방향"])]:
        print(f"\n  --- {label} ---")
        base = abs(d["M0_정속"] - d["actual_s"]).mean()
        for name, c in [("M0 정속 4km/h", "M0_정속"),
                        ("M2 개인화 (신호 없음)", "M2_개인화"),
                        ("M3 + 신호(기존 영통구 검출기)", "M3_공공_기존"),
                        ("M3 + 신호(이 모델)", "M3_공공_신규"),
                        ("M3 + 신호(실측·오라클)", "M3_신호"),
                        ("M4 웜스타트 + 신호(기존)", "M4_공공_기존"),
                        ("M4 웜스타트 + 신호(이 모델)", "M4_공공_신규"),
                        ("M4 웜스타트 + 신호(실측·오라클)", "M4_캘리브")]:
            e = d[c] - d["actual_s"]
            print(f"    {name:34s} MAE {abs(e).mean():6.1f}초  편향 {e.mean():+7.1f}  "
                  f"MAPE {(abs(e) / d['actual_s']).mean() * 100:5.1f}%  "
                  f"M0대비 {(1 - abs(e).mean() / base) * 100:3.0f}%↓")


def main() -> int:
    ap = argparse.ArgumentParser(description="신호대기 모델 → 최종 ETA 평가")
    ap.add_argument("--root", type=Path, default=PROJ)
    ap.add_argument("--out", type=Path, default=PROJ / "data/processed")
    a = ap.parse_args()

    D, R = evaluate(a.root)
    report(D, R)
    compare_with_validation_results(R, a.root)
    D.to_csv(a.out / "eval_signal_detail.csv", index=False, encoding="utf-8-sig")
    R.to_csv(a.out / "eval_signal_route.csv", index=False, encoding="utf-8-sig")
    print(f"\n  -> {a.out/'eval_signal_detail.csv'}\n  -> {a.out/'eval_signal_route.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
