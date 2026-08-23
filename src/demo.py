"""ETA 계산기 직접 써보기 (역할③ 데모).

실행: python -m src.demo
아무것도 입력 안 해도 예시가 쭉 나오고, 마지막에 직접 숫자를 넣어볼 수 있습니다.
"""
from __future__ import annotations

import pandas as pd

from src.eta import EtaEngine


def fmt(sec: float) -> str:
    """초 -> '3분 12초' 형태."""
    m, s = divmod(int(round(sec)), 60)
    return f"{m}분 {s}초" if m else f"{s}초"


def simple_eta(engine: EtaEngine, person: str, dist_m: float, slope_pct: float) -> float:
    """거리와 경사도만 알 때의 ETA(초). 구간 1개짜리 가상 경로로 계산."""
    seg = pd.DataFrame([{"dist_m": dist_m, "slope_pct": slope_pct}])
    return engine.route_eta(seg, person, include_signals=False)


def main() -> None:
    engine = EtaEngine.load()
    people = sorted(engine.v_user, key=lambda p: -engine.v_user[p])

    print("\n" + "=" * 60)
    print("  WalkFit ETA 계산기")
    print("=" * 60)

    # --- 1. 등록된 사람들의 속도 ---
    print("\n[1] 등록된 사람과 평지 속도")
    for p in people:
        v = engine.v_user[p]
        print(f"  {p}: {v:.3f} m/s  (시속 {v*3.6:.2f} km)")
    print(f"  참고) 네이버지도 가정: 1.111 m/s (시속 4.00 km)")

    # --- 2. 경사도에 따른 속도 배율 ---
    print("\n[2] 경사도가 속도에 미치는 영향 (평지=1.000)")
    print("  경사도    배율     의미")
    for s in [-15, -10, -5, 0, 5, 10, 15]:
        r = float(engine.predicted_ratio(s))
        label = "내리막" if s < 0 else ("평지" if s == 0 else "오르막")
        print(f"  {s:+4d}%   {r:.3f}   {label} — 평지 대비 {(r-1)*100:+.1f}%")

    # --- 3. 같은 길, 사람별 비교 ---
    print("\n[3] 같은 길을 걸으면? (1km, 오르막 5%)")
    for p in people:
        t = simple_eta(engine, p, 1000, 5)
        print(f"  {p}: {fmt(t)}")
    print(f"  네이버지도 방식: {fmt(1000 / (4/3.6))}  <- 누구든 똑같이 계산")

    # --- 4. 같은 사람, 지형별 비교 ---
    who = people[len(people) // 2]
    print(f"\n[4] {who}님이 1km를 걸을 때, 지형에 따라")
    for s in [-10, -5, 0, 5, 10]:
        t = simple_eta(engine, who, 1000, s)
        base = simple_eta(engine, who, 1000, 0)
        print(f"  경사 {s:+3d}%: {fmt(t)}  (평지 대비 {t-base:+.0f}초)")

    # --- 5. 실제 걸었던 길로 검증 ---
    print("\n[5] 실제로 걸었던 길 — 예측이 맞았나?")
    try:
        seg = pd.read_csv("data/processed/segments.csv")
        print(f"  {'경로':<28} {'실측':>8} {'WalkFit':>10} {'네이버방식':>10}")
        for f, g in list(seg.groupby("source_file"))[:6]:
            p = g["person"].iloc[0]
            actual = g["dt_s"].sum()
            ours = engine.route_eta(g, p)
            base = EtaEngine.route_eta_baseline(g)
            name = f.replace(".gpx", "")[:26]
            print(f"  {name:<28} {fmt(actual):>8} {fmt(ours):>10} {fmt(base):>10}")
    except FileNotFoundError:
        print("  (segments.csv를 찾지 못했습니다)")

    # --- 6. 직접 입력 ---
    print("\n" + "-" * 60)
    print("[6] 직접 넣어보기  (그냥 Enter 치면 종료)")
    print("-" * 60)
    while True:
        try:
            raw = input("\n거리(m)와 경사도(%)를 띄어쓰기로 [예: 800 3] > ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not raw:
            break
        try:
            parts = raw.split()
            dist, slope = float(parts[0]), float(parts[1]) if len(parts) > 1 else 0.0
        except (ValueError, IndexError):
            print("  숫자 두 개를 넣어주세요. 예: 800 3")
            continue

        print(f"\n  [{dist:.0f}m, 경사 {slope:+.1f}%]")
        base = dist / (4 / 3.6)
        for p in people:
            t = simple_eta(engine, p, dist, slope)
            diff = t - base
            print(f"    {p}: {fmt(t):>10}   (네이버방식 대비 {diff:+.0f}초)")
        print(f"    {'네이버지도':<4}: {fmt(base):>10}")

    print("\n종료합니다.\n")


if __name__ == "__main__":
    main()
