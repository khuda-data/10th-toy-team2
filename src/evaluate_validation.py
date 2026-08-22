"""검증 경로 최종 평가.

검증용 GPX(`data/raw/validation/`)가 도착하면 이 스크립트 하나로 최종 결과가 나온다.

산출물:
  1. 사람별 × 경로별 예측 오차 (MAE/MAPE)      <- 회의 결정: 풀링하지 않고 사람별 산출
  2. 요인별 기여도 분해 (M0 정속 -> M1 +경사 -> M2 +개인화 -> M3 +신호)
  3. 통계 검정 (대응표본 t-검정 + Wilcoxon)
  4. 경로 간 오차 패턴 비교 (모델 편향 vs 경로 특성 판별)

사용법:
    # 실제 검증 데이터로
    python -m src.evaluate_validation --dem-dir data/raw/dem

    # 데이터 도착 전 동작 확인 (학습 데이터를 검증 데이터인 척 사용)
    python -m src.evaluate_validation --dry-run

⚠️ 실측 정답은 dt_s(총 경과시간)를 사용한다. moving_dt_s(정지 제외)를 쓰면
   신호 대기가 정답에서 빠져 검증이 무의미해진다.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.eta import EtaEngine, DEFAULT_WALK_SPEED_KMH


# ---------------------------------------------------------------- 데이터 적재
def load_validation_segments(raw_dir: Path, dem_dir: Path, out: Path) -> pd.DataFrame:
    """검증 GPX -> 구간 테이블. build_dataset과 동일 파이프라인."""
    from src.build_dataset import sample_dem_df
    from src.gpx_parser import parse_all
    from src.segmentation import add_step_distance, build_segments, remove_gps_jumps

    dem_paths = sorted({p for ext in ("*.tif", "*.tiff", "*.img", "*.asc") for p in dem_dir.glob(ext)})
    if not dem_paths:
        raise SystemExit(f"{dem_dir}에서 DEM 파일을 찾지 못했습니다")

    pts = parse_all(raw_dir)
    print(f"  GPX {pts['source_file'].nunique()}개, {len(pts)} 포인트")
    pts = remove_gps_jumps(pts)
    pts = sample_dem_df(pts, dem_paths)
    n_nan = int(pts["ele_dem"].isna().sum())
    if n_nan:
        print(f"  ⚠️ DEM 범위 밖 포인트 {n_nan}개 — 해당 지역 DEM 도엽이 필요합니다")
    pts = add_step_distance(pts)
    seg = build_segments(pts)
    out.parent.mkdir(parents=True, exist_ok=True)
    seg.to_csv(out, index=False, encoding="utf-8-sig")
    print(f"  -> {len(seg)}개 구간 저장: {out}")
    return seg


def load_stopwatch(path: Path) -> pd.DataFrame | None:
    """스톱워치 신호대기 실측 기록(선택). 컬럼: person, route, trial, wait_s"""
    if not path.exists():
        return None
    df = pd.read_csv(path)
    print(f"  스톱워치 기록 {len(df)}건 로드: {path}")
    return df


# ---------------------------------------------------------------- 4단계 예측
def predict_stages(seg_route: pd.DataFrame, person: str, engine: EtaEngine,
                   team_mean_v: float) -> dict[str, float]:
    """한 회차(경로 1개)에 대한 4단계 누적 예측(초)."""
    dist = seg_route["dist_m"].sum()
    slope = seg_route["slope_pct"].fillna(0.0)
    ratio = engine.predicted_ratio(slope)
    d = seg_route["dist_m"].to_numpy()

    m0 = dist / (DEFAULT_WALK_SPEED_KMH / 3.6)          # 정속 4km/h (네이버 방식)
    m1 = float((d / (team_mean_v * ratio)).sum())       # + 경사 (개인화 없음)
    m2 = float((d / (engine.v_user.get(person, team_mean_v) * ratio)).sum())  # + 개인화
    wait = engine.route_signal_wait_s(seg_route)
    return {"M0_정속": m0, "M1_경사": m1, "M2_개인화": m2, "M3_신호": m2 + wait, "신호대기": wait}


def build_results(seg: pd.DataFrame, engine: EtaEngine,
                  stopwatch: pd.DataFrame | None = None) -> pd.DataFrame:
    """회차 단위 결과 테이블. category를 '경로'로 사용한다(검증A/검증B)."""
    team_mean_v = float(np.mean(list(engine.v_user.values())))
    rows = []
    for (person, route, trial), g in seg.groupby(["person", "category", "trial"]):
        actual = g["dt_s"].sum()          # ⚠️ 총 경과시간 = 정답
        r = {"person": person, "route": route, "trial": trial,
             "dist_m": g["dist_m"].sum(), "n_seg": len(g), "actual_s": actual,
             "moving_s": g["moving_dt_s"].sum()}
        r.update(predict_stages(g, person, engine, team_mean_v))

        if stopwatch is not None:  # 실측 신호대기가 있으면 그것으로 대체
            m = stopwatch[(stopwatch.person == person) & (stopwatch.route == route)
                          & (stopwatch.trial == trial)]
            if len(m):
                r["신호대기_실측"] = float(m["wait_s"].iloc[0])
                r["M3_신호"] = r["M2_개인화"] + r["신호대기_실측"]
        rows.append(r)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- 리포트
STAGES = ["M0_정속", "M1_경사", "M2_개인화", "M3_신호"]


def fmt(s: float) -> str:
    m, sec = divmod(int(round(abs(s))), 60)
    return f"{'-' if s < 0 else ''}{m}분 {sec:02d}초"


def report_by_person(res: pd.DataFrame) -> None:
    """회의 결정: 풀링하지 않고 사람별로 산출."""
    print("\n" + "=" * 74)
    print("  [1] 사람별 예측 오차 — 회차 평균")
    print("=" * 74)
    for person, g in res.groupby("person"):
        print(f"\n  {person}  (측정 {len(g)}회)")
        print(f"    {'경로':<10}{'실측':>10}{'우리모델':>11}{'네이버':>10}{'우리오차':>10}{'네이버오차':>11}")
        for route, gr in g.groupby("route"):
            a = gr["actual_s"].mean()
            print(f"    {route:<10}{fmt(a):>10}{fmt(gr['M3_신호'].mean()):>11}"
                  f"{fmt(gr['M0_정속'].mean()):>10}"
                  f"{(gr['M3_신호'] - gr['actual_s']).abs().mean():>9.0f}초"
                  f"{(gr['M0_정속'] - gr['actual_s']).abs().mean():>10.0f}초")
        ours = (g["M3_신호"] - g["actual_s"]).abs()
        base = (g["M0_정속"] - g["actual_s"]).abs()
        print(f"    {'전체':<10}{'':>10}{'':>11}{'':>10}{ours.mean():>9.0f}초{base.mean():>10.0f}초"
              f"   (MAPE {ours.mean()/g['actual_s'].mean()*100:.1f}% vs {base.mean()/g['actual_s'].mean()*100:.1f}%)")


def report_ablation(res: pd.DataFrame) -> None:
    """요인별 기여도 분해."""
    print("\n" + "=" * 74)
    print("  [2] 요인별 기여도 — 무엇이 오차를 줄였나")
    print("=" * 74)
    print(f"\n  {'단계':<14}{'MAE':>10}{'MAPE':>9}{'직전 대비 개선':>16}")
    prev = None
    for s in STAGES:
        err = (res[s] - res["actual_s"]).abs()
        mae, mape = err.mean(), (err / res["actual_s"]).mean() * 100
        delta = "—" if prev is None else f"{prev - mae:+.0f}초"
        print(f"  {s:<14}{mae:>9.0f}초{mape:>8.1f}%{delta:>16}")
        prev = mae
    tot = (res["M0_정속"] - res["actual_s"]).abs().mean()
    fin = (res["M3_신호"] - res["actual_s"]).abs().mean()
    print(f"\n  총 개선: {tot:.0f}초 -> {fin:.0f}초 ({(1-fin/tot)*100:.0f}% 감소)")


def report_route_effect(res: pd.DataFrame) -> None:
    """경로 2개 설계의 핵심: 모델 편향인가 경로 특성인가."""
    print("\n" + "=" * 74)
    print("  [3] 경로 간 비교 — 모델 편향 vs 경로 특성")
    print("=" * 74)
    print(f"\n  {'경로':<10}{'n':>4}{'평균 부호오차':>15}{'MAE':>9}")
    signs = {}
    for route, g in res.groupby("route"):
        signed = (g["M3_신호"] - g["actual_s"]).mean()   # 부호 유지 = 편향 방향
        signs[route] = signed
        print(f"  {route:<10}{len(g):>4}{signed:>+14.0f}초{(g['M3_신호']-g['actual_s']).abs().mean():>8.0f}초")
    if len(signs) >= 2:
        vals = list(signs.values())
        same = all(v > 0 for v in vals) or all(v < 0 for v in vals)
        print(f"\n  -> 부호가 {'일치' if same else '불일치'}: "
              f"{'모델의 계통 편향으로 해석' if same else '경로 특성 차이로 해석 (모델 편향 근거 약함)'}")
    else:
        print("\n  -> 경로가 1개뿐이라 모델 편향과 경로 특성을 분리할 수 없음")


def report_tests(res: pd.DataFrame) -> None:
    from scipy import stats
    print("\n" + "=" * 74)
    print("  [4] 통계 검정 — 우리 모델이 정속 방식보다 나은가")
    print("=" * 74)
    ours = (res["M3_신호"] - res["actual_s"]).abs()
    base = (res["M0_정속"] - res["actual_s"]).abs()
    d = base - ours
    print(f"\n  n = {len(res)}회 측정, 회차당 평균 개선 {d.mean():+.0f}초")
    if len(res) >= 3:
        t = stats.ttest_rel(base, ours)
        w = stats.wilcoxon(base, ours) if len(res) >= 6 else None
        print(f"  대응표본 t-검정 : t = {t.statistic:.2f}, p = {t.pvalue:.2e}"
              f"  {'✅ 유의' if t.pvalue < 0.05 else '❌ 유의수준 미달'}")
        if w:
            print(f"  Wilcoxon 부호순위: p = {w.pvalue:.2e}"
                  f"  {'✅ 유의' if w.pvalue < 0.05 else '❌ 유의수준 미달'}")
        print("\n  (표본이 작아 정규성 가정이 불확실하므로 두 검정을 병기한다)")
    else:
        print("  측정 수가 부족해 검정을 생략한다")


# ---------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--raw-dir", type=Path, default=Path("data/raw/validation"))
    ap.add_argument("--dem-dir", type=Path, default=Path("data/raw/dem"))
    ap.add_argument("--segments", type=Path, default=Path("data/processed/validation_segments.csv"))
    ap.add_argument("--stopwatch", type=Path, default=Path("data/processed/validation_stopwatch.csv"),
                    help="신호대기 실측 기록(선택). 없으면 공공데이터 기대대기를 사용")
    ap.add_argument("--out", type=Path, default=Path("data/processed/validation_results.csv"))
    ap.add_argument("--dry-run", action="store_true",
                    help="검증 데이터 도착 전 동작 확인 — 학습 데이터로 대체 실행")
    args = ap.parse_args()

    engine = EtaEngine.load()

    if args.dry_run:
        print("\n[DRY RUN] 학습 데이터를 검증 데이터인 척 사용합니다 (동작 확인용)")
        seg = pd.read_csv("data/processed/segments.csv")
        seg = seg[seg.person.isin(["권동하", "김재형", "박준서"])]
        seg = seg[seg.category.isin(["평지", "완만오르막"])]   # '경로 2개'인 척
    elif args.segments.exists():
        print(f"기존 구간 테이블 사용: {args.segments}")
        seg = pd.read_csv(args.segments)
    else:
        print(f"[1/2] 검증 GPX 처리: {args.raw_dir}")
        seg = load_validation_segments(args.raw_dir, args.dem_dir, args.segments)

    seg = seg.dropna(subset=["speed_mps"])
    stopwatch = None if args.dry_run else load_stopwatch(args.stopwatch)
    res = build_results(seg, engine, stopwatch)

    report_by_person(res)
    report_ablation(res)
    report_route_effect(res)
    report_tests(res)

    if not args.dry_run:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        res.to_csv(args.out, index=False, encoding="utf-8-sig")
        print(f"\n저장: {args.out} (그래프용 원자료)")
    print()


if __name__ == "__main__":
    main()
