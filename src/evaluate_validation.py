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
    return {"기존_정속": m0, "경사반영": m1, "개인속도_실험값": m2, "신호반영": m2 + wait, "신호대기": wait}


# 수집 설계상의 정방향: 검증1 = 공대(82m) -> 영통역(55m) 내리막,
# 검증2 = 영통역(55m) -> 레이지하우스(68m) 오르막.
# 박준서·홍민기의 2회차는 돌아오는 길(역방향)이라 경사가 뒤집혀 있다.
FORWARD_IS_DOWNHILL = {"검증1": True, "검증2": False}


def direction_of(route: str, net_elev_m: float) -> str:
    """순고도차 부호로 정/역방향을 판정한다. 미등록 경로는 '정방향'으로 둔다."""
    fwd_down = FORWARD_IS_DOWNHILL.get(route)
    if fwd_down is None:
        return "정방향"
    is_down = net_elev_m < 0
    return "정방향" if is_down == fwd_down else "역방향"


def build_results(seg: pd.DataFrame, engine: EtaEngine,
                  stopwatch: pd.DataFrame | None = None) -> pd.DataFrame:
    """회차 단위 결과 테이블. category를 '경로'로 사용한다(검증A/검증B).

    같은 경로라도 왕복 방향에 따라 오르막/내리막이 뒤집히므로 `방향`을
    순고도차 부호로 판정해 따로 기록한다. 경로만으로 묶으면 [3]의
    '모델 편향 vs 경로 특성' 판별이 방향과 교란된다.
    """
    team_mean_v = float(np.mean(list(engine.v_user.values())))
    rows = []
    for (person, route, trial), g in seg.groupby(["person", "category", "trial"]):
        g = g.sort_values("segment_id")
        actual = g["dt_s"].sum()          # ⚠️ 총 경과시간 = 정답
        net_elev = float(g["elev_end_m"].iloc[-1] - g["elev_start_m"].iloc[0])
        r = {"person": person, "route": route, "trial": trial,
             "경사": "내리막" if net_elev < 0 else "오르막", "순고도차_m": net_elev,
             "방향": direction_of(route, net_elev),
             "dist_m": g["dist_m"].sum(), "n_seg": len(g), "actual_s": actual,
             "moving_s": g["moving_dt_s"].sum()}
        r.update(predict_stages(g, person, engine, team_mean_v))

        if stopwatch is not None:  # 실측 신호대기가 있으면 그것으로 대체
            m = stopwatch[(stopwatch.person == person) & (stopwatch.route == route)
                          & (stopwatch.trial == trial)]
            if len(m):
                r["신호대기_실측"] = float(m["wait_s"].iloc[0])
                r["신호반영"] = r["개인속도_실험값"] + r["신호대기_실측"]
        # 이 회차가 말해주는 평지환산 속도 (M4 캘리브레이션용)
        ratio = engine.predicted_ratio(g["slope_pct"].fillna(0.0))
        r["평지환산거리_m"] = float((g["dist_m"].to_numpy() / ratio).sum())
        r["대기_s"] = r.get("신호대기_실측", r["신호대기"])
        moving = actual - r["대기_s"]
        r["v_hat"] = r["평지환산거리_m"] / moving if moving > 0 else np.nan
        r["생속도"] = r["dist_m"] / moving if moving > 0 else np.nan
        rows.append(r)
    return add_warm_start(pd.DataFrame(rows))


def add_warm_start(res: pd.DataFrame) -> pd.DataFrame:
    """M4: 회차 leave-one-out 개인 캘리브레이션.

    학습 v_user는 경사 실험용 반복 보행에서 나왔고 검증은 목적지를 향한
    연속 보행이라 맥락이 다르다(최대 +21% 괴리). 실제 서비스에는 사용자의
    '같은 맥락' 이력이 쌓이므로 그 상황을 누수 없이 모사한다 — 평가 대상
    회차를 제외한 나머지 회차로만 v_user를 재추정해 그 회차를 예측한다.

    M2(학습 v_user) = 콜드스타트, M4(회차 LOO) = 웜스타트.
    """
    res = res.copy()
    m4, v_loo = [], []
    for i, row in res.iterrows():
        others = res[(res.person == row.person) & (res.index != i)]
        if len(others) == 0 or others["v_hat"].isna().all():
            m4.append(np.nan); v_loo.append(np.nan); continue
        v = float(others["v_hat"].mean())
        v_loo.append(v)
        m4.append(row["평지환산거리_m"] / v + row["대기_s"])
    res["v_LOO"] = v_loo
    res["개인속도_실제이력"] = m4
    return res


# ---------------------------------------------------------------- 리포트
STAGES = ["기존_정속", "경사반영", "개인속도_실험값", "신호반영", "개인속도_실제이력"]


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
            print(f"    {route:<10}{fmt(a):>10}{fmt(gr['신호반영'].mean()):>11}"
                  f"{fmt(gr['기존_정속'].mean()):>10}"
                  f"{(gr['신호반영'] - gr['actual_s']).abs().mean():>9.0f}초"
                  f"{(gr['기존_정속'] - gr['actual_s']).abs().mean():>10.0f}초")
        ours = (g["신호반영"] - g["actual_s"]).abs()
        base = (g["기존_정속"] - g["actual_s"]).abs()
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
    tot = (res["기존_정속"] - res["actual_s"]).abs().mean()
    m3 = (res["신호반영"] - res["actual_s"]).abs().mean()
    m4 = (res["개인속도_실제이력"] - res["actual_s"]).abs().mean()
    print(f"\n  총 개선(기존 실험값 사용): {tot:.0f}초 -> {m3:.0f}초 ({(1-m3/tot)*100:.0f}% 감소)")
    print(f"  총 개선(실제 이력 사용): {tot:.0f}초 -> {m4:.0f}초 ({(1-m4/tot)*100:.0f}% 감소)")


def report_route_effect(res: pd.DataFrame) -> None:
    """경로 2개 설계의 핵심: 모델 편향인가 경로 특성인가.

    ⚠️ 같은 '경로' 안에 왕복 두 방향이 섞여 있다. 박준서·홍민기는 2회차를
    돌아오는 길로 걸었다(역방향 4회). 경로로만 묶으면 부호오차가 경사
    방향과 교란되므로 경로 × 방향으로 나눠서 본다.
    """
    print("\n" + "=" * 74)
    print("  [3] 경로 × 방향 비교 — 모델 편향 vs 경로/방향 특성")
    print("=" * 74)
    print(f"\n  {'경로':<8}{'방향':<9}{'경사':<7}{'n':>3}{'①부호오차':>13}{'②부호오차':>13}")
    signs = []
    for (route, direc), g in res.groupby(["route", "방향"]):
        s3 = (g["신호반영"] - g["actual_s"]).mean()      # 부호 유지 = 편향 방향
        s4 = (g["개인속도_실제이력"] - g["actual_s"]).mean()
        signs.append(s3)
        print(f"  {route:<8}{direc:<9}{g['경사'].iloc[0]:<7}{len(g):>3}"
              f"{s3:>+12.0f}초{s4:>+12.0f}초")
    if len(signs) >= 2:
        same = all(v > 0 for v in signs) or all(v < 0 for v in signs)
        print(f"\n  -> ①의 부호가 {'일치' if same else '불일치'}: "
              f"{'경로·방향과 무관한 계통 편향' if same else '경로/방향 특성이 섞여 있음'}")
    s4_all = (res["개인속도_실제이력"] - res["actual_s"]).mean()
    s3_all = (res["신호반영"] - res["actual_s"]).mean()
    print(f"  -> 전체 부호오차: ① {s3_all:+.0f}초 -> ② {s4_all:+.0f}초  "
          f"(실제 이력을 쓰면 한쪽으로 쏠리는 편향이 {'사라진다' if abs(s4_all) < abs(s3_all) / 2 else '남는다'})")


def report_roundtrip(res: pd.DataFrame) -> None:
    """왕복 대칭 — k_slope가 방향 비대칭을 얼마나 설명하는가.

    역방향 회차가 있는 사람만 대상. 같은 길을 반대로 걸었으므로 생속도
    차이는 (경사 + 그 외)이고, 평지환산 속도(v_hat) 차이는 (그 외)만
    남는다. 둘의 간극이 k_slope가 설명한 몫이다.
    """
    pairs = []
    for (person, route), g in res.groupby(["person", "route"]):
        if set(g["방향"]) != {"정방향", "역방향"}:
            continue
        f = g[g.방향 == "정방향"].iloc[0]
        b = g[g.방향 == "역방향"].iloc[0]
        raw = abs(b["생속도"] / f["생속도"] - 1) * 100
        adj = abs(b["v_hat"] / f["v_hat"] - 1) * 100
        pairs.append({"person": person, "route": route, "보정전": raw, "보정후": adj})
    if not pairs:
        return
    P = pd.DataFrame(pairs)
    print("\n" + "=" * 74)
    print("  [5] 왕복 대칭 — 경사 모델이 방향 비대칭을 설명하는가")
    print("=" * 74)
    print(f"\n  {'사람':<8}{'경로':<8}{'생속도 차':>11}{'경사보정 후':>13}{'설명한 몫':>12}")
    for _, r in P.iterrows():
        print(f"  {r.person:<8}{r.route:<8}{r.보정전:>10.1f}%{r.보정후:>12.1f}%"
              f"{r.보정전 - r.보정후:>11.1f}%p")
    share = (1 - P.보정후.mean() / P.보정전.mean()) * 100
    print(f"\n  평균: {P.보정전.mean():.1f}% -> {P.보정후.mean():.1f}%  "
          f"(k_slope가 왕복 비대칭의 {share:.0f}%만 설명)")
    if share < 50:
        print("  -> 경사 모델이 방향 비대칭을 과소보정한다. 남은 비대칭은 미모델링 요인"
              "(계단·피로·노면).")


def report_tests(res: pd.DataFrame) -> None:
    from scipy import stats
    print("\n" + "=" * 74)
    print("  [4] 통계 검정 — 우리 모델이 정속 방식보다 나은가")
    print("=" * 74)
    base = (res["기존_정속"] - res["actual_s"]).abs()
    print(f"\n  n = {len(res)}회 측정")
    if len(res) < 3:
        print("  측정 수가 부족해 검정을 생략한다")
        return
    for col, label in [("신호반영", "① 개인속도를 기존 실험값으로"),
                       ("개인속도_실제이력", "② 개인속도를 실제 이력으로")]:
        ours = (res[col] - res["actual_s"]).abs()
        if ours.isna().any():
            continue
        t = stats.ttest_rel(base, ours)
        w = stats.wilcoxon(base, ours) if len(res) >= 6 else None
        print(f"\n  {label}  회차당 평균 개선 {(base - ours).mean():+.0f}초 "
              f"({(1 - ours.mean() / base.mean()) * 100:.0f}% 감소)")
        print(f"    대응표본 t-검정 : t = {t.statistic:.2f}, p = {t.pvalue:.2e}"
              f"  {'✅ 유의' if t.pvalue < 0.05 else '❌ 유의수준 미달'}")
        if w:
            print(f"    Wilcoxon 부호순위: p = {w.pvalue:.2e}"
                  f"  {'✅ 유의' if w.pvalue < 0.05 else '❌ 유의수준 미달'}")
    print("\n  (표본이 작아 정규성 가정이 불확실하므로 두 검정을 병기한다)")


# ---------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--raw-dir", type=Path, default=Path("data/raw/validation"))
    ap.add_argument("--dem-dir", type=Path, default=Path("data/raw/dem"))
    ap.add_argument("--segments", type=Path, default=Path("data/processed/validation_segments.csv"))
    ap.add_argument("--stopwatch", type=Path, default=Path("data/processed/validation_stopwatch.csv"),
                    help="신호대기 실측 기록(선택). 없으면 공공데이터 기대대기를 사용")
    ap.add_argument("--out", type=Path, default=Path("data/processed/validation_results.csv"))
    ap.add_argument("--all-runs", action="store_true",
                    help="역방향 회차까지 포함해 16회 전체를 주분석으로 사용")
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

    fwd = res[res.방향 == "정방향"]
    rev = res[res.방향 == "역방향"]
    if len(rev) and not args.all_runs:
        print(f"\n{'*' * 74}")
        print(f"  주분석: 정방향 {len(fwd)}회 (수집 설계대로 걸은 회차)")
        print(f"  역방향 {len(rev)}회(박준서·홍민기 2회차 = 돌아오는 길)는 [5]에서 별도 분석")
        print(f"  전체 {len(res)}회 통합 결과는 --all-runs 로 확인")
        print("*" * 74)
        primary = fwd
    else:
        primary = res

    report_by_person(primary)
    report_ablation(primary)
    report_route_effect(res)      # 정/역 대비가 목적이라 항상 전체로
    report_tests(primary)
    report_roundtrip(res)

    if not args.dry_run:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        res.to_csv(args.out, index=False, encoding="utf-8-sig")
        print(f"\n저장: {args.out} (그래프용 원자료)")
    print()


if __name__ == "__main__":
    main()
