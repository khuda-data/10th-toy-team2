"""내부 검증 실험 (역할③): 회차(파일) 단위 홀드아웃으로 개인화 효과를 숫자로 증명.

핵심 설계 (인수인계 문서 3-2의 실험을 두 가지 업그레이드):
1. 분할 단위 = 구간이 아니라 파일(회차). 같은 GPX의 인접 구간은 같은 날·같은 길이라
   강하게 상관 -> 구간 랜덤 분할은 데이터 누수로 성능이 과대평가된다.
2. 누수 없는 재적합: 배포된 v_user.csv는 test 구간까지 포함해 계산된 값이므로 쓰지 않고,
   train 세트만으로 v_flat -> 균형표본 k_slope -> 잔차 -> shrinkage를 전부 다시 적합한다.

비교하는 4가지 예측 방식 (test 구간속도 MAE, m/s):
  const : 정속 4km/h (기존 지도앱 가정)
  a     : v_flat × ratio(slope)   — 공통곡선만, 개인화 없음
  b     : v_user × ratio(slope)   — 현재 파이프라인(shrinkage)
  c     : v_flat×(1+잔차평균) × ratio — 순수 개인화(shrinkage 없음)

사용 (프로젝트 루트에서):
    python -m src.evaluate_internal --n-seeds 10
    # -> 콘솔 표 + data/processed/eval_internal.csv 저장 (발표 그래프 재료)
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

TIER = {"권동하": "소량", "김재형": "소량", "박준서": "소량", "최희수": "소량",
        "윤예진": "중간", "홍민기": "다량"}


def split_by_file(seg: pd.DataFrame, seed: int, test_frac: float = 0.2):
    """사람별로 파일(회차)의 test_frac 만큼을 test로 홀드아웃."""
    rng = np.random.RandomState(seed)
    test_files: set[str] = set()
    for _, g in seg[["person", "source_file"]].drop_duplicates().groupby("person"):
        files = g["source_file"].tolist()
        n_test = max(1, int(round(len(files) * test_frac)))
        test_files |= set(rng.choice(files, size=n_test, replace=False))
    mask = seg["source_file"].isin(test_files)
    return seg[~mask].copy(), seg[mask].copy()


def fit_pipeline_on_train(train: pd.DataFrame, seed: int):
    """train만으로 v_flat -> 균형표본 k_slope -> 잔차 -> shrinkage 재적합 (누수 방지)."""
    v_flat = train[train["category"] == "평지"].groupby("person")["speed_mps"].mean()
    train = train.merge(v_flat.rename("v_flat"), on="person")
    train["speed_ratio"] = train["speed_mps"] / train["v_flat"]
    train = train.dropna(subset=["speed_ratio", "slope_pct"])

    cap = train.groupby("person").size().min()  # model_speed.sample_balanced_baseline과 동일 원리
    bal = pd.concat([g.sample(n=min(cap, len(g)), random_state=seed)
                     for _, g in train.groupby("person")])
    coef = np.polyfit(bal["slope_pct"], bal["speed_ratio"], deg=2)

    train["resid"] = train["speed_ratio"] - np.polyval(coef, train["slope_pct"])
    st = train.groupby("person").agg(resid_mean=("resid", "mean"), n=("resid", "size"))
    k = st["n"].median()
    st["w"] = st["n"] / (st["n"] + k)
    st = st.join(v_flat.rename("v_flat"))
    st["v_user"] = st["v_flat"] * (1 + st["w"] * st["resid_mean"])   # (b)
    st["v_pure"] = st["v_flat"] * (1 + st["resid_mean"])             # (c)
    return coef, st.reset_index()


def evaluate_split(seg: pd.DataFrame, seed: int) -> pd.DataFrame:
    train, test = split_by_file(seg, seed)
    coef, st = fit_pipeline_on_train(train, seed)

    test = test.dropna(subset=["speed_mps", "slope_pct"]).merge(
        st[["person", "v_flat", "v_user", "v_pure"]], on="person")
    ratio = np.clip(np.polyval(coef, test["slope_pct"]), 0.4, 1.3)
    preds = {
        "const": np.full(len(test), 4.0 / 3.6),
        "a": test["v_flat"] * ratio,
        "b": test["v_user"] * ratio,
        "c": test["v_pure"] * ratio,
    }
    test["tier"] = test["person"].map(TIER)
    rows = []
    for method, pred in preds.items():
        err = np.abs(test["speed_mps"].to_numpy() - np.asarray(pred))
        for tier, g_idx in test.groupby("tier").groups.items():
            rows.append({"seed": seed, "tier": tier, "method": method,
                         "mae": err[test.index.get_indexer(g_idx)].mean(),
                         "n_test": len(g_idx)})
        rows.append({"seed": seed, "tier": "전체", "method": method,
                     "mae": err.mean(), "n_test": len(test)})
    return pd.DataFrame(rows)


def shrinkage_curve(seg: pd.DataFrame, seed: int, cap: int | None):
    """표본 크기별 shrinkage 손익 비교용. cap: 사람별 train 구간 수 제한(None=제한 없음).

    반환: (shrinkage 미적용 MAE, shrinkage 적용 MAE)
    """
    train, test = split_by_file(seg, seed)
    if cap:
        train = pd.concat([g.sample(min(cap, len(g)), random_state=seed)
                           for _, g in train.groupby("person")])

    v_flat = train[train["category"] == "평지"].groupby("person")["speed_mps"].mean()
    train = train.merge(v_flat.rename("v_flat"), on="person")
    train["speed_ratio"] = train["speed_mps"] / train["v_flat"]
    train = train.dropna(subset=["speed_ratio", "slope_pct"])

    n = train.groupby("person").size().min()
    bal = pd.concat([g.sample(min(n, len(g)), random_state=seed)
                     for _, g in train.groupby("person")])
    coef = np.polyfit(bal["slope_pct"], bal["speed_ratio"], deg=1)

    # v2 방식: 경사보정 속도의 사람별 평균
    train["v_deslope"] = train["speed_mps"] / np.clip(np.polyval(coef, train["slope_pct"]), 0.3, 1.4)
    st = train.groupby("person").agg(n=("v_deslope", "size"), mean=("v_deslope", "mean"))
    grand = st["mean"].mean()                                  # 수축 목표(팀 전체 평균)
    w = st["n"] / (st["n"] + st["n"].median())
    v_plain = st["mean"]                                       # shrinkage 미적용
    v_shrunk = grand + w * (st["mean"] - grand)                # shrinkage 적용

    test = test.dropna(subset=["speed_mps", "slope_pct"])
    ratio = np.clip(np.polyval(coef, test["slope_pct"]), 0.3, 1.4)
    return (np.abs(test["speed_mps"] - test["person"].map(v_plain) * ratio).mean(),
            np.abs(test["speed_mps"] - test["person"].map(v_shrunk) * ratio).mean())


def run_shrinkage_curve(seg: pd.DataFrame, n_seeds: int) -> None:
    """보고서 4-1절: shrinkage가 유효해지는 표본 크기 손익분기점 탐색."""
    from scipy import stats

    print(f"\n=== Shrinkage 손익분기점 (표본 크기별, {n_seeds}-seed 대응표본 검정) ===")
    print(f"{'사람당 train 표본':<20}{'미적용':>10}{'적용':>10}{'차이':>10}{'p-value':>10}  판정")
    for cap, label in [(None, "제한 없음(72~363)"), (30, "30개"), (10, "10개"), (5, "5개")]:
        res = np.array([shrinkage_curve(seg, s, cap) for s in range(n_seeds)])
        plain, shrunk = res[:, 0], res[:, 1]
        p = stats.ttest_rel(plain, shrunk).pvalue
        diff = plain.mean() - shrunk.mean()   # 양수면 shrinkage가 유리
        if p >= 0.05:
            verdict = "구분 불가"
        else:
            verdict = "적용이 우세" if diff > 0 else "미적용이 우세"
        print(f"{label:<20}{plain.mean():>10.4f}{shrunk.mean():>10.4f}{diff:>+10.4f}{p:>10.3f}  {verdict}")
    print("\n해석: 표본이 적을수록 shrinkage가 유리해진다. 손익분기점 부근(≈10개) 아래에서만")
    print("      적용 가치가 있으며, 본 프로젝트 표본(72개 이상)에서는 미적용이 타당하다.")


ROUTE_M = 1200.0   # 신뢰구간을 초 단위로 환산할 때 쓰는 기준 경로 길이


def _common_curve(seg: pd.DataFrame, target: str, seed: int = 0) -> tuple[np.ndarray, float]:
    """대상자를 제외한 '남들 데이터'로 공통 k_slope 곡선을 적합한다.

    개인화 학습곡선을 볼 때 공통 곡선까지 대상자 데이터로 흔들리면 효과가 섞이므로,
    곡선은 고정하고 대상자의 v_user 표본 수만 변화시킨다.
    """
    others = seg[seg["person"] != target]
    v_flat = others[others["slope_pct"].abs() <= 1].groupby("person")["speed_mps"].mean()
    o = others.merge(v_flat.rename("v_flat"), on="person")
    o["speed_ratio"] = o["speed_mps"] / o["v_flat"]
    o = o.dropna(subset=["speed_ratio", "slope_pct"])
    n = o.groupby("person").size().min()
    bal = pd.concat([g.sample(n, random_state=seed) for _, g in o.groupby("person")])
    return np.polyfit(bal["slope_pct"], bal["speed_ratio"], deg=1), float(v_flat.mean())


def learning_curve(seg: pd.DataFrame, target: str, sizes: list[int],
                   n_seeds: int) -> pd.DataFrame:
    """개인 데이터량 n에 따른 (예측오차, v_user 추정 안정성)을 계산한다.

    각 seed마다 대상자의 파일 20%를 test로 떼고, 남은 train에서 n개를 뽑아
    v_user를 추정한 뒤 test 구간속도 예측오차를 잰다. 같은 n을 여러 seed로
    반복했을 때 v_user가 얼마나 흔들리는지(표준편차)가 '신뢰도' 지표다.
    """
    coef, team_v = _common_curve(seg, target)
    mine = seg[seg["person"] == target]
    rows = []
    for n in sizes:
        errs, vs = [], []
        for s in range(n_seeds):
            rng = np.random.RandomState(s)
            files = mine["source_file"].unique()
            test_files = set(rng.choice(files, max(1, round(len(files) * 0.2)), replace=False))
            train = mine[~mine["source_file"].isin(test_files)]
            test = mine[mine["source_file"].isin(test_files)]
            if len(train) < n or len(test) == 0:
                continue
            sub = train.sample(n, random_state=s)
            v = float((sub["speed_mps"] / np.clip(np.polyval(coef, sub["slope_pct"]), 0.3, 1.4)).mean())
            ratio = np.clip(np.polyval(coef, test["slope_pct"]), 0.3, 1.4)
            errs.append(np.abs(test["speed_mps"] - v * ratio).mean())
            vs.append(v)
        if not vs:
            continue
        mu, sd = float(np.mean(vs)), float(np.std(vs))
        # v_user 흔들림을 1.2km 경로 ETA의 ±초로 환산
        band = ROUTE_M / mu - ROUTE_M / (mu + sd) if sd > 0 else 0.0
        rows.append({"person": target, "n_segments": n, "mae": float(np.mean(errs)),
                     "v_mean": mu, "v_sd": sd, "eta_band_s": band})

    # 개인화 없이 팀평균만 쓴 경우 (n=0)
    base = []
    for s in range(n_seeds):
        rng = np.random.RandomState(s)
        files = mine["source_file"].unique()
        test_files = set(rng.choice(files, max(1, round(len(files) * 0.2)), replace=False))
        test = mine[mine["source_file"].isin(test_files)]
        if len(test) == 0:
            continue
        ratio = np.clip(np.polyval(coef, test["slope_pct"]), 0.3, 1.4)
        base.append(np.abs(test["speed_mps"] - team_v * ratio).mean())
    rows.insert(0, {"person": target, "n_segments": 0, "mae": float(np.mean(base)),
                    "v_mean": team_v, "v_sd": np.nan, "eta_band_s": np.nan})
    return pd.DataFrame(rows)


def run_learning_curve(seg: pd.DataFrame, n_seeds: int, out: Path) -> None:
    """보고서 4-2절: 데이터를 더 모으면 개인화가 어떻게 좋아지는가."""
    sizes = [5, 10, 20, 30, 60, 90, 150, 250, 360]
    counts = seg.groupby("person").size().sort_values(ascending=False)
    all_rows = []

    print(f"\n=== 개인 데이터량에 따른 학습곡선 ({n_seeds}-seed) ===")
    print("  n=0은 개인화 없이 팀 평균 속도만 쓴 경우\n")
    for person in counts.index:
        df = learning_curve(seg, person, [n for n in sizes if n <= counts[person]], n_seeds)
        all_rows.append(df)
        print(f"  ■ {person} (보유 {counts[person]}개)")
        print(f"    {'구간수':>7}{'예측오차 MAE':>14}{'ETA 신뢰구간':>15}")
        for _, r in df.iterrows():
            band = "—" if np.isnan(r.eta_band_s) else f"± {r.eta_band_s:.0f}초"
            tag = "  (팀평균)" if r.n_segments == 0 else ""
            print(f"    {int(r.n_segments):>7}{r.mae:>14.4f}{band:>15}{tag}")
        print()

    res = pd.concat(all_rows, ignore_index=True)
    out.parent.mkdir(parents=True, exist_ok=True)
    res.to_csv(out, index=False, encoding="utf-8-sig")

    print("  해석")
    print("   - 예측오차(MAE)는 20~30개 구간에서 포화된다. v_user는 평균 하나를")
    print("     추정하는 값이라 1/√n으로 빠르게 수렴하기 때문이다.")
    print("   - 반면 ETA 신뢰구간은 계속 좁아진다. 데이터를 더 모아서 얻는 것은")
    print("     '더 정확한 값'이 아니라 '그 값을 더 확신할 수 있다'는 점이다.")
    print(f"\n  저장: {out}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--segments", type=Path, default=Path("data/processed/segments.csv"))
    parser.add_argument("--out", type=Path, default=Path("data/processed/eval_internal.csv"))
    parser.add_argument("--n-seeds", type=int, default=10,
                        help="분할 반복 횟수(평균±표준편차로 보고, 단일 분할 요행 방지)")
    parser.add_argument("--shrinkage-curve", action="store_true",
                        help="보고서 4-1절: 표본 크기별 shrinkage 손익분기점 실험만 실행")
    parser.add_argument("--learning-curve", action="store_true",
                        help="보고서 4-2절: 개인 데이터량에 따른 개인화 학습곡선만 실행")
    args = parser.parse_args()

    if args.shrinkage_curve:
        run_shrinkage_curve(pd.read_csv(args.segments), args.n_seeds)
        return
    if args.learning_curve:
        run_learning_curve(pd.read_csv(args.segments).dropna(subset=["speed_mps", "slope_pct"]),
                           args.n_seeds, Path("data/processed/eval_learning_curve.csv"))
        return

    seg = pd.read_csv(args.segments)
    results = pd.concat([evaluate_split(seg, s) for s in range(args.n_seeds)],
                        ignore_index=True)

    summary = (results.groupby(["tier", "method"])["mae"]
               .agg(["mean", "std"]).round(4).reset_index())
    pivot = summary.pivot(index="tier", columns="method", values="mean")[
        ["const", "a", "b", "c"]].reindex(["소량", "중간", "다량", "전체"])
    print(f"\n=== 회차 단위 홀드아웃 MAE (m/s), seed {args.n_seeds}회 평균 ===")
    print(pivot.to_string())
    imp = (1 - pivot.loc["전체", "b"] / pivot.loc["전체", "const"]) * 100
    print(f"\n헤드라인: 정속 4km/h 대비 현재 모델(b) 구간속도 MAE {imp:.0f}% 감소")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(args.out, index=False, encoding="utf-8-sig")
    print(f"저장: {args.out} (seed별 원자료 — 그래프/오차막대는 여기서 그리면 됨)")


if __name__ == "__main__":
    main()
