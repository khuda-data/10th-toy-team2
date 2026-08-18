"""speed_ratio 정규화 -> k_slope 다항회귀 -> 잔차 기반 v_user(Shrinkage) 산출.

설계 결정 (README가 정확한 수식까지 규정하진 않아 아래와 같이 확정함 — 팀 검토 후 조정 가능):
- "개인 평지속도"(정규화 분모)는 category=='평지' 구간들의 speed_mps 평균(실측 raw값)을 사용.
- k_slope 회귀: pooled speed_ratio ~ slope_pct + slope_pct^2 (2차 다항회귀).
  전원 데이터를 그대로 합치면 데이터를 많이 모은 사람(예: 홍민기) 쪽으로 회귀가 쏠리는
  문제가 있어서, **사람별로 동일한 개수(인원 중 최소 구간 수)만큼만 무작위 추출한
  "균형 표본"으로 이 곡선만 적합**한다. 잔차/v_user는 이후 각자의 전체 데이터로 계산해서
  버리지 않는다.
- v_user: 그 회귀의 잔차(전체 데이터 기준)를 사람별로 평균 낸 값에 shrinkage(w_i = n_i/(n_i+k))를
  적용해 0(전체 평균)과 개인 고유값 사이로 보간한 뒤, 평지속도(m/s) 스케일로 환산.
  구간 수가 많은 사람일수록 shrink_w가 커져 개인 고유 패턴이 더 많이 반영된다
  (데이터 수집량에 따른 "개인화 그라데이션"을 보여주는 지점).

사용법 (프로젝트 루트에서, segments.csv가 이미 있어야 함):
    python -m src.model_speed --segments data/processed/segments.csv
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def compute_flat_speed(segments: pd.DataFrame) -> pd.Series:
    """사람별 평지 구간 평균 속도(m/s). speed_ratio 정규화 분모로 사용."""
    flat = segments[segments["category"] == "평지"]
    return flat.groupby("person")["speed_mps"].mean().rename("v_flat_raw")


def add_speed_ratio(segments: pd.DataFrame, v_flat: pd.Series) -> pd.DataFrame:
    df = segments.merge(v_flat, on="person", how="left")
    df["speed_ratio"] = df["speed_mps"] / df["v_flat_raw"]
    return df


def sample_balanced_baseline(df: pd.DataFrame, cap: int | None = None, seed: int = 42) -> pd.DataFrame:
    """pooled k_slope 회귀용 인원별 균형 표본.

    사람별로 최대 cap개까지만 무작위 추출해서, 데이터를 많이 모은 사람의 개인
    패턴이 "공통 경향"인 것처럼 회귀에 과대 반영되는 것을 막는다. cap 미지정 시
    사람별 구간 수의 최솟값을 사용(전원이 동일한 가중치로 회귀에 기여).
    """
    valid = df.dropna(subset=["speed_ratio", "slope_pct"])
    counts = valid.groupby("person").size()
    if cap is None:
        cap = int(counts.min())
    parts = [g.sample(n=min(cap, len(g)), random_state=seed) for _, g in valid.groupby("person")]
    return pd.concat(parts, ignore_index=True)


def fit_k_slope(df: pd.DataFrame) -> np.ndarray:
    """pooled 2차 다항회귀: speed_ratio ~ slope_pct + slope_pct^2.
    반환값은 np.polyfit 순서(높은 차수 우선): [b2, b1, b0]
    """
    valid = df.dropna(subset=["speed_ratio", "slope_pct"])
    return np.polyfit(valid["slope_pct"], valid["speed_ratio"], deg=2)


def r_squared(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    return 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")


def add_residual(df: pd.DataFrame, coef: np.ndarray) -> pd.DataFrame:
    df = df.copy()
    df["predicted_ratio"] = np.polyval(coef, df["slope_pct"])
    df["residual"] = df["speed_ratio"] - df["predicted_ratio"]
    return df


def estimate_v_user(df: pd.DataFrame, v_flat: pd.Series, shrink_k: float | None = None) -> pd.DataFrame:
    """잔차의 사람별 평균 -> shrinkage(w_i = n_i/(n_i+k)) -> v_user(m/s)."""
    stats = df.groupby("person").agg(
        residual_mean=("residual", "mean"),
        n_segments=("residual", "size"),
    )
    if shrink_k is None:
        shrink_k = stats["n_segments"].median()
    stats["shrink_k"] = shrink_k
    stats["shrink_w"] = stats["n_segments"] / (stats["n_segments"] + shrink_k)
    stats["shrunk_residual"] = stats["shrink_w"] * stats["residual_mean"]
    stats = stats.join(v_flat)
    stats["v_user_mps"] = stats["v_flat_raw"] * (1 + stats["shrunk_residual"])
    return stats.reset_index()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--segments", type=Path, default=Path("data/processed/segments.csv"))
    parser.add_argument("--out", type=Path, default=Path("data/processed/v_user.csv"))
    parser.add_argument(
        "--k-slope-out", type=Path, default=Path("data/processed/k_slope_model.json"), help="k_slope 계수 저장 경로"
    )
    parser.add_argument("--shrink-k", type=float, default=None, help="미지정 시 사람별 구간수 중앙값을 사용")
    parser.add_argument(
        "--baseline-cap", type=int, default=None, help="k_slope 균형 표본 인원별 상한(미지정 시 최소 구간 수)"
    )
    parser.add_argument("--seed", type=int, default=42, help="균형 표본 무작위 추출 시드")
    args = parser.parse_args()

    segments = pd.read_csv(args.segments)

    print("[1/4] 개인 평지속도(정규화 분모) 계산")
    v_flat = compute_flat_speed(segments)
    print(v_flat.to_string())

    print("[2/4] speed_ratio 계산 및 k_slope 다항회귀 적합 (인원별 균형 표본)")
    df = add_speed_ratio(segments, v_flat)
    baseline = sample_balanced_baseline(df, cap=args.baseline_cap, seed=args.seed)
    cap_used = baseline.groupby("person").size()
    print(f"  균형 표본 인원별 구간 수(cap={cap_used.max()}):")
    print("  " + cap_used.to_string().replace("\n", "\n  "))

    coef = fit_k_slope(baseline)
    print(f"  k_slope coef [2차, 1차, 절편] = {coef}")

    r2_baseline = r_squared(baseline["speed_ratio"].to_numpy(), np.polyval(coef, baseline["slope_pct"].to_numpy()))
    valid_full = df.dropna(subset=["speed_ratio", "slope_pct"])
    r2_full = r_squared(valid_full["speed_ratio"].to_numpy(), np.polyval(coef, valid_full["slope_pct"].to_numpy()))
    print(f"  R^2 (균형 표본, n={len(baseline)}) = {r2_baseline:.4f}")
    print(f"  R^2 (전체 데이터, n={len(valid_full)}) = {r2_full:.4f}")

    print("[3/4] 잔차 계산(전체 데이터 기준) 및 사람별 v_user(Shrinkage) 추정")
    df = add_residual(df, coef)
    v_user = estimate_v_user(df, v_flat, shrink_k=args.shrink_k)
    print(v_user.to_string(index=False))

    print(f"[4/4] 저장: {args.out}, {args.k_slope_out}")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    v_user.to_csv(args.out, index=False, encoding="utf-8-sig")

    k_slope_model = {
        "description": "predicted_ratio(slope_pct) = coef[0]*slope_pct**2 + coef[1]*slope_pct + coef[2]",
        "coef": coef.tolist(),
        "baseline_cap": int(cap_used.max()),
        "baseline_n": len(baseline),
        "seed": args.seed,
        "r2_baseline": r2_baseline,
        "r2_full": r2_full,
    }
    args.k_slope_out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.k_slope_out, "w", encoding="utf-8") as f:
        json.dump(k_slope_model, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
