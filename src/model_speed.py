"""speed_ratio 정규화 -> k_slope 다항회귀 -> 잔차 기반 v_user(Shrinkage) 산출.

설계 결정 (README가 정확한 수식까지 규정하진 않아 아래와 같이 확정함 — 팀 검토 후 조정 가능):
- "개인 평지속도"(정규화 분모)는 category=='평지' 구간들의 speed_mps 평균(실측 raw값)을 사용.
- k_slope 회귀: pooled speed_ratio ~ slope_pct + slope_pct^2 (2차 다항회귀, 전원 데이터를 합쳐서 1개 모델).
- v_user: 그 회귀의 잔차를 사람별로 평균 낸 값에 shrinkage(w_i = n_i/(n_i+k))를 적용해
  0(전체 평균)과 개인 고유값 사이로 보간한 뒤, 평지속도(m/s) 스케일로 환산.

사용법 (프로젝트 루트에서, segments.csv가 이미 있어야 함):
    python -m src.model_speed --segments data/processed/segments.csv
"""
from __future__ import annotations

import argparse
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
    parser.add_argument("--shrink-k", type=float, default=None, help="미지정 시 사람별 구간수 중앙값을 사용")
    args = parser.parse_args()

    segments = pd.read_csv(args.segments)

    print("[1/4] 개인 평지속도(정규화 분모) 계산")
    v_flat = compute_flat_speed(segments)
    print(v_flat.to_string())

    print("[2/4] speed_ratio 계산 및 k_slope 다항회귀 적합")
    df = add_speed_ratio(segments, v_flat)
    coef = fit_k_slope(df)
    valid = df.dropna(subset=["speed_ratio", "slope_pct"])
    r2 = r_squared(valid["speed_ratio"].to_numpy(), np.polyval(coef, valid["slope_pct"].to_numpy()))
    print(f"  k_slope coef [2차, 1차, 절편] = {coef}")
    print(f"  R^2 = {r2:.4f}")

    print("[3/4] 잔차 계산 및 사람별 v_user(Shrinkage) 추정")
    df = add_residual(df, coef)
    v_user = estimate_v_user(df, v_flat, shrink_k=args.shrink_k)
    print(v_user.to_string(index=False))

    print(f"[4/4] 저장: {args.out}")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    v_user.to_csv(args.out, index=False, encoding="utf-8-sig")


if __name__ == "__main__":
    main()
