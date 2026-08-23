"""검증 결과 시각화 3장.

  fig_검증_사람별오차.png    사람별 우리모델 vs 네이버 오차 (개인화가 누구에게 이득인가)
  fig_검증_요인기여도.png    M0->M4 폭포수 (무엇이 오차를 줄였나)
  fig_검증_캘리브레이션.png  학습 v_user vs 검증 실측 속도 (도메인 시프트와 그 해소)
  fig_검증_왕복대칭.png      역방향 4회 — k_slope가 방향 비대칭을 얼마나 설명하나

주분석은 정방향 12회다(박준서·홍민기 2회차는 돌아오는 길이라 역방향).
[1][2][3]은 정방향만, 왕복 그림만 역방향을 쓴다.

사용법: python -m src.plot_validation
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.rcParams["font.family"] = "AppleGothic"
plt.rcParams["axes.unicode_minus"] = False

C_BASE, C_COLD, C_WARM = "#B0B7C3", "#F2A65A", "#2E7DD1"
OUT = Path("figures")


def fig_person(res: pd.DataFrame) -> None:
    people = sorted(res.person.unique())
    base = [(res[res.person == p].M0_정속 - res[res.person == p].actual_s).abs().mean() for p in people]
    cold = [(res[res.person == p].M3_신호 - res[res.person == p].actual_s).abs().mean() for p in people]
    warm = [(res[res.person == p].M4_캘리브 - res[res.person == p].actual_s).abs().mean() for p in people]

    x = np.arange(len(people)); w = 0.27
    fig, ax = plt.subplots(figsize=(9, 5))
    for off, v, c, lab in [(-w, base, C_BASE, "네이버 정속 4km/h"),
                           (0, cold, C_COLD, "우리 모델 (콜드스타트)"),
                           (w, warm, C_WARM, "우리 모델 (웜스타트)")]:
        b = ax.bar(x + off, v, w, color=c, label=lab)
        ax.bar_label(b, fmt="%.0f", fontsize=9, padding=2)
    ax.set_xticks(x, people)
    ax.set_ylabel("회차 평균 절대오차 (초)")
    ax.set_title(f"사람별 ETA 예측 오차 — 검증 정방향 {len(res)}회차", fontsize=13, pad=12)
    ax.legend(frameon=False, ncols=3, loc="upper center", bbox_to_anchor=(0.5, -0.08))
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUT / "fig_검증_사람별오차.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def fig_waterfall(res: pd.DataFrame) -> None:
    stages = ["M0_정속", "M1_경사", "M2_개인화", "M3_신호", "M4_캘리브"]
    labels = ["M0\n정속 4km/h", "M1\n+경사", "M2\n+개인화", "M3\n+신호", "M4\n+개인 캘리브"]
    mae = [(res[s] - res.actual_s).abs().mean() for s in stages]

    fig, ax = plt.subplots(figsize=(9, 5))
    colors = [C_BASE, C_BASE, C_BASE, C_COLD, C_WARM]
    b = ax.bar(labels, mae, color=colors, width=0.6)
    ax.bar_label(b, fmt="%.0f초", fontsize=10, padding=3)
    for i in range(1, len(mae)):
        d = mae[i] - mae[i - 1]
        ax.annotate(f"{d:+.0f}초", xy=(i - 0.5, max(mae[i], mae[i - 1]) + 12),
                    ha="center", fontsize=9, color="#C0392B" if d > 0 else "#1E8449")
    ax.set_ylabel("MAE (초)")
    ax.set_ylim(0, max(mae) * 1.25)
    ax.set_title(f"요인별 기여도 — 무엇이 오차를 줄였나 (정방향 n={len(res)}회차)", fontsize=13, pad=12)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUT / "fig_검증_요인기여도.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def fig_calibration(res: pd.DataFrame) -> None:
    vu = pd.read_csv("models/v_user_v2.csv")
    vu = dict(zip(vu.person, vu.v_user_mps))
    people = sorted(res.person.unique())
    trained = [vu[p] for p in people]
    observed = [res[res.person == p].v_hat.mean() for p in people]

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(12, 5), gridspec_kw={"width_ratios": [1.1, 1]})

    x = np.arange(len(people)); w = 0.34
    b1 = ax.bar(x - w / 2, trained, w, color=C_COLD, label="학습 v_user (경사 실험 보행)")
    b2 = ax.bar(x + w / 2, observed, w, color=C_WARM, label="검증 실측 (연속 목적지 보행)")
    ax.bar_label(b1, fmt="%.2f", fontsize=9); ax.bar_label(b2, fmt="%.2f", fontsize=9)
    for i, p in enumerate(people):
        d = (observed[i] / trained[i] - 1) * 100
        ax.annotate(f"{d:+.0f}%", xy=(i, max(trained[i], observed[i]) + 0.19),
                    ha="center", fontsize=11,
                    color="#C0392B" if abs(d) > 10 else "#555")
    ax.axhline(1000 / 900, ls="--", lw=1, color="#888")
    ax.annotate("네이버 가정 1.111", xy=(len(people) - 0.4, 1000 / 900 + 0.03),
                ha="right", fontsize=8, color="#888")
    ax.set_xticks(x, people); ax.set_ylabel("평지환산 보행속도 (m/s)")
    ax.set_ylim(0, max(observed) * 1.42)
    ax.set_title("학습–검증 도메인 시프트", fontsize=12, pad=10)
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    ax.spines[["top", "right"]].set_visible(False); ax.grid(axis="y", alpha=0.25)

    shift = [(observed[i] / trained[i] - 1) * 100 for i in range(len(people))]
    cold = [(res[res.person == p].M3_신호 - res[res.person == p].actual_s).abs().mean() for p in people]
    warm = [(res[res.person == p].M4_캘리브 - res[res.person == p].actual_s).abs().mean() for p in people]
    ax2.scatter(np.abs(shift), cold, s=110, color=C_COLD, label="콜드스타트 (학습 v_user)", zorder=3)
    ax2.scatter(np.abs(shift), warm, s=110, color=C_WARM, label="웜스타트 (회차 LOO)", zorder=3)
    for i, p in enumerate(people):
        ax2.annotate(p, (abs(shift[i]), cold[i]), textcoords="offset points",
                     xytext=(8, 4), fontsize=9)
        ax2.plot([abs(shift[i])] * 2, [cold[i], warm[i]], color="#CCC", lw=1, zorder=1)
    ax2.set_xlabel("학습 대비 검증 속도 괴리 (%, 절댓값)")
    ax2.set_ylabel("회차 평균 절대오차 (초)")
    ax2.set_title("괴리가 클수록 콜드스타트가 무너진다", fontsize=12, pad=10)
    ax2.legend(frameon=False, fontsize=9)
    ax2.spines[["top", "right"]].set_visible(False); ax2.grid(alpha=0.25)

    fig.tight_layout()
    fig.savefig(OUT / "fig_검증_캘리브레이션.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def fig_roundtrip(res: pd.DataFrame) -> None:
    """역방향 회차가 있는 사람만: 왕복 속도차를 경사 보정이 얼마나 지우나."""
    rows = []
    for (person, route), g in res.groupby(["person", "route"]):
        if set(g["방향"]) != {"정방향", "역방향"}:
            continue
        f = g[g.방향 == "정방향"].iloc[0]; b = g[g.방향 == "역방향"].iloc[0]
        rows.append({"lab": f"{person}\n{route}",
                     "보정전": abs(b.생속도 / f.생속도 - 1) * 100,
                     "보정후": abs(b.v_hat / f.v_hat - 1) * 100})
    if not rows:
        return
    P = pd.DataFrame(rows)
    x = np.arange(len(P)); w = 0.34
    fig, ax = plt.subplots(figsize=(8, 5))
    b1 = ax.bar(x - w / 2, P.보정전, w, color=C_BASE, label="원래 왕복 속도차")
    b2 = ax.bar(x + w / 2, P.보정후, w, color=C_WARM, label="k_slope 경사보정 후 남은 차")
    ax.bar_label(b1, fmt="%.1f%%", fontsize=9); ax.bar_label(b2, fmt="%.1f%%", fontsize=9)
    ax.set_xticks(x, P.lab, fontsize=10)
    ax.set_ylabel("같은 길 왕복 시 보행속도 차이 (%)")
    ax.set_ylim(0, P.보정전.max() * 1.3)
    share = (1 - P.보정후.mean() / P.보정전.mean()) * 100
    ax.set_title(f"왕복 대칭 — 경사 모델이 방향 비대칭의 {share:.0f}%만 설명한다",
                 fontsize=13, pad=12)
    ax.legend(frameon=False, fontsize=10)
    ax.spines[["top", "right"]].set_visible(False); ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUT / "fig_검증_왕복대칭.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    OUT.mkdir(exist_ok=True)
    res = pd.read_csv("data/processed/validation_results.csv")
    fwd = res[res.방향 == "정방향"] if "방향" in res else res
    fig_person(fwd); fig_waterfall(fwd); fig_calibration(fwd); fig_roundtrip(res)
    print(f"주분석 정방향 {len(fwd)}회 / 왕복 그림은 역방향 포함 {len(res)}회")
    print("저장 완료:")
    for f in ["fig_검증_사람별오차.png", "fig_검증_요인기여도.png",
              "fig_검증_캘리브레이션.png", "fig_검증_왕복대칭.png"]:
        print(f"  figures/{f}")


if __name__ == "__main__":
    main()
