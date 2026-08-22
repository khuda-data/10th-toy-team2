"""최종 ETA 통합 엔진.

속도 모델(k_slope, v_user)과 신호대기 모델 산출물을 조립해
"경로 + 사람 -> 예상 소요시간(초)"를 계산한다.

인터페이스 계약 (팀원 산출물이 갱신되면 파일만 교체하면 됨):
- 속도 모델: data/processed/k_slope_model.json  {"coef": [b2, b1, b0]}  (np.polyfit 순서)
          data/processed/v_user.csv         person, v_user_mps 컬럼 필수
- 신호 모델: data/processed/signal_model.json   {"alpha": float, "beta": float}
          data/processed/signals.csv         crossing_id, lat, lon, cycle_C_s, red_R_s
          (아직 없으면 신호대기 0초로 동작 — 엔진은 오늘부터 사용 가능)

사용 예:
    from src.eta import EtaEngine
    engine = EtaEngine.load()
    eta_s = engine.route_eta(segments_df, person="권동하")
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

# 다항식은 학습 범위 밖에서 발산하므로 배율을 물리적 범위로 제한 (v2 문서 규약: 0.3~1.4)
RATIO_CLIP = (0.3, 1.4)
DEFAULT_WALK_SPEED_KMH = 4.0  # 비교 기준: 기존 지도앱의 정속 가정


class EtaEngine:
    def __init__(self, coef: list[float], v_user: dict[str, float],
                 signal_model: dict | None = None, signals: pd.DataFrame | None = None,
                 slope_range: tuple[float, float] | None = None):
        self.coef = np.asarray(coef, dtype=float)
        self.v_user = v_user
        self.signal_model = signal_model  # {"alpha":..., "beta":...} or None
        self.signals = signals            # 횡단보도 테이블 or None
        self.slope_range = slope_range    # v2: 측정 범위 밖 경사도는 경계값으로 클립(외삽 방지)

    # ---------- 로딩 ----------
    @classmethod
    def load(cls, root: str | Path = ".") -> "EtaEngine":
        """v2 산출물(models/)을 우선 사용, 없으면 v1(data/processed/) fallback.

        v2: 1차식 + valid_slope_range 클립 + 경사보정 v_user.
        """
        root = Path(root)
        v2_model = root / "models/k_slope_model_v2.json"
        v2_vuser = root / "models/v_user_v2.csv"
        if v2_model.exists() and v2_vuser.exists():
            mj = json.load(open(v2_model, encoding="utf-8"))
            coef, slope_range = mj["coef"], tuple(mj["valid_slope_range"])
            vu = pd.read_csv(v2_vuser)
            print("[eta] v2 모델 사용 (models/k_slope_model_v2.json, 1차식)")
        else:
            mj = json.load(open(root / "data/processed/k_slope_model.json", encoding="utf-8"))
            coef, slope_range = mj["coef"], None
            vu = pd.read_csv(root / "data/processed/v_user.csv")
            print("[eta] v1 모델 사용 (data/processed/, 2차식)")
        v_user = dict(zip(vu["person"], vu["v_user_mps"]))

        d = root / "data/processed"
        signal_model, signals = None, None
        if (root / "models/yeongtong_signal_params.json").exists():
            signal_model = json.load(open(root / "models/yeongtong_signal_params.json", encoding="utf-8"))
        elif (d / "signal_model.json").exists():
            signal_model = json.load(open(d / "signal_model.json", encoding="utf-8"))

        # 신호대기 산출물: 횡단보도별 기대대기가 이미 계산되어 있음
        sig_path = d / "영통구_횡단보도별_신호대기.csv"
        if sig_path.exists():
            raw = pd.read_csv(sig_path)
            signals = raw.rename(columns={"횡단보도관리번호": "crossing_id", "위도": "lat",
                                          "경도": "lon", "기대대기_s": "wait_s"})[
                ["crossing_id", "lat", "lon", "wait_s"]]
            print(f"[eta] 신호대기 데이터 {len(signals)}개 지점 로드 (영통구)")
        elif (d / "signals.csv").exists():
            signals = pd.read_csv(d / "signals.csv")
        return cls(coef, v_user, signal_model, signals, slope_range=slope_range)

    # ---------- 구성 요소 ----------
    def predicted_ratio(self, slope_pct) -> np.ndarray:
        """경사도(%) -> 평지 대비 속도 배율 (공통 곡선).
        v2 규약: 경사도를 측정 범위로 먼저 클립한 뒤 배율도 [0.3, 1.4]로 제한."""
        s = np.asarray(slope_pct, dtype=float)
        if self.slope_range is not None:
            s = np.clip(s, *self.slope_range)
        return np.clip(np.polyval(self.coef, s), *RATIO_CLIP)

    def expected_wait_s(self, cycle_C_s: float, red_R_s: float) -> float:
        """이론공식 E(Wait)=R^2/(2C)에 보정회귀(alpha+beta*이론값)를 적용."""
        theo = red_R_s ** 2 / (2 * cycle_C_s)
        if self.signal_model is None:
            return theo
        return self.signal_model["alpha"] + self.signal_model["beta"] * theo

    def route_signal_wait_s(self, segments: pd.DataFrame, radius_m: float = 25.0,
                            detail: bool = False):
        """경로가 횡단보도 반경 radius_m 내를 지나면 그 지점의 기대대기를 합산.

        신호대기 산출물(`영통구_횡단보도별_신호대기.csv`)은 지점별 기대대기(`wait_s`)를
        이미 계산해 두었으므로 그 값을 그대로 사용한다. 신호 데이터가 없거나 경로에
        좌표가 없으면 0초(= 신호 미반영)로 동작한다.
        """
        if self.signals is None or len(self.signals) == 0 or "end_lat" not in segments:
            return (0.0, []) if detail else 0.0

        pts = segments[["end_lat", "end_lon"]].to_numpy()
        total, hits = 0.0, []
        for _, s in self.signals.iterrows():
            if "wait_s" in s:
                wait = float(s["wait_s"])
            else:  # 원래 계약 형식(cycle_C_s, red_R_s)일 때
                wait = self.expected_wait_s(s["cycle_C_s"], s["red_R_s"])
            if wait <= 0:
                continue
            d = _haversine_m(pts[:, 0], pts[:, 1], s["lat"], s["lon"])
            if (d < radius_m).any():
                total += wait
                hits.append((s.get("crossing_id", "?"), wait))
        return (total, hits) if detail else total

    # ---------- 최종 ETA ----------
    def route_eta(self, segments: pd.DataFrame, person: str,
                  include_signals: bool = True) -> float:
        """segments: dist_m, slope_pct 컬럼 필수 (segmentation.build_segments 출력 형식).
        person이 v_user에 없으면(신규 사용자) 팀 평균 속도로 fallback."""
        v = self.v_user.get(person, float(np.mean(list(self.v_user.values()))))
        ratio = self.predicted_ratio(segments["slope_pct"].fillna(0.0))
        moving_s = float((segments["dist_m"].to_numpy() / (v * ratio)).sum())
        wait_s = self.route_signal_wait_s(segments) if include_signals else 0.0
        return moving_s + wait_s

    @staticmethod
    def route_eta_baseline(segments: pd.DataFrame,
                           speed_kmh: float = DEFAULT_WALK_SPEED_KMH) -> float:
        """비교 대상: 기존 지도앱 방식(정속·경사·신호 무시)."""
        return float(segments["dist_m"].sum() / (speed_kmh / 3.6))


def _haversine_m(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, (np.asarray(lat1, float), np.asarray(lon1, float),
                                              float(lat2), float(lon2)))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 2 * 6371000.0 * np.arcsin(np.sqrt(a))


if __name__ == "__main__":
    # 스모크 테스트: 실제 데이터의 GPX 한 회차를 "경로"로 간주해 ETA를 계산해본다
    seg = pd.read_csv("data/processed/segments.csv")
    engine = EtaEngine.load()
    one = seg[seg["source_file"] == seg["source_file"].iloc[0]]
    person = one["person"].iloc[0]
    actual = one["dt_s"].sum()  # 정지 포함 실제 경과시간(초) = ETA의 정답 기준
    pred = engine.route_eta(one, person)
    base = engine.route_eta_baseline(one)
    print(f"경로: {one['source_file'].iloc[0]} ({one['dist_m'].sum():.0f}m, {len(one)}개 구간)")
    print(f"실측 {actual:.0f}s | 우리 모델 {pred:.0f}s (오차 {abs(pred-actual):.0f}s) "
          f"| 정속 4km/h {base:.0f}s (오차 {abs(base-actual):.0f}s)")
