"""최종 ETA 통합 엔진.

속도 모델(k_slope, v_user)과 신호대기 모델 산출물을 조립해
"경로 + 사람 -> 예상 소요시간(초)"를 계산한다.

인터페이스 계약 (팀원 산출물이 갱신되면 파일만 교체하면 됨):
- 속도 모델: data/processed/k_slope_model.json  {"coef": [b2, b1, b0]}  (np.polyfit 순서)
          data/processed/v_user.csv         person, v_user_mps 컬럼 필수
- 신호 모델: data/processed/영통구_횡단보도별_신호대기.csv (한글 스키마, 우선)
          또는 data/processed/signals.csv  crossing_id, lat, lon, cycle_C_s, red_R_s
          (아직 없으면 신호대기 0초로 동작 — 엔진은 오늘부터 사용 가능)
          경로가 어느 횡단보도를 건넜는지는 route_wait.crossed() 가 판정한다.

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

try:                      # 패키지로 import 될 때(src.eta)와 직접 실행 둘 다 지원
    from . import route_wait
except ImportError:
    import route_wait

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

        # 신호대기 산출물: 횡단보도별 기대대기가 이미 계산되어 있음.
        # 한글 스키마 그대로 둔다 — 통과 판정을 route_wait 에 위임하는데
        # 거기서 횡단거리_m 로 임계값을 정하므로 컬럼을 버리면 안 된다.
        # 수원시 전체 + 신호등 보충본(signal_suwon.py)이 있으면 그쪽을 쓴다.
        # 영통구본은 검증 경로에서 확정 9곳 중 3곳이 아예 없었다.
        suwon = d / "수원시_횡단보도별_신호대기.csv"
        sig_path = suwon if suwon.exists() else d / "영통구_횡단보도별_신호대기.csv"
        if sig_path.exists():
            signals = pd.read_csv(sig_path, encoding="utf-8-sig")
            print(f"[eta] 신호대기 데이터 {len(signals)}개 지점 로드 ({sig_path.stem.split('_')[0]})")
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
        """이론공식 E(Wait)=R^2/(2C)에 보정회귀(alpha+beta*이론값)를 적용.

        alpha/beta 는 검증 걷기 후에나 정해진다. yeongtong_signal_params.json
        에는 없으므로 기본값(보정 없음)으로 동작해야 한다.
        """
        theo = red_R_s ** 2 / (2 * cycle_C_s)
        if self.signal_model is None:
            return theo
        return (self.signal_model.get("alpha", 0.0)
                + self.signal_model.get("beta", 1.0) * theo)

    def _route_track(self, segments: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """구간 테이블 → 경로 폴리라인. 첫 구간의 시작점부터 이어 붙인다."""
        lat = segments["end_lat"].to_numpy(float)
        lon = segments["end_lon"].to_numpy(float)
        if "start_lat" in segments:
            lat = np.concatenate([segments["start_lat"].to_numpy(float)[:1], lat])
            lon = np.concatenate([segments["start_lon"].to_numpy(float)[:1], lon])
        return lat, lon

    def route_signal_wait_s(self, segments: pd.DataFrame, detail: bool = False):
        """경로가 실제로 건넌 횡단보도의 기대대기를 합산.

        통과 판정은 `route_wait.crossed()` 에 위임한다. 고정 반경으로 잡으면
        세 가지가 어긋난다(근거·회귀시험은 route_wait 모듈 docstring 참조).

          - 옆을 스쳐 지나기만 해도 붙는다      → 임계값을 횡단보도 길이 기반 + 상한
          - 구간(40m) 중간의 횡단을 놓친다      → 점이 아니라 선분으로 거리 측정
          - 교차로 사방 3~4개가 모두 더해진다   → 교차로당 가장 가까운 1개만

        신호 데이터가 없거나 경로에 좌표가 없으면 0초(= 신호 미반영)로 동작한다.
        """
        if self.signals is None or len(self.signals) == 0 or "end_lat" not in segments:
            return (0.0, []) if detail else 0.0

        lat, lon = self._route_track(segments)
        if "위도" not in self.signals.columns:       # 옛 계약 형식(signals.csv)
            return self._legacy_wait(lat, lon, detail)

        hit = route_wait.crossed(self.signals, lat, lon)
        # 이론값은 실측 대비 과대추정한다. 검증 걷기 16회차로 잰 보정계수를 건다.
        # signal_model 에 alpha/beta 가 있으면 그것을, 없으면 route_wait.LAMBDA.
        w = self._calibrate(hit["기대대기_s"].to_numpy(float))
        total = float(w.sum())
        hits = list(zip(hit["횡단보도관리번호"], w))
        return (total, hits) if detail else total

    def _calibrate(self, theo: np.ndarray) -> np.ndarray:
        """이론 기대대기 → 보정 기대대기 (alpha + beta·theo)."""
        if self.signal_model and "beta" in self.signal_model:
            return (self.signal_model.get("alpha", 0.0)
                    + self.signal_model["beta"] * theo)
        return getattr(route_wait, "LAMBDA", 1.0) * theo

    def _legacy_wait(self, lat, lon, detail: bool, radius_m: float = 25.0):
        """`signals.csv`(crossing_id, lat, lon, cycle_C_s, red_R_s) 대비책.

        길이 정보가 없어 길이 기반 임계값을 못 쓴다. 고정 반경으로 두되,
        이 경로로 들어오면 과대추정될 수 있음을 알린다.
        """
        total, hits = 0.0, []
        for _, s in self.signals.iterrows():
            wait = self.expected_wait_s(s["cycle_C_s"], s["red_R_s"])
            if wait <= 0:
                continue
            if (_haversine_m(lat, lon, s["lat"], s["lon"]) < radius_m).any():
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
