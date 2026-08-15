"""보행 트랙을 30~50m 단위로 구간화하고, 구간별 경사도·속도를 산출한다.

GPX 타임스탬프 간격이 불규칙해서(포인트 누락/몰림) speed_mps에 물리적으로
불가능한 이상치가 섞이는 문제가 있어 두 가지 정제를 거친다:
- GPS 튐(순간속도 > MAX_SPEED_KMH): 원본 포인트 단계에서 제거 후 재구간화.
  구간 평균에서 걸러내면 이미 오염된 cum_dist_m까지 같이 틀어져 있어
  근본 해결이 안 되기 때문에 포인트 단계에서 잘라낸다.
- 정지구간(연속 STATIONARY_GAP_S초 이상 이동 없음): 포인트를 지워도 남은
  앞뒤 포인트의 실제 경과시간(wall-clock)은 그대로라 dt_s가 부풀어 있는
  문제가 안 고쳐진다. 대신 정지 시간을 뺀 이동시간(moving_dt_s)을 따로
  합산해 speed_mps 계산의 분모로 쓴다.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

EARTH_RADIUS_M = 6371000.0
TARGET_SEGMENT_M = 40.0  # 30~50m 목표 구간 길이
MAX_SPEED_KMH = 15.0  # 이 순간속도를 넘는 point-to-point 이동은 GPS 튐으로 간주해 제거
STATIONARY_GAP_S = 3.0  # 이 시간 이상 거의 안 움직이면 정지구간으로 간주해 이동시간에서 제외
STATIONARY_DIST_M = 1.0  # 정지 판정 시 "거의 안 움직임"의 기준 거리


def haversine_m(lat1, lon1, lat2, lon2) -> np.ndarray:
    lat1, lon1, lat2, lon2 = (np.radians(v.astype(float)) for v in (lat1, lon1, lat2, lon2))
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 2 * EARTH_RADIUS_M * np.arcsin(np.sqrt(a))


def remove_gps_jumps(df: pd.DataFrame, max_speed_kmh: float = MAX_SPEED_KMH) -> pd.DataFrame:
    """파일별로 순간속도가 max_speed_kmh를 넘는 포인트(GPS 튐)를 제거한다.

    제거된 포인트 다음 포인트는 "마지막으로 남은 정상 포인트"를 기준으로
    다시 순간속도를 검사한다(튐 포인트와 비교하면 연쇄적으로 다음 포인트도
    잘못 튐으로 판정될 수 있기 때문).
    """
    max_speed_mps = max_speed_kmh / 3.6
    df = df.sort_values(["source_file", "point_idx"]).reset_index(drop=True)

    keep_mask = np.ones(len(df), dtype=bool)
    n_dropped = 0
    for _, g in df.groupby("source_file", sort=False):
        idx = g.index.to_numpy()
        lats, lons, times = g["lat"].to_numpy(), g["lon"].to_numpy(), g["time"].to_numpy()
        last = 0
        for i in range(1, len(idx)):
            dt = (times[i] - times[last]) / np.timedelta64(1, "s")
            d = float(haversine_m(lats[last:last + 1], lons[last:last + 1], lats[i:i + 1], lons[i:i + 1])[0])
            if dt > 0 and d / dt > max_speed_mps:
                keep_mask[idx[i]] = False
                n_dropped += 1
            else:
                last = i

    if n_dropped:
        print(f"  GPS 튐(순간속도 > {max_speed_kmh}km/h) 포인트 {n_dropped}개 제거")
    return df[keep_mask].reset_index(drop=True)


def add_step_distance(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["source_file", "point_idx"]).reset_index(drop=True)
    df["step_dist_m"] = 0.0
    df["step_time_s"] = 0.0
    for _, g in df.groupby("source_file", sort=False):
        idx = g.index
        d = haversine_m(g["lat"].shift(1), g["lon"].shift(1), g["lat"], g["lon"])
        dt = (g["time"] - g["time"].shift(1)).dt.total_seconds()
        df.loc[idx, "step_dist_m"] = d.fillna(0.0).to_numpy()
        df.loc[idx, "step_time_s"] = dt.fillna(0.0).to_numpy()
    df["is_stationary_step"] = (df["step_time_s"] >= STATIONARY_GAP_S) & (df["step_dist_m"] < STATIONARY_DIST_M)
    df["cum_dist_m"] = df.groupby("source_file")["step_dist_m"].cumsum()
    return df


def build_segments(df: pd.DataFrame, target_m: float = TARGET_SEGMENT_M) -> pd.DataFrame:
    """cum_dist_m 기준 target_m 간격으로 트랙을 구간화하고 구간별 지표를 계산한다."""
    rows = []
    for source_file, g in df.groupby("source_file", sort=False):
        g = g.reset_index(drop=True)
        bin_id = (g["cum_dist_m"] // target_m).astype(int)
        for seg_id, seg in g.groupby(bin_id):
            if len(seg) < 2:
                continue
            start, end = seg.iloc[0], seg.iloc[-1]
            dist = end["cum_dist_m"] - start["cum_dist_m"]
            if dist <= 0:
                continue

            dt = (end["time"] - start["time"]).total_seconds()
            # 구간 내부 스텝(첫 포인트 제외)의 이동시간만 합산 -> 정지구간 제외
            internal = seg.iloc[1:]
            moving_dt = internal.loc[~internal["is_stationary_step"], "step_time_s"].sum()

            elev_start, elev_end = start["ele_dem"], end["ele_dem"]
            slope_pct = np.nan
            if pd.notna(elev_start) and pd.notna(elev_end):
                slope_pct = (elev_end - elev_start) / dist * 100
            speed_mps = dist / moving_dt if moving_dt > 0 else np.nan

            rows.append(
                {
                    "person": start["person"],
                    "category": start["category"],
                    "trial": start["trial"],
                    "source_file": source_file,
                    "segment_id": int(seg_id),
                    "n_points": len(seg),
                    "dist_m": dist,
                    "dt_s": dt,
                    "moving_dt_s": moving_dt,
                    "speed_mps": speed_mps,
                    "elev_start_m": elev_start,
                    "elev_end_m": elev_end,
                    "slope_pct": slope_pct,
                    "start_lat": start["lat"],
                    "start_lon": start["lon"],
                    "end_lat": end["lat"],
                    "end_lon": end["lon"],
                    "start_time": start["time"],
                    "end_time": end["time"],
                }
            )

    return pd.DataFrame(rows)
