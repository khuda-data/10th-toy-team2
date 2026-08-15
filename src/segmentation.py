"""보행 트랙을 30~50m 단위로 구간화하고, 구간별 경사도·속도를 산출한다."""
from __future__ import annotations

import numpy as np
import pandas as pd

EARTH_RADIUS_M = 6371000.0
TARGET_SEGMENT_M = 40.0  # 30~50m 목표 구간 길이


def haversine_m(lat1, lon1, lat2, lon2) -> np.ndarray:
    lat1, lon1, lat2, lon2 = (np.radians(v.astype(float)) for v in (lat1, lon1, lat2, lon2))
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 2 * EARTH_RADIUS_M * np.arcsin(np.sqrt(a))


def add_step_distance(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["source_file", "point_idx"]).reset_index(drop=True)
    df["step_dist_m"] = 0.0
    for _, g in df.groupby("source_file", sort=False):
        idx = g.index
        d = haversine_m(g["lat"].shift(1), g["lon"].shift(1), g["lat"], g["lon"])
        df.loc[idx, "step_dist_m"] = d.fillna(0.0).to_numpy()
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
            elev_start, elev_end = start["ele_dem"], end["ele_dem"]
            slope_pct = np.nan
            if pd.notna(elev_start) and pd.notna(elev_end):
                slope_pct = (elev_end - elev_start) / dist * 100
            speed_mps = dist / dt if dt > 0 else np.nan

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
