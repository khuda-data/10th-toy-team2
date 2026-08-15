"""GPX + DEM -> 구간별 경사도/속도 테이블(data/processed/segments.csv) 생성.

고도는 국토정보플랫폼 등고선에서 만든 5m TIN DEM(src/dem_utils.py의
build_dem_from_contours/sample_dem)으로 샘플링한다. 90m DEM 대비 30m
구간 경사도의 분산이 크게 줄어드는 것을 확인했다(계단식 픽셀 경계
아티팩트 감소).

사용법 (프로젝트 루트에서):
    python -m src.build_dataset --dem-dir <DEM(.tif/.img) 파일이 있는 폴더>
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.dem_utils import sample_dem
from src.gpx_parser import parse_all
from src.segmentation import add_step_distance, build_segments, remove_gps_jumps


def sample_dem_df(
    df: pd.DataFrame,
    dem_paths: list[Path],
    lon_col: str = "lon",
    lat_col: str = "lat",
    out_col: str = "ele_dem",
) -> pd.DataFrame:
    """여러 DEM 타일에 대해 dem_utils.sample_dem을 순서대로 시도해 첫 유효값을 채운다.

    타일 간 경계가 겹치지 않는다고 가정한다(국토정보플랫폼 도엽 단위 DEM).
    """
    df = df.copy()
    lon = df[lon_col].to_numpy()
    lat = df[lat_col].to_numpy()
    elev = np.full(len(df), np.nan)

    remaining = np.isnan(elev)
    for dem_path in dem_paths:
        if not remaining.any():
            break
        idx = np.where(remaining)[0]
        vals = sample_dem(dem_path, lon[idx], lat[idx])
        filled = idx[~np.isnan(vals)]
        elev[filled] = vals[~np.isnan(vals)]
        remaining = np.isnan(elev)

    df[out_col] = elev
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--dem-dir", type=Path, required=True, help="DEM(.tif/.img) 파일이 있는 폴더")
    parser.add_argument("--out", type=Path, default=Path("data/processed/segments.csv"))
    parser.add_argument("--target-m", type=float, default=40.0, help="목표 구간 길이(m), 30~50m 권장")
    args = parser.parse_args()

    dem_exts = ("*.tif", "*.tiff", "*.img", "*.asc")
    dem_paths = sorted({p for ext in dem_exts for p in args.dem_dir.glob(ext)})
    if not dem_paths:
        raise SystemExit(f"{args.dem_dir}에서 DEM 파일(.tif/.tiff/.img/.asc)을 찾지 못했습니다")

    print(f"[1/5] GPX 파싱: {args.raw_dir}")
    points = parse_all(args.raw_dir)
    print(f"  -> {len(points)} 포인트, {points['source_file'].nunique()} 파일")

    print("[2/5] GPS 튐 포인트 제거")
    points = remove_gps_jumps(points)
    print(f"  -> {len(points)} 포인트 남음")

    print(f"[3/5] DEM 고도 샘플링: {len(dem_paths)}개 타일")
    points = sample_dem_df(points, dem_paths)
    n_nan = int(points["ele_dem"].isna().sum())
    if n_nan:
        print(f"  경고: DEM 범위를 벗어났거나 nodata인 포인트 {n_nan}개 (NaN)")

    print("[4/5] 누적거리 계산 및 구간화 (정지구간 제외 이동시간 반영)")
    points = add_step_distance(points)
    segments = build_segments(points, target_m=args.target_m)

    print(f"[5/5] 저장: {args.out}")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    segments.to_csv(args.out, index=False, encoding="utf-8-sig")
    print(f"  -> {len(segments)}개 구간 저장 완료")


if __name__ == "__main__":
    main()
