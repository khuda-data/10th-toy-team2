"""GPX + DEM -> 구간별 경사도/속도 테이블(data/processed/segments.csv) 생성.

사용법 (프로젝트 루트에서):
    python -m src.build_dataset --dem-dir <DEM(.tif/.img) 파일이 있는 폴더>
"""
from __future__ import annotations

import argparse
from pathlib import Path

from src.dem import DEMSampler
from src.gpx_parser import parse_all
from src.segmentation import add_step_distance, build_segments


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

    print(f"[1/4] GPX 파싱: {args.raw_dir}")
    points = parse_all(args.raw_dir)
    print(f"  -> {len(points)} 포인트, {points['source_file'].nunique()} 파일")

    print(f"[2/4] DEM 고도 샘플링: {len(dem_paths)}개 타일")
    sampler = DEMSampler(dem_paths)
    points = sampler.sample_df(points)
    sampler.close()
    n_nan = int(points["ele_dem"].isna().sum())
    if n_nan:
        print(f"  경고: DEM 범위를 벗어났거나 nodata인 포인트 {n_nan}개 (NaN)")

    print("[3/4] 누적거리 계산 및 구간화")
    points = add_step_distance(points)
    segments = build_segments(points, target_m=args.target_m)

    print(f"[4/4] 저장: {args.out}")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    segments.to_csv(args.out, index=False, encoding="utf-8-sig")
    print(f"  -> {len(segments)}개 구간 저장 완료")


if __name__ == "__main__":
    main()
