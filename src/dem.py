"""국토지리정보원 정밀 DEM(.tif/.img) 래스터에서 좌표별 고도 샘플링.

여러 도엽(타일)로 나뉘어 다운로드된 DEM을 하나의 모자이크로 합친 뒤,
GPX의 WGS84(EPSG:4326) 좌표를 DEM의 좌표계(보통 EPSG:5186/5179)로 변환하여 픽셀값을 읽는다.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from rasterio.merge import merge
from rasterio.warp import transform as warp_transform


class DEMSampler:
    def __init__(self, dem_paths: list[Path]):
        if not dem_paths:
            raise ValueError("DEM 파일이 하나도 없습니다")
        self._srcs = [rasterio.open(p) for p in dem_paths]

        if len(self._srcs) == 1:
            src = self._srcs[0]
            self._mosaic = src.read(1)
            self._transform = src.transform
            self._nodata = src.nodata
        else:
            mosaic, transform = merge(self._srcs)
            self._mosaic = mosaic[0]
            self._transform = transform
            self._nodata = self._srcs[0].nodata

        self._crs = self._srcs[0].crs

    def close(self) -> None:
        for s in self._srcs:
            s.close()

    def sample(self, lon: np.ndarray, lat: np.ndarray) -> np.ndarray:
        """WGS84(EPSG:4326) lon/lat 배열 -> DEM 고도(m) 배열. 범위 밖/nodata는 NaN."""
        xs, ys = warp_transform("EPSG:4326", self._crs, list(lon), list(lat))

        h, w = self._mosaic.shape
        elev = np.full(len(lon), np.nan)
        for i, (x, y) in enumerate(zip(xs, ys)):
            row, col = rasterio.transform.rowcol(self._transform, x, y)
            if 0 <= row < h and 0 <= col < w:
                value = self._mosaic[row, col]
                if self._nodata is None or value != self._nodata:
                    elev[i] = value

        return elev

    def sample_df(
        self,
        df: pd.DataFrame,
        lon_col: str = "lon",
        lat_col: str = "lat",
        out_col: str = "ele_dem",
    ) -> pd.DataFrame:
        df = df.copy()
        df[out_col] = self.sample(df[lon_col].to_numpy(), df[lat_col].to_numpy())
        return df
