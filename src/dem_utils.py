"""국토지리정보원 1:5,000 수치지형도(Ver2.0) 등고선 SHP -> 5m DEM(GeoTIFF).

등고선(라인) 레이어 정점(x, y, 등고수치)을 Delaunay TIN 기반 선형보간
(scipy.interpolate.LinearNDInterpolator)으로 격자화해 DEM을 만든다.
결과 GeoTIFF는 원본 SHP의 CRS(보통 EPSG:5186 등 GRS80/중부원점 계열)를
그대로 담으므로, src/dem.py의 DEMSampler가 WGS84(EPSG:4326) 좌표를
알아서 이 CRS로 변환해 샘플링할 수 있다.
"""
from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.transform import from_origin
from rasterio.warp import transform as warp_transform
from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator
from scipy.spatial import cKDTree

DEFAULT_ELEVATION_FIELD = "등고수치"


def build_dem_from_contours(
    contour_shp: str | Path,
    out_tif: str | Path,
    elevation_field: str = DEFAULT_ELEVATION_FIELD,
    resolution: float = 5.0,
) -> Path:
    """등고선 SHP -> TIN(선형보간) DEM GeoTIFF 생성.

    Delaunay 볼록껍질 밖(등고선 범위 경계 바깥) 픽셀은 최근접 등고선
    정점 값으로 채운다. 볼록껍질 안쪽은 선형보간이라 정점값 범위를
    벗어나는 오버슈트가 나올 수 없다.
    """
    gdf = gpd.read_file(contour_shp)
    if elevation_field not in gdf.columns:
        raise ValueError(
            f"'{elevation_field}' 필드가 없습니다. 사용 가능한 컬럼: {list(gdf.columns)}"
        )
    if gdf.crs is None:
        raise ValueError(f"{contour_shp}에 CRS(.prj)가 없습니다")

    xs: list[float] = []
    ys: list[float] = []
    zs: list[float] = []
    for geom, z in zip(gdf.geometry, gdf[elevation_field]):
        for x, y in geom.coords:
            xs.append(x)
            ys.append(y)
            zs.append(z)
    xs_arr, ys_arr, zs_arr = np.array(xs), np.array(ys), np.array(zs)

    minx, miny, maxx, maxy = gdf.total_bounds
    width = int(np.ceil((maxx - minx) / resolution))
    height = int(np.ceil((maxy - miny) / resolution))

    col_centers = minx + (np.arange(width) + 0.5) * resolution
    row_centers = maxy - (np.arange(height) + 0.5) * resolution  # row 0 = 북쪽
    grid_x, grid_y = np.meshgrid(col_centers, row_centers)

    points = np.column_stack([xs_arr, ys_arr])
    grid_z = LinearNDInterpolator(points, zs_arr)(grid_x, grid_y)

    nan_mask = np.isnan(grid_z)
    if nan_mask.any():
        grid_z[nan_mask] = NearestNDInterpolator(points, zs_arr)(
            grid_x[nan_mask], grid_y[nan_mask]
        )

    grid_z = grid_z.astype("float32")
    transform = from_origin(minx, maxy, resolution, resolution)

    out_path = Path(out_tif)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(
        out_path,
        "w",
        driver="GTiff",
        height=height,
        width=width,
        count=1,
        dtype="float32",
        crs=gdf.crs,
        transform=transform,
        nodata=np.nan,
        compress="lzw",
    ) as dst:
        dst.write(grid_z, 1)

    return out_path


def sample_dem(dem_tif: str | Path, lon, lat) -> np.ndarray:
    """WGS84(EPSG:4326) lon/lat 배열 -> DEM 고도(m) 배열.

    src/dem.py의 DEMSampler와 동일한 좌표변환 관례(EPSG:4326 입력)를
    따른다. 래스터 범위 밖/nodata는 NaN.
    """
    lon_arr = np.atleast_1d(np.asarray(lon, dtype=float))
    lat_arr = np.atleast_1d(np.asarray(lat, dtype=float))

    with rasterio.open(dem_tif) as src:
        xs, ys = warp_transform("EPSG:4326", src.crs, list(lon_arr), list(lat_arr))
        band = src.read(1)
        h, w = band.shape
        elev = np.full(len(lon_arr), np.nan)
        for i, (x, y) in enumerate(zip(xs, ys)):
            row, col = rasterio.transform.rowcol(src.transform, x, y)
            if 0 <= row < h and 0 <= col < w:
                val = band[row, col]
                if not np.isnan(val):
                    elev[i] = val

    return elev


def detect_sparse_artifacts(
    contour_shp: str | Path,
    dem_tif: str | Path,
    elevation_field: str = DEFAULT_ELEVATION_FIELD,
    sparse_pct: float = 95,
    slope_pct: float = 99,
) -> dict:
    """등고선 밀도가 낮은 구간(건물 밀집지역 등)에서 보간이 비정상적으로
    튀는 픽셀을 찾는다.

    각 DEM 픽셀에서 가장 가까운 등고선 정점까지의 거리(희소도)와 인접
    픽셀 간 경사(np.gradient 기반)를 계산해, '등고선에서 멀리 떨어져
    있으면서(sparse_pct 백분위 이상) 경사가 비정상적으로 큰(slope_pct
    백분위 이상)' 픽셀을 이상치 후보로 표시한다. 대부분 Delaunay 볼록껍질
    경계부의 최근접값 채움(nearest-fill) 경계에서 발생한다.
    """
    gdf = gpd.read_file(contour_shp)
    xs: list[float] = []
    ys: list[float] = []
    for geom in gdf.geometry:
        for x, y in geom.coords:
            xs.append(x)
            ys.append(y)
    tree = cKDTree(np.column_stack([xs, ys]))

    with rasterio.open(dem_tif) as src:
        z = src.read(1)
        transform = src.transform
        res = transform.a
        crs = src.crs

    h, w = z.shape
    rows, cols = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
    grid_x, grid_y = rasterio.transform.xy(transform, rows.ravel(), cols.ravel())
    dist, _ = tree.query(np.column_stack([grid_x, grid_y]), k=1)
    dist = dist.reshape(h, w)

    dz_dy, dz_dx = np.gradient(z, res, res)
    slope_deg = np.degrees(np.arctan(np.sqrt(dz_dx**2 + dz_dy**2)))

    sparse_thresh = float(np.percentile(dist, sparse_pct))
    steep_thresh = float(np.nanpercentile(slope_deg, slope_pct))
    flagged = (dist > sparse_thresh) & (slope_deg > steep_thresh)

    ys_idx, xs_idx = np.where(flagged)
    flon, flat = ([], [])
    if len(ys_idx):
        fx, fy = rasterio.transform.xy(transform, ys_idx, xs_idx)
        flon, flat = warp_transform(crs, "EPSG:4326", fx, fy)

    return {
        "n_flagged": int(flagged.sum()),
        "n_total": int(flagged.size),
        "sparse_dist_threshold_m": sparse_thresh,
        "slope_threshold_deg": steep_thresh,
        "flagged_lon": list(flon),
        "flagged_lat": list(flat),
    }
