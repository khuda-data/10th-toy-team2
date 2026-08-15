"""GPX 트랙 포인트 파싱.

파일명 규칙: {이름}_{카테고리}_{회차}[회차].gpx ("회차" 접미사는 선택.
예: 권동하_급경사내리막_1회차.gpx, 홍민기_평지_6.gpx)
GPX 자체의 <ele>는 신뢰도가 낮아 좌표(lat/lon)만 사용하고, 고도는 dem.py에서 별도 매핑한다.
"""
from __future__ import annotations

import re
import unicodedata
from pathlib import Path

import gpxpy
import pandas as pd

FILENAME_RE = re.compile(r"^(?P<person>[^_]+)_(?P<category>[^_]+)_(?P<trial>\d+)(?:회차)?\.gpx$")


def parse_filename(path: Path) -> dict:
    # macOS(APFS)는 파일마다 NFC/NFD가 섞여서 반환될 수 있어 정규화 후 매칭한다.
    name = unicodedata.normalize("NFC", path.name)
    m = FILENAME_RE.match(name)
    if not m:
        raise ValueError(f"파일명 규칙과 맞지 않음: {path.name}")
    return {
        "person": m.group("person"),
        "category": m.group("category"),
        "trial": int(m.group("trial")),
    }


def parse_gpx_file(path: Path) -> pd.DataFrame:
    meta = parse_filename(path)
    with open(path, "r", encoding="utf-8") as f:
        gpx = gpxpy.parse(f)

    rows = []
    for track in gpx.tracks:
        for segment in track.segments:
            for point in segment.points:
                rows.append(
                    {
                        **meta,
                        "lat": point.latitude,
                        "lon": point.longitude,
                        "ele_gps": point.elevation,
                        "time": point.time,
                        "source_file": path.name,
                    }
                )

    df = pd.DataFrame(rows)
    df = df.sort_values("time").reset_index(drop=True)
    df["point_idx"] = df.index
    return df


def parse_all(raw_dir: Path) -> pd.DataFrame:
    files = sorted(Path(raw_dir).glob("*.gpx"))
    if not files:
        raise FileNotFoundError(f"{raw_dir}에서 .gpx 파일을 찾지 못했습니다")
    frames = [parse_gpx_file(p) for p in files]
    return pd.concat(frames, ignore_index=True)
