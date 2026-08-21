"""전국 표준데이터 → 영통구 부분집합 추출.

행안부 표준데이터 전국 파일은 각 10~12MB 라 저장소에 올리기 부적절하다.
모델 학습에 실제로 쓴 영통구 행만 뽑아 작은 CSV 두 개로 남긴다.

  전국횡단보도표준데이터.csv (50,000행)  →  영통구_횡단보도_표준데이터.csv (351행)
  전국신호등표준데이터.csv   (50,000행)  →  영통구_신호등_표준데이터.csv  (379행)

컬럼은 원본 그대로 둔다. 가공은 signal_yeongtong.py 가 하고,
이 파일은 "어느 행을 썼는가"만 고정하는 역할이다.

실행:
  python extract_yeongtong.py
  python extract_yeongtong.py --src <전국파일폴더> --out <저장폴더>
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

HERE = Path(__file__).resolve().parent

SIDO, SGG_KEY = "경기도", "영통"

#: (전국 파일명, 내보낼 파일명)
PAIRS = [
    ("전국횡단보도표준데이터.csv", "영통구_횡단보도_표준데이터.csv"),
    ("전국신호등표준데이터.csv", "영통구_신호등_표준데이터.csv"),
]


def extract(src: Path, dst: Path) -> pd.DataFrame:
    d = pd.read_csv(src, encoding="cp949", dtype=str, on_bad_lines="skip")
    hay = (d["시군구명"].fillna("") + " "
           + d["소재지도로명주소"].fillna("") + " "
           + d["소재지지번주소"].fillna(""))
    out = d[hay.str.contains(SGG_KEY, na=False) & (d["시도명"] == SIDO)].copy()

    # utf-8-sig 로 저장한다. cp949 로 두면 다른 OS 에서 못 읽는다.
    dst.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(dst, index=False, encoding="utf-8-sig")

    base = d["데이터기준일자"].dropna().unique() if "데이터기준일자" in d else []
    print(f"  {src.name}")
    print(f"    전체 {len(d):,}행 → 영통구 {len(out):,}행  ({dst.name})")
    print(f"    기준일자 {sorted(out['데이터기준일자'].dropna().unique())[:3]}"
          if "데이터기준일자" in out else "")
    if len(d) == 50000:
        print(f"    ※ 전국 파일이 정확히 50,000행 = data.go.kr 그리드 상한. "
              f"수원시 행은 이 안에 온전히 포함됨(확인 완료).")
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="전국 표준데이터 → 영통구 추출")
    ap.add_argument("--src", type=Path, default=HERE.parent,
                    help="전국 CSV 가 있는 폴더")
    ap.add_argument("--out", type=Path, default=HERE.parent / "data" / "raw",
                    help="추출본을 저장할 폴더")
    a = ap.parse_args()

    print(f"\n원본 폴더: {a.src}\n저장 폴더: {a.out}\n")
    missing = [n for n, _ in PAIRS if not (a.src / n).exists()]
    if missing:
        print(f"  ❌ 전국 파일이 없습니다: {missing}")
        print(f"     data.go.kr 에서 받아 {a.src} 에 두세요.")
        return 1

    for src_name, dst_name in PAIRS:
        extract(a.src / src_name, a.out / dst_name)
    print(f"\n완료 → {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
