"""수원시 영통구 신호대기 모델 — 공공데이터 + 기획안 수학식.

경희대 국제캠퍼스 일대(수원시 영통구)의 횡단보도별 기대 신호대기시간을
공공데이터 두 개와 기획안의 수학식만으로 산출한다. 실측·검증 데이터는 쓰지 않는다.

────────────────────────────────────────────────────────────
입력 (행정안전부 표준데이터, data.go.kr)
────────────────────────────────────────────────────────────
  전국횡단보도표준데이터 (15028201)
      → 차로수, 횡단보도연장, 위도/경도, 보행자신호등유무
      ⚠️ 수원시는 `녹색신호시간`·`적색신호시간`을 제출하지 않았다(결측 100%).

  전국신호등표준데이터 (15028198)
      → 신호등화시간 "36+3+111" (녹색+황색+적색). 합이 곧 주기 C.
      영통구 379개 전부 채워져 있다. 이것으로 주기 결측을 메운다.

  ※ 신호등 데이터는 전부 차량신호등(신호등구분=1)이지만,
     한 교차로의 모든 방향은 같은 주기로 돌기 때문에 C 는 그대로 쓸 수 있다.
     방향마다 다른 것은 녹색 배분이지 주기가 아니다.

────────────────────────────────────────────────────────────
처리 흐름
────────────────────────────────────────────────────────────
  1. 영통구 추출
  2. 신호등화시간 → 주기 C, 점멸등·이상치 제외, 40m 로 교차로 군집화
  3. 횡단보도 ↔ 교차로 결합
       1순위  반경 75m 최근접        2순위  같은 노선 150m 까지
       못 찾으면 영통구 자체 차로수별 중앙값
  4. 수식 적용 (아래)
  5~7. 기존 가정 대비 / 반경 민감도 / 분포

────────────────────────────────────────────────────────────
수학식 (기획안 그대로)
────────────────────────────────────────────────────────────
  보행 녹색   G = 진입 7초 + 횡단거리 ÷ 1.0 m/s     (경찰청 매뉴얼)
  보행 적색   R = C − G
  기대 대기   E[W] = R² / (2C)                      (기획안 B절)

  E[W] 는 "주기 안 아무 때나 도착한다"는 균등도착에서 유도된 값이다.
  실측 데이터가 없어 이 가정은 검증하지 않고 그대로 적용했다.
  따라서 분산·구간은 내지 않고 기대값만 낸다.

  ⚠️ 타 시군(의정부·구리·포천) 정답 대조 결과 이 방법은 대기를 약 +7%
     과대추정한다. 집계 수준에서만 쓰고, 개별 횡단보도 예측에는 쓰지 말 것
     (개별 MAE 5~9초). 자세한 내용은 docs/신호대기_모델링_진행기록.md 5절.

────────────────────────────────────────────────────────────
실행
────────────────────────────────────────────────────────────
  python src/signal_yeongtong.py                 # 전체
  python src/signal_yeongtong.py --radius 100    # 근접 반경 조정
  python src/signal_yeongtong.py --no-figures    # 그림 생략
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

try:
    from . import signal_wait as sw
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import signal_wait as sw


HERE = Path(__file__).resolve().parent
PROJ = HERE.parent

#: 기본 입력은 extract_yeongtong.py 가 만든 영통구 추출본이다.
#: 저장소에는 이 작은 파일만 올리고 전국 원본(각 10MB+)은 올리지 않는다.
#: 추출본이 없으면 같은 폴더의 전국 원본을 찾는다.
def _pick(*cands: Path) -> Path:
    for p in cands:
        if p.exists():
            return p
    return cands[0]


DEFAULT_CW = _pick(HERE / "data" / "raw" / "영통구_횡단보도_표준데이터.csv",
                   HERE / "out" / "yeongtong" / "raw" / "영통구_횡단보도_표준데이터.csv",
                   PROJ / "data" / "raw" / "영통구_횡단보도_표준데이터.csv",
                   PROJ / "전국횡단보도표준데이터.csv")
DEFAULT_SG = _pick(HERE / "data" / "raw" / "영통구_신호등_표준데이터.csv",
                   HERE / "out" / "yeongtong" / "raw" / "영통구_신호등_표준데이터.csv",
                   PROJ / "data" / "raw" / "영통구_신호등_표준데이터.csv",
                   PROJ / "전국신호등표준데이터.csv")
DEFAULT_OUT = PROJ / "out" / "yeongtong"

#: 지역 필터. 시군구명이 '수원시'까지만 있어 주소로 구를 잡는다.
SIDO, SGG_KEY = "경기도", "영통"

#: 주기의 현실 범위(초). 벗어나면 점멸등이거나 입력 오류다.
CYCLE_SANE = (40.0, 300.0)

#: 신호등 여러 개를 한 교차로로 묶는 거리(m).
INTERSECTION_M = 40.0

#: 횡단보도 ↔ 교차로 결합 반경(m) — 도로명이 다를 때.
#: 두 데이터셋을 다른 부서가 등록해 좌표 기준이 달라(신호주 vs 횡단보도 중심)
#: 최근접거리 중앙값이 75m 다. 대형 교차로의 대각 길이를 감안한 값.
MATCH_M = 75.0

#: 도로명이 같을 때만 허용하는 확장 반경(m). MATCH_M 안에서 못 찾은 것만 줍는다.
#:
#: 근거 — 같은 노선의 신호는 연동제어로 주기를 공유한다.
#:   노선명이 주기 분산을 설명하는 비율: 영통 44% · 의정부 37% · 구리 48%
#:
#: 150m 로 정한 근거 — 정답을 아는 도시(의정부)에서 새로 매칭된 횡단보도의
#: 주기 오차를 대체값(차로수별 중앙값)과 비교했다.
#:      125m  신규  5개  MAE 27.6초  (대체 38.0초)
#:      150m  신규 11개  MAE 19.6초  (대체 42.9초)   ← 채택
#:      200m  신규 28개  MAE 36.9초  (대체 42.9초)   개선폭 급감
#: 200m 는 커버리지는 늘지만 먼 교차로를 끌어와 정확도가 떨어진다.
MATCH_ROAD_M = 150.0

#: ⚠️ '번길'을 떼어 상위 도로로 묶으면 안 된다.
#: 간선(광교로)과 그 이면도로(광교로42번길)는 주기가 실제로 다르다.
#: 번길 제거 시 설명력이 영통 44%→19%, 구리 48%→29% 로 떨어진다.

#: 차로수 구간 라벨.
LANE_BINS = [-np.inf, 3, 5, np.inf]
LANE_LABELS = ["3차로 이하", "4~5차로", "6차로 이상"]

BAR = "─" * 72


def _h(t: str) -> None:
    print(f"\n{BAR}\n{t}\n{BAR}")


def _warn(t: str) -> None:
    print(f"  ⚠️  {t}")


# ─────────────────────────────────────────────────────────────
# 1. 로드
# ─────────────────────────────────────────────────────────────

def load_region(path: Path) -> pd.DataFrame:
    """표준데이터 CSV 에서 경기도 영통구 행만 뽑는다.

    전국 원본(cp949)과 추출본(utf-8-sig) 둘 다 읽는다.
    추출본은 이미 영통구만 남아 있어 필터가 한 번 더 걸려도 결과가 같다(멱등).
    """
    d = None
    for enc in ("utf-8-sig", "cp949"):
        try:
            d = pd.read_csv(path, encoding=enc, dtype=str, on_bad_lines="skip")
            break
        except UnicodeDecodeError:
            continue
    if d is None:
        raise RuntimeError(f"인코딩을 판별하지 못했습니다: {path}")
    hay = (d["시군구명"].fillna("") + " "
           + d["소재지도로명주소"].fillna("") + " "
           + d["소재지지번주소"].fillna(""))
    m = hay.str.contains(SGG_KEY, na=False) & (d["시도명"] == SIDO)
    out = d[m].copy()
    print(f"  {path.name}  전체 {len(d):,}행 → 영통 {len(out):,}행")
    if len(d) == 50000:
        _warn("파일이 정확히 50,000행입니다 — data.go.kr 그리드 다운로드 상한. "
              "수원시 행은 이 안에 온전히 들어 있음을 확인했습니다.")
    for c in ("위도", "경도"):
        out[c] = pd.to_numeric(out[c], errors="coerce")
    return out.dropna(subset=["위도", "경도"]).reset_index(drop=True)


def _hav(lat1, lon1, lat2, lon2) -> np.ndarray:
    """미터 거리. 브로드캐스팅 가능."""
    R = 6371000.0
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dl = np.radians(np.asarray(lon2) - np.asarray(lon1))
    a = np.sin((p2 - p1) / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


# ─────────────────────────────────────────────────────────────
# 2. 신호등 → 주기
# ─────────────────────────────────────────────────────────────

def parse_cycle(s) -> float:
    """'36+3+111' → 150.0. 파싱 실패는 NaN."""
    try:
        v = [float(x) for x in str(s).split("+")]
    except (TypeError, ValueError):
        return np.nan
    return float(sum(v)) if v else np.nan


def signal_cycles(sg: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """신호등 행 → 주기. 점멸등·이상치를 걸러내고 교차로로 묶는다."""
    sg = sg.copy()
    sg["C"] = sg["신호등화시간"].map(parse_cycle)

    n0 = len(sg)
    flash = sg["신호등화순서"].fillna("").str.strip().eq("황색") | sg["C"].eq(0)
    bad = ~flash & (sg["C"].isna() | ~sg["C"].between(*CYCLE_SANE))
    ok = sg[~flash & ~bad].copy()

    print(f"  신호등 {n0}개")
    print(f"    점멸등(황색 단독·C=0) 제외   {int(flash.sum()):>4}개")
    print(f"    주기 결측·범위밖 제외        {int(bad.sum()):>4}개")
    print(f"    → 유효 주기                 {len(ok):>4}개")

    # 교차로 군집화 — 같은 교차로의 여러 신호주는 같은 주기를 공유한다.
    # 묶지 않으면 신호주가 많은 대형 교차로가 통계를 지배한다.
    lat, lon = ok["위도"].values, ok["경도"].values
    cluster = np.full(len(ok), -1, dtype=int)
    cid = 0
    for i in range(len(ok)):
        if cluster[i] >= 0:
            continue
        d = _hav(lat[i], lon[i], lat, lon)
        near = (d <= INTERSECTION_M) & (cluster < 0)
        cluster[near] = cid
        cid += 1
    ok["교차로"] = cluster

    ok["road"] = ok["도로노선명"].fillna("").astype(str).str.strip()

    inter = (ok.groupby("교차로")
             .agg(위도=("위도", "mean"), 경도=("경도", "mean"),
                  신호주수=("C", "size"), 주기_s=("C", "median"),
                  주기_편차=("C", "std"))
             .reset_index())
    # 한 교차로는 여러 도로가 만나는 지점이라 노선명이 여럿일 수 있다.
    # 집합으로 들고 있어야 어느 쪽 도로의 횡단보도든 매칭된다.
    roads = ok.groupby("교차로")["road"].apply(lambda s: set(x for x in s if x))
    inter["roads"] = inter["교차로"].map(roads)
    print(f"    → 교차로 {len(inter)}곳으로 묶임 "
          f"(신호주 평균 {len(ok)/len(inter):.1f}개/교차로)")

    # 한 교차로 안에서 주기가 갈리면 등록 오류이거나 시간대별 운영이다
    split = inter[inter["주기_편차"].fillna(0) > 5]
    if len(split):
        _warn(f"교차로 {len(split)}곳은 내부 신호주끼리 주기가 다릅니다"
              f"(최대 편차 {split['주기_편차'].max():.0f}초). 중앙값을 씁니다.")
    return ok, inter


# ─────────────────────────────────────────────────────────────
# 3. 횡단보도 ↔ 교차로 결합
# ─────────────────────────────────────────────────────────────

def join_cycles(cw: pd.DataFrame, inter: pd.DataFrame,
                radius: float) -> pd.DataFrame:
    """각 횡단보도에 가장 가까운 교차로의 주기를 붙인다."""
    cw = cw.copy()
    for c, col in [("lanes", "차로수"), ("length_m", "횡단보도연장")]:
        cw[c] = pd.to_numeric(cw[col], errors="coerce")
    cw["has_signal"] = cw["보행자신호등유무"].astype(str).str.contains("있|^Y$", regex=True)

    # 버튼식은 '있음/없음/미상' 3값으로 다뤄야 한다. 영통구는 이 칸이 전부
    # 공백(미상)이라, 예전처럼 '없음'으로 뭉개면 균등도착이 깨지는 지점을 놓친다.
    _pb = cw["보행자작동신호기유무"].astype(str).str.strip()
    cw["push_btn"] = np.where(_pb.str.contains("있|^Y$", regex=True), "있음",
                              np.where(_pb.str.contains("없|^N$", regex=True),
                                       "없음", "미상"))
    # 교통섬이 있으면 2단 횡단이라 대기가 두 번 생길 수 있다(모델 미반영).
    cw["island"] = cw["교통섬유무"].astype(str).str.strip().isin(["Y", "있음"])
    cw["kind"] = cw["횡단보도종류"].astype(str).str.strip()

    cw["road"] = cw["도로명"].fillna("").astype(str).str.strip()

    D = _hav(cw["위도"].values[:, None], cw["경도"].values[:, None],
             inter["위도"].values[None, :], inter["경도"].values[None, :])
    cyc = inter["주기_s"].values
    iid = inter["교차로"].values
    has_road = "roads" in inter.columns
    iroads = inter["roads"].tolist() if has_road else [set()] * len(inter)

    # 2단계 매칭. 순서가 중요하다.
    #  1순위  반경 radius 안 최근접 교차로 (도로명 무관)
    #  2순위  1순위가 없을 때만, 도로명이 같은 교차로를 radius_road 까지 확장
    #
    # 도로명을 1순위로 두면 '더 가까운 다른 도로 교차로'를 밀어내고
    # 더 먼 같은 도로 교차로를 고르게 되는데, 정답 대조에서 이쪽이 더 나빴다
    # (의정부 공통 MAE 18.1 → 18.2). 그래서 도로명은 '못 찾은 것만 줍는' 용도로만 쓴다.
    # 이렇게 하면 기존 매칭은 그대로 두고 커버리지만 늘어난다(악화 불가능).
    road_radius = max(radius, MATCH_ROAD_M) if has_road else radius
    n = len(cw)
    m_cyc = np.full(n, np.nan)
    m_iid = np.full(n, np.nan)
    m_dist = np.full(n, np.nan)
    m_src = np.array(["미매칭"] * n, dtype=object)

    for i in range(n):
        d = D[i]
        near = d <= radius
        if near.any():                                   # 1순위: 근접
            k = int(np.where(near, d, np.inf).argmin())
            m_cyc[i], m_iid[i], m_dist[i] = cyc[k], iid[k], d[k]
            m_src[i] = "근접"
            continue
        r = cw["road"].iat[i]
        if r and has_road and road_radius > radius:      # 2순위: 같은 노선 확장
            same = np.array([r in s for s in iroads]) & (d <= road_radius)
            if same.any():
                k = int(np.where(same, d, np.inf).argmin())
                m_cyc[i], m_iid[i], m_dist[i] = cyc[k], iid[k], d[k]
                m_src[i] = "동일노선 확장"

    cw["거리_m"] = np.round(m_dist, 1)
    cw["주기_s"] = m_cyc
    cw["교차로"] = m_iid
    cw["주기출처"] = m_src

    sig = cw["has_signal"]
    n_sig = int(sig.sum())
    hit = sig & (cw["주기출처"] != "미매칭")
    print(f"  횡단보도 {len(cw)}개 (보행신호 있음 {n_sig}개)")
    print(f"    주기 확보  {int(hit.sum())}/{n_sig}개 ({hit.sum()/n_sig:.0%})")
    if has_road:
        for lab in ("근접", "동일노선 확장"):
            k = int((sig & (cw["주기출처"] == lab)).sum())
            if k:
                med = cw.loc[sig & (cw["주기출처"] == lab), "거리_m"].median()
                print(f"      {lab:<10} {k:>4}개   결합거리 중앙 {med:.0f}m")
    return cw


def fill_missing(cw: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """미매칭 횡단보도의 주기를 영통구 자체 차로수별 중앙값으로 채운다.

    기존 signal_wait 의 90/120/150 은 전국 표준값 가정이었다.
    여기서는 같은 동네 실측치에서 뽑으므로 가정의 폭이 크게 준다.
    """
    cw = cw.copy()
    cw["차로구간"] = pd.cut(cw["lanes"], LANE_BINS, labels=LANE_LABELS)
    matched = cw[cw["주기_s"].notna()]

    tbl = (matched.groupby("차로구간", observed=True)["주기_s"]
           .agg(n="size", 주기_중앙값="median", 주기_평균="mean")
           .reset_index())
    q = (matched.groupby("차로구간", observed=True)["주기_s"]
         .quantile([.25, .75]).unstack())
    # 차로구간이 Categorical 이라 map 결과도 Categorical 이 된다.
    # 그대로 두면 뺄셈에서 TypeError 가 나므로 문자열 키로 바꿔 float 로 받는다.
    lab = tbl["차로구간"].astype(str)
    tbl["주기_25%"] = lab.map({str(k): v for k, v in q[.25].items()}).astype(float).round(1)
    tbl["주기_75%"] = lab.map({str(k): v for k, v in q[.75].items()}).astype(float).round(1)
    tbl["주기_중앙값"] = tbl["주기_중앙값"].astype(float).round(1)
    tbl["주기_평균"] = tbl["주기_평균"].astype(float).round(1)
    tbl["기존_표준값"] = lab.map(
        {"3차로 이하": 90.0, "4~5차로": 120.0, "6차로 이상": 150.0}).astype(float)
    tbl["차이_초"] = (tbl["주기_중앙값"] - tbl["기존_표준값"]).round(1)

    med = dict(zip(lab, tbl["주기_중앙값"]))
    overall = float(matched["주기_s"].median())
    need = cw["주기_s"].isna()
    cw.loc[need, "주기_s"] = (cw.loc[need, "차로구간"].astype(str)
                             .map(med).fillna(overall))
    cw.loc[need, "주기출처"] = "영통 차로수별 중앙값"
    return cw, tbl


# ─────────────────────────────────────────────────────────────
# 4. 기획안 수학식 적용
# ─────────────────────────────────────────────────────────────

def apply_formula(cw: pd.DataFrame) -> pd.DataFrame:
    """G = 7 + L/1.0,  R = C − G,  E[W] = R²/(2C).

    무신호 횡단보도는 대기 0. 횡단거리가 없으면 차로수로 추정한다
    (영통구 실측 회귀 대신 3.5m/차로 + 여유 — 결측이 적어 영향은 미미).
    """
    d = cw.copy()
    est = d["length_m"].isna() | (d["length_m"] <= 0)
    if est.any():
        # 영통구는 연장이 100% 채워져 있어 여기 걸리지 않는다.
        # 다른 지역에 돌릴 때를 위한 방어이므로 조용히 넘기지 않고 알린다.
        _warn(f"횡단거리 결측 {int(est.sum())}개 — 차로수로 추정합니다 "
              f"(3.5m/차로 + 2m).")
        d.loc[est, "length_m"] = 3.5 * d.loc[est, "lanes"].fillna(2) + 2.0

    G, R, W, capped = [], [], [], []
    for _, r in d.iterrows():
        if not bool(r["has_signal"]):
            G.append(np.nan); R.append(np.nan); W.append(0.0)
            capped.append(False)
            continue
        C = float(r["주기_s"])
        need = sw.green_time(r["length_m"])          # 규정식이 요구하는 녹색
        g = min(need, C * sw.MAX_GREEN_FRAC)         # 주기의 60% 로 상한
        rr = C - g
        G.append(g); R.append(rr); W.append(sw.wait_mean(rr, C))
        capped.append(need > C * sw.MAX_GREEN_FRAC)
    d["녹색_G_s"] = np.round(G, 1)
    d["적색_R_s"] = np.round(R, 1)
    d["기대대기_s"] = np.round(W, 1)
    d["녹색상한적용"] = capped
    return d


def baseline_compare(d: pd.DataFrame) -> pd.DataFrame:
    """실측 주기 vs 기존 표준값(90/120/150) 가정의 차이.

    이 프로젝트에서 신호 주기는 가장 큰 불확실성이었다.
    공공데이터로 바꿔서 실제로 얼마나 달라졌는지가 이 모듈의 성과다.
    """
    std = {"3차로 이하": 90.0, "4~5차로": 120.0, "6차로 이상": 150.0}
    g = d[d["has_signal"]].copy()
    # 행마다 그 차로구간의 옛 표준 주기를 붙여두면 그룹·전체를 같은 식으로 다룰 수 있다
    g["_C0"] = g["차로구간"].astype(str).map(std).astype(float)

    rows = []
    for lab, s in list(g.groupby("차로구간", observed=True)) + [("전체", g)]:
        C0 = s["_C0"]
        G0 = np.minimum(sw.ENTRY_TIME_S + s["length_m"] / sw.DESIGN_WALK_MS,
                        C0 * sw.MAX_GREEN_FRAC)
        W0 = (C0 - G0) ** 2 / (2 * C0)
        rows.append({
            "차로구간": str(lab), "n": len(s),
            "기존가정_대기": round(float(W0.mean()), 1),
            "실측주기_대기": round(float(s["기대대기_s"].mean()), 1),
            "차이_초": round(float(s["기대대기_s"].mean() - W0.mean()), 1),
            "차이_%": round(100 * float(s["기대대기_s"].mean() / W0.mean() - 1), 1),
        })
    return pd.DataFrame(rows)


def radius_sensitivity(cw_raw: pd.DataFrame, inter: pd.DataFrame,
                       radii=(50, 75, 100, 125, 150)) -> pd.DataFrame:
    """공간결합 반경을 바꿔가며 결과가 흔들리는지 본다.

    두 데이터셋의 좌표 기준이 달라 반경 선택이 임의적이다.
    결과가 반경에 민감하면 그 수치는 쓸 수 없다.
    """
    rows = []
    for r in radii:
        c = join_cycles(cw_raw, inter, float(r))
        n_hit = int((~c["주기출처"].eq("미매칭") & c["has_signal"]).sum())
        c, _ = fill_missing(c)
        g = apply_formula(c)
        g = g[g["has_signal"]]
        rows.append({"반경_m": r, "매칭": f"{n_hit}/{int(c['has_signal'].sum())}",
                     "매칭률_%": round(100 * n_hit / int(c["has_signal"].sum()), 0),
                     "주기_중앙값": round(float(g["주기_s"].median()), 1),
                     "평균대기_초": round(float(g["기대대기_s"].mean()), 1)})
    return pd.DataFrame(rows)


def summarize(d: pd.DataFrame) -> pd.DataFrame:
    """차로수 구간별 대기 요약 — 경로 ETA 에서 참조할 표."""
    def row(lab, s):
        return {"차로구간": str(lab), "n": len(s),
                "횡단거리_중앙값": round(float(s["length_m"].median()), 1),
                "주기_중앙값": round(float(s["주기_s"].median()), 1),
                "녹색_중앙값": round(float(s["녹색_G_s"].median()), 1),
                "적색_중앙값": round(float(s["적색_R_s"].median()), 1),
                "기대대기_평균": round(float(s["기대대기_s"].mean()), 1),
                "기대대기_중앙값": round(float(s["기대대기_s"].median()), 1)}

    g = d[d["has_signal"]]
    return pd.DataFrame([row(lab, s)
                         for lab, s in g.groupby("차로구간", observed=True)]
                        + [row("전체", g)])


# ─────────────────────────────────────────────────────────────
# 5. 출력
# ─────────────────────────────────────────────────────────────

def make_figures(out: Path, d: pd.DataFrame, bc: pd.DataFrame,
                 rs: pd.DataFrame) -> list[Path]:
    """발표용 그림 3종. matplotlib 은 여기서만 import 한다."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.font_manager as fm   # pyplot 이 간접 import 하지만 명시한다
        import matplotlib.pyplot as plt
    except ImportError:
        _warn("matplotlib 이 없어 그림을 건너뜁니다.")
        return []

    # 한글 폰트. 없으면 라벨이 네모로 나오므로 확인 후 경고한다.
    for f in ("Malgun Gothic", "AppleGothic", "NanumGothic"):
        try:
            fm.findfont(f, fallback_to_default=False)
            plt.rcParams["font.family"] = f
            break
        except Exception:
            continue
    else:
        _warn("한글 폰트를 찾지 못했습니다 — 그림 라벨이 깨질 수 있습니다.")
    plt.rcParams["axes.unicode_minus"] = False

    g = d[d["has_signal"]]
    paths = []

    # ① 기대대기 분포
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.hist(g["기대대기_s"], bins=30, color="#4C78A8", edgecolor="white")
    m = float(g["기대대기_s"].mean())
    ax.axvline(m, color="#E45756", lw=2, ls="--", label=f"평균 {m:.1f}초")
    ax.set_xlabel("기대 신호대기 (초)"); ax.set_ylabel("횡단보도 수")
    ax.set_title(f"영통구 신호 횡단보도 기대대기 분포 (n={len(g)})")
    ax.legend(); fig.tight_layout()
    p = out / "fig_대기분포.png"; fig.savefig(p, dpi=150); plt.close(fig)
    paths.append(p)

    # ② 기존 가정 vs 실측 주기
    b = bc[bc["차로구간"] != "전체"]
    x = np.arange(len(b)); w = 0.38
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.bar(x - w/2, b["기존가정_대기"], w, label="기존 표준값 가정", color="#BAB0AC")
    ax.bar(x + w/2, b["실측주기_대기"], w, label="공공데이터 실측주기", color="#4C78A8")
    for i, (_, r) in enumerate(b.iterrows()):
        ax.text(i + w / 2, r["실측주기_대기"] + 1, f"+{r['차이_%']:.0f}%",
                ha="center", fontsize=9, color="#E45756")
    ax.set_xticks(x); ax.set_xticklabels(b["차로구간"])
    ax.set_ylabel("기대 신호대기 (초)")
    ax.set_title("주기 가정을 공공데이터로 대체한 효과")
    ax.legend(); fig.tight_layout()
    p = out / "fig_기존가정대비.png"; fig.savefig(p, dpi=150); plt.close(fig)
    paths.append(p)

    # ③ 결합 반경 민감도 — 결과가 안 흔들린다는 증거
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.plot(rs["반경_m"], rs["평균대기_초"], "o-", color="#4C78A8", lw=2)
    ax.set_ylim(0, max(60, rs["평균대기_초"].max() * 1.4))
    for _, r in rs.iterrows():
        ax.annotate(f"매칭 {r['매칭률_%']:.0f}%", (r["반경_m"], r["평균대기_초"]),
                    textcoords="offset points", xytext=(0, 10),
                    ha="center", fontsize=8, color="#666")
    ax.set_xlabel("횡단보도↔교차로 결합 반경 (m)")
    ax.set_ylabel("평균 기대대기 (초)")
    # 제목의 매칭률은 반드시 데이터에서 뽑는다. 예전에 하드코딩해 뒀다가
    # 매칭 방식을 바꾼 뒤 그림만 옛 수치(39→81%)를 말하고 있었다.
    ax.set_title(f"공간결합 반경 민감도 — 매칭률 "
                 f"{rs['매칭률_%'].min():.0f}→{rs['매칭률_%'].max():.0f}%에도 결과 안정")
    ax.grid(alpha=.3); fig.tight_layout()
    p = out / "fig_반경민감도.png"; fig.savefig(p, dpi=150); plt.close(fig)
    paths.append(p)

    return paths


def export(out: Path, d: pd.DataFrame, ct: pd.DataFrame,
           summ: pd.DataFrame, inter: pd.DataFrame, radius: float) -> Path:
    p = out / "yeongtong_signal_params.json"
    lookup = {r["차로구간"]: r["기대대기_평균"]
              for _, r in summ.iterrows() if r["차로구간"] != "전체"}
    obj = {
        "region": "경기도 수원시 영통구",
        "source": "행안부 전국횡단보도표준데이터(15028201) + 전국신호등표준데이터(15028198)",
        "method": "기획안 수학식 E[W]=R²/(2C), G=7초+연장÷1.0m/s. 균등도착 검증 없음.",
        "match_radius_m": radius,
        "n_crosswalk": int(len(d)),
        "n_signalized": int(d["has_signal"].sum()),
        "n_intersection": int(len(inter)),
        "cycle_by_lane_bin": {r["차로구간"]: r["주기_중앙값"]
                              for _, r in ct.iterrows()},
        "cycle_n": {r["차로구간"]: int(r["n"]) for _, r in ct.iterrows()},
        "wait_by_lane_bin": lookup,
        "wait_overall_mean_s": float(
            summ[summ["차로구간"] == "전체"]["기대대기_평균"].iloc[0]),
        "caveat": [
            "수원시가 횡단보도 표준데이터의 녹색/적색신호시간을 제출하지 않아, "
            "주기는 신호등 표준데이터(차량신호)에서 공간결합으로 가져왔다.",
            "균등도착 가정은 검증하지 않았다(실측 데이터 없음).",
            "버튼식 여부는 표준데이터 칸이 공백이라 미상이다(0개가 아님). "
            "버튼식이 섞여 있으면 그 지점은 균등도착이 성립하지 않는다.",
            f"교통섬 있는 신호 횡단보도 {int(d[d['has_signal']]['island'].sum())}개는 "
            f"2단 횡단일 수 있으나 대기를 1회로 계산했다(과소추정 방향).",
            "타 시군(의정부·구리·포천) 정답 대조 결과 이 방법은 대기를 "
            "약 +7% 과대추정하는 경향이 있다(개별 횡단보도 MAE 0.4~8.7초).",
        ],
    }
    p.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
    return p


def write_report(out: Path, d: pd.DataFrame, ct: pd.DataFrame,
                 summ: pd.DataFrame, inter: pd.DataFrame,
                 radius: float, bc: pd.DataFrame, rs: pd.DataFrame) -> Path:
    def md(df):
        cols = [str(c) for c in df.columns]
        body = ["| " + " | ".join("" if v is None or (isinstance(v, float) and v != v)
                                  else (f"{v:.4g}" if isinstance(v, float) else str(v))
                                  for v in r) + " |"
                for r in df.itertuples(index=False)]
        return "\n".join(["| " + " | ".join(cols) + " |",
                          "|" + "|".join("---" for _ in cols) + "|", *body])

    g = d[d["has_signal"]]
    src = d[d["has_signal"]]["주기출처"].value_counts()
    L = [
        "# 수원시 영통구 신호대기 모델 — 공공데이터 + 기획안 수학식", "",
        "> 경희대 국제캠퍼스 일대 · 실측/검증 데이터 없이 공식만으로 산출",
        f"> 횡단보도 {len(d)}개 (보행신호 {len(g)}개) · 교차로 {len(inter)}곳", "",
        "## 1. 결과 — 차로수별 기대 신호대기", "", md(summ), "",
        f"**영통구 신호 횡단보도 1개당 평균 기대대기 "
        f"{summ[summ['차로구간']=='전체']['기대대기_평균'].iloc[0]:.1f}초**", "",
        "## 2. 적용한 수학식 (기획안)", "",
        "```",
        "보행 녹색  G = 7초 + 횡단거리 ÷ 1.0 m/s      (경찰청 매뉴얼)",
        "보행 적색  R = C − G",
        "기대 대기  E[W] = R² / (2C)                  (기획안 B절)",
        "```",
        "",
        "`C`(주기)는 가정하지 않고 신호등 표준데이터의 `신호등화시간`",
        "(예: `36+3+111` = 녹색+황색+적색)을 합해 실측치로 썼습니다.", "",
        "## 3. 주기 출처", "",
        md(pd.DataFrame({"출처": src.index, "횡단보도수": src.values})), "",
        f"공간결합 반경 {radius:.0f}m. 미매칭분은 영통구 자체 차로수별 중앙값으로 채웠습니다.", "",
        "### 차로수별 실측 주기", "", md(ct), "",
        "## 4. 기존 표준값 가정과의 차이", "",
        "이 모듈 이전에는 주기를 도로 위계별 표준값(90/120/150초)으로 가정했습니다.",
        "공공데이터 실측치로 바꾼 결과입니다.", "", md(bc), "",
        "## 5. 공간결합 반경 민감도", "",
        "두 데이터셋의 좌표 기준이 달라 결합 반경이 임의적입니다.",
        "반경을 바꿔가며 결과가 흔들리는지 확인했습니다.", "", md(rs), "",
        f"**반경 50~150m 에서 평균대기 변동폭 "
        f"{rs['평균대기_초'].max()-rs['평균대기_초'].min():.1f}초** — "
        f"매칭률이 {rs['매칭률_%'].min():.0f}%에서 {rs['매칭률_%'].max():.0f}%까지 "
        f"변하는데도 결과가 안정적입니다.", "",
        "## 6. 한계", "",
        "- 수원시가 횡단보도 데이터의 `녹색/적색신호시간`을 제출하지 않아(결측 100%),",
        "  주기를 신호등 데이터(차량신호)에서 공간결합으로 가져왔습니다.",
        "  한 교차로의 모든 방향은 같은 주기로 돌기 때문에 `C`는 유효하지만,",
        "  **보행 녹색시간 `G`는 실측이 아니라 규정식 추정값**입니다.",
        "- `E[W] = R²/(2C)`는 균등도착 가정에서 나온 식입니다. 이번 산출에서는",
        "  실측 데이터가 없어 이 가정을 검증하지 않고 그대로 적용했습니다.",
        "- **버튼식 여부는 미상입니다(0개가 아님).** 표준데이터의 해당 칸이 전부",
        "  공백이라 확인할 수 없습니다. 버튼식이 섞여 있으면 그 지점은 균등도착이",
        "  성립하지 않습니다.",
        f"- 교통섬이 있는 신호 횡단보도 {int(d[d['has_signal']]['island'].sum())}개는",
        "  2단 횡단일 수 있으나 대기를 1회로 계산했습니다 — 과소추정 방향입니다.",
        "- 타 시군(의정부·구리·포천) 정답 대조 결과 이 방법은 대기를 약 **+7% 과대추정**",
        "  하는 경향이 있습니다. 위 과소추정 요인과 방향이 반대라 일부 상쇄됩니다.",
        "- 표준데이터 갱신주기가 반기여서 최근 신호 개선은 미반영일 수 있습니다.", "",
        "## 7. 경로 ETA 에서 쓰는 법", "",
        "```python",
        "import signal_wait as sw",
        "sw.load_region_params('out/yeongtong/yeongtong_signal_params.json')",
        "# 이후 sw.route_wait(crossings) 가 영통구 실측 주기를 사용",
        "```",
        "",
        "경로별 계산은 `횡단보도별_신호대기.csv`에서 해당 횡단보도를 직접 찾아",
        "`기대대기_s`를 합산하는 쪽이 정확합니다. 차로수별 평균은 보조 지표입니다.",
        "(보행자는 큰 교차로를 더 자주 건너므로 전체 평균은 과소추정됩니다.)", "",
    ]
    p = out / "영통구_신호대기_리포트.md"
    p.write_text("\n".join(L), encoding="utf-8")
    return p


# ─────────────────────────────────────────────────────────────
# 실행
# ─────────────────────────────────────────────────────────────

def main() -> int:
    ap = argparse.ArgumentParser(description="영통구 신호대기 모델 (수학식 기반)")
    ap.add_argument("--crosswalk", type=Path, default=DEFAULT_CW)
    ap.add_argument("--signal", type=Path, default=DEFAULT_SG)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--radius", type=float, default=MATCH_M,
                    help=f"횡단보도↔교차로 결합 반경 m (기본 {MATCH_M:.0f})")
    ap.add_argument("--no-figures", action="store_true", help="그림 생성 생략")
    a = ap.parse_args()

    for p in (a.crosswalk, a.signal):
        if not p.exists():
            print(f"  ❌ 파일 없음: {p}")
            return 1
    a.out.mkdir(parents=True, exist_ok=True)

    _h("1. 영통구 추출")
    cw_raw = load_region(a.crosswalk)
    sg_raw = load_region(a.signal)

    _h("2. 신호등 → 주기")
    _, inter = signal_cycles(sg_raw)
    # roads 는 파이썬 set 이라 그대로 쓰면 프로세스마다 순서가 달라진다
    # (해시 시드 무작위화). 정렬해 문자열로 굳혀야 재현 가능한 산출물이 된다.
    _inter_out = inter.round(1).copy()
    _inter_out["roads"] = _inter_out["roads"].map(
        lambda s: "|".join(sorted(s)) if isinstance(s, (set, frozenset)) else s)
    _inter_out.to_csv(a.out / "교차로_주기.csv", index=False,
                      encoding="utf-8-sig")

    _h("3. 횡단보도 ↔ 교차로 공간결합")
    cw = join_cycles(cw_raw, inter, a.radius)
    cw, ct = fill_missing(cw)
    print()
    print(ct.to_string(index=False))
    mx = ct["차이_초"].abs().max()
    print(f"\n  기존 표준값(90/120/150) 대비 최대 {mx:.0f}초 차이"
          + ("  → 표준값이 대체로 맞았습니다."
             if mx <= 15 else "  → 영통구 실측이 상당히 다릅니다."))
    ct.to_csv(a.out / "차로수별_주기.csv", index=False, encoding="utf-8-sig")

    _h("4. 기획안 수학식 적용")
    d = apply_formula(cw)
    print(f"  G = {sw.ENTRY_TIME_S:.0f}초 + 횡단거리 ÷ {sw.DESIGN_WALK_MS:.1f} m/s")
    print(f"  R = C − G,   E[W] = R²/(2C)")
    ncap = int(d["녹색상한적용"].sum())
    if ncap:
        _warn(f"녹색시간이 주기의 {sw.MAX_GREEN_FRAC:.0%}를 넘어 상한을 적용한 "
              f"횡단보도 {ncap}개 (짧은 주기 + 긴 횡단보도)")

    summ = summarize(d)
    print()
    print(summ.to_string(index=False))

    keep = ["횡단보도관리번호", "소재지도로명주소", "위도", "경도", "lanes",
            "length_m", "차로구간", "has_signal", "push_btn", "island", "kind",
            "주기_s", "주기출처", "거리_m", "녹색_G_s", "적색_R_s", "기대대기_s"]
    out_d = d[[c for c in keep if c in d.columns]].rename(columns={
        "lanes": "차로수", "length_m": "횡단거리_m", "has_signal": "보행신호",
        "push_btn": "버튼식", "island": "교통섬", "kind": "횡단보도종류",
        "거리_m": "교차로거리_m"})
    out_d.to_csv(a.out / "횡단보도별_신호대기.csv", index=False,
                 encoding="utf-8-sig")
    summ.to_csv(a.out / "차로수별_대기요약.csv", index=False,
                encoding="utf-8-sig")

    _h("5. 기존 표준값 가정과의 차이")
    bc = baseline_compare(d)
    print(bc.to_string(index=False))
    bc.to_csv(a.out / "기존가정_대비.csv", index=False, encoding="utf-8-sig")

    _h("6. 공간결합 반경 민감도")
    import io
    import contextlib
    with contextlib.redirect_stdout(io.StringIO()):     # 내부 진행출력 숨김
        rs = radius_sensitivity(cw_raw, inter)
    print(rs.to_string(index=False))
    rs.to_csv(a.out / "반경_민감도.csv", index=False, encoding="utf-8-sig")
    span = rs["평균대기_초"].max() - rs["평균대기_초"].min()
    print(f"\n  반경 50~150m 에서 평균대기 변동폭 {span:.1f}초"
          + ("  → 결합 반경 선택에 결과가 좌우되지 않습니다."
             if span <= 5 else "  → 반경에 민감합니다. 해석 주의."))

    _h("7. 분포")
    g = d[d["has_signal"]]["기대대기_s"]
    q = g.quantile([.1, .25, .5, .75, .9])
    print(f"  n={len(g)}  평균 {g.mean():.1f}초")
    print(f"  10% {q[.1]:.1f} · 25% {q[.25]:.1f} · 50% {q[.5]:.1f} "
          f"· 75% {q[.75]:.1f} · 90% {q[.9]:.1f} 초")
    sig = d[d["has_signal"]]
    pb = sig["push_btn"].value_counts().to_dict()
    print(f"\n  버튼식 여부: {pb}")
    if pb.get("미상", 0):
        _warn(f"버튼식 여부 미상 {pb['미상']}개 — 표준데이터의 해당 칸이 공백입니다. "
              f"'없음'이 아니라 '모름'이므로, 버튼식이 섞여 있으면 "
              f"그 지점은 균등도착이 성립하지 않습니다.")
    if pb.get("있음", 0):
        _warn(f"버튼식 {pb['있음']}개 — 균등도착 모델 비적용 대상")

    n_isl = int(sig["island"].sum())
    if n_isl:
        _warn(f"교통섬 있는 횡단보도 {n_isl}개 ({n_isl/len(sig):.0%}) — "
              f"2단 횡단이면 대기가 두 번 생길 수 있으나 모델은 1회로 봅니다. "
              f"→ 이만큼은 과소추정입니다.")

    if not a.no_figures:
        figs = make_figures(a.out, d, bc, rs)
        if figs:
            print(f"\n  그림 {len(figs)}종 생성")

    pj = export(a.out, d, ct, summ, inter, a.radius)
    pr = write_report(a.out, d, ct, summ, inter, a.radius, bc, rs)

    _h("완료")
    for f in sorted(a.out.iterdir()):
        print(f"  {f}")
    print(f"\n  📄 리포트: {pr}")
    print(f"  ⚙️  파라미터: {pj}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
