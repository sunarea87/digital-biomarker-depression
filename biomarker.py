# -*- coding: utf-8 -*-
"""디지털 바이오마커 분석에 쓰는 함수를 전부 모아 둔 곳.

이 파일 하나만 import 하면 전처리부터 그림까지 다 됩니다.
step1_preprocess.py / step2_analyze.py / step3_figures.py 가 여기서 함수를 가져다 씁니다.
직접 실행하지 않습니다.

--------------------------------------------------------------------------
찾아보기 (무엇을 하고 싶을 때 어느 함수를 보면 되는지)
--------------------------------------------------------------------------
[1] 설정과 상수
    F16                     분석에 쓰는 지표 16개 이름
    VISITS / WEEK_OF        방문 코드(v1~v4)와 주차(0/2/4/6)
    POLY_CONTRAST           직교 다항 대비 가중치 (P0 평균, P1 기울기, P2 2차, P3 3차)
    MIN_DAYS_PER_WINDOW     한 관찰 창에서 며칠 이상 있어야 인정할지 (3일)
    load_config             config.json 읽기

[2] 원자료 파싱 : Fitbit 내보내기 폴더 -> 하루 단위 표
    read_sleep_days         수면 JSON
    read_hrv_days           심박변이 CSV (★ 두 파일을 같이 읽어야 함. 아래 주의사항 참고)
    read_activity_days      활동 JSON
    visit_windows           방문별 7일 관찰 창 계산
    daily_table             한 대상자의 네 창을 하루 단위 표로
    window_coverage         창별로 며칠 확보되었는지 세기
    select_cohort           품질 기준을 넘은 대상자만 남기기

[3] 특징 생성 : 하루 단위 표 -> 모델 입력
    visit_summary           창 단위 평균 (1행 = 1대상자, 열 = 지표__방문)
    trajectory_summary      지표별 기울기/총변화/최대변화
    polynomial_basis        직교 다항 계수 (P0, P1, ...)
    representation          표현 방식을 이름으로 고르기 (raw2 / raw4 / traj / poly)

[4] 모형과 교차검증
    make_models             LR / RF / XGB / SVM 파이프라인 (대치·스케일 포함)
    cross_val_predict_patient   교차검증 out-of-fold 예측 확률
    fold_aucs               겹마다의 AUC
    compare_representations 표현 방식끼리 같은 조건으로 비교
    record_vs_subject       방문 단위와 대상자 단위 분할의 성능 차이 측정

[5] 지표 계산
    full_metrics            정확도/민감도/특이도/PPV/NPV/F1/AUC 한 번에
    classification_metrics  위의 축약판
    bootstrap_ci            부트스트랩 신뢰구간

[6] 해석
    shap_importance         SHAP 기여도 (평가 겹에서만 모음)
    KOR                     지표 영문 이름 -> 한글 이름

[7] 그림
    use_korean_font         한글 폰트 등록
    save_figure             TIFF(600dpi) + PNG 동시 저장
    plot_roc / plot_importance

--------------------------------------------------------------------------
꼭 알아야 할 주의사항 세 가지
--------------------------------------------------------------------------
1) 심박변이 저주파수(LF)와 고주파수(HF)는 'Daily Heart Rate Variability Summary'
   파일에 없습니다. 'Heart Rate Variability Details' 파일에 5분 간격으로 들어 있어
   하루 평균으로 접어서 써야 합니다. 이것을 빠뜨리면 16지표 중 2개가 통째로 빕니다.
   -> read_hrv_days 가 두 파일을 모두 읽도록 되어 있습니다.

2) 6주(v4) 관찰 창은 마지막 방문일에서 거꾸로 셉니다. 실제 방문 간격이 계획된
   42일에서 벗어나는 대상자가 많아, 앞에서부터 세면 임상 평가 시점과 어긋납니다.
   -> visit_windows 를 보세요.

3) 결측 대치와 스케일링은 반드시 교차검증 '분할 안에서' 해야 합니다.
   전체 자료로 먼저 대치하면 평가 자료 정보가 학습에 새어 성능이 높게 나옵니다.
   -> make_models 가 대치와 스케일을 파이프라인 안에 넣어 두었습니다.
"""
from __future__ import annotations

import collections
import json
import os
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

# =====================================================================
# 1. 설정과 상수
# =====================================================================

#: 분석에 쓰는 16개 지표. 렘 수면 잠복기(SQ_rem_latency)는 결측이 많아 제외한다.
SLEEP_KEYS = ["SQ_deep_minutes", "SQ_wake_minutes", "SQ_light_minutes",
              "SQ_rem_minutes", "SQ_efficiency", "SQ_time_in_bed"]
HRV_KEYS = ["HRV_low_frequency", "HRV_high_frequency", "HRV_entropy",
            "HRV_nremhr", "HRV_rmssd"]
ACT_KEYS = ["ACT_distances_weekday", "ACT_distances_weekend",
            "ACT_calories_weekday", "ACT_calories_weekend",
            "ACT_total_active_minutes"]
F16 = SLEEP_KEYS + HRV_KEYS + ACT_KEYS

#: 지표를 모달리티 블록으로 묶은 것. 블록 단위 실험에서 쓴다.
BLOCKS = {"sleep": SLEEP_KEYS, "hrv": HRV_KEYS, "act": ACT_KEYS}

#: 방문 코드와 주차. v1=0주, v2=2주, v3=4주, v4=6주.
VISITS = ["v1", "v2", "v3", "v4"]
WEEK_OF = {"v1": 0, "v2": 2, "v3": 4, "v4": 6}

#: 4시점 등간격 직교 다항 대비. P0=평균, P1=선형, P2=2차, P3=3차.
POLY_CONTRAST = {
    "P0": np.array([1.0, 1.0, 1.0, 1.0]),
    "P1": np.array([-3.0, -1.0, 1.0, 3.0]),
    "P2": np.array([1.0, -1.0, -1.0, 1.0]),
    "P3": np.array([-1.0, 3.0, -3.0, 1.0]),
}

#: 한 창(7일)에서 최소 며칠이 있어야 그 방문을 인정할지.
MIN_DAYS_PER_WINDOW = 3

SEED = 42
N_SPLITS = 5
N_REPEATS = 5


def load_config(path: str = "config.json") -> dict:
    """데이터 경로 설정을 읽는다.

    config.json 예시::

        {"raw_root": "D:/원본데이터",
         "subject_xlsx": "D:/피험자 DATA.xlsx",
         "work_dir": "./work"}

    환경변수 DBD_RAW_ROOT / DBD_SUBJECT_XLSX / DBD_WORK_DIR 로도 줄 수 있다.
    """
    cfg = {}
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            cfg = json.load(f)
    for key, env in [("raw_root", "DBD_RAW_ROOT"),
                     ("subject_xlsx", "DBD_SUBJECT_XLSX"),
                     ("work_dir", "DBD_WORK_DIR")]:
        if os.environ.get(env):
            cfg[key] = os.environ[env]
    return cfg


def work_path(cfg: dict, *parts: str) -> str:
    p = os.path.join(cfg["work_dir"], *parts)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    return p


# =====================================================================
# 2. 원자료 파싱
# =====================================================================

def _export_dirs(raw_root: str, pid: str, sub: str) -> list:
    """Fitbit 내보내기 폴더는 두 가지 형태로 온다. 존재하는 것만 돌려준다."""
    cand = [os.path.join(raw_root, pid, "Fitbit", sub),
            os.path.join(raw_root, pid, "Takeout", "Fitbit", sub)]
    return [d for d in cand if os.path.isdir(d)]


def read_sleep_days(raw_root: str, pid: str, start, end) -> dict:
    """수면 단계별 분 수를 날짜별로 모은다. 반환 {date: {지표: 값}}."""
    acc = collections.defaultdict(lambda: collections.defaultdict(list))
    for base in _export_dirs(raw_root, pid, "Global Export Data"):
        for name in os.listdir(base):
            if not (name.startswith("sleep-") and name.endswith(".json")):
                continue
            try:
                with open(os.path.join(base, name), encoding="utf-8") as f:
                    entries = json.load(f)
            except Exception:
                continue
            for e in entries:
                try:
                    d = datetime.strptime(e.get("dateOfSleep", ""), "%Y-%m-%d").date()
                except Exception:
                    continue
                if not (start.date() <= d <= end.date()):
                    continue
                s = e.get("levels", {}).get("summary", {})
                acc[d]["SQ_deep_minutes"].append(s.get("deep", {}).get("minutes", 0))
                acc[d]["SQ_wake_minutes"].append(s.get("wake", {}).get("minutes", 0))
                acc[d]["SQ_light_minutes"].append(s.get("light", {}).get("minutes", 0))
                acc[d]["SQ_rem_minutes"].append(s.get("rem", {}).get("minutes", 0))
                acc[d]["SQ_efficiency"].append(e.get("efficiency", 0) or 0)
                acc[d]["SQ_time_in_bed"].append(e.get("timeInBed", 0) or 0)
    return {d: {k: float(np.mean(v)) for k, v in kv.items()} for d, kv in acc.items()}


def read_hrv_days(raw_root: str, pid: str, start, end) -> dict:
    """일별 심박변이도를 모은다.

    Fitbit 은 심박변이도를 두 파일에 나눠 내보낸다.
      Daily ... Summary  하루 한 줄. rmssd, nremhr, entropy
      ... Details        5분 간격. low_frequency, high_frequency
    저주파와 고주파는 Details 에만 있으므로 하루 단위로 평균 내어 합친다.
    """
    acc = {}
    for base in _export_dirs(raw_root, pid, "Heart Rate Variability"):
        for name in os.listdir(base):
            if not name.endswith(".csv"):
                continue
            is_summary = name.startswith("Daily Heart Rate Variability Summary")
            is_detail = name.startswith("Heart Rate Variability Details")
            if not (is_summary or is_detail):
                continue
            try:
                df = pd.read_csv(os.path.join(base, name))
            except Exception:
                continue
            if df.empty or "timestamp" not in df.columns:
                continue
            df["_d"] = pd.to_datetime(df["timestamp"], errors="coerce").dt.date
            df = df[df["_d"].between(start.date(), end.date())]
            if df.empty:
                continue
            if is_summary:
                ren = {"rmssd": "HRV_rmssd", "nremhr": "HRV_nremhr",
                       "entropy": "HRV_entropy"}
                for _, r in df.iterrows():
                    rec = acc.setdefault(r["_d"], {})
                    for src, dst in ren.items():
                        if src in df.columns and pd.notna(r[src]):
                            rec[dst] = float(r[src])
            else:
                # 5분 간격 기록을 하루 평균으로 접는다
                ren = {"low_frequency": "HRV_low_frequency",
                       "high_frequency": "HRV_high_frequency"}
                cols = [c for c in ren if c in df.columns]
                if not cols:
                    continue
                for d, g in df.groupby("_d"):
                    rec = acc.setdefault(d, {})
                    for src in cols:
                        v = pd.to_numeric(g[src], errors="coerce").dropna()
                        if len(v):
                            rec[ren[src]] = float(v.mean())
    return acc


def read_activity_days(raw_root: str, pid: str, start, end) -> dict:
    """거리·칼로리·활동시간을 날짜별로 합산한다."""
    minute_files = {"lightly_active_minutes": "_light",
                    "moderately_active_minutes": "_mod",
                    "very_active_minutes": "_very"}
    mins = collections.defaultdict(lambda: collections.defaultdict(float))
    other = collections.defaultdict(lambda: collections.defaultdict(list))
    for base in _export_dirs(raw_root, pid, "Global Export Data"):
        for name in os.listdir(base):
            if not name.endswith(".json"):
                continue
            head = name.split("-")[0]
            path = os.path.join(base, name)
            if head in minute_files:
                key = minute_files[head]
            elif head in ("distance", "calories"):
                key = "ACT_distances" if head == "distance" else "ACT_calories"
            else:
                continue
            try:
                with open(path, encoding="utf-8") as f:
                    entries = json.load(f)
            except Exception:
                continue
            for e in entries:
                dt = _parse_fitbit_dt(e.get("dateTime", ""))
                if dt is None or not (start.date() <= dt.date() <= end.date()):
                    continue
                val = float(e.get("value", 0) or 0)
                if head in minute_files:
                    mins[dt.date()][key] += val
                else:
                    other[dt.date()][key].append(val)
    out = {}
    for d in set(mins) | set(other):
        rec = {}
        if d in mins:
            rec["ACT_total_active_minutes"] = sum(mins[d].values())
        for k, v in other.get(d, {}).items():
            if v:
                rec[k] = float(np.mean(v))
        if rec:
            out[d] = rec
    return out


def _parse_fitbit_dt(s: str):
    for fmt in ("%m/%d/%y %H:%M:%S", "%m/%d/%y"):
        try:
            return datetime.strptime(s, fmt)
        except Exception:
            pass
    return None


def visit_windows(v1_date, v4_date) -> dict:
    """방문별 7일 창.

    0·2·4주는 첫 방문일부터 전진하고, 6주는 마지막 방문일에서 역산한다.
    실제 v1-v4 간격이 42일에서 벗어나는 대상자가 많기 때문이다.
    """
    v1 = pd.to_datetime(v1_date).to_pydatetime().replace(hour=0, minute=0, second=0)
    v4 = pd.to_datetime(v4_date).to_pydatetime().replace(hour=0, minute=0, second=0)
    return {"v1": (v1, v1 + timedelta(days=6)),
            "v2": (v1 + timedelta(days=14), v1 + timedelta(days=20)),
            "v3": (v1 + timedelta(days=28), v1 + timedelta(days=34)),
            "v4": (v4 - timedelta(days=6), v4)}


def daily_table(raw_root: str, pid: str, v1_date, v4_date) -> pd.DataFrame:
    """한 대상자의 네 창을 일 단위 표로 만든다. 1행 = 1날짜."""
    rows = []
    for visit, (st, en) in visit_windows(v1_date, v4_date).items():
        sleep = read_sleep_days(raw_root, pid, st, en)
        hrv = read_hrv_days(raw_root, pid, st, en)
        act = read_activity_days(raw_root, pid, st, en)
        for i in range(7):
            d = (st + timedelta(days=i)).date()
            rec = {"pid": pid, "visit": visit, "day_in_window": i, "date": d,
                   "is_weekend": int(d.weekday() >= 5)}
            rec.update(sleep.get(d, {}))
            rec.update(hrv.get(d, {}))
            rec.update(act.get(d, {}))
            rows.append(rec)
    return pd.DataFrame(rows)


def window_coverage(daily: pd.DataFrame) -> pd.DataFrame:
    """창별·모달리티별로 값이 있는 날 수를 센다. 코호트 선정 기준이 된다."""
    out = []
    for (pid, visit), g in daily.groupby(["pid", "visit"]):
        rec = {"pid": pid, "visit": visit}
        for name, keys in BLOCKS.items():
            cols = [k for k in keys if k in g.columns]
            base = [c for c in cols if not c.startswith("ACT_distances_")
                    and not c.startswith("ACT_calories_")]
            use = base or cols
            rec[name] = int(g[use].notna().any(axis=1).sum()) if use else 0
        rec["min_modality"] = min(rec[n] for n in BLOCKS)
        out.append(rec)
    return pd.DataFrame(out)


def select_cohort(cov: pd.DataFrame, min_days: int = MIN_DAYS_PER_WINDOW) -> list:
    """네 창 모두에서 세 모달리티가 min_days 이상인 대상자만 남긴다."""
    w = cov.pivot(index="pid", columns="visit", values="min_modality")
    need = [v for v in VISITS if v in w.columns]
    return sorted(w.index[(w[need] >= min_days).all(axis=1)])


# =====================================================================
# 3. 특징 생성
# =====================================================================

def visit_summary(daily: pd.DataFrame) -> pd.DataFrame:
    """일 단위 표를 방문 요약으로 접는다. 1행 = 1대상자, 열 = 지표__방문."""
    simple = [k for k in F16 if not k.startswith(("ACT_distances_", "ACT_calories_"))]
    simple = [k for k in simple if k in daily.columns]
    g = daily.groupby(["pid", "visit"])
    out = g[simple].mean()
    # 거리·칼로리는 평일과 주말을 나눈다. 활동량의 주중 패턴이 다르기 때문이다.
    for src in ["ACT_distances", "ACT_calories"]:
        if src not in daily.columns:
            continue
        for label, flag in [("weekday", 0), ("weekend", 1)]:
            sel = daily[daily["is_weekend"] == flag]
            out[f"{src}_{label}"] = sel.groupby(["pid", "visit"])[src].mean()
    wide = out.unstack("visit")
    wide.columns = [f"{a}__{b}" for a, b in wide.columns]
    keep = [f"{f}__{v}" for f in F16 for v in VISITS if f"{f}__{v}" in wide.columns]
    return wide[keep]


def trajectory_summary(wide: pd.DataFrame, feats=None) -> pd.DataFrame:
    """지표별 궤적 요약 세 가지.

    SLP 선형 기울기, TOT 총변화(6주-0주), MXD 최대 인접변화.
    제거 실험에서 MXD 는 기여가 없었으므로 기본 표현에서는 빼도 된다.
    """
    feats = feats or F16
    t = np.arange(len(VISITS), dtype=float)
    t -= t.mean()
    out = {}
    for f in feats:
        cols = [f"{f}__{v}" for v in VISITS if f"{f}__{v}" in wide.columns]
        if len(cols) < len(VISITS):
            continue
        m = wide[cols].values.astype(float)
        out[f"SLP_{f}"] = np.nansum(m * t, axis=1) / np.sum(t ** 2)
        out[f"TOT_{f}"] = m[:, -1] - m[:, 0]
        out[f"MXD_{f}"] = np.nanmax(np.abs(np.diff(m, axis=1)), axis=1)
    return pd.DataFrame(out, index=wide.index)


def polynomial_basis(wide: pd.DataFrame, orders=("P0", "P1"), feats=None) -> pd.DataFrame:
    """직교 다항 계수. 기본값 P0+P1 이 실험에서 가장 좋았다.

    P0 는 6주간 평균 수준, P1 은 선형 방향이다. P2 이상은 성능을 떨어뜨렸다.
    """
    feats = feats or F16
    out = {}
    for f in feats:
        cols = [f"{f}__{v}" for v in VISITS if f"{f}__{v}" in wide.columns]
        if len(cols) < len(VISITS):
            continue
        m = wide[cols].values.astype(float)
        for o in orders:
            c = POLY_CONTRAST[o]
            out[f"{o}_{f}"] = (m * c).sum(axis=1) / (c ** 2).sum()
    return pd.DataFrame(out, index=wide.index)


def representation(wide: pd.DataFrame, kind: str = "poly") -> pd.DataFrame:
    """표현 방식을 이름으로 고른다.

    raw2  0주와 6주 원값만            (임상에서 쓰는 전후 비교)
    raw4  네 시점 원값
    traj  네 시점 원값 + 궤적 요약
    poly  직교 다항 P0+P1            (권장, 차원이 가장 작고 성능이 가장 좋다)
    """
    if kind == "raw2":
        cols = [f"{f}__{v}" for v in ("v1", "v4") for f in F16 if f"{f}__{v}" in wide.columns]
        return wide[cols]
    if kind == "raw4":
        cols = [f"{f}__{v}" for v in VISITS for f in F16 if f"{f}__{v}" in wide.columns]
        return wide[cols]
    if kind == "traj":
        raw = representation(wide, "raw4")
        tr = trajectory_summary(wide)
        return pd.concat([raw, tr[[c for c in tr.columns
                                   if not c.startswith("MXD_")]]], axis=1)
    if kind == "poly":
        return polynomial_basis(wide, orders=("P0", "P1"))
    raise ValueError(f"모르는 표현 방식: {kind}")


# =====================================================================
# 4. 모형과 교차검증
# =====================================================================

def make_models(seed: int = SEED) -> dict:
    """네 가지 분류기. 결측 대치와 스케일을 파이프라인 안에 둔다.

    파이프라인에 넣어야 학습 겹의 통계만으로 대치·스케일이 이루어지고
    평가 겹의 정보가 새지 않는다.
    """
    from sklearn.pipeline import Pipeline
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import MinMaxScaler
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.svm import SVC
    imp = lambda: ("im", SimpleImputer(strategy="median"))
    sc = lambda: ("sc", MinMaxScaler())
    out = {
        "LR": Pipeline([imp(), sc(),
                        ("m", LogisticRegression(max_iter=5000, class_weight="balanced",
                                                 random_state=seed))]),
        "RF": Pipeline([imp(),
                        ("m", RandomForestClassifier(n_estimators=500, min_samples_leaf=3,
                                                     max_features="sqrt",
                                                     class_weight="balanced",
                                                     random_state=seed, n_jobs=1))]),
        "SVM": Pipeline([imp(), sc(),
                         ("m", SVC(probability=True, class_weight="balanced",
                                   random_state=seed))]),
    }
    try:
        from xgboost import XGBClassifier
        out["XGB"] = Pipeline([imp(),
                               ("m", XGBClassifier(n_estimators=300, max_depth=3,
                                                   learning_rate=0.05, subsample=0.8,
                                                   colsample_bytree=0.8, reg_lambda=1.0,
                                                   eval_metric="logloss",
                                                   random_state=seed, n_jobs=1))])
    except ImportError:
        pass
    return out


def cross_val_predict_patient(X, y, model, seed=SEED, n_splits=N_SPLITS,
                              n_repeats=N_REPEATS, groups=None):
    """대상자 단위 교차검증으로 out-of-fold 예측을 만든다.

    groups 를 주면 같은 사람의 여러 행이 학습과 평가로 갈라지지 않는다.
    한 사람이 한 행이면 groups 없이도 자동으로 보장된다.
    """
    from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedGroupKFold
    X = np.asarray(X, float)
    y = np.asarray(y, int)
    if groups is None:
        splits = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats,
                                         random_state=seed).split(X, y)
    else:
        splits = StratifiedGroupKFold(n_splits=n_splits, shuffle=True,
                                      random_state=seed).split(X, y, groups=groups)
    oof = np.zeros(len(y))
    cnt = np.zeros(len(y))
    for tr, te in splits:
        m = make_models(seed)[model].fit(X[tr], y[tr])
        oof[te] += m.predict_proba(X[te])[:, 1]
        cnt[te] += 1
    ok = cnt > 0
    return y[ok], oof[ok] / cnt[ok]


def fold_aucs(X, y, model, seed=SEED, n_splits=N_SPLITS, n_repeats=N_REPEATS):
    """겹마다의 AUC 를 벡터로 돌려준다. 표현 간 짝지은 비교에 쓴다."""
    from sklearn.model_selection import RepeatedStratifiedKFold
    from sklearn.metrics import roc_auc_score
    X = np.asarray(X, float)
    y = np.asarray(y, int)
    out = []
    for tr, te in RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats,
                                          random_state=seed).split(X, y):
        m = make_models(seed)[model].fit(X[tr], y[tr])
        try:
            out.append(roc_auc_score(y[te], m.predict_proba(X[te])[:, 1]))
        except ValueError:
            pass
    return np.array(out)


def compare_representations(reps: dict, y, models=None, seeds=(42, 7, 101, 2024)):
    """표현 여러 개를 같은 조건에서 비교한다. 시드를 바꿔 안정성도 본다."""
    models = models or list(make_models())
    rows = []
    for name, X in reps.items():
        for m in models:
            au = [fold_aucs(X, y, m, seed=s).mean() * 100 for s in seeds]
            rows.append({"표현": name, "차원": np.shape(X)[1], "모형": m,
                         "AUC": float(np.mean(au)), "SD": float(np.std(au)),
                         "최소": float(np.min(au)), "최대": float(np.max(au))})
    return pd.DataFrame(rows)


def record_vs_subject(X, y, groups, model, seed=SEED, n_repeats=20):
    """같은 자료를 방문 단위와 대상자 단위로 나눠 평가해 차이를 잰다.

    같은 사람의 여러 방문을 독립 표본처럼 다루면 성능이 과대추정된다.
    그 크기를 직접 재는 함수다.
    """
    from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold
    from sklearn.metrics import roc_auc_score
    X = np.asarray(X, float)
    y = np.asarray(y, int)
    res = {}
    for mode in ("record", "subject"):
        vals = []
        for r in range(n_repeats):
            s = seed + r
            sp = (StratifiedKFold(N_SPLITS, shuffle=True, random_state=s).split(X, y)
                  if mode == "record" else
                  StratifiedGroupKFold(N_SPLITS, shuffle=True, random_state=s)
                  .split(X, y, groups=groups))
            oof = np.zeros(len(y)); cnt = np.zeros(len(y))
            for tr, te in sp:
                m = make_models(s)[model].fit(X[tr], y[tr])
                oof[te] += m.predict_proba(X[te])[:, 1]; cnt[te] += 1
            ok = cnt > 0
            vals.append(100 * roc_auc_score(y[ok], oof[ok] / cnt[ok]))
        res[mode] = (float(np.mean(vals)), float(np.std(vals)))
    return {"방문 단위": res["record"][0], "방문 단위 SD": res["record"][1],
            "대상자 단위": res["subject"][0], "대상자 단위 SD": res["subject"][1],
            "부풀림": res["record"][0] - res["subject"][0]}


# =====================================================================
# 5. 지표 계산
# =====================================================================

def classification_metrics(y, p, threshold=0.5) -> dict:
    from sklearn.metrics import confusion_matrix, f1_score, roc_auc_score
    y = np.asarray(y, int)
    pred = (np.asarray(p) >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {"ACC": 100 * (tp + tn) / len(y),
            "Sen": 100 * tp / max(tp + fn, 1),
            "Spec": 100 * tn / max(tn + fp, 1),
            "F1": 100 * f1_score(y, pred, zero_division=0),
            "AUC": 100 * roc_auc_score(y, p)}


def full_metrics(y, p, threshold=0.5) -> dict:
    """보고서에 넣는 일곱 가지 지표를 한 번에 낸다.

    민감도(sensitivity)와 재현율(recall)은 같은 값이라 한 번만 낸다.
    PPV 는 양성으로 예측한 것 중 실제 양성 비율, NPV 는 그 반대다.
    """
    from sklearn.metrics import confusion_matrix, roc_auc_score
    y = np.asarray(y, int)
    pred = (np.asarray(p) >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    sen = tp / max(tp + fn, 1)
    spec = tn / max(tn + fp, 1)
    ppv = tp / max(tp + fp, 1)
    npv = tn / max(tn + fn, 1)
    f1 = 2 * ppv * sen / max(ppv + sen, 1e-9)
    return {"ACC": 100 * (tp + tn) / len(y), "Sen": 100 * sen, "Spec": 100 * spec,
            "PPV": 100 * ppv, "NPV": 100 * npv, "F1": 100 * f1,
            "AUC": 100 * roc_auc_score(y, p)}


def oof_predict(X, y, model, splitter="record", groups=None, n_repeats=10, seed=SEED):
    """반복 교차검증으로 out-of-fold 예측 확률을 모아 평균한다.

    splitter="record"  표본을 그냥 무작위로 나눈다 (방문 단위)
    splitter="subject" groups 로 준 대상자를 통째로 한쪽에만 넣는다 (대상자 단위)
    """
    from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold
    X = np.asarray(X, float)
    y = np.asarray(y, int)
    acc = np.zeros(len(y))
    cnt = np.zeros(len(y))
    for r in range(n_repeats):
        s = seed + r
        sp = (StratifiedKFold(N_SPLITS, shuffle=True, random_state=s).split(X, y)
              if splitter == "record" else
              StratifiedGroupKFold(N_SPLITS, shuffle=True, random_state=s)
              .split(X, y, groups))
        for tr, te in sp:
            m = make_models(s)[model].fit(X[tr], y[tr])
            acc[te] += m.predict_proba(X[te])[:, 1]
            cnt[te] += 1
    ok = cnt > 0
    return y[ok], acc[ok] / cnt[ok]


def bootstrap_ci(y, p, fn, n_boot=1000, seed=0):
    """지표의 95% 신뢰구간. fn 은 (y, p) 를 받아 값 하나를 돌려주는 함수."""
    rng = np.random.default_rng(seed)
    y = np.asarray(y); p = np.asarray(p)
    vals = []
    for _ in range(n_boot):
        i = rng.integers(0, len(y), len(y))
        if len(np.unique(y[i])) < 2:
            continue
        try:
            vals.append(fn(y[i], p[i]))
        except Exception:
            pass
    if not vals:
        return (np.nan, np.nan)
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def metrics_with_ci(y, p, n_boot=1000) -> dict:
    out = classification_metrics(y, p)
    for k in list(out):
        lo, hi = bootstrap_ci(y, p, lambda a, b, kk=k: classification_metrics(a, b)[kk],
                              n_boot=n_boot)
        out[f"{k}_CI"] = f"{lo:.1f}-{hi:.1f}"
    return out


# =====================================================================
# 6. 해석
# =====================================================================

#: 지표 영문 이름을 보고서에 쓰는 한글 이름으로 바꾼다.
KOR = {
    "SQ_deep_minutes": "깊은 수면 시간", "SQ_wake_minutes": "수면 중 각성 시간",
    "SQ_light_minutes": "얕은 수면 시간", "SQ_rem_minutes": "렘 수면 시간",
    "SQ_efficiency": "수면 효율", "SQ_time_in_bed": "침대에 머무른 총시간",
    "HRV_low_frequency": "저주파수(LF)", "HRV_high_frequency": "고주파수(HF)",
    "HRV_entropy": "엔트로피", "HRV_nremhr": "비렘수면 심박수", "HRV_rmssd": "RMSSD",
    "ACT_distances_weekday": "평일 이동 거리", "ACT_distances_weekend": "주말 이동 거리",
    "ACT_calories_weekday": "평일 소모 칼로리", "ACT_calories_weekend": "주말 소모 칼로리",
    "ACT_total_active_minutes": "총 활동 시간",
}

#: 판별 표적 세 가지. (이름, 라벨 열, 양성 절단점)
TARGETS = [("중증도", "HAMD_v4", 14), ("자살사고", "SUI_v4", 1), ("불안", "BAI_v4", 16)]


def load_cohort(data_dir: str):
    """코호트 폴더에서 방문 요약과 라벨을 함께 읽는다.

    data_dir 예: "../3.데이터/정형_146명"
    """
    w = pd.read_csv(os.path.join(data_dir, "visit_wide.csv"), index_col=0,
                    encoding="utf-8-sig")
    w.index = w.index.astype(str)
    L = pd.read_csv(os.path.join(data_dir, "labels.csv"), index_col=0,
                    encoding="utf-8-sig")
    L.index = L.index.astype(str)
    return w, L.reindex(w.index)


def to_visit_rows(wide: pd.DataFrame, visits=("v1", "v4")) -> pd.DataFrame:
    """1대상자 1행짜리 표를 1방문 1행짜리 표로 편다 (정형 분석용).

    같은 대상자가 두 줄을 갖게 되므로, 학습·평가를 나눌 때 pid 를 그룹으로 줘야
    같은 사람이 양쪽에 들어가지 않는다.
    """
    out = []
    for v in visits:
        s = wide[[f"{f}__{v}" for f in F16]].copy()
        s.columns = F16
        s["pid"] = wide.index
        s["visit"] = v
        out.append(s)
    return pd.concat(out, ignore_index=True)


def shap_importance(X, y, feature_names, model="RF", seed=SEED, n_repeats=3):
    """평가 겹에서만 SHAP 값을 모아 평균한다.

    학습 겹에서 적합한 모형으로 평가 겹의 기여도를 구하므로,
    같은 자료로 적합하고 해석하는 낙관적 편향을 피한다.
    """
    import shap
    from sklearn.model_selection import RepeatedStratifiedKFold
    X = np.asarray(X, float)
    y = np.asarray(y, int)
    acc = np.zeros(X.shape[1])
    n = 0
    for tr, te in RepeatedStratifiedKFold(n_splits=N_SPLITS, n_repeats=n_repeats,
                                          random_state=seed).split(X, y):
        pipe = make_models(seed)[model].fit(X[tr], y[tr])
        Xt = pipe[:-1].transform(X[te])
        sv = shap.TreeExplainer(pipe[-1]).shap_values(Xt)
        if isinstance(sv, list):
            sv = sv[1]
        if np.ndim(sv) == 3:
            sv = sv[:, :, 1]
        acc += np.abs(sv).mean(axis=0)
        n += 1
    return (pd.DataFrame({"특징": list(feature_names), "SHAP": acc / max(n, 1)})
            .sort_values("SHAP", ascending=False).reset_index(drop=True))


# =====================================================================
# 7. 그림
# =====================================================================

def use_korean_font(path: str = r"C:\Windows\Fonts\malgun.ttf"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.font_manager as fm
    if os.path.exists(path):
        fm.fontManager.addfont(path)
        plt.rcParams["font.family"] = fm.FontProperties(fname=path).get_name()
    plt.rcParams["axes.unicode_minus"] = False


def save_figure(fig, stem: str, dpi: int = 600):
    """논문 제출용 TIFF 와 미리보기용 PNG 를 함께 저장한다."""
    for ext, kw in [("tiff", {"pil_kwargs": {"compression": "tiff_lzw"}}), ("png", {})]:
        fig.savefig(f"{stem}.{ext}", format=ext, bbox_inches="tight", dpi=dpi,
                    facecolor="white", **kw)
    import matplotlib.pyplot as plt
    plt.close(fig)


def plot_roc(curves: dict, stem: str):
    """curves: {모형이름: (y, p)}. 그림 안에 제목은 넣지 않는다."""
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_curve, roc_auc_score
    colors = ["#C62E4F", "#5B8FF9", "#E67E22", "#27AE60"]
    fig, ax = plt.subplots(figsize=(8, 8), dpi=600)
    for (name, (y, p)), c in zip(curves.items(), colors):
        fpr, tpr, _ = roc_curve(y, p)
        ax.plot(fpr, tpr, color=c, lw=2.2,
                label=f"{name} (AUC = {roc_auc_score(y, p):.3f})")
    ax.plot([0, 1], [0, 1], ls="--", lw=1.2, color="#999999")
    ax.set_xlabel("1 - Specificity", fontsize=14)
    ax.set_ylabel("Sensitivity", fontsize=14)
    ax.set_xlim(-0.01, 1.01); ax.set_ylim(-0.01, 1.01)
    ax.tick_params(labelsize=12)
    ax.legend(loc="lower right", fontsize=12)
    ax.grid(alpha=0.15)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    save_figure(fig, stem)


def plot_importance(imp: pd.DataFrame, stem: str, top_n: int = 16):
    """SHAP 중요도 가로 막대."""
    import matplotlib.pyplot as plt
    d = imp.head(top_n).sort_values("SHAP")
    fig, ax = plt.subplots(figsize=(9, 7), dpi=600)
    ax.barh(d["특징"], d["SHAP"], color="#FF0051", height=0.55)
    mx = d["SHAP"].max()
    for i, v in enumerate(d["SHAP"]):
        ax.text(v + mx * 0.012, i, f"+{v:.3f}", va="center", fontsize=10,
                color="#FF0051", fontweight="bold")
    ax.set_xlabel("mean(|SHAP value|)", fontsize=13)
    ax.set_xlim(0, mx * 1.16)
    ax.tick_params(labelsize=11)
    ax.grid(axis="x", alpha=0.15)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    save_figure(fig, stem)
