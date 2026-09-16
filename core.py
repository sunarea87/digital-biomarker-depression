# -*- coding: utf-8 -*-
"""공통 함수 모음.

preprocess.py / train.py / evaluate.py 가 필요한 모든 기능을 여기서 가져다 쓴다.
데이터 파일은 저장소에 포함되지 않으며, 경로는 config.json 또는 환경변수로 준다.

구성
  1. 설정과 상수
  2. 원자료 파싱      (Fitbit JSON/CSV -> 일 단위 표)
  3. 특징 생성        (방문 요약, 궤적 요약, 직교 다항 기저)
  4. 모형과 교차검증
  5. 지표 계산
  6. 해석 (SHAP)
  7. 그림
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
    cfg.setdefault("work_dir", "./work")
    os.makedirs(cfg["work_dir"], exist_ok=True)
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
    """일별 심박변이도 요약을 모은다."""
    acc = {}
    for base in _export_dirs(raw_root, pid, "Heart Rate Variability"):
        for name in os.listdir(base):
            if not name.startswith("Daily Heart Rate Variability Summary"):
                continue
            try:
                df = pd.read_csv(os.path.join(base, name))
            except Exception:
                continue
            if df.empty or "timestamp" not in df.columns:
                continue
            df["_d"] = pd.to_datetime(df["timestamp"], errors="coerce").dt.date
            ren = {"rmssd": "HRV_rmssd", "nremhr": "HRV_nremhr",
                   "entropy": "HRV_entropy",
                   "low_frequency": "HRV_low_frequency",
                   "high_frequency": "HRV_high_frequency"}
            for _, r in df.iterrows():
                d = r["_d"]
                if d is None or not (start.date() <= d <= end.date()):
                    continue
                rec = acc.setdefault(d, {})
                for src, dst in ren.items():
                    if src in df.columns and pd.notna(r[src]):
                        rec[dst] = float(r[src])
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
    """환자 단위 교차검증으로 out-of-fold 예측을 만든다.

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
    """같은 자료를 방문 단위와 환자 단위로 나눠 평가해 부풀림을 잰다.

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
            "환자 단위": res["subject"][0], "환자 단위 SD": res["subject"][1],
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
