# -*- coding: utf-8 -*-
"""[5단계] 디지털치료기기 효과 평가를 보조하는 방문 단위 도구입니다.

무엇을 하는 도구인가요
    한 번의 방문에 해당하는 7일 관찰 창의 16지표만 받아서,
    그 시점의 우울 중증도를 4단계(정상·경도·중등도·중증)로 돌려줍니다.
    궤적을 쓰지 않습니다. 0주에 한 번, 6주에 한 번 독립적으로 씁니다.

    치료 전후를 비교하고 싶을 때는 '반응자 판별' 을 함께 씁니다.
    이쪽은 두 시점의 지표 변화량을 직접 입력으로 받습니다.

    python step5_assess.py 학습
        방문 단위 단계 모형 3개와 반응자 모형 1개를 모형/ 아래에 저장합니다.

    python step5_assess.py 평가 <visit_wide.csv 가 있는 폴더>
        각 대상자의 0주와 6주 중증도 단계를 각각 매겨
        <폴더>/평가결과.csv 로 저장합니다.

    python step5_assess.py 검증
        교차검증으로 이 도구의 성능을 다시 계산해 결과/ 에 저장합니다.

    python step5_assess.py 기기비교
        Fitbit 이 아닌 스마트워치에서 얻을 수 있는 지표만 썼을 때의 성능을
        네 가지 구성(엄격공통8 / 기기공통12 / 공통13 / 전체16)으로 비교해
        결과/device_common_performance.csv 로 저장합니다.

다른 스마트워치에서 쓰기
    학습·평가·검증 뒤에 --지표 <구성> 을 붙이면 그 구성의 지표만 씁니다.
        python step5_assess.py 학습 --지표 기기공통12
        python step5_assess.py 평가 <폴더> --지표 기기공통12
    기기공통12 는 Fitbit · Apple · Galaxy · Garmin 모두에서 얻거나 계산할 수 있는
    12개 지표입니다. 수면 효율과 침대 체류는 기기의 값 대신 수면 단계 시간으로
    다시 계산해 씁니다(biomarker.add_derived). 구성 목록은 biomarker.FEATURE_SETS.
    지정하지 않으면 전체16 입니다.

반드시 알아야 할 한 가지
    이 도구는 '그 시점의 상태' 는 잘 맞히지만 '두 시점 사이의 변화' 는 맞히지 못합니다.
    실제 코호트 146명에서 임상 척도상 집단 평균 단계는 1.77 에서 1.45 로
    유의하게 내려갔으나(p = 0.0001), 이 도구의 추정값은 1.59 에서 1.52 로
    움직이지 않았습니다(p = 0.64).
    개인 단위로도 단계 변화의 방향 일치율이 50.7% 로 우연 수준입니다.

    따라서 두 시점의 단계 차이를 치료 효과의 크기로 쓰면 안 됩니다.
    치료 전후 비교가 목적이라면 아래 '반응자 판별' 을 쓰십시오.
    이쪽은 곡선하면적 74.2(95% 신뢰구간 62.8-85.1)로 검증되었습니다.
"""
import os
import sys

import joblib
import numpy as np
import pandas as pd

import biomarker as bm

HERE = os.path.dirname(os.path.abspath(__file__))
CFG = bm.load_config(os.path.join(HERE, "config.json"))
DATA = os.path.abspath(os.path.join(HERE, CFG["out_dir"]))
RES = os.path.join(HERE, "결과")
MODEL_DIR = os.path.join(HERE, "모형")

#: 평가 도구가 공식적으로 지원하는 시점. 임상 척도가 측정된 시점만 둡니다.
VISITS_SUPPORTED = [("v1", "0주", "HAMD_v1"), ("v4", "6주", "HAMD_v4")]

#: 쓸 지표 구성. 명령줄의 --지표 로 바꿉니다.
FSET = "전체16"
FEATS = bm.feature_set(FSET)


def _tag():
    """모형·결과 파일 이름 꼬리. 전체16 은 기존 이름을 그대로 씁니다."""
    return "" if FSET == "전체16" else f"_{FSET}"


# =====================================================================
# 자료 만들기
# =====================================================================
def visit_rows(wide, lab, feats=None):
    """1행 = 1방문. 입력은 그 방문 창의 지표, 라벨은 그 방문의 HAMD 입니다."""
    feats = feats or FEATS
    wide = bm.add_derived(wide)
    X, y, pid, vis = [], [], [], []
    for p in wide.index:
        for v, nm, col in VISITS_SUPPORTED:
            cols = [f"{f}__{v}" for f in feats]
            if not all(c in wide.columns for c in cols) or pd.isna(lab.loc[p, col]):
                continue
            X.append(wide.loc[p, cols].values)
            y.append(lab.loc[p, col]); pid.append(p); vis.append(nm)
    return (np.array(X, float), np.array(y, float),
            np.array(pid), np.array(vis))


def change_rows(wide, lab):
    """1행 = 1대상자. 입력은 지표 변화량과 0주 원값입니다."""
    wide = bm.add_derived(wide)
    idx = lab.index[lab["HAMD_v1"].notna() & lab["HAMD_v4"].notna()]
    v1 = wide.loc[idx, [f"{f}__v1" for f in FEATS]].values
    v4 = wide.loc[idx, [f"{f}__v4" for f in FEATS]].values
    h1 = lab.loc[idx, "HAMD_v1"].values
    h4 = lab.loc[idx, "HAMD_v4"].values
    with np.errstate(divide="ignore", invalid="ignore"):
        resp = np.where(h1 > 0, (h1 - h4) / h1, 0) >= 0.5
    return np.hstack([v4 - v1, v1]), resp.astype(int), idx


# =====================================================================
# 학습
# =====================================================================
def 학습():
    os.makedirs(MODEL_DIR, exist_ok=True)
    wide, lab = bm.load_cohort(bm.find_cohort_dir(DATA, "정형_"))

    X, y, pid, vis = visit_rows(wide, lab)
    print(f"지표 구성 {FSET} ({len(FEATS)}개) · 방문 단위 표본 {len(y)}개 (대상자 {len(set(pid))}명)")
    for cut in bm.BAND_CUTS:
        t = (y >= cut).astype(int)
        model = bm.make_models()["RF"]
        model.fit(X, t)
        path = os.path.join(MODEL_DIR, f"평가_단계_{cut}{_tag()}.joblib")
        joblib.dump({"모형": model, "열": list(FEATS), "지표구성": FSET,
                     "기준": f"그 방문의 HAMD >= {cut}",
                     "학습표본": int(len(t)), "양성": int(t.sum())}, path)
        print(f"  단계절단 HAMD>={cut:2d}  양성 {t.sum():3d}  -> {path}")

    Xc, rc, idx = change_rows(wide, lab)
    model = bm.make_models()["RF"]
    model.fit(Xc, rc)
    path = os.path.join(MODEL_DIR, f"평가_반응자{_tag()}.joblib")
    joblib.dump({"모형": model, "열": [f"{f}__차이" for f in FEATS] + [f"{f}__v1" for f in FEATS],
                 "지표구성": FSET,
                 "기준": "HAMD 가 0주 대비 50% 이상 감소",
                 "학습표본": int(len(rc)), "양성": int(rc.sum())}, path)
    print(f"  반응자 판별   양성 {rc.sum():3d}/{len(rc)}  -> {path}")
    print("\n저장했습니다. 성능은 '검증' 으로 다시 확인하십시오.")


# =====================================================================
# 평가
# =====================================================================
def 평가(folder):
    csv = os.path.join(folder, "visit_wide.csv")
    if not os.path.isfile(csv):
        raise FileNotFoundError(f"visit_wide.csv 가 없습니다: {folder}")
    wide = bm.add_derived(pd.read_csv(csv, index_col=0))
    names = [b[2] for b in bm.HAMD_BANDS]

    paths = [os.path.join(MODEL_DIR, f"평가_단계_{c}{_tag()}.joblib") for c in bm.BAND_CUTS]
    if not all(os.path.isfile(q) for q in paths):
        print("모형이 없습니다. 먼저 '학습' 을 돌리세요."); return

    rows = []
    for v, nm, _ in VISITS_SUPPORTED:
        cols = [f"{f}__{v}" for f in FEATS]
        if not all(c in wide.columns for c in cols):
            lack = [c for c in cols if c not in wide.columns]
            print(f"  {nm} 창에 없는 지표가 있어 건너뜁니다: {lack[:4]}"); continue
        Xv = wide[cols].values
        pr, mat = {}, []
        for c, q in zip(bm.BAND_CUTS, paths):
            pack = joblib.load(q)
            pp = pack["모형"].predict_proba(Xv)[:, 1]
            pr[f"P(HAMD>={c})"] = np.round(pp, 3)
            mat.append(pp)
        step = bm.monotone_steps(np.column_stack(mat))
        d = pd.DataFrame(pr, index=wide.index)
        d.insert(0, "시점", nm)
        d["중증도_4단계"] = [names[i] for i in step]
        rows.append(d.reset_index().rename(columns={d.index.name or "index": "pid"}))
    out = pd.concat(rows, ignore_index=True).sort_values(["pid", "시점"])

    # 치료 전후 비교가 가능하면 반응자 판별을 덧붙입니다
    rpath = os.path.join(MODEL_DIR, f"평가_반응자{_tag()}.joblib")
    have_both = all(all(f"{f}__{v}" in wide.columns for f in FEATS) for v in ("v1", "v4"))
    if os.path.isfile(rpath) and have_both:
        pack = joblib.load(rpath)
        v1 = wide[[f"{f}__v1" for f in FEATS]].values
        v4 = wide[[f"{f}__v4" for f in FEATS]].values
        p = pack["모형"].predict_proba(np.hstack([v4 - v1, v1]))[:, 1]
        resp = pd.DataFrame({"pid": wide.index, "반응자_확률": np.round(p, 3),
                             "반응자_판정": np.where(p >= 0.5, "반응", "비반응")})
        resp.to_csv(os.path.join(folder, "반응자판정.csv"), index=False, encoding="utf-8-sig")
        print(f"반응자 판정을 저장했습니다: {os.path.join(folder, '반응자판정.csv')}")

    dst = os.path.join(folder, "평가결과.csv")
    out.to_csv(dst, index=False, encoding="utf-8-sig")
    print()
    print(out.head(8).to_string(index=False))
    print(f"\n{out['pid'].nunique()}명 x {out['시점'].nunique()}시점을 저장했습니다: {dst}")
    print()
    print("읽는 법")
    print("  중증도_4단계 는 그 시점의 상태를 임상의 평가에 준해 추정한 값입니다.")
    print("  0주와 6주의 단계 차이를 치료 효과의 크기로 해석하지 마십시오.")
    print("  치료 전후 비교에는 반응자판정.csv 를 쓰십시오.")


# =====================================================================
# 검증
# =====================================================================
def 검증():
    from sklearn.model_selection import GroupKFold, KFold
    from sklearn.metrics import roc_auc_score
    from scipy import stats

    os.makedirs(RES, exist_ok=True)
    wide, lab = bm.load_cohort(bm.find_cohort_dir(DATA, "정형_"))
    X, y, pid, vis = visit_rows(wide, lab)

    probs = np.zeros((len(y), 3))
    rows = []
    for k, cut in enumerate(bm.BAND_CUTS):
        t = (y >= cut).astype(int); acc = np.zeros(len(t)); n = 0
        for seed in (42, 7, 101, 2024):
            q = np.zeros(len(t))
            for tr, te in GroupKFold(n_splits=5).split(X, t, pid):
                m = bm.make_models(seed)["RF"]
                m.fit(X[tr], t[tr]); q[te] = m.predict_proba(X[te])[:, 1]
            acc += q; n += 1
        probs[:, k] = acc / n
        a = 100 * roc_auc_score(t, probs[:, k])
        rows.append({"항목": f"단계 절단점 HAMD>={cut}", "표본": len(t),
                     "양성": int(t.sum()), "AUC": round(a, 1)})
        print(f"  절단점 HAMD>={cut:2d}  AUC {a:.1f}")

    pred = bm.monotone_steps(probs)
    true = np.array([bm.band_index(v) for v in y])
    m = bm.band_metrics(true, pred)
    print(f"  4단계  정확 {m['정확']:.1f}%  인접1단계 {m['인접1단계']:.1f}%  가중카파 {m['가중카파']:.3f}")
    rows.append({"항목": "4단계 종합", "표본": len(y), "양성": None, "AUC": None,
                 "정확": round(m["정확"], 1), "인접1단계": round(m["인접1단계"], 1),
                 "가중카파": round(m["가중카파"], 3)})

    # 변화 감지 능력 : 쓰면 안 되는 이유를 수치로 남깁니다
    t1, t4 = true[vis == "0주"], true[vis == "6주"]
    p1, p4 = pred[vis == "0주"], pred[vis == "6주"]
    wt = stats.wilcoxon(t1, t4, zero_method="zsplit").pvalue
    wp = stats.wilcoxon(p1, p4, zero_method="zsplit").pvalue
    print(f"  집단 평균단계  실제 {t1.mean():.2f}->{t4.mean():.2f} (p={wt:.4f})  "
          f"추정 {p1.mean():.2f}->{p4.mean():.2f} (p={wp:.4f})")
    rows.append({"항목": "집단 평균단계 변화(실제)", "표본": len(t1),
                 "값": round(t4.mean() - t1.mean(), 3), "p": round(wt, 4)})
    rows.append({"항목": "집단 평균단계 변화(추정)", "표본": len(p1),
                 "값": round(p4.mean() - p1.mean(), 3), "p": round(wp, 4)})

    # 반응자 판별
    Xc, rc, idx = change_rows(wide, lab)
    acc = np.zeros(len(rc)); n = 0
    for seed in (42, 7, 101, 2024):
        q = np.zeros(len(rc))
        for tr, te in KFold(5, shuffle=True, random_state=seed).split(Xc):
            mm = bm.make_models(seed)["RF"]
            mm.fit(Xc[tr], rc[tr]); q[te] = mm.predict_proba(Xc[te])[:, 1]
        acc += q; n += 1
    acc /= n
    a = 100 * roc_auc_score(rc, acc)
    lo, hi = np.percentile(bm.bootstrap_ci(rc, acc, lambda yy, pp: 100 * roc_auc_score(yy, pp)),
                           [2.5, 97.5])
    print(f"  반응자 판별  AUC {a:.1f}  95% CI [{lo:.1f}, {hi:.1f}]")
    rows.append({"항목": "반응자 판별", "표본": len(rc), "양성": int(rc.sum()),
                 "AUC": round(a, 1), "CI하한": round(lo, 1), "CI상한": round(hi, 1)})

    dst = os.path.join(RES, f"assess_metrics{_tag()}.csv")
    pd.DataFrame(rows).to_csv(dst, index=False, encoding="utf-8-sig")
    print(f"\n저장: {dst}")


# =====================================================================
# 기기비교 : 다른 스마트워치에서 얻을 수 있는 지표만 쓸 때의 성능
# =====================================================================
def 기기비교(seeds=(42, 7, 101, 2024, 11)):
    """FEATURE_SETS 의 네 구성을 같은 분할로 비교합니다.

    AUC 는 그 방문의 HAMD 14점 이상 판별, 나머지는 4단계 평가 성능입니다.
    대상자 단위 분할(StratifiedGroupKFold 5겹)을 씨앗 5개로 반복해 평균합니다.
    """
    from sklearn.model_selection import StratifiedGroupKFold
    from sklearn.metrics import roc_auc_score

    os.makedirs(RES, exist_ok=True)
    wide, lab = bm.load_cohort(bm.find_cohort_dir(DATA, "정형_"))
    need = list(dict.fromkeys(f for fs, _ in bm.FEATURE_SETS.values() for f in fs))
    X, y, pid, _ = visit_rows(wide, lab, feats=need)
    X = pd.DataFrame(X, columns=need)
    y14 = (y >= 14).astype(int)
    true = np.array([bm.band_index(v) for v in y])

    rows = []
    print(f"방문 {len(y)}개 (대상자 {len(set(pid))}명) · 대상자 단위 5겹 · 씨앗 {len(seeds)}개\n")
    for name, (feats, devices) in bm.FEATURE_SETS.items():
        aucs, kap, ex, adj = [], [], [], []
        for seed in seeds:
            probs = np.zeros((len(y), 3))
            for tr, te in StratifiedGroupKFold(5, shuffle=True, random_state=seed).split(X, y14, pid):
                for j, cut in enumerate(bm.BAND_CUTS):
                    m = bm.make_models(seed)["RF"]
                    m.fit(X.iloc[tr][feats].values, (y[tr] >= cut).astype(int))
                    probs[te, j] = m.predict_proba(X.iloc[te][feats].values)[:, 1]
            aucs.append(100 * roc_auc_score(y14, probs[:, 1]))
            b = bm.band_metrics(true, bm.monotone_steps(probs))
            kap.append(b["가중카파"]); ex.append(b["정확"]); adj.append(b["인접1단계"])
        rows.append({"구성": name, "지표수": len(feats), "기기": devices,
                     "AUC": round(np.mean(aucs), 1), "AUC_SD": round(np.std(aucs), 1),
                     "가중카파": round(np.mean(kap), 3), "4단계정확": round(np.mean(ex), 1),
                     "인접1단계": round(np.mean(adj), 1),
                     "지표": " · ".join(bm.KOR.get(f, f) for f in feats)})
        r = rows[-1]
        print(f"  {name:6s} {len(feats):2d}종  AUC {r['AUC']:.1f} ± {r['AUC_SD']:.1f}  "
              f"카파 {r['가중카파']:.3f}  정확 {r['4단계정확']:.1f}%  "
              f"한 단계 이내 {r['인접1단계']:.1f}%   [{devices}]")
    dst = os.path.join(RES, "device_common_performance.csv")
    pd.DataFrame(rows).to_csv(dst, index=False, encoding="utf-8-sig")
    print(f"\n저장: {dst}")


if __name__ == "__main__":
    args = sys.argv[1:]
    if "--지표" in args:
        i = args.index("--지표")
        FSET = args[i + 1]
        FEATS = bm.feature_set(FSET)
        del args[i:i + 2]
    what = args[0] if args else "학습"
    if what == "학습":
        학습()
    elif what == "평가":
        평가(args[1] if len(args) > 1 else bm.find_cohort_dir(DATA, "정형_"))
    elif what == "검증":
        검증()
    elif what == "기기비교":
        기기비교()
    else:
        print(__doc__)
