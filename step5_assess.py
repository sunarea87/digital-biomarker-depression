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


# =====================================================================
# 자료 만들기
# =====================================================================
def visit_rows(wide, lab):
    """1행 = 1방문. 입력은 그 방문 창의 16지표, 라벨은 그 방문의 HAMD 입니다."""
    X, y, pid, vis = [], [], [], []
    for p in wide.index:
        for v, nm, col in VISITS_SUPPORTED:
            cols = [f"{f}__{v}" for f in bm.F16]
            if not all(c in wide.columns for c in cols) or pd.isna(lab.loc[p, col]):
                continue
            X.append(wide.loc[p, cols].values)
            y.append(lab.loc[p, col]); pid.append(p); vis.append(nm)
    return (np.array(X, float), np.array(y, float),
            np.array(pid), np.array(vis))


def change_rows(wide, lab):
    """1행 = 1대상자. 입력은 지표 변화량 16개와 0주 원값 16개입니다."""
    idx = lab.index[lab["HAMD_v1"].notna() & lab["HAMD_v4"].notna()]
    v1 = wide.loc[idx, [f"{f}__v1" for f in bm.F16]].values
    v4 = wide.loc[idx, [f"{f}__v4" for f in bm.F16]].values
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
    print(f"방문 단위 표본 {len(y)}개 (대상자 {len(set(pid))}명)")
    for cut in bm.BAND_CUTS:
        t = (y >= cut).astype(int)
        model = bm.make_models()["RF"]
        model.fit(X, t)
        path = os.path.join(MODEL_DIR, f"평가_단계_{cut}.joblib")
        joblib.dump({"모형": model, "열": [f"{f}" for f in bm.F16],
                     "기준": f"그 방문의 HAMD >= {cut}",
                     "학습표본": int(len(t)), "양성": int(t.sum())}, path)
        print(f"  단계절단 HAMD>={cut:2d}  양성 {t.sum():3d}  -> {path}")

    Xc, rc, idx = change_rows(wide, lab)
    model = bm.make_models()["RF"]
    model.fit(Xc, rc)
    path = os.path.join(MODEL_DIR, "평가_반응자.joblib")
    joblib.dump({"모형": model, "열": [f"{f}__차이" for f in bm.F16] + [f"{f}__v1" for f in bm.F16],
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
    wide = pd.read_csv(csv, index_col=0)
    names = [b[2] for b in bm.HAMD_BANDS]

    paths = [os.path.join(MODEL_DIR, f"평가_단계_{c}.joblib") for c in bm.BAND_CUTS]
    if not all(os.path.isfile(q) for q in paths):
        print("모형이 없습니다. 먼저 '학습' 을 돌리세요."); return

    rows = []
    for v, nm, _ in VISITS_SUPPORTED:
        cols = [f"{f}__{v}" for f in bm.F16]
        if not all(c in wide.columns for c in cols):
            print(f"  {nm} 창의 지표가 없어 건너뜁니다."); continue
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
    rpath = os.path.join(MODEL_DIR, "평가_반응자.joblib")
    have_both = all(all(f"{f}__{v}" in wide.columns for f in bm.F16) for v in ("v1", "v4"))
    if os.path.isfile(rpath) and have_both:
        pack = joblib.load(rpath)
        v1 = wide[[f"{f}__v1" for f in bm.F16]].values
        v4 = wide[[f"{f}__v4" for f in bm.F16]].values
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

    pd.DataFrame(rows).to_csv(os.path.join(RES, "assess_metrics.csv"),
                              index=False, encoding="utf-8-sig")
    print(f"\n저장: {os.path.join(RES, 'assess_metrics.csv')}")


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "학습"
    if what == "학습":
        학습()
    elif what == "평가":
        평가(sys.argv[2] if len(sys.argv) > 2 else bm.find_cohort_dir(DATA, "정형_"))
    elif what == "검증":
        검증()
    else:
        print(__doc__)
