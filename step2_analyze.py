# -*- coding: utf-8 -*-
"""[2단계] 실험을 전부 돌리고 결과표를 만듭니다.

실행
    python step2_analyze.py            전부 실행 (10~20분)
    python step2_analyze.py 정형       한 부분만 실행
    (고를 수 있는 이름: 정형 / 비정형 / 서브그룹 / 불안)

네 가지 실험을 합니다
    [정형]   한 대상자의 0주·6주를 각각 하나의 표본으로 봅니다 (146명 -> 292 방문).
             표본을 '방문 단위'로 나눈 경우와 '대상자 단위'로 나눈 경우를 모두 내어,
             같은 사람이 학습과 평가에 동시에 들어갈 때 성능이 얼마나 부풀려지는지 봅니다.
    [비정형] 한 대상자의 네 시점을 하나의 표본으로 묶습니다 (141명 -> 141 표본).
             시간 정보를 넣는 방법 여섯 가지를 비교한 뒤, 가장 좋은 방법으로 최종 성능을 냅니다.
    [서브그룹] 우울군 안에서 수면장애 동반 / 불안장애 동반으로 좁혀 다시 판별해 봅니다.
    [불안]   0주에 불안이 심했던 사람과 아닌 사람의 6주 예후를 비교합니다.

무엇이 만들어지나요 (결과/ 아래 CSV)
    full_metrics.csv                정확도·민감도·특이도·PPV·NPV·F1·AUC  (보고서 표의 출처)
    structured_perf.csv             정형: 방문 단위 vs 대상자 단위 AUC와 그 차이
    structured_shap.csv             정형: 표현형별 16지표 기여도 순위
    longitudinal_representations.csv 비정형: 표현 방식 6종 x 모형 4종 AUC
    longitudinal_perf.csv           비정형: 표현형별 최고 조합
    longitudinal_shap.csv           비정형: 궤적 성분별 기여도
    subgroup_perf.csv               동반 질환별 서브그룹 판별 성능
    anxiety_clinical.csv            0주 불안 유무에 따른 6주 결과 (위험비)
    anxiety_adjusted.csv            보정 단계별 교차비
    anxiety_transition.csv          자살사고 0주->6주 상태 변화

보고서에 들어가는 성능 숫자는 모두 full_metrics.csv 한 곳에서 나옵니다.
표를 손으로 옮겨 적지 말고 이 파일을 그대로 읽어 쓰세요.
"""
import os
import sys
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
os.environ.setdefault("JOBLIB_TEMP_FOLDER", "C:/jltmp")

from scipy import stats
import statsmodels.api as sm

import biomarker as bm

HERE = os.path.dirname(os.path.abspath(__file__))
CFG = bm.load_config(os.path.join(HERE, "config.json"))
DATA = os.path.abspath(os.path.join(HERE, CFG["out_dir"]))
RES = os.path.join(HERE, "결과")
os.makedirs(RES, exist_ok=True)

DIR_TWO = os.path.join(DATA, "정형_146명")
DIR_LONG = os.path.join(DATA, "비정형_141명")
MODELS = ["LR", "RF", "XGB", "SVM"]
SEEDS = (42, 7, 101, 2024)

#: 비정형 분석에서 표현형마다 쓰기로 정한 표현 방식 (아래 compare 실험 결과)
BEST_REP = {"중증도": "traj", "자살사고": "poly", "불안": "traj"}

save = lambda df, name: df.to_csv(os.path.join(RES, name), index=False,
                                  encoding="utf-8-sig")


# =====================================================================
# 1. 정형 : 방문 하나가 표본 하나
# =====================================================================
def run_structured(metrics):
    print("=" * 78)
    print("[정형] 0주·6주를 각각 하나의 표본으로 (146명 x 2방문)")
    print("=" * 78)
    wide, lab = bm.load_cohort(DIR_TWO)
    rows_long = bm.to_visit_rows(wide, ("v1", "v4"))
    perf, ranks = [], {}

    for tgt, col, th in bm.TARGETS:
        ok = lab[col].notna()
        m = rows_long[rows_long["pid"].isin(ok.index[ok])].reset_index(drop=True)
        y = (lab[col].reindex(m["pid"]) >= th).astype(int).values
        X = m[bm.F16].values.astype(float)
        g = m["pid"].values
        print(f"\n  {tgt}: 방문 {len(y)}개 (대상자 {m['pid'].nunique()}명), 양성 {y.sum()}")

        for mo in MODELS:
            # (1) 두 분할 방식의 AUC 차이 = 부풀림
            r = bm.record_vs_subject(X, y, g, mo, n_repeats=20)
            perf.append({"표현형": tgt, "모형": mo, "n방문": len(y),
                         "n대상자": int(m["pid"].nunique()), "양성": int(y.sum()), **r})
            # (2) 지표 일곱 가지를 두 분할 방식 모두에 대해
            for mode, nm in (("record", "방문 단위"), ("subject", "대상자 단위")):
                yy, pp = bm.oof_predict(X, y, mo, mode, g)
                metrics.append({"코호트": "정형", "분할": nm, "표현형": tgt, "모형": mo,
                                "표본": len(y), "양성": int(y.sum()),
                                **{k: round(v, 1) for k, v in bm.full_metrics(yy, pp).items()}})
            print(f"    {mo:4s} 방문 {r['방문 단위']:5.1f}  대상자 {r['대상자 단위']:5.1f}"
                  f"  차이 {r['부풀림']:+5.1f}", flush=True)

        # (3) 어떤 지표가 기여했는지
        d = bm.shap_importance(X, y, bm.F16, model="XGB", n_repeats=2)
        d["순위"] = range(1, len(d) + 1)
        ranks[tgt] = d.set_index("특징")["순위"]
        print("    기여 상위 4개: " + ", ".join(bm.KOR[f] for f in d.head(4)["특징"]))

    save(pd.DataFrame(perf), "structured_perf.csv")
    pd.DataFrame(ranks).to_csv(os.path.join(RES, "structured_shap.csv"),
                               encoding="utf-8-sig")


# =====================================================================
# 2. 비정형 : 대상자 한 명이 표본 하나
# =====================================================================
def run_longitudinal(metrics):
    print("\n" + "=" * 78)
    print("[비정형] 한 대상자의 0·2·4·6주를 하나의 표본으로 (141명)")
    print("=" * 78)
    wide, lab = bm.load_cohort(DIR_LONG)

    # 시간 정보를 넣는 여섯 가지 방법
    reps = {"raw2": bm.representation(wide, "raw2"),
            "raw4": bm.representation(wide, "raw4"),
            "traj": bm.representation(wide, "traj"),
            "poly": bm.representation(wide, "poly"),
            "poly012": bm.polynomial_basis(wide, ("P0", "P1", "P2")),
            "poly0123": bm.polynomial_basis(wide, ("P0", "P1", "P2", "P3"))}

    allrows, best, imp = [], [], {}
    for tgt, col, th in bm.TARGETS:
        idx = lab.index[lab[col].notna()]
        y = (lab.loc[idx, col] >= th).astype(int).values
        print(f"\n  {tgt}: 대상자 {len(y)}명, 양성 {y.sum()} ({100 * y.mean():.1f}%)")

        # (1) 표현 방식끼리 같은 조건으로 비교 -> 무엇을 쓸지 정한다
        R = bm.compare_representations({k: v.loc[idx] for k, v in reps.items()},
                                       y, seeds=SEEDS)
        R["표현형"] = tgt
        allrows.append(R)
        print(R.pivot(index="표현", columns="모형", values="AUC")
              .reindex(list(reps)).round(1).to_string().replace("\n", "\n    "))

        # (2) 정해진 표현으로 최종 지표
        X = reps[BEST_REP[tgt]].loc[idx].values
        for mo in MODELS:
            yy, pp = bm.oof_predict(X, y, mo, "record")   # 1명 1행이라 누수 없음
            metrics.append({"코호트": "비정형", "분할": f"대상자 단위 ({BEST_REP[tgt]})",
                            "표현형": tgt, "모형": mo, "표본": len(y), "양성": int(y.sum()),
                            **{k: round(v, 1) for k, v in bm.full_metrics(yy, pp).items()}})
        b = R.loc[R["AUC"].idxmax()]
        best.append({"표현형": tgt, "n": len(y), "양성": int(y.sum()),
                     "최고표현": b["표현"], "최고모형": b["모형"], "AUC": round(b["AUC"], 1)})

        # (3) 궤적 성분 가운데 무엇이 기여했는지 (평균 성분인지 기울기 성분인지)
        Xp = reps["poly"].loc[idx]
        names = [c.replace("P0_", "평균 · ").replace("P1_", "기울기 · ") for c in Xp.columns]
        names = [n.split(" · ")[0] + " · " + bm.KOR.get(n.split(" · ")[1], n.split(" · ")[1])
                 for n in names]
        d = bm.shap_importance(Xp.values, y, names, model="RF", n_repeats=2)
        imp[tgt] = d.set_index("특징")["SHAP"]

    save(pd.concat(allrows, ignore_index=True), "longitudinal_representations.csv")
    save(pd.DataFrame(best), "longitudinal_perf.csv")
    pd.DataFrame(imp).to_csv(os.path.join(RES, "longitudinal_shap.csv"),
                             encoding="utf-8-sig")


# =====================================================================
# 3. 서브그룹 : 동반 질환이 있는 사람만 따로
# =====================================================================
def run_subgroup():
    print("\n" + "=" * 78)
    print("[서브그룹] 수면장애 동반 / 불안장애 동반 우울군")
    print("=" * 78)
    wide, lab = bm.load_cohort(DIR_LONG)

    # 불면 관련 HAMD 세 문항(초기·중기·말기)의 합으로 수면장애 동반을 정의
    a = pd.read_excel(os.path.abspath(os.path.join(HERE, CFG["subject_xlsx"])),
                      sheet_name=0)
    a["pid"] = a["대상자번호"].astype(str).str.strip()
    a = a.drop_duplicates("pid").set_index("pid").reindex(wide.index)
    cols = ["Baseline_HAMD_4_초기불면증", "Baseline_HAMD_5_중기불면증",
            "Baseline_HAMD_6_말기불면증"]
    lab["SLEEP"] = pd.concat([pd.to_numeric(a[c], errors="coerce") for c in cols],
                             axis=1).sum(axis=1, min_count=3)

    reps = {"traj": bm.representation(wide, "traj"), "poly": bm.representation(wide, "poly")}
    dep = lab.index[(lab["grp"] == 1) & lab["HAMD_v4"].notna()]
    groups = {"동반 질환 구분 없음": dep,
              "수면장애 동반": dep[lab.loc[dep, "SLEEP"] >= 3],
              "불안장애 동반": dep[lab.loc[dep, "BAI_v1"] >= 16]}

    rows = []
    for name, idx in groups.items():
        y = (lab.loc[idx, "HAMD_v4"] >= 14).astype(int).values
        best = ("-", -1.0)
        for rep_name, X in reps.items():
            for mo in MODELS:
                au = np.mean([bm.fold_aucs(X.loc[idx].values, y, mo, seed=s).mean()
                              for s in SEEDS]) * 100
                if au > best[1]:
                    best = (f"{rep_name}+{mo}", au)
        rows.append({"서브그룹": name, "n": len(idx), "양성": int(y.sum()),
                     "최고모형": best[0], "AUC": round(best[1], 1)})
        print(f"  {name:16s} {len(idx):3d}명  양성 {y.sum():3d}"
              f" ({100 * y.mean():4.1f}%)  최고 AUC {best[1]:.1f} ({best[0]})")
    save(pd.DataFrame(rows), "subgroup_perf.csv")
    print("  ※ 불안 동반군의 AUC 가 낮은 것은 양성률이 높아 음성 사례가 10명뿐이기 때문입니다.")


# =====================================================================
# 4. 0주 불안에 따른 6주 예후
# =====================================================================
def run_anxiety():
    print("\n" + "=" * 78)
    print("[불안] 0주 불안이 심했던 사람과 아닌 사람의 6주 결과 비교")
    print("=" * 78)
    _, lab = bm.load_cohort(DIR_LONG)
    d = lab[(lab["grp"] == 1) & lab["BAI_v1"].notna()].copy()
    d["불안군"] = (d["BAI_v1"] >= 16).astype(float)
    print(f"  우울군 {len(d)}명 = 불안군 {int(d['불안군'].sum())}"
          f" + 비불안군 {int((d['불안군'] == 0).sum())}")

    # (1) 결과별 위험비
    rows = []
    for nm, cond, col in [("6주 중등도 이상 우울", d["HAMD_v4"] >= 14, "HAMD_v4"),
                          ("baseline 자살사고", d["SUI_v1"] >= 1, "SUI_v1"),
                          ("6주 자살사고", d["SUI_v4"] >= 1, "SUI_v4")]:
        ok = d[col].notna()
        s, c = d[ok], cond[ok]
        a_, na = int(c[s["불안군"] == 1].sum()), int((s["불안군"] == 1).sum())
        b_, nb = int(c[s["불안군"] == 0].sum()), int((s["불안군"] == 0).sum())
        _, p = stats.fisher_exact([[a_, na - a_], [b_, nb - b_]])
        rr = (a_ / na) / (b_ / nb)
        se = np.sqrt(1 / a_ - 1 / na + 1 / b_ - 1 / nb)
        rows.append({"결과": nm, "불안군": f"{a_}/{na} ({100 * a_ / na:.1f}%)",
                     "비불안군": f"{b_}/{nb} ({100 * b_ / nb:.1f}%)", "위험비": round(rr, 2),
                     "CI": f"{rr * np.exp(-1.96 * se):.2f}-{rr * np.exp(1.96 * se):.2f}",
                     "p": round(float(p), 4)})
        print(f"  {nm:22s} {rows[-1]['불안군']:>16s} vs {rows[-1]['비불안군']:>16s}"
              f"  위험비 {rr:.2f}  p={p:.4f}")
    save(pd.DataFrame(rows), "anxiety_clinical.csv")

    # (2) 보정하면 효과가 남는지
    print("\n  보정 회귀 (0주 우울 중증도, 나이, 성별을 차례로 보정)")
    d["여성"] = (d["성별"] == 2).astype(float)
    adj = []
    for tag, yv, col in [("6주 중등도 이상 우울", d["HAMD_v4"] >= 14, "HAMD_v4"),
                         ("6주 자살사고", d["SUI_v4"] >= 1, "SUI_v4")]:
        for nm, cs in [("불안 단독", ["불안군"]),
                       ("+ baseline HAMD", ["불안군", "HAMD_v1"]),
                       ("+ 나이·성별", ["불안군", "HAMD_v1", "나이", "여성"])]:
            t = d[cs + [col]].copy()
            t["y"] = yv.astype(float)
            t = t.drop(columns=[col]).dropna()
            r = sm.Logit(t["y"].values, sm.add_constant(t[cs].astype(float))).fit(disp=0)
            b_, se_ = r.params["불안군"], r.bse["불안군"]
            adj.append({"표적": tag, "보정": nm, "n": len(t), "OR": round(np.exp(b_), 2),
                        "CI하한": round(np.exp(b_ - 1.96 * se_), 2),
                        "CI상한": round(np.exp(b_ + 1.96 * se_), 2),
                        "p": round(float(r.pvalues["불안군"]), 4)})
            print(f"    {tag:22s} {nm:18s} 교차비 {adj[-1]['OR']:5.2f}"
                  f" [{adj[-1]['CI하한']:.2f}-{adj[-1]['CI상한']:.2f}]  p={adj[-1]['p']:.4f}")
    save(pd.DataFrame(adj), "anxiety_adjusted.csv")

    # (3) 자살사고가 생기고 사라지는 방향
    t = d[d["SUI_v1"].notna() & d["SUI_v4"].notna()].copy()
    t["전이"] = np.where(t["SUI_v1"] >= 1,
                        np.where(t["SUI_v4"] >= 1, "P→P", "P→N"),
                        np.where(t["SUI_v4"] >= 1, "N→P", "N→N"))
    ct = pd.crosstab(t["불안군"], t["전이"]).reindex(
        columns=["N→N", "N→P", "P→N", "P→P"], fill_value=0)
    ct.index = ["비불안군", "불안군"]
    ct.to_csv(os.path.join(RES, "anxiety_transition.csv"), encoding="utf-8-sig")
    print("\n  자살사고 상태 변화")
    print("   " + ct.to_string().replace("\n", "\n   "))


# =====================================================================
def main():
    want = sys.argv[1] if len(sys.argv) > 1 else "전부"
    metrics = []
    if want in ("전부", "정형"):
        run_structured(metrics)
    if want in ("전부", "비정형"):
        run_longitudinal(metrics)
    if metrics:
        old = os.path.join(RES, "full_metrics.csv")
        df = pd.DataFrame(metrics)
        if want != "전부" and os.path.exists(old):
            keep = pd.read_csv(old, encoding="utf-8-sig")
            keep = keep[~keep["코호트"].isin(df["코호트"].unique())]
            df = pd.concat([keep, df], ignore_index=True)
        save(df, "full_metrics.csv")
    if want in ("전부", "서브그룹"):
        run_subgroup()
    if want in ("전부", "불안"):
        run_anxiety()
    print(f"\n모든 결과표를 저장했습니다: {RES}")


if __name__ == "__main__":
    main()
