# -*- coding: utf-8 -*-
"""[3단계] 보고서와 발표자료에 들어가는 그림을 전부 만듭니다.

실행
    python step3_figures.py

만들어지는 그림 (결과/그림/ 아래, 같은 이름으로 TIFF 와 PNG 를 함께 저장)

  정형 (0주·6주를 각각 하나의 표본으로)
    s_roc_severity / s_roc_suicide / s_roc_anxiety      표현형별 ROC 곡선
    s_shap_severity / s_shap_suicide / s_shap_anxiety   표현형별 16지표 기여도
    s_split_compare                                     방문 단위 vs 대상자 단위
    s_rank_heatmap                                      세 표현형의 지표 순위 비교

  비정형 (대상자 한 명이 표본 하나)
    poly_concept                     직교 다항 대비가 무엇인지 설명하는 그림
    l_representation_3panel          표현 방식 6종 비교
    l_structured_vs_longitudinal     정형과 비정형 성능 비교
    l_shap_severity / l_shap_suicide / l_shap_anxiety   궤적 성분별 기여도

  0주 불안에 따른 예후
    a_outcomes                       불안군 vs 비불안군 6주 결과
    a_forest                         보정 단계별 교차비
    a_transition                     자살사고 상태 변화

규칙 두 가지
  1. 그림 안에는 제목과 설명을 넣지 않습니다. 캡션은 문서 본문에 씁니다.
     (같은 그림을 보고서와 발표자료에 쓸 때 캡션만 바꾸면 되기 때문입니다.)
  2. 막대에 적는 숫자는 결과/full_metrics.csv 에서 읽어 옵니다.
     표와 그림이 어긋나지 않게 하려는 것입니다.
"""
import os
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
os.environ.setdefault("JOBLIB_TEMP_FOLDER", "C:/jltmp")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, roc_auc_score

import biomarker as bm

HERE = os.path.dirname(os.path.abspath(__file__))
CFG = bm.load_config(os.path.join(HERE, "config.json"))
DATA = os.path.abspath(os.path.join(HERE, CFG["out_dir"]))
RES = os.path.join(HERE, "결과")
FIG = os.path.join(RES, "그림")
os.makedirs(FIG, exist_ok=True)

bm.use_korean_font()
VLAG = sns.color_palette("vlag", 256)
RED, BLUE = VLAG[-1], VLAG[0]
COLORS = [RED, BLUE, "#e67e22", "#27ae60"]
MODELS = ["LR", "RF", "XGB", "SVM"]
FULLNAME = {"LR": "Logistic Regression", "RF": "Random Forest",
            "XGB": "Gradient Boosting", "SVM": "Support Vector Machine"}
TG = ["중증도", "자살사고", "불안"]
SLUG = {"중증도": "severity", "자살사고": "suicide", "불안": "anxiety"}

M = pd.read_csv(os.path.join(RES, "full_metrics.csv"), encoding="utf-8-sig")
REP = pd.read_csv(os.path.join(RES, "longitudinal_representations.csv"),
                  encoding="utf-8-sig")


def auc(cohort, split, tgt, model):
    q = M[(M["코호트"] == cohort) & M["분할"].str.startswith(split)
          & (M["표현형"] == tgt) & (M["모형"] == model)]
    return float(q["AUC"].iloc[0])


def save(fig, stem):
    for ext, kw in [("tiff", {"pil_kwargs": {"compression": "tiff_lzw"}}), ("png", {})]:
        fig.savefig(os.path.join(FIG, f"{stem}.{ext}"), format=ext,
                    bbox_inches="tight", dpi=600, facecolor="white", **kw)
    plt.close(fig)
    print(f"    {stem}")


def bare(ax, axis="y"):
    ax.grid(axis=axis, alpha=.15)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


# =====================================================================
def figures_structured():
    print("  정형 그림")
    wide, lab = bm.load_cohort(bm.find_cohort_dir(DATA, "정형_"))
    rows = bm.to_visit_rows(wide, ("v1", "v4"))

    for tgt, col, th in bm.TARGETS:
        ok = lab[col].notna()
        m = rows[rows["pid"].isin(ok.index[ok])].reset_index(drop=True)
        y = (lab[col].reindex(m["pid"]) >= th).astype(int).values
        X = m[bm.F16].values.astype(float)

        # ROC (방문 단위 분할 기준)
        fig, ax = plt.subplots(figsize=(8, 8), dpi=600)
        for mo, c in zip(MODELS, COLORS):
            yy, pp = bm.oof_predict(X, y, mo, "record", n_repeats=3)
            fpr, tpr, _ = roc_curve(yy, pp)
            ax.plot(fpr, tpr, color=c, lw=2.2,
                    label=f"{FULLNAME[mo]} (AUC = {roc_auc_score(yy, pp):.3f})")
        ax.plot([0, 1], [0, 1], ls="--", lw=1.2, color="#999")
        ax.set_xlabel("1 - Specificity", fontsize=14)
        ax.set_ylabel("Sensitivity", fontsize=14)
        ax.set_xlim(-.01, 1.01); ax.set_ylim(-.01, 1.01)
        ax.tick_params(labelsize=12); ax.legend(loc="lower right", fontsize=12)
        bare(ax, "both"); save(fig, f"s_roc_{SLUG[tgt]}")

        # SHAP 가로 막대
        imp = bm.shap_importance(X, y, [bm.KOR[f] for f in bm.F16],
                                 model="XGB", n_repeats=2)
        d = imp.sort_values("SHAP")
        fig, ax = plt.subplots(figsize=(9, 7), dpi=600)
        ax.barh(d["특징"], d["SHAP"], color="#ff0051", height=.55)
        mx = d["SHAP"].max()
        for i, v in enumerate(d["SHAP"]):
            ax.text(v + mx * .012, i, f"+{v:.3f}", va="center", fontsize=10,
                    color="#ff0051", fontweight="bold")
        ax.set_xlabel("mean(|SHAP value|)", fontsize=13)
        ax.set_xlim(0, mx * 1.16); ax.tick_params(labelsize=11)
        bare(ax, "x"); save(fig, f"s_shap_{SLUG[tgt]}")

    # 분할 방식 비교
    fig, ax = plt.subplots(figsize=(9.5, 5.6), dpi=600)
    x = np.arange(12, dtype=float) + np.repeat(np.arange(3), 4) * .9
    rec = [auc("정형", "방문", t, m) for t in TG for m in MODELS]
    sub = [auc("정형", "대상자", t, m) for t in TG for m in MODELS]
    ax.bar(x - .19, rec, .38, color=RED, label="방문 단위 분할")
    ax.bar(x + .19, sub, .38, color=BLUE, label="대상자 단위 분할")
    ax.axhline(50, ls=":", lw=1.2, color="#999")
    ax.set_xticks(x); ax.set_xticklabels(MODELS * 3, fontsize=10)
    for i, t in enumerate(TG):
        ax.text(x[i * 4:(i + 1) * 4].mean(), 96, t, ha="center",
                fontsize=12.5, fontweight="bold")
    ax.set_ylim(40, 100); ax.set_ylabel("AUC (%)", fontsize=12)
    ax.legend(fontsize=10.5, ncol=2, loc="upper center",
              bbox_to_anchor=(.5, -.07), frameon=False)
    bare(ax); save(fig, "s_split_compare")

    # 지표 순위 히트맵
    S = pd.read_csv(os.path.join(RES, "structured_shap.csv"), index_col=0,
                    encoding="utf-8-sig")
    S.index = [bm.KOR.get(i, i) for i in S.index]
    S = S.loc[S.mean(axis=1).sort_values().index]
    fig, ax = plt.subplots(figsize=(7.2, 8), dpi=600)
    sns.heatmap(S, annot=True, fmt=".0f", cmap="vlag_r", ax=ax, cbar=False,
                linewidths=.6, linecolor="white",
                annot_kws={"fontsize": 11, "fontweight": "bold"})
    ax.set_xticklabels(S.columns, fontsize=12)
    ax.set_yticklabels(S.index, fontsize=11, rotation=0)
    ax.tick_params(left=False, bottom=False)
    save(fig, "s_rank_heatmap")


# =====================================================================
def figures_longitudinal():
    print("  비정형 그림")
    order = ["raw2", "raw4", "traj", "poly", "poly012", "poly0123"]
    lab = {"raw2": "원값\n2시점", "raw4": "원값\n4시점", "traj": "궤적\n요약",
           "poly": "P0+P1", "poly012": "+2차", "poly0123": "+3차"}

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), dpi=600, sharey=True)
    for ax, tgt in zip(axes, TG):
        s = REP[REP["표현형"] == tgt]
        x = np.arange(len(order), dtype=float)
        for m, c in zip(MODELS, COLORS):
            v = [s[(s["표현"] == o) & (s["모형"] == m)]["AUC"].iloc[0] for o in order]
            ax.plot(x, v, "o-", color=c, lw=2, ms=6.5, mec="white", mew=.9, label=m)
        ax.set_xticks(x); ax.set_xticklabels([lab[o] for o in order], fontsize=10)
        ax.axvline(3, ls=":", lw=1.3, color="#999")
        ax.set_title(tgt, fontsize=14, fontweight="bold", pad=10)
        bare(ax)
    axes[0].set_ylabel("AUC (%)", fontsize=12)
    axes[0].set_ylim(70, 92)
    axes[0].legend(fontsize=10, ncol=2, loc="lower left")
    fig.tight_layout(); save(fig, "l_representation_3panel")

    # 궤적 성분 기여도
    S = pd.read_csv(os.path.join(RES, "longitudinal_shap.csv"), index_col=0,
                    encoding="utf-8-sig")
    for tgt in TG:
        d = S[tgt].dropna().sort_values().tail(14)
        fig, ax = plt.subplots(figsize=(9, 6.5), dpi=600)
        ax.barh(list(d.index), d.values, color="#ff0051", height=.55)
        mx = d.max()
        for i, v in enumerate(d.values):
            ax.text(v + mx * .012, i, f"+{v:.3f}", va="center", fontsize=10,
                    color="#ff0051", fontweight="bold")
        ax.set_xlabel("mean(|SHAP value|)", fontsize=13)
        ax.set_xlim(0, mx * 1.18); ax.tick_params(labelsize=10.5)
        bare(ax, "x"); save(fig, f"l_shap_{SLUG[tgt]}")

    # 정형 대 비정형
    fig, ax = plt.subplots(figsize=(8.6, 5.2), dpi=600)
    st = [max(auc("정형", "대상자", t, m) for m in MODELS) for t in TG]
    lo = [max(auc("비정형", "대상자", t, m) for m in MODELS) for t in TG]
    x = np.arange(3)
    for off, v, c, l in [(-.19, st, BLUE, "정형 0·6주 (대상자 단위)"),
                         (.19, lo, RED, "비정형 0~6주 궤적")]:
        b = ax.bar(x + off, v, .38, color=c, label=l)
        for r in b:
            ax.text(r.get_x() + r.get_width() / 2, r.get_height() + .8,
                    f"{r.get_height():.1f}", ha="center", fontsize=11, color="#333")
    for i in range(3):
        ax.annotate(f"+{lo[i] - st[i]:.1f}", xy=(i, max(st[i], lo[i]) + 4.0),
                    ha="center", fontsize=12, fontweight="bold", color=RED)
    ax.set_xticks(x); ax.set_xticklabels(TG, fontsize=13)
    ax.set_ylim(60, 95); ax.set_ylabel("AUC (%)", fontsize=12)
    ax.legend(fontsize=11, ncol=2, loc="upper center",
              bbox_to_anchor=(.5, -.09), frameon=False)
    bare(ax); save(fig, "l_structured_vs_longitudinal")


# =====================================================================
def figure_poly_concept():
    """직교 다항 대비를 말로 설명하기 어려워 그림으로 만들어 둔 것."""
    print("  직교 다항 설명 그림")
    wk = np.array([0, 2, 4, 6], float)
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.6), dpi=600)
    ax = axes[0]
    style = {"P0": ("평균 성분 (P0)", "#1f4e9c", "-"), "P1": ("선형 성분 (P1)", "#d62839", "-"),
             "P2": ("2차 성분 (P2)", "#e58f27", "--"), "P3": ("3차 성분 (P3)", "#2a9d5c", "--")}
    for k, (l, c, ls) in style.items():
        ax.plot(wk, bm.POLY_CONTRAST[k], ls, color=c, lw=2.2, marker="o", ms=7,
                mec="white", mew=1.1, label=l)
    ax.axhline(0, color="#aaa", lw=1)
    ax.set_xticks(wk); ax.set_xticklabels(["0주", "2주", "4주", "6주"], fontsize=12)
    ax.set_ylabel("대비 계수(weight)", fontsize=12)
    ax.set_title("네 시점에 부여하는 가중치", fontsize=13.5, fontweight="bold", pad=10)
    ax.legend(fontsize=10.5, loc="upper left", ncol=2)
    ax.tick_params(labelsize=11); bare(ax, "both")

    ax = axes[1]
    obs = np.array([320.0, 285.0, 250.0, 205.0])          # 설명용 예시 궤적
    p0 = (obs * bm.POLY_CONTRAST["P0"]).sum() / (bm.POLY_CONTRAST["P0"] ** 2).sum()
    p1 = (obs * bm.POLY_CONTRAST["P1"]).sum() / (bm.POLY_CONTRAST["P1"] ** 2).sum()
    ax.plot(wk, obs, "-o", color="#333", lw=2.4, ms=8, mec="white", mew=1.2,
            label="측정된 총 활동 시간")
    ax.axhline(p0, color="#1f4e9c", lw=2, label=f"평균 성분 P0 = {p0:.0f}분")
    ax.plot(wk, p0 + p1 * bm.POLY_CONTRAST["P1"], "--", color="#d62839", lw=2,
            label=f"평균 + 선형 성분 (P1 = {p1:.1f})")
    ax.set_xticks(wk); ax.set_xticklabels(["0주", "2주", "4주", "6주"], fontsize=12)
    ax.set_ylabel("총 활동 시간 (분)", fontsize=12)
    ax.set_title("한 대상자의 궤적을 두 숫자로 요약", fontsize=13.5, fontweight="bold", pad=10)
    ax.legend(fontsize=10.5, loc="upper right")
    ax.tick_params(labelsize=11); bare(ax, "both")
    fig.tight_layout(); save(fig, "poly_concept")


# =====================================================================
def figures_anxiety():
    print("  불안 그림")
    C = pd.read_csv(os.path.join(RES, "anxiety_clinical.csv"), encoding="utf-8-sig")
    fig, ax = plt.subplots(figsize=(8.4, 5.0), dpi=600)
    labs = ["6주 중등도\n이상 우울", "baseline\n자살사고", "6주\n자살사고"]
    anx = [float(s.split("(")[1].rstrip("%)")) for s in C["불안군"]]
    non = [float(s.split("(")[1].rstrip("%)")) for s in C["비불안군"]]
    x = np.arange(3)
    for off, v, c, l in [(-.18, anx, RED, "불안군 (n=64)"), (.18, non, BLUE, "비불안군 (n=36)")]:
        b = ax.bar(x + off, v, .36, color=c, label=l)
        for r in b:
            ax.text(r.get_x() + r.get_width() / 2, r.get_height() + 1.5,
                    f"{r.get_height():.1f}", ha="center", fontsize=10.5, color="#333")
    ax.set_xticks(x); ax.set_xticklabels(labs, fontsize=11)
    ax.set_ylim(0, 100); ax.set_ylabel("해당 대상자 비율 (%)", fontsize=12)
    ax.legend(fontsize=11, loc="upper right"); bare(ax); save(fig, "a_outcomes")

    D = pd.read_csv(os.path.join(RES, "anxiety_adjusted.csv"), encoding="utf-8-sig")
    fig, ax = plt.subplots(figsize=(8.6, 4.6), dpi=600)
    yp = np.arange(len(D))[::-1]
    for y_, r in zip(yp, D.itertuples()):
        c = RED if r.CI하한 > 1 else "#9aa3b2"
        ax.plot([r.CI하한, r.CI상한], [y_, y_], color=c, lw=2.4, solid_capstyle="round")
        ax.plot(r.OR, y_, "o", color=c, ms=9.5, mec="white", mew=1.3, zorder=3)
        ax.text(r.CI상한 * 1.12, y_, f"{r.OR:.2f}", va="center", fontsize=10.5,
                color=c, fontweight="bold")
    ax.axvline(1, ls="--", lw=1.4, color="#333")
    ax.set_yticks(yp)
    ax.set_yticklabels([f"{r.표적} · {r.보정}" if r.보정 == "불안 단독" else f"    {r.보정}"
                        for r in D.itertuples()], fontsize=10.5)
    ax.set_xscale("log"); ax.set_xlim(.4, 32)
    ax.set_xticks([.5, 1, 2, 5, 10, 20])
    ax.set_xticklabels(["0.5", "1", "2", "5", "10", "20"], fontsize=11)
    ax.set_xlabel("교차비 (95% 신뢰구간, 로그 눈금)", fontsize=12)
    ax.grid(axis="x", alpha=.15)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    save(fig, "a_forest")

    T = pd.read_csv(os.path.join(RES, "anxiety_transition.csv"), index_col=0,
                    encoding="utf-8-sig")
    fig, ax = plt.subplots(figsize=(8.4, 5.0), dpi=600)
    st = ["N→N", "N→P", "P→N", "P→P"]
    x = np.arange(4)
    for off, idx, c, l in [(-.18, "불안군", RED, "불안군"), (.18, "비불안군", BLUE, "비불안군")]:
        v = [int(T.loc[idx, s]) for s in st]
        b = ax.bar(x + off, v, .36, color=c, label=l)
        for r in b:
            ax.text(r.get_x() + r.get_width() / 2, r.get_height() + .6,
                    f"{int(r.get_height())}", ha="center", fontsize=11, color="#333")
    ax.set_xticks(x); ax.set_xticklabels(st, fontsize=13)
    ax.set_ylabel("대상자 수 (명)", fontsize=12)
    ax.legend(fontsize=11, loc="upper left"); bare(ax); save(fig, "a_transition")


if __name__ == "__main__":
    figures_structured()
    figure_poly_concept()
    figures_longitudinal()
    figures_anxiety()
    print(f"\n모든 그림을 저장했습니다: {FIG}")
