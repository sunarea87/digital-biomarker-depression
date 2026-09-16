# -*- coding: utf-8 -*-
"""학습: 시계열 표현 방식을 비교한다.

실행::

    python train.py --config config.json                 # 표현 비교 (기본)
    python train.py --task ablation                      # 궤적 요약 제거 실험
    python train.py --task basis                         # 직교 다항 차수 실험
    python train.py --task leakage                       # 방문 혼재의 영향

무엇을 묻는가
    같은 대상자를 6주간 네 번 관찰했을 때, 그 네 점을 어떤 형태로 모형에
    넣어야 하는가. 시점을 그냥 늘리는 것과, 궤적의 형태를 뽑아 주는 것과,
    직교 다항 기저로 분해하는 것 중 무엇이 나은가.

결론 요약 (자세한 내용은 README)
    시점을 늘리기만 하면 선형 모형에서는 오히려 나빠진다.
    지표당 평균(P0)과 선형 방향(P1) 두 계수면 충분하며, 2차 이상은 해롭다.
"""
import argparse

import numpy as np
import pandas as pd

import core


def load_matrix(cfg):
    """전처리 결과를 읽어 (특징표, 라벨) 로 돌려준다."""
    wide = pd.read_csv(core.work_path(cfg, "visit_wide.csv"), index_col=0,
                       encoding="utf-8-sig")
    lab = pd.read_csv(core.work_path(cfg, "cohort.csv"), index_col=0,
                      encoding="utf-8-sig")
    lab = lab[lab["y_moderate_w6"].notna()]
    idx = wide.index.intersection(lab.index)
    return wide.loc[idx], lab.loc[idx, "y_moderate_w6"].astype(int).values


def task_compare(cfg, seeds):
    """표현 네 가지를 같은 조건에서 비교한다."""
    wide, y = load_matrix(cfg)
    reps = {name: core.representation(wide, name)
            for name in ("raw2", "raw4", "traj", "poly")}
    print(f"대상자 {len(y)}명, 양성 {y.sum()} ({100*y.mean():.1f}%)")
    for k, v in reps.items():
        print(f"  {k:6s} 차원 {v.shape[1]:4d}")
    R = core.compare_representations(reps, y, seeds=seeds)
    R.to_csv(core.work_path(cfg, "compare_representations.csv"), index=False,
             encoding="utf-8-sig")
    print()
    print(R.pivot(index="표현", columns="모형", values="AUC").round(1).to_string())
    return R


def task_ablation(cfg, seeds):
    """궤적 요약 세 가지 중 무엇이 이득을 만드는지 하나씩 넣어 가른다."""
    wide, y = load_matrix(cfg)
    raw4 = core.representation(wide, "raw4")
    tr = core.trajectory_summary(wide)
    pick = lambda pre: tr[[c for c in tr.columns if c.startswith(pre)]]
    reps = {"원값 4시점": raw4,
            "+ 기울기": pd.concat([raw4, pick("SLP_")], axis=1),
            "+ 총변화": pd.concat([raw4, pick("TOT_")], axis=1),
            "+ 최대변화": pd.concat([raw4, pick("MXD_")], axis=1),
            "+ 셋 다": pd.concat([raw4, tr], axis=1)}
    R = core.compare_representations(reps, y, seeds=seeds)
    R.to_csv(core.work_path(cfg, "ablation_trajectory.csv"), index=False,
             encoding="utf-8-sig")
    P = R.pivot(index="표현", columns="모형", values="AUC")
    base = P.loc["원값 4시점"]
    print((P - base).round(2).to_string())
    print("\n원값 4시점 대비 이득. 최대 인접변화는 기여가 없다.")
    return R


def task_basis(cfg, seeds):
    """직교 다항 기저를 차수별로 쌓아가며 어디서 이득이 멈추는지 본다."""
    wide, y = load_matrix(cfg)
    reps = {"원값 2시점": core.representation(wide, "raw2"),
            "원값 4시점": core.representation(wide, "raw4"),
            "P0 평균만": core.polynomial_basis(wide, ("P0",)),
            "P0+P1 평균·선형": core.polynomial_basis(wide, ("P0", "P1")),
            "P0..P2 +2차": core.polynomial_basis(wide, ("P0", "P1", "P2")),
            "P0..P3 +3차": core.polynomial_basis(wide, ("P0", "P1", "P2", "P3"))}
    R = core.compare_representations(reps, y, seeds=seeds)
    R.to_csv(core.work_path(cfg, "basis_orders.csv"), index=False, encoding="utf-8-sig")
    print(R.pivot(index="표현", columns="모형", values="AUC").round(1).to_string())
    print("\nP0+P1 이 가장 좋고 2차 이상은 성능을 떨어뜨린다.")
    return R


def task_leakage(cfg, seeds):
    """같은 대상자의 방문을 독립 표본으로 다뤘을 때의 부풀림을 잰다."""
    wide, y = load_matrix(cfg)
    rows = []
    for v in ("v1", "v4"):
        cols = [f"{f}__{v}" for f in core.F16 if f"{f}__{v}" in wide.columns]
        sub = wide[cols].copy()
        sub.columns = core.F16[:len(cols)]
        sub["_pid"] = wide.index
        sub["_y"] = y
        rows.append(sub)
    long = pd.concat(rows, ignore_index=True)
    X = long[[c for c in long.columns if not c.startswith("_")]].values
    for m in core.make_models():
        r = core.record_vs_subject(X, long["_y"].values, long["_pid"].values, m)
        rows_out = {"모형": m, **{k: round(v, 2) for k, v in r.items()}}
        print("  " + "  ".join(f"{k} {v}" for k, v in rows_out.items()))
    print("\n방문 단위 분할은 같은 사람이 학습과 평가에 동시에 들어가 성능을 부풀린다.")


def main():
    ap = argparse.ArgumentParser(description="시계열 표현 비교 학습")
    ap.add_argument("--config", default="config.json")
    ap.add_argument("--task", default="compare",
                    choices=["compare", "ablation", "basis", "leakage"])
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 7, 101, 2024],
                    help="교차검증 분할 시드. 여러 개를 주면 안정성을 함께 본다")
    args = ap.parse_args()
    cfg = core.load_config(args.config)
    {"compare": task_compare, "ablation": task_ablation,
     "basis": task_basis, "leakage": task_leakage}[args.task](cfg, args.seeds)


if __name__ == "__main__":
    main()
