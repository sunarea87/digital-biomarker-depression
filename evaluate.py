# -*- coding: utf-8 -*-
"""평가: 최종 표현으로 성능표와 그림을 만든다.

실행::

    python evaluate.py --config config.json
    python evaluate.py --rep traj          # 다른 표현으로 평가

만들어지는 파일 (work_dir 아래)
    metrics.csv          모형별 정확도·민감도·특이도·F1·AUC 와 95% 신뢰구간
    importance.csv       SHAP 중요도
    fig_roc.tiff/.png    ROC 곡선
    fig_importance.*     중요도 막대

모든 평가는 한 대상자를 하나의 표본으로 고정한 교차검증으로 이루어진다.
그림 안에는 제목이나 설명을 넣지 않는다. 캡션은 문서에서 붙이는 편이
같은 그림을 논문과 발표자료에 함께 쓰기 좋기 때문이다.
"""
import argparse

import pandas as pd

import core
from train import load_matrix


def main():
    ap = argparse.ArgumentParser(description="최종 평가와 그림 생성")
    ap.add_argument("--config", default="config.json")
    ap.add_argument("--rep", default="poly", choices=["raw2", "raw4", "traj", "poly"],
                    help="사용할 표현. 기본 poly = 직교 다항 P0+P1")
    ap.add_argument("--shap-model", default="RF", help="SHAP 을 뽑을 트리 모형")
    ap.add_argument("--n-boot", type=int, default=1000, help="신뢰구간 부트스트랩 반복")
    args = ap.parse_args()

    cfg = core.load_config(args.config)
    core.use_korean_font()

    wide, y = load_matrix(cfg)
    X = core.representation(wide, args.rep)
    print(f"표현 {args.rep}  |  대상자 {len(y)}명  |  특징 {X.shape[1]}개  "
          f"|  양성 {y.sum()} ({100*y.mean():.1f}%)")

    # ---- 성능표와 ROC
    rows, curves = [], {}
    for name in core.make_models():
        yy, pp = core.cross_val_predict_patient(X.values, y, name)
        curves[name] = (yy, pp)
        rows.append({"모형": name, **{k: (round(v, 1) if isinstance(v, float) else v)
                                     for k, v in core.metrics_with_ci(
                                         yy, pp, n_boot=args.n_boot).items()}})
    M = pd.DataFrame(rows)
    M.to_csv(core.work_path(cfg, "metrics.csv"), index=False, encoding="utf-8-sig")
    print()
    print(M[["모형", "ACC", "Sen", "Spec", "F1", "AUC", "AUC_CI"]].to_string(index=False))

    core.plot_roc(curves, core.work_path(cfg, "fig_roc"))
    print("\nfig_roc.tiff / .png 저장")

    # ---- 중요도
    imp = core.shap_importance(X.values, y, X.columns, model=args.shap_model)
    imp.to_csv(core.work_path(cfg, "importance.csv"), index=False, encoding="utf-8-sig")
    core.plot_importance(imp, core.work_path(cfg, "fig_importance"))
    print("fig_importance.tiff / .png 저장\n")
    print("상위 8개 특징")
    print(imp.head(8).to_string(index=False))


if __name__ == "__main__":
    main()
