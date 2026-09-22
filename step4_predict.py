# -*- coding: utf-8 -*-
"""[4단계] 모형을 학습해 파일로 저장하고, 새 대상자에게 예측합니다.

2단계(step2_analyze.py)는 교차검증으로 성능을 재기만 합니다. 모형을 남기지 않습니다.
새로 들어온 대상자에게 실제로 써 보려면 이 단계가 필요합니다.

    python step4_predict.py 학습
        코호트 전체로 표현형 세 개의 모형을 학습해 모형/ 폴더에 저장합니다.
        저장물에는 학습에 쓴 열 목록과 표현 방식이 함께 들어갑니다.

    python step4_predict.py 예측 <visit_wide.csv 가 있는 폴더>
        저장된 모형으로 그 폴더의 대상자들에게 확률을 매겨
        <폴더>/예측결과.csv 로 저장합니다.

주의
    학습은 가진 자료를 전부 써서 한 번 적합시키는 것입니다. 여기서 나온 정확도는
    학습 자료에 대한 값이므로 성능 근거로 쓰면 안 됩니다.
    보고할 성능은 2단계의 교차검증 결과(결과/full_metrics.csv)입니다.
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
MODEL_DIR = os.path.join(HERE, "모형")

#: 표현형마다 쓰기로 정한 표현 방식과 분류기 (2단계 비교 실험에서 고른 조합)
CHOICE = {"중증도": ("traj", "RF"), "자살사고": ("poly", "RF"), "불안": ("traj", "RF")}


def 학습():
    """비정형 코호트 전체로 모형 세 개를 적합시켜 저장합니다."""
    os.makedirs(MODEL_DIR, exist_ok=True)
    wide, lab = bm.load_cohort(bm.find_cohort_dir(DATA, "비정형_"))

    for name, col, cut in bm.TARGETS:
        rep, algo = CHOICE[name]
        idx = lab.index[lab[col].notna()]
        y = (lab.loc[idx, col] >= cut).astype(int).values
        X = bm.representation(wide, rep).loc[idx]

        model = bm.make_models()[algo]
        model.fit(X.values, y)

        path = os.path.join(MODEL_DIR, f"{name}.joblib")
        joblib.dump({"모형": model, "표현방식": rep, "분류기": algo,
                     "열": list(X.columns), "표현형": name,
                     "기준": f"{col} >= {cut}", "학습표본": int(len(y)),
                     "양성": int(y.sum())}, path)
        print(f"  {name:6s} {algo}/{rep}  학습 {len(y)}명(양성 {y.sum()})  -> {path}")

    print()
    print("저장했습니다. 여기 적힌 정확도가 아니라 결과/full_metrics.csv 를 성능으로 쓰세요.")


def 예측(folder):
    """저장된 모형으로 확률을 매깁니다. visit_wide.csv 만 있으면 됩니다."""
    csv = os.path.join(folder, "visit_wide.csv")
    if not os.path.isfile(csv):
        raise FileNotFoundError(f"visit_wide.csv 가 없습니다: {folder}")
    wide = pd.read_csv(csv, index_col=0)
    out = pd.DataFrame(index=wide.index)

    for name, _, _ in bm.TARGETS:
        path = os.path.join(MODEL_DIR, f"{name}.joblib")
        if not os.path.isfile(path):
            print(f"  {name} 모형이 없습니다. 먼저 '학습' 을 돌리세요.")
            continue
        pack = joblib.load(path)
        X = bm.representation(wide, pack["표현방식"])
        # 학습 때와 열 순서·구성을 똑같이 맞춥니다. 빠진 열은 결측으로 두면
        # 파이프라인 안의 대치기가 채웁니다.
        X = X.reindex(columns=pack["열"])
        missing = int(X.isna().all().sum())
        if missing:
            print(f"  [주의] {name}: 학습에 쓰인 열 {missing}개가 통째로 비어 있습니다.")
        p = pack["모형"].predict_proba(X.values)[:, 1]
        out[f"{name}_확률"] = np.round(p, 4)
        out[f"{name}_판정"] = np.where(p >= 0.5, "양성", "음성")

    dst = os.path.join(folder, "예측결과.csv")
    out.to_csv(dst, encoding="utf-8-sig")
    print()
    print(out.head(10).to_string())
    print(f"\n{len(out)}명의 예측을 저장했습니다: {dst}")
    print("판정 임계값은 0.5 입니다. 선별 목적이면 낮추는 편이 낫습니다.")


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "학습"
    if what == "학습":
        학습()
    elif what == "예측":
        target = sys.argv[2] if len(sys.argv) > 2 else bm.find_cohort_dir(DATA, "비정형_")
        예측(target)
    else:
        print(__doc__)
