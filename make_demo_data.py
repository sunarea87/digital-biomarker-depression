# -*- coding: utf-8 -*-
"""예제 데이터 만들기.

코드를 처음 받아보는 분이 실제 환자 데이터 없이도 전체 흐름을
돌려볼 수 있도록, 가짜 환자 데이터를 만들어 줍니다.

중요: 여기서 만드는 숫자는 **전부 컴퓨터가 지어낸 것**입니다.
실제 환자분의 기록이 아니며, 한 줄도 실제 값이 들어가지 않습니다.
다만 실제 데이터와 비슷하게 보이도록 평균과 퍼짐 정도만 맞췄습니다.

실행::

    python make_demo_data.py                 # demo_data/ 에 120명 생성
    python make_demo_data.py --n 30          # 인원을 바꾸고 싶을 때
                                             # 30명 아래면 결과가 크게 흔들립니다

만들어지는 파일
    demo_data/visit_wide.csv   1행 = 1명. 16지표 x 4시점 = 64열
    demo_data/cohort.csv       1행 = 1명. 6주 우울 중증도 정답
"""
import argparse
import os

import numpy as np
import pandas as pd

import core

# 실제 코호트에서 관찰된 지표별 평균과 표준편차.
# 이 두 숫자만 쓰고 개인 값은 전혀 쓰지 않는다.
STATS = {
    "SQ_deep_minutes":          (51.7, 21.2),
    "SQ_wake_minutes":          (54.2, 19.0),
    "SQ_light_minutes":        (217.3, 66.0),
    "SQ_rem_minutes":           (67.4, 27.7),
    "SQ_efficiency":            (93.0, 2.1),
    "SQ_time_in_bed":          (412.7, 100.3),
    "HRV_low_frequency":       (639.4, 931.9),
    "HRV_high_frequency":      (307.2, 575.8),
    "HRV_entropy":               (2.27, 0.48),
    "HRV_nremhr":               (64.9, 9.5),
    "HRV_rmssd":                (26.1, 15.6),
    "ACT_distances_weekday":   (821.6, 379.1),
    "ACT_distances_weekend":   (759.6, 363.0),
    "ACT_calories_weekday":      (1.43, 0.39),
    "ACT_calories_weekend":      (1.40, 0.36),
    "ACT_total_active_minutes": (241.9, 82.4),
}

#: 같은 사람의 네 시점은 서로 닮아 있다. 실제 자료에서 0.70 이었다.
WITHIN_PERSON_R = 0.70

#: 6주 중증도와 관련이 있다고 알려진 지표들. 예제에서도 약한 신호를 넣어둔다.
SIGNAL_FEATURES = ["ACT_distances_weekday", "ACT_distances_weekend",
                   "SQ_deep_minutes", "ACT_total_active_minutes",
                   "SQ_efficiency", "HRV_rmssd"]


def make_demo(n_patients: int = 120, positive_rate: float = 0.65, seed: int = 0):
    """가짜 환자 n명의 방문 요약표와 정답을 만든다.

    사람마다 고유한 기본값을 먼저 뽑고(개인차), 그 위에 시점별 흔들림을 얹는다.
    실제 웨어러블 자료가 그런 구조이기 때문이다. 개인차가 크고 시점 변화는 작다.
    """
    rng = np.random.default_rng(seed)
    pids = [f"DEMO-{i:03d}" for i in range(1, n_patients + 1)]
    y = (rng.random(n_patients) < positive_rate).astype(int)

    cols = {}
    for feat, (mu, sd) in STATS.items():
        # 1) 개인 고유 수준
        person = rng.normal(mu, sd * np.sqrt(WITHIN_PERSON_R), n_patients)
        # 2) 중증도가 높은 사람은 활동과 깊은 잠이 조금 낮게
        if feat in SIGNAL_FEATURES:
            person = person - y * sd * 0.55
        for k, v in enumerate(core.VISITS):
            noise = rng.normal(0, sd * np.sqrt(1 - WITHIN_PERSON_R), n_patients)
            # 3) 시간이 지나며 조금씩 회복되는 경향 (양성군은 덜 회복)
            drift = (k / 3.0) * sd * 0.20 * np.where(y == 1, 0.3, 1.0)
            val = person + noise + (drift if feat in SIGNAL_FEATURES else 0)
            if feat == "SQ_efficiency":
                val = np.clip(val, 70, 100)
            else:
                val = np.clip(val, mu * 0.05, None)
            cols[f"{feat}__{v}"] = np.round(val, 2)

    wide = pd.DataFrame(cols, index=pd.Index(pids, name="pid"))
    wide = wide[[f"{f}__{v}" for f in core.F16 for v in core.VISITS]]
    cohort = pd.DataFrame({"y_moderate_w6": y}, index=pd.Index(pids, name="pid"))
    return wide, cohort


def main():
    ap = argparse.ArgumentParser(description="예제용 가짜 데이터 만들기")
    ap.add_argument("--n", type=int, default=120, help="만들 사람 수")
    ap.add_argument("--out", default="demo_data", help="저장할 폴더")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    wide, cohort = make_demo(args.n, seed=args.seed)
    wide.to_csv(os.path.join(args.out, "visit_wide.csv"), encoding="utf-8-sig")
    cohort.to_csv(os.path.join(args.out, "cohort.csv"), encoding="utf-8-sig")

    n_pos = int(cohort["y_moderate_w6"].sum())
    print(f"{args.out}/visit_wide.csv   {wide.shape[0]}명 x {wide.shape[1]}열")
    print(f"{args.out}/cohort.csv       양성 {n_pos}명 / 음성 {len(cohort)-n_pos}명")
    print()
    print("이 숫자들은 전부 지어낸 것입니다. 실제 환자 기록이 아닙니다.")
    print()
    print("이제 이렇게 돌려보실 수 있습니다.")
    print("  python train.py --config demo_config.json --task basis")
    print("  python evaluate.py --config demo_config.json")


if __name__ == "__main__":
    main()
