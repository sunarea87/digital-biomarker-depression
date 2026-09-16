# -*- coding: utf-8 -*-
"""전처리: Fitbit 원자료를 분석용 표로 만든다.

실행::

    python preprocess.py --config config.json

만들어지는 파일 (work_dir 아래)
    daily_long.csv       1행 = 1대상자 x 1날짜. 궤적 인코딩의 원재료
    coverage.csv         창별 확보 일수. 코호트 선정 근거를 남긴다
    visit_wide.csv       1행 = 1대상자. 지표 x 방문 요약
    cohort.csv           최종 코호트 명단과 라벨

코호트 기준은 core.MIN_DAYS_PER_WINDOW 로 정한다. 기본 3일이며,
네 창 모두에서 수면·심박변이·활동이 각각 3일 이상 있어야 통과한다.
"""
import argparse
import os

import pandas as pd

import core


def load_subjects(xlsx: str) -> pd.DataFrame:
    """피험자 표에서 방문일과 라벨을 읽는다.

    열 이름이 기관마다 다를 수 있어 위치가 아니라 이름으로 찾는다.
    필요한 것은 대상자번호, 첫 방문일, 마지막 방문일, HAMD 총점, 군 구분이다.
    """
    df = pd.read_excel(xlsx, sheet_name=0)
    df.columns = [str(c).strip() for c in df.columns]

    def pick(*cands, required=True):
        for c in cands:
            for col in df.columns:
                if c in col:
                    return col
        if required:
            raise KeyError(f"열을 찾지 못했습니다: {cands}")
        return None

    out = pd.DataFrame()
    out["pid"] = df[pick("대상자번호", "subject_id")].astype(str).str.strip()
    out["v1_date"] = pd.to_datetime(df[pick("V1날짜", "V1_방문일", "방문일")],
                                    errors="coerce")
    out["v4_date"] = pd.to_datetime(df[pick("V4 날짜", "V4날짜", "V4_방문일")],
                                    errors="coerce")
    out["hamd_v1"] = pd.to_numeric(df[pick("HamD점수", "Baseline_HAMD_total")],
                                   errors="coerce")
    out["hamd_v4"] = pd.to_numeric(df[pick("V4_HAMD점수", "F/U_HAMD_total")],
                                   errors="coerce")
    g = pick("우울증_0무1유", "group", required=False)
    out["is_patient"] = pd.to_numeric(df[g], errors="coerce") if g else 1
    return out.drop_duplicates("pid").dropna(subset=["v1_date", "v4_date"])


def main():
    ap = argparse.ArgumentParser(description="Fitbit 원자료 전처리")
    ap.add_argument("--config", default="config.json")
    ap.add_argument("--min-days", type=int, default=core.MIN_DAYS_PER_WINDOW,
                    help="창(7일)당 최소 확보 일수")
    args = ap.parse_args()

    cfg = core.load_config(args.config)
    raw_root = cfg["raw_root"]
    subs = load_subjects(cfg["subject_xlsx"])
    print(f"피험자 표에서 방문일이 온전한 대상자 {len(subs)}명")

    # ---- 1) 일 단위 표
    frames = []
    for i, r in enumerate(subs.itertuples(), 1):
        if not os.path.isdir(os.path.join(raw_root, r.pid)):
            continue
        frames.append(core.daily_table(raw_root, r.pid, r.v1_date, r.v4_date))
        if i % 20 == 0:
            print(f"  {i}/{len(subs)} 처리", flush=True)
    daily = pd.concat(frames, ignore_index=True)
    daily.to_csv(core.work_path(cfg, "daily_long.csv"), index=False,
                 encoding="utf-8-sig")
    print(f"daily_long.csv  {len(daily)}행, 대상자 {daily['pid'].nunique()}명")

    # ---- 2) 커버리지와 코호트
    cov = core.window_coverage(daily)
    cov.to_csv(core.work_path(cfg, "coverage.csv"), index=False, encoding="utf-8-sig")
    cohort = core.select_cohort(cov, args.min_days)
    print(f"coverage.csv    창당 {args.min_days}일 기준 통과 {len(cohort)}명")

    # ---- 3) 방문 요약
    wide = core.visit_summary(daily[daily["pid"].isin(cohort)])
    wide.to_csv(core.work_path(cfg, "visit_wide.csv"), encoding="utf-8-sig")
    print(f"visit_wide.csv  {wide.shape[0]}명 x {wide.shape[1]}열")

    # ---- 4) 라벨
    lab = subs[subs["pid"].isin(cohort)].set_index("pid").copy()
    # 6주 해밀턴 우울척도 14점 이상을 중등도 이상으로 본다 (표준 절단점).
    lab["y_moderate_w6"] = (lab["hamd_v4"] >= 14).astype("Int64")
    lab.to_csv(core.work_path(cfg, "cohort.csv"), encoding="utf-8-sig")
    n_pos = int(lab["y_moderate_w6"].sum())
    print(f"cohort.csv      {len(lab)}명 (양성 {n_pos} / 음성 {len(lab)-n_pos})")


if __name__ == "__main__":
    main()
