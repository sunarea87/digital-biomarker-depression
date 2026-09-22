# -*- coding: utf-8 -*-
"""[1단계] Fitbit 원자료를 분석용 표로 만듭니다.

실행
    python step1_preprocess.py

무엇을 하나요
    1. 대상자 명부에서 첫 방문일(0주)과 마지막 방문일(6주)을 읽습니다.
    2. 방문마다 7일짜리 관찰 창을 잡고, 그 안의 Fitbit 자료를 하루 단위로 모읍니다.
    3. 창마다 며칠이 확보되었는지 세어, 품질 기준(영역별 3일 이상)을 넘은 대상자만 남깁니다.
    4. 두 가지 코호트를 만듭니다.
         정형   0주와 6주 두 창만 요구  -> 사람이 가장 많습니다 (146명)
         비정형 0·2·4·6주 네 창 모두 요구 -> 궤적을 쓸 수 있습니다 (141명)

무엇이 만들어지나요 (3.데이터 아래)
    폴더 이름 끝의 숫자는 그때 실제로 남은 인원수입니다. 대상자가 늘면 숫자가 바뀝니다.
    2·3단계는 앞글자(정형_, 비정형_)로 폴더를 찾으므로 이름이 바뀌어도 그대로 돕니다.

    비정형_141명/daily_long.csv    1행 = 1대상자 x 1날짜. 모든 계산의 원재료
    비정형_141명/coverage.csv      창별 확보 일수. 코호트 선정 근거
    비정형_141명/visit_wide.csv    1행 = 1대상자, 열 = 지표__방문
    비정형_141명/labels.csv        임상 라벨 (HAMD, 자살 문항, BAI, 나이, 성별)
    정형_146명/visit_wide.csv      위와 같은 형식이되 0주·6주 두 창만
    정형_146명/labels.csv

주의
    원본 Fitbit 폴더는 용량이 커서 이 폴더에 복사해 두지 않았습니다.
    config.json 의 raw_root 가 원본 위치를 가리킵니다. 접속이 안 되면
    이 단계는 건너뛰고 2단계부터 실행하면 됩니다. 이미 만들어 둔 표가
    3.데이터 아래에 그대로 있습니다.
"""
import os

import pandas as pd

import biomarker as bm

HERE = os.path.dirname(os.path.abspath(__file__))
CFG = bm.load_config(os.path.join(HERE, "config.json"))
OUT = os.path.abspath(os.path.join(HERE, CFG["out_dir"]))
# 저장 폴더 이름은 아래 main() 에서 실제 인원수를 세어 붙입니다.
# (예: 정형_146명, 비정형_141명). 대상자가 늘면 새 이름의 폴더가 생깁니다.


# ---------------------------------------------------------------------
# 대상자 명부 읽기
# ---------------------------------------------------------------------
def load_subjects(xlsx: str) -> pd.DataFrame:
    """명부에서 방문일과 임상 점수를 읽습니다.

    열 이름이 조금씩 바뀌는 일이 있어 위치가 아니라 이름 일부로 찾습니다.
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

    num = lambda col: pd.to_numeric(df[col], errors="coerce")
    out = pd.DataFrame()
    out["pid"] = df[pick("대상자번호")].astype(str).str.strip()
    out["v1_date"] = pd.to_datetime(df[pick("V1날짜")], errors="coerce")
    out["v4_date"] = pd.to_datetime(df[pick("V4 날짜", "V4날짜")], errors="coerce")
    out["grp"] = num(pick("우울증_0무1유"))
    out["HAMD_v1"] = num(pick("HamD점수"))
    out["HAMD_v4"] = num(pick("V4_HAMD점수"))
    out["SUI_v1"] = num(pick("Baseline_HAMD_3_자살"))
    out["SUI_v4"] = num(pick("F/U_HAMD_3_자살"))
    out["BAI_v1"] = num(pick("V1_백 불안 척도 점수(BAI)"))
    out["BAI_v4"] = num(pick("V4_BAI점수"))
    out["나이"] = num(pick("나이(만)"))
    out["성별"] = num(pick("성별_1남_2여"))
    out = out.drop_duplicates("pid")

    # 명부에 잘못 적혔거나 비어 있는 방문일은 별도 파일로 덮어씁니다.
    # 원본 엑셀은 연구간호사가 관리하므로 코드가 직접 고치지 않습니다.
    fix = os.path.join(os.path.dirname(xlsx), "날짜보정.csv")
    if os.path.exists(fix):
        f = pd.read_csv(fix, encoding="utf-8-sig")
        f["pid"] = f["pid"].astype(str).str.strip()
        m = out.set_index("pid")
        n = 0
        for r in f.itertuples():
            if r.pid in m.index:
                m.loc[r.pid, "v4_date"] = pd.to_datetime(r.V4날짜, errors="coerce")
                n += 1
        out = m.reset_index()
        print(f"  방문일 보정 {n}건 적용 (날짜보정.csv)")
    return out


def save_cohort(daily, subs, pids, folder, visits):
    """한 코호트의 방문 요약과 라벨을 저장합니다."""
    os.makedirs(folder, exist_ok=True)
    d = daily[daily["pid"].isin(pids) & daily["visit"].isin(visits)]
    wide = bm.visit_summary(d)
    keep = [c for c in wide.columns if c.rsplit("__", 1)[1] in visits]
    wide = wide[keep]
    wide.to_csv(os.path.join(folder, "visit_wide.csv"), encoding="utf-8-sig")
    lab = subs.set_index("pid").reindex(wide.index)
    lab = lab.drop(columns=["v1_date", "v4_date"])
    lab.to_csv(os.path.join(folder, "labels.csv"), encoding="utf-8-sig")
    return wide, lab


def main():
    xlsx = os.path.abspath(os.path.join(HERE, CFG["subject_xlsx"]))
    subs = load_subjects(xlsx)
    print(f"명부 등록 {len(subs)}명")
    subs = subs.dropna(subset=["v1_date", "v4_date"])
    print(f"0주·6주 방문일 확보 {len(subs)}명")

    # ---- 1) 하루 단위 표 만들기 (원본 폴더를 읽는 유일한 단계)
    raw = CFG["raw_root"]
    frames = []
    for i, r in enumerate(subs.itertuples(), 1):
        if not os.path.isdir(os.path.join(raw, r.pid)):
            continue
        frames.append(bm.daily_table(raw, r.pid, r.v1_date, r.v4_date))
        if i % 20 == 0:
            print(f"    {i}/{len(subs)} 처리 중", flush=True)
    daily = pd.concat(frames, ignore_index=True)
    print(f"웨어러블 일 자료 확인 {daily['pid'].nunique()}명 ({len(daily)}행)")

    # ---- 2) 창별 확보 일수
    cov = bm.window_coverage(daily)
    piv = cov.pivot(index="pid", columns="visit", values="min_modality").fillna(0)

    # ---- 3) 두 코호트로 나누기
    ok2 = sorted(piv.index[(piv[["v1", "v4"]] >= bm.MIN_DAYS_PER_WINDOW).all(axis=1)])
    ok4 = sorted(piv.index[(piv[bm.VISITS] >= bm.MIN_DAYS_PER_WINDOW).all(axis=1)])

    # 폴더 이름에 실제 인원수를 붙입니다. 2·3단계는 앞글자로 찾으므로 고칠 것이 없습니다.
    dir_two = os.path.join(OUT, f"정형_{len(ok2)}명")
    dir_long = os.path.join(OUT, f"비정형_{len(ok4)}명")
    os.makedirs(dir_long, exist_ok=True)
    daily.to_csv(os.path.join(dir_long, "daily_long.csv"), index=False,
                 encoding="utf-8-sig")
    cov.to_csv(os.path.join(dir_long, "coverage.csv"), index=False,
               encoding="utf-8-sig")
    w2, l2 = save_cohort(daily, subs, ok2, dir_two, ("v1", "v4"))
    w4, l4 = save_cohort(daily, subs, ok4, dir_long, tuple(bm.VISITS))

    print()
    print("코호트 확정")
    for nm, w, l in [("정형   (0주·6주)", w2, l2), ("비정형 (네 시점)", w4, l4)]:
        print(f"  {nm} {len(w):3d}명"
              f"  우울군 {int((l['grp'] == 1).sum()):3d}"
              f"  건강 대조군 {int((l['grp'] == 0).sum()):3d}")
    print()
    print("표현형별 양성·음성")
    for nm, l in [("정형", l2), ("비정형", l4)]:
        for tgt, col, th in bm.TARGETS:
            ok = l[col].notna()
            y = (l[col] >= th)[ok]
            print(f"  [{nm}] {tgt:5s} 유효 {int(ok.sum()):3d}명"
                  f"  양성 {int(y.sum()):3d}  음성 {int((~y).sum()):3d}")
    print(f"\n저장 위치: {OUT}")


if __name__ == "__main__":
    main()
