"""핵심 원칙 테스트: look-ahead 차단(단계별 시각), 예측 불변성/해시 체인, Overnight 결과 계산, 보정."""
from datetime import date, datetime, time

import numpy as np
import pandas as pd
import pytest
from sqlalchemy import text
from sqlalchemy.orm import sessionmaker

from app.collectors import synthetic
from app.db import Prediction, init_db, make_engine, verify_chain
from app.features.builder import build_features
from app.features.store import PITData
from app.prediction import calibration as cal
from app.config import settings
from app.evaluation.outcomes import overnight
from app.pipeline import STAGE_ORDER, run_day, run_grading, run_stage


@pytest.fixture(scope="module")
def env(tmp_path_factory):
    eng = make_engine(f"sqlite:///{tmp_path_factory.mktemp('db') / 't.db'}")
    init_db(eng)
    s = sessionmaker(bind=eng, expire_on_commit=False)()
    days = synthetic.generate(s, date(2025, 1, 6), 180, 30, seed=1)
    return s, days


def test_lookahead_features_identical_with_or_without_future(env):
    """T 14:00 피처는 T 이후 데이터가 DB 에 있든 없든 동일해야 한다."""
    s, days = env
    T = days[150]
    as_of = datetime.combine(T, time(14, 0))
    full = PITData.load(s)
    trunc = PITData.load(s)  # T 이후(T 포함) 원천 데이터를 아예 제거한 버전
    trunc = PITData(trunc.bars_long[trunc.bars_long.date < T], trunc.flows_long[trunc.flows_long.date < T],
                    trunc.shorts_long[trunc.shorts_long.date < T], trunc.index_long[trunc.index_long.date < T], trunc.instruments,
                    trunc.disclosures, trunc.snapshots, trunc.conflicts)
    a = build_features(full.as_of(as_of)).table.drop(columns=["name"])
    b = build_features(trunc.as_of(as_of)).table.drop(columns=["name"])
    pd.testing.assert_frame_equal(a.sort_index(), b.sort_index(), check_dtype=False)


def test_as_of_excludes_same_day_bar_before_close(env):
    s, days = env
    T = days[100]
    view = PITData.load(s).as_of(datetime.combine(T, time(14, 0)))
    assert view.last_bar_date == days[99]          # 14:00 에는 T 일봉(16:00 확정) 사용 불가
    assert view.flows["foreign_net"].index[-1] == days[99]  # T-1 수급(18:00 확정)까지 사용 가능
    view2 = PITData.load(s).as_of(datetime.combine(T, time(17, 0)))
    assert view2.last_bar_date == T
    assert view2.flows["foreign_net"].index[-1] == days[99]  # 17:00 에는 T 수급 아직 미확정


def _truncate(data: PITData, as_of: datetime) -> PITData:
    """as_of 시점 이후에 생긴 원천 데이터를 아예 제거한 버전 (당일 일봉·수급, 이후 스냅샷, 이후 공시)."""
    T = as_of.date()
    sn = data.snapshots
    sn = sn[sn.ts <= as_of] if not sn.empty else sn
    disc = data.disclosures
    disc = disc[disc.available_at <= as_of] if not disc.empty else disc
    return PITData(data.bars_long[data.bars_long.date < T], data.flows_long[data.flows_long.date < T],
                   data.shorts_long[data.shorts_long.date < T], data.index_long[data.index_long.date < T],
                   data.instruments, disc, sn, data.conflicts)


@pytest.mark.parametrize("hhmm", [(13, 57), (15, 18)])
def test_stage_features_ignore_post_snapshot_data(env, hhmm):
    """14:00/15:20 단계 피처는 그 시각 이후 스냅샷·당일 종가·당일 수급이 DB 에 있든 없든 동일해야 한다."""
    s, days = env
    T = days[140]
    as_of = datetime.combine(T, time(*hhmm))
    full = PITData.load(s)
    view = full.as_of(as_of)
    assert not view.snapshot.empty and view.snapshot.ts.max() <= pd.Timestamp(as_of)
    assert view.last_bar_date == days[139]
    a = build_features(view).table.drop(columns=["name"])
    b = build_features(_truncate(full, as_of).as_of(as_of)).table.drop(columns=["name"])
    pd.testing.assert_frame_equal(a.sort_index(), b.sort_index(), check_dtype=False)
    assert a["snap_price"].notna().all()


def test_final_stage_snapshot_time_before_1520():
    st = {name: (t, snap) for name, t, snap, _ in settings.stages}
    t, snap = st[settings.final_stage]
    assert snap < t <= time(15, 20)


def test_overnight_outcome_costs():
    o = overnight(100.0, 101.0, 103.0, 99.0, 102.0)
    assert o["open_ret"] == pytest.approx(0.01) and o["close_ret"] == pytest.approx(0.02)
    assert o["net_close"] == pytest.approx(0.02 - settings.round_trip_cost)
    assert o["gap_up"] and o["high_hit"]
    assert settings.round_trip_cost == pytest.approx((2 * settings.fee_bps + settings.tax_bps + 2 * settings.slippage_bps) / 1e4)


def test_predictions_immutable_and_chain(env):
    s, days = env
    data = PITData.load(s)
    for T in days[120:124]:
        runs = run_day(s, T, data)
        assert [r.stage for r in runs] == STAGE_ORDER
        final = runs[-1]
        assert final.as_of <= datetime.combine(T, time(15, 20))
        run_grading(s, now=datetime.combine(T, settings.grading_time))
    # 최종 단계 후보는 최대 max_final_picks
    n_final_cand = s.query(Prediction).filter(Prediction.stage == settings.final_stage, Prediction.is_candidate).count()
    assert n_final_cand <= 4 * settings.max_final_picks
    assert verify_chain(s)["ok"]
    for tbl in ("predictions", "prediction_outcomes", "prediction_runs"):
        with pytest.raises(Exception):
            s.execute(text(f"UPDATE {tbl} SET rowid = rowid WHERE rowid = 1"))
            s.commit()
        s.rollback()
        with pytest.raises(Exception):
            s.execute(text(f"DELETE FROM {tbl} WHERE rowid = 1"))
            s.commit()
        s.rollback()
    # 같은 날·같은 단계 재실행해도 기존 예측을 덮어쓰지 않는다
    n = s.query(Prediction).count()
    run_stage(s, settings.final_stage, days[120], data, use_claude=False)
    assert s.query(Prediction).count() == n


def test_chain_detects_tampering(env):
    s, _ = env
    # 트리거를 우회한 변조(예: DB 파일 직접 수정)를 해시 체인이 잡는지 확인
    s.execute(text("DROP TRIGGER predictions_no_update"))
    s.execute(text("UPDATE predictions SET p_up = 0.99 WHERE id = (SELECT min(id) FROM predictions)"))
    s.commit()
    s.expire_all()
    assert verify_chain(s)["ok"] is False
    init_db(s.bind)


def test_calibration_shrinks_to_base_rate_without_data():
    c = cal.fit([], [])
    assert np.allclose(cal.predict_proba(c, [10, -10, 0]), 0.5)
    rng = np.random.default_rng(0)
    s = rng.normal(0, 1, 5000)
    y = (rng.random(5000) < cal.sigmoid(0.5 * s)).astype(float)
    c = cal.fit(s, y)
    assert 0.35 < c["b"] < 0.65
    p = cal.predict_proba(c, s)
    tbl = cal.calibration_table(p, y)
    assert all(abs(b["mean_pred"] - b["actual_up_rate"]) < 0.1 for b in tbl if b["n"] > 200)


def test_ondemand_launch_runs_and_official_final(env):
    """앱 실행 시 분석: 실행 시각까지의 데이터만, 15:30 이후엔 생성 안 함, 같은 날 여러 번이면 마지막만 성과 집계."""
    from app.pipeline import graded_frame, official_finals, run_ondemand
    s, days = env
    data = PITData.load(s)
    T = days[130]
    r1, _ = run_ondemand(s, datetime.combine(T, time(14, 35)), data, use_claude=False)
    r2, _ = run_ondemand(s, datetime.combine(T, time(15, 10)), data, use_claude=False)
    assert r1.stage == "L1435" and r2.stage == "L1510"
    assert r1.as_of == datetime.combine(T, time(14, 35))
    assert len(r2.summary.get("candidate_tickers", [])) <= settings.max_final_picks
    late, msg = run_ondemand(s, datetime.combine(T, time(15, 40)), data, use_claude=False)
    assert late is None and "15:30" in msg
    run_grading(s, now=datetime.combine(days[131], settings.grading_time))
    g = graded_frame(s)
    fin = official_finals(g[g.trade_date == T])
    assert set(fin.stage) == {"L1510"}
