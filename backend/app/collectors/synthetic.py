"""!!! 테스트 전용 합성 데이터 !!!

네트워크가 막힌 개발 환경에서 파이프라인(수집→단계별 분석→저장→Overnight 채점→개선)을 end-to-end 로 검증하기 위한 것.
- source 는 항상 "SYNTHETIC(TEST ONLY)", 화면에 '실제 데이터 아님' 배지가 붙는다. 종목명도 TEST001 형식.
- 일봉 + 장중 체크포인트 스냅샷 + KOSPI 장중 지수를 생성한다.
- 시스템이 '발견해야 할' 약한 신호를 의도적으로 심는다 (모델에는 알려주지 않음):
    1) 14:00 이후 장 후반 모멘텀(15:18 vs 13:57) → 다음날 시가 갭 (양의 관계)
    2) T-1 까지 5일 외국인 순매수 → 다음날 장중 수익률 (약한 양의 관계)
  나머지 지표(RSI, MACD 등)는 아무 관계 없음 → 모델/검증 리포트가 이를 구분하는지 확인할 수 있다.
"""
from __future__ import annotations

from datetime import date, datetime, time, timedelta

import numpy as np
from sqlalchemy.orm import Session

from ..config import settings
from ..db import DailyBar, IndexBar, Instrument, IntradaySnapshot, InvestorFlow, ShortData
from .base import at, upsert_ignore

SOURCE = "SYNTHETIC(TEST ONLY)"
SECTORS = [("S01", "TEST-반도체"), ("S02", "TEST-2차전지"), ("S03", "TEST-바이오"), ("S04", "TEST-금융"),
           ("S05", "TEST-자동차"), ("S06", "TEST-유통")]
OPEN_MIN, DAY_MIN = 9 * 60, 390


def synthetic_calendar(start: date, n_days: int) -> list[date]:
    days, d = [], start
    while len(days) < n_days:
        if d.weekday() < 5:
            days.append(d)
        d += timedelta(days=1)
    return days


def _frac(t: time) -> float:
    return (t.hour * 60 + t.minute - OPEN_MIN) / DAY_MIN


def generate(session: Session, start: date, n_days: int, n_tickers: int = 120, seed: int = 7) -> list[date]:
    rng = np.random.default_rng(seed)
    days = synthetic_calendar(start, n_days)
    T, N = len(days), n_tickers
    retrieved = datetime.combine(days[-1] + timedelta(days=1), time(19, 0))
    snap_times = [st[2] for st in settings.stages]

    tickers = [f"TEST{i:03d}" for i in range(1, N + 1)]
    sec_of = rng.integers(0, len(SECTORS), N)
    for i, t in enumerate(tickers):
        s_code, s_name = SECTORS[sec_of[i]]
        session.merge(Instrument(ticker=t, name=f"테스트종목{i + 1:03d}", market="KOSPI" if i % 3 else "KOSDAQ",
                                 sector=s_name, sector_index=s_code, is_halted=(i == N - 1),
                                 is_admin_issue=(i == N - 2), status_source=SOURCE, status_checked_at=retrieved))

    drift = np.interp(np.arange(T), [0, T * .3, T * .5, T * .7, T * .85, T], [0.0012, 0.0010, 0.0, -0.0012, -0.0004, 0.0010])
    mvol = np.interp(np.arange(T), [0, T * .6, T * .7, T * .85, T], [0.008, 0.009, 0.020, 0.018, 0.010])
    us = rng.normal(0.0004, 0.011, T)
    beta = rng.uniform(0.6, 1.5, N)
    sigma = rng.uniform(0.012, 0.035, N)
    sigma[:3] = 0.09
    avg_value = np.exp(rng.uniform(np.log(3e8), np.log(3e11), N))
    avg_value[3:6] = 2e8
    base_px = rng.uniform(2000, 200000, N)

    f = np.zeros(N)
    f_hist: list[np.ndarray] = []
    prev_close = base_px.copy()
    late_mom = np.zeros(N)
    prev_mkt = 2500.0
    sec_level = np.full(len(SECTORS), 1000.0)
    us_level = np.array([5000.0, 16000.0, 4500.0])
    bars, flows, shorts, snaps, idx_rows = [], [], [], [], []

    for ti, d in enumerate(days):
        m_gap = (0.4 * us[ti - 1] if ti else 0.0) + 0.3 * mvol[ti] * rng.normal()
        m_intra = drift[ti] + 0.8 * mvol[ti] * rng.normal()
        s_gap, s_intra = 0.003 * rng.normal(size=len(SECTORS)), 0.005 * rng.normal(size=len(SECTORS))
        f5 = np.sum(f_hist[-5:], axis=0) / np.sqrt(5) if f_hist else np.zeros(N)
        gap = beta * m_gap + s_gap[sec_of] + 0.30 * sigma * np.tanh(late_mom) + 0.4 * sigma * rng.normal(size=N)
        intra = beta * m_intra + s_intra[sec_of] + 0.15 * sigma * np.tanh(f5) + 0.9 * sigma * rng.normal(size=N)
        opn = prev_close * (1 + np.clip(gap, -0.2, 0.2))
        close = np.maximum(opn * (1 + np.clip(intra, -0.25, 0.25)), 1)
        value = avg_value * np.exp(rng.normal(0, 0.35, N) + 8 * np.abs(close / prev_close - 1))

        # 장중 경로: 체크포인트별 가격(시가→종가 브라운 브리지), 누적 고가/저가/거래대금
        cp = {}
        run_hi, run_lo = opn.copy(), opn.copy()
        last_fr, last_dev = 0.0, np.zeros(N)
        for st in snap_times:
            fr = _frac(st)
            dev = last_dev * (1 - fr) / max(1e-9, 1 - last_fr) + \
                0.6 * sigma * opn * np.sqrt(max(0.0, fr - last_fr) * (1 - fr)) * rng.normal(size=N)
            px = np.maximum(opn + (close - opn) * fr + dev, 1)
            run_hi = np.maximum(run_hi, px * (1 + np.abs(rng.normal(0, 0.15, N)) * sigma))
            run_lo = np.minimum(run_lo, px * (1 - np.abs(rng.normal(0, 0.15, N)) * sigma))
            cp[st] = (px, run_hi.copy(), run_lo.copy(), value * fr ** 0.9 * 0.92)
            last_fr, last_dev = fr, dev
        hi, lo = np.maximum(run_hi, close), np.minimum(run_lo, close)
        late_mom = (cp[snap_times[-1]][0] / cp[snap_times[0]][0] - 1) / (0.3 * sigma)   # 다음날 갭에 반영

        f = 0.55 * f + rng.normal(size=N)
        f_hist.append(f.copy())
        inst = rng.normal(size=N)
        b_av, f_av = at(d, settings.bar_available_time), at(d, settings.flow_available_time)
        for i, t in enumerate(tickers):
            if i == N - 1 and ti > T - 6:
                continue  # 거래정지 종목: 최근 5일 체결 없음
            bars.append(dict(ticker=t, date=d, open=round(opn[i]), high=round(hi[i]), low=round(lo[i]),
                             close=round(close[i]), volume=round(value[i] / close[i]), value=float(value[i]),
                             change_pct=float((close[i] / prev_close[i] - 1) * 100),
                             market_cap=float(avg_value[i] * 300 / base_px[i] * close[i]),
                             source=SOURCE, retrieved_at=retrieved, data_timestamp=b_av, available_at=b_av))
            fn, inn = f[i] * 0.04 * avg_value[i], inst[i] * 0.03 * avg_value[i]
            flows.append(dict(ticker=t, date=d, foreign_net=float(fn), institution_net=float(inn),
                              individual_net=float(-fn - inn), source=SOURCE, retrieved_at=retrieved,
                              data_timestamp=f_av, available_at=f_av))
            shorts.append(dict(ticker=t, date=d, short_volume=float(value[i] / close[i] * rng.uniform(0, 0.08)),
                               short_value=None, short_balance=None, source=SOURCE, retrieved_at=retrieved,
                               data_timestamp=f_av, available_at=f_av))
            for st, (px, h_, l_, cv) in cp.items():
                ts = datetime.combine(d, st)
                snaps.append(dict(ticker=t, ts=ts, price=round(px[i]), open=round(opn[i]), high=round(h_[i]),
                                  low=round(l_[i]), cum_volume=float(cv[i] / px[i]), cum_value=float(cv[i]),
                                  source=SOURCE, retrieved_at=ts, data_timestamp=ts, available_at=ts))
        m_open = prev_mkt * (1 + m_gap)
        mkt_level = m_open * (1 + m_intra)
        for st in snap_times:
            ts = datetime.combine(d, st)
            snaps.append(dict(ticker="IDX:KOSPI", ts=ts, price=float(m_open * (1 + m_intra * _frac(st))),
                              open=float(m_open), high=None, low=None, cum_volume=None, cum_value=None,
                              source=SOURCE, retrieved_at=ts, data_timestamp=ts, available_at=ts))
        sec_level = sec_level * (1 + m_gap + s_gap) * (1 + m_intra + s_intra)
        us_level = us_level * (1 + np.array([1.0, 1.3, 1.8]) * us[ti])
        us_av = at(d, time(7, 0), plus_days=1)
        series = [("KOSPI", mkt_level, b_av), ("KOSDAQ", 800 * (mkt_level / 2500) ** 1.2, b_av),
                  ("US:SPX", us_level[0], us_av), ("US:NASDAQ", us_level[1], us_av), ("US:SOX", us_level[2], us_av),
                  ("US:VIX", 16 * np.exp(-3 * us[ti] + 0.05 * rng.normal()), us_av),
                  ("FX:USDKRW", 1350 * (1 - 0.3 * (mkt_level / 2500 - 1)) * (1 + 0.003 * rng.normal()),
                   at(d, time(9, 0), plus_days=1))]
        series += [(f"SECTOR:{c}", sec_level[k], b_av) for k, (c, _) in enumerate(SECTORS)]
        for sym, val, av in series:
            idx_rows.append(dict(symbol=sym, date=d, open=None, high=None, low=None, close=float(val), volume=None,
                                 source=SOURCE, retrieved_at=retrieved, data_timestamp=av, available_at=av))
        prev_close, prev_mkt = close, mkt_level

    upsert_ignore(session, DailyBar, bars, ["ticker", "date", "source"])
    upsert_ignore(session, InvestorFlow, flows, ["ticker", "date", "source"])
    upsert_ignore(session, ShortData, shorts, ["ticker", "date", "source"])
    upsert_ignore(session, IntradaySnapshot, snaps, ["ticker", "ts", "source"])
    upsert_ignore(session, IndexBar, idx_rows, ["symbol", "date", "source"])
    session.commit()
    return days
