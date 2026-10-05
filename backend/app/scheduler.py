"""자동화 스케줄러 (Asia/Seoul).  python -m app.scheduler

거래일:
  06:30  미국 지수·환율·거시 수집 (전일 미국장 확정치)
  각 단계 스냅샷 시각(13:57, 14:27, 14:58, 15:08, 15:18)  KIS 장중 스냅샷 (단계 유니버스)
  각 단계 분석 시각(14:00, 14:30, 15:00, 15:10, 15:20)      단계 분석·저장 (15:20 = Overnight 후보 확정)
  18:30  당일 확정치 수집 → 전 거래일 예측 Overnight 채점
토 09:00  전략 재평가
휴장일에는 아무것도 하지 않는다. 14:00 단계 직전에 전 거래일 확정치가 없으면 보충 수집한다.
"""
from __future__ import annotations

import logging
from datetime import timedelta

from apscheduler.schedulers.blocking import BlockingScheduler

from .calendar_kr import is_trading_day, previous_trading_day
from .collectors import ecos, global_markets
from .config import KST, settings
from .db import engine, get_session, init_db, now_kst
from .pipeline import collect, collect_stage_snapshot, run_grading, run_stage
from .strategy.manager import reevaluate

log = logging.getLogger("scheduler")


def _trading_today() -> bool:
    ok = is_trading_day(now_kst().date())
    if not ok:
        log.info("휴장일: skip")
    return ok


def job_morning():
    if not _trading_today():
        return
    with get_session() as s:
        d = now_kst().date()
        global_markets.collect_global(s, d - timedelta(days=10), d)
        ecos.collect_ecos(s, d - timedelta(days=10), d)
        prev = previous_trading_day(d)
        collect(s, prev, prev)


def job_snapshot(stage: str):
    def run():
        if _trading_today():
            with get_session() as s:
                collect_stage_snapshot(s, now_kst().date(), stage)
    return run


def job_stage(stage: str):
    def run():
        if _trading_today():
            with get_session() as s:
                r = run_stage(s, stage)
                log.info("%s 완료: %s", stage, r.summary.get("no_candidate_message") or r.summary["candidate_tickers"])
    return run


def job_close():
    if not _trading_today():
        return
    with get_session() as s:
        d = now_kst().date()
        collect(s, d, d)
        log.info("채점: %s", run_grading(s))


def job_weekly():
    with get_session() as s:
        log.info("재평가: %s", reevaluate(s).get("status"))


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    init_db(engine)
    sch = BlockingScheduler(timezone=KST)
    sch.add_job(job_morning, "cron", day_of_week="mon-fri", hour=6, minute=30)
    for stage, t_run, t_snap, _ in settings.stages:
        sch.add_job(job_snapshot(stage), "cron", day_of_week="mon-fri", hour=t_snap.hour, minute=t_snap.minute,
                    misfire_grace_time=60)
        sch.add_job(job_stage(stage), "cron", day_of_week="mon-fri", hour=t_run.hour, minute=t_run.minute,
                    misfire_grace_time=300)
    sch.add_job(job_close, "cron", day_of_week="mon-fri", hour=18, minute=30, misfire_grace_time=3600)
    sch.add_job(job_weekly, "cron", day_of_week="sat", hour=9, minute=0)
    log.info("scheduler started")
    sch.start()


if __name__ == "__main__":
    main()
