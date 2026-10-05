"""명령행 진입점.  python -m app.cli <command>

  init-db                        테이블/불변성 트리거 생성
  collect --start --end          실데이터 수집 (KRX/yfinance/ECOS/DART)
  import-intraday --csv          (선택) 외부 분봉 CSV 로 과거 장중 스냅샷 백필 → 장중 특징 백테스트 가능
  bootstrap                      과거 데이터 워크포워드 백테스트 → 학습표본 적재 → v1 모델/OOS 성과 기록
  stage S1400|S1430|S1500|S1510|S1520   해당 단계 분석·저장 (스케줄러가 호출, 실데이터는 해당 시각에만)
  grade                          확정치로 Overnight 채점 + 오류분석 + 학습표본 + 성과 스냅샷
  reevaluate                     champion/challenger 재평가 (새 OOS 홀드아웃, 비용 차감 수익 기준)
  study                          특징별 OOS 예측력 검증 리포트 출력
  backtest [--start --end --top-k --exit open|close --fee-bps ...]
  verify                         예측 해시 체인 무결성 검사
  demo-synthetic                 [테스트 전용] 합성 데이터 → 부트스트랩 → N일 단계별 라이브 재생 → 재평가
"""
from __future__ import annotations

import argparse
import json
import logging
from datetime import date, datetime, timedelta

from .backtest.engine import BacktestParams, run_backtest
from .config import settings
from .db import BacktestRun, engine, get_session, init_db, now_kst, verify_chain
from .features.store import PITData
from .pipeline import STAGE_ORDER, catch_up, collect, run_day, run_grading, run_ondemand, run_stage
from .research.feature_study import study
from .strategy.manager import champion, ensure_initial, load_samples, reevaluate

log = logging.getLogger("cli")


def _d(s: str) -> date:
    return datetime.strptime(s, "%Y-%m-%d").date()


def cmd_bootstrap(session, end=None, source="historical", synthetic=False):
    data = PITData.load(session)
    params = BacktestParams(end=end)
    metrics, preds = run_backtest(data, params, progress=lambda i, n, d: i % 25 == 0 and log.info("bootstrap %d/%d %s", i, n, d))
    if preds.empty:
        print("부트스트랩 불가: 데이터 부족")
        return
    v = ensure_initial(session, preds, source=source, is_synthetic=synthetic)
    session.add(BacktestRun(created_at=now_kst(), params=metrics.get("params", {}), strategy_version=v.version,
                            metrics=metrics, is_synthetic=synthetic))
    session.commit()
    print(f"전략 {v.version}: {v.reason}")


def main(argv=None):
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("init-db")
    c = sub.add_parser("collect")
    c.add_argument("--start", type=_d, required=True)
    c.add_argument("--end", type=_d, required=True)
    im = sub.add_parser("import-intraday")
    im.add_argument("--csv", required=True)
    sub.add_parser("bootstrap")
    st = sub.add_parser("stage")
    st.add_argument("stage", choices=STAGE_ORDER)
    st.add_argument("--no-claude", action="store_true")
    sub.add_parser("launch", help="앱 실행 시 분석: 밀린 데이터 수집·채점 → 지금 시점 분석")
    sub.add_parser("grade")
    sub.add_parser("reevaluate")
    sub.add_parser("study")
    b = sub.add_parser("backtest")
    b.add_argument("--start", type=_d)
    b.add_argument("--end", type=_d)
    b.add_argument("--top-k", type=int, default=settings.max_final_picks)
    b.add_argument("--exit", choices=["close", "open"], default="close")
    b.add_argument("--fee-bps", type=float, default=settings.fee_bps)
    b.add_argument("--tax-bps", type=float, default=settings.tax_bps)
    b.add_argument("--slippage-bps", type=float, default=settings.slippage_bps)
    sub.add_parser("verify")
    s = sub.add_parser("demo-synthetic")
    s.add_argument("--days", type=int, default=320)
    s.add_argument("--tickers", type=int, default=120)
    s.add_argument("--replay", type=int, default=100, help="라이브처럼 재생할 마지막 N 거래일")
    a = ap.parse_args(argv)

    init_db(engine)
    session = get_session()
    if a.cmd == "init-db":
        print("ok:", settings.database_url)
    elif a.cmd == "collect":
        collect(session, a.start, a.end)
    elif a.cmd == "import-intraday":
        from .collectors.kis import import_minute_csv
        import_minute_csv(session, a.csv, [s[2] for s in settings.stages])
    elif a.cmd == "bootstrap":
        cmd_bootstrap(session)
    elif a.cmd == "stage":
        run = run_stage(session, a.stage, use_claude=not a.no_claude)
        print(run.run_id, run.summary.get("no_candidate_message") or run.summary["candidate_tickers"])
    elif a.cmd == "launch":
        print(cmd_launch(session))
    elif a.cmd == "grade":
        print(run_grading(session))
    elif a.cmd == "reevaluate":
        print(json.dumps(reevaluate(session), ensure_ascii=False, indent=1, default=str))
    elif a.cmd == "study":
        r = study(load_samples(session))
        for f in r.get("features", []):
            print(f"{f['label']:<20} IC(in) {f.get('ic_in')} t={f.get('t_in')} | IC(OOS) {f.get('ic_oos')} t={f.get('t_oos')} → {f['verdict']}")
    elif a.cmd == "backtest":
        params = BacktestParams(start=a.start, end=a.end, top_k=a.top_k, exit=a.exit, fee_bps=a.fee_bps,
                                tax_bps=a.tax_bps, slippage_bps=a.slippage_bps)
        metrics, _ = run_backtest(PITData.load(session), params)
        session.add(BacktestRun(created_at=now_kst(), params=metrics.get("params", {}), strategy_version="walk-forward",
                                metrics=metrics))
        session.commit()
        print(json.dumps({k: v for k, v in metrics.get("trading", {}).items() if k not in ("equity_curve", "monthly")},
                         ensure_ascii=False, indent=1))
    elif a.cmd == "verify":
        print(verify_chain(session))
    elif a.cmd == "demo-synthetic":
        demo_synthetic(session, a.days, a.tickers, a.replay)


def cmd_launch(session) -> str:
    """앱을 켤 때마다: 밀린 확정 데이터 수집 → 결과 나온 예측 채점 → (7일 지났으면) 전략 재평가 → 지금 시점 분석."""
    from sqlalchemy import func, select
    from .db import StrategyVersion
    g = catch_up(session)
    log.info("데이터 갱신·채점: %s", g)
    last = session.execute(select(func.max(StrategyVersion.created_at))).scalar()
    ev = (champion(session).evaluation or {}).get("last_reevaluated_at") if champion(session) else None
    ref = max([x for x in (last, datetime.fromisoformat(ev) if ev else None) if x], default=None)
    if ref is None or now_kst() - ref >= timedelta(days=7):
        r = reevaluate(session)
        log.info("전략 재평가: %s — %s", r.get("status"), r.get("reason", ""))
        ch = champion(session)
        if ch:
            ch.evaluation = {**(ch.evaluation or {}), "last_reevaluated_at": now_kst().isoformat()}
            session.commit()
    run, msg = run_ondemand(session)
    return f"[{run.stage if run else '분석 없음'}] {msg}"


def demo_synthetic(session, n_days: int, n_tickers: int, replay: int):
    from .collectors import synthetic
    if not settings.allow_synthetic:
        raise SystemExit("ALLOW_SYNTHETIC=1 이 필요합니다 (테스트 전용 DB 에서만 사용하세요)")
    days = synthetic.generate(session, date(2025, 1, 6), n_days, n_tickers)
    log.info("합성 데이터 %d일 × %d종목 생성. 부트스트랩 ~%s, 라이브 재생 %s~", len(days), n_tickers,
             days[-replay - 2], days[-replay])
    cmd_bootstrap(session, end=days[-replay - 2], source="SYNTHETIC walk-forward", synthetic=True)
    data = PITData.load(session)
    for i, T in enumerate(days[-replay:-1]):
        run_day(session, T, data)
        res = run_grading(session, now=datetime.combine(T, settings.grading_time), snapshot=(i % 10 == 0))
        if i % 10 == 0:
            log.info("replay %s: %s", T, res)
        if i and i % 25 == 0:
            r = reevaluate(session)
            log.info("재평가: %s — %s", r["status"], r.get("reason", ""))
    run_grading(session, now=datetime.combine(days[-1], settings.grading_time))
    r = reevaluate(session)
    log.info("최종 재평가: %s — %s", r["status"], r.get("reason", ""))
    print(verify_chain(session))


if __name__ == "__main__":
    main()
