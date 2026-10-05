"""네이버 수집기 파서·공개시각 규칙 테스트 (네트워크 없이, 응답 형식 예시로 검증)."""
from datetime import date, datetime

from sqlalchemy.orm import sessionmaker

from app.collectors import naver
from app.db import DailyBar, IndexBar, Instrument, InvestorFlow, init_db, make_engine

SISE = """[['날짜', '시가', '고가', '저가', '종가', '거래량', '외국주문한도소진율'],
["20261001", 70000, 71000, 69500, 70500, 12345678, 50.1],
["20261002", 70500, 72000, 70000, 71800, 23456789, 50.2]
]"""
MARKET_SUM = """<td class="no">1</td><td><a href="/item/main.naver?code=005930" class="tltle">삼성전자</a></td>
<td class="no">2</td><td><a href="/item/main.naver?code=000660" class="tltle">SK하이닉스</a></td>"""
FRGN = """<table><tr onMouseOver="x"><td class="tc"><span class="tah p10 gray03">2026.10.02</span></td>
<td class="num"><span class="tah p11">71,800</span></td><td class="num"><span>1,300</span></td>
<td class="num"><span>+1.84%</span></td><td class="num"><span class="tah p11">23,456,789</span></td>
<td class="num"><span class="tah p11 red01">+1,000</span></td><td class="num"><span class="tah p11 nv01">-2,000</span></td>
<td class="num"><span>3,000,000,000</span></td><td class="num"><span>50.20%</span></td></tr></table>"""


def test_parsers():
    rows = naver.parse_sise_json(SISE)
    assert rows[0] == {"date": date(2026, 10, 1), "open": 70000, "high": 71000, "low": 69500, "close": 70500,
                       "volume": 12345678}
    assert naver.parse_market_sum(MARKET_SUM) == [("005930", "삼성전자"), ("000660", "SK하이닉스")]
    f = naver.parse_frgn(FRGN)[0]
    assert f["date"] == date(2026, 10, 2) and f["close"] == 71800
    assert f["inst_net_shares"] == 1000 and f["foreign_net_shares"] == -2000


def test_collect_range_respects_availability(tmp_path, monkeypatch):
    eng = make_engine(f"sqlite:///{tmp_path / 'n.db'}")
    init_db(eng)
    s = sessionmaker(bind=eng, expire_on_commit=False)()
    s.add(Instrument(ticker="005930", name="삼성전자", market="KOSPI"))
    s.commit()

    def fake_get(url, params=None, encoding=None, pause=0):
        if "siseJson" in url:
            return SISE
        if "frgn" in url:
            return FRGN if params.get("page") == 1 else ""
        return ""
    monkeypatch.setattr(naver, "_get", fake_get)
    # 10/2 17:00: 10/2 일봉(16:00)은 확정, 10/2 수급(18:00)은 아직 미확정 → 저장 안 함
    monkeypatch.setattr(naver, "now_kst", lambda: datetime(2026, 10, 2, 17, 0))
    naver.collect_range(s, date(2026, 9, 1), date(2026, 10, 2))
    assert {b.date for b in s.query(DailyBar).all()} == {date(2026, 10, 1), date(2026, 10, 2)}
    assert s.query(InvestorFlow).count() == 0
    assert s.query(IndexBar).filter_by(symbol="KOSPI").count() == 2
    b = s.query(DailyBar).filter_by(date=date(2026, 10, 2)).one()
    assert b.value == 71800 * 23456789 and b.source == naver.SOURCE
    # 18:00 이후에는 수급 저장 (수량 × 종가 = 금액 근사)
    monkeypatch.setattr(naver, "now_kst", lambda: datetime(2026, 10, 2, 18, 30))
    naver.collect_range(s, date(2026, 10, 2), date(2026, 10, 2))
    fl = s.query(InvestorFlow).one()
    assert fl.foreign_net == -2000 * 71800 and fl.institution_net == 1000 * 71800 and fl.individual_net is None
