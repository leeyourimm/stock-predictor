const $ = s => document.querySelector(s);
const pct = (x, d = 1) => x == null ? '<span class="muted">데이터 없음</span>' : (x * 100).toFixed(d) + '%';
const spct = (x, d = 1) => x == null ? '<span class="muted">데이터 없음</span>' : `<span class="${x > 0 ? 'up' : x < 0 ? 'down' : ''}">${x > 0 ? '+' : ''}${(x * 100).toFixed(d)}%</span>`;
const num = (x, d = 2) => x == null ? '<span class="muted">데이터 없음</span>' : Number(x).toLocaleString('ko-KR', {maximumFractionDigits: d});
const conf = c => `<span class="badge ${c}">${c}</span>`;
const esc = s => String(s ?? '').replace(/[&<>"]/g, c => ({'&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;'}[c]));
async function api(p, opt) { const r = await fetch(p, opt); if (!r.ok) throw new Error(await r.text()); return r.json(); }
function nav(on) {
  const pages = [['index', '대시보드'], ['performance', '성과·검증'], ['research', '특징 검증'], ['backtest', '백테스트'], ['data', '데이터 상태']];
  document.body.insertAdjacentHTML('afterbegin', `<header><b>KR Overnight Predictor</b><nav>${pages.map(([p, l]) =>
    `<a href="/${p === 'index' ? '' : p + '.html'}" class="${p === on ? 'on' : ''}">${l}</a>`).join('')}</nav>
    <span class="muted" style="margin-left:auto;font-size:12px">확률은 과거 실제 결과로 보정한 통계치이며 수익을 보장하지 않습니다</span></header>`);
}
function synthBanner(isSynth) { return isSynth ? '<div class="banner">⚠ SYNTHETIC 테스트 데이터 — 실제 시장 데이터가 아닙니다. 파이프라인 검증용입니다.</div>' : ''; }
function hitStr(h) { return !h || h.hit_rate == null ? '<span class="muted">표본 없음</span>' : `${pct(h.hit_rate)} <span class="muted">(n=${h.n})</span>`; }
const css = v => getComputedStyle(document.documentElement).getPropertyValue(v).trim();

const STAGE_LABEL = {S1400: '14:00 1차 분석', S1430: '14:30 재평가', S1500: '15:00 압축', S1510: '15:10 최종 분석', S1520: '15:20 후보 확정'};
const rng = (a, b) => `${spct(a)} ~ ${spct(b)}`;
function factorBadges(fs, positive) {
  const items = Object.values(fs || {}).flatMap(g => g.items || []).filter(i => i.points && (i.points > 0) === positive).sort((a, b) => Math.abs(b.points) - Math.abs(a.points)).slice(0, 3);
  return items.map(i => `<span class="badge">${esc(i.label)} ${i.points > 0 ? '+' : ''}${i.points}</span>`).join('') || '<span class="muted">-</span>';
}
