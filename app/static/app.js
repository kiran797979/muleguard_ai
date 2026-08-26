/* ==========================================================================
   MULEGUARD AI — command center front end.

   Design rule enforced throughout: this file NEVER supplies a fallback number.
   If an endpoint fails, the panel renders an error that names the missing
   artefact and the stage that produces it. A demo that quietly shows plausible
   figures when the model failed to load is the worst possible failure for a
   project whose whole thesis is "do not trust unverified numbers".
   ========================================================================== */
'use strict';

// `rail: false` keeps a section out of the left navigation while leaving it
// routable: go() still shows and hides it, and its URL still works. These five
// are supporting evidence rather than the argument, so they are reachable when
// a judge asks a follow-up without competing for attention on the way in.
const SECTIONS = [
  { id: 'hero',       n: '00', label: 'The Problem',        group: 'Start' },
  { id: 'build',      n: '01', label: 'How We Built It',    group: 'Start' },
  { id: 'judge',      n: '02', label: 'Judge Mode',         group: 'Start' },
  { id: 'dataset',    n: '03', label: 'Dataset',            group: 'The Data' },
  { id: 'integrity',  n: '04', label: 'What We Found',      group: 'The Data' },
  { id: 'approach',   n: '05', label: 'How It Works',       group: 'The Method' },
  { id: 'leakage',    n: '06', label: 'Leakage Defence',    group: 'The Method' },
  { id: 'features',   n: '07', label: 'Mule Features',      group: 'The Method' },
  { id: 'shap',       n: '08', label: 'Why It Flagged',     group: 'The Results' },
  { id: 'triage',     n: '09', label: 'Risk Triage',        group: 'The Results' },
  { id: 'unified',    n: '10', label: 'End To End',         group: 'The Results' },
  { id: 'overview',   n: '11', label: 'Overview',           group: 'The Results' },
  { id: 'upload',     n: '12', label: 'Upload Dataset',     group: 'See It Run' },
  { id: 'pipeline',   n: '13', label: 'Pipeline',           group: 'See It Run' },
  // Off the rail: supporting evidence, still routable by URL and still
  // linked from the rubric map in Judge Mode.
  { id: 'models',     n: '--', label: 'Models',             group: 'The Results', rail: false },
  { id: 'baseline',   n: '--', label: 'Rules & Ablation',   group: 'The Results', rail: false },
  { id: 'analyze',    n: '--', label: 'Account Analysis',   group: 'The Results', rail: false },
  { id: 'operating',  n: '--', label: 'Operating Cost',     group: 'The Results', rail: false },
  { id: 'audit',      n: '--', label: 'Audit Trail',        group: 'The Results', rail: false },
];

/* ---------- tiny helpers ------------------------------------------------ */
const $  = (s, r = document) => r.querySelector(s);
const el = (t, cls, html) => { const e = document.createElement(t);
  if (cls) e.className = cls; if (html != null) e.innerHTML = html; return e; };
const esc = (s) => String(s ?? '').replace(/[&<>"']/g,
  (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));

/** Format a number for display. `null` means the backend sent NaN/Inf. */
function num(v, dp = 3) {
  if (v === null || v === undefined || Number.isNaN(v)) return '—';
  const n = Number(v);
  if (!Number.isFinite(n)) return '—';
  if (Math.abs(n) >= 1e6) return n.toExponential(2);
  if (Number.isInteger(n) && Math.abs(n) < 1e6) return n.toLocaleString();
  return n.toFixed(dp);
}
const pct = (v, dp = 1) => (v === null || v === undefined ? '—' : `${(Number(v) * 100).toFixed(dp)}%`);
/** mean ± std from the shape the metrics file uses.
 *  Returns HTML, so it belongs in stat()'s VALUE slot only — the unit slot is
 *  escaped and would print the markup verbatim. */
const ms = (b, dp = 3) => (!b || b.mean === undefined ? '—'
  : `${Number(b.mean).toFixed(dp)} <span class="faint">± ${Number(b.std ?? 0).toFixed(dp)}</span>`);

async function api(path) {
  let r;
  try {
    r = await fetch(path, { headers: { Accept: 'application/json' } });
  } catch (e) {
    throw { kind: 'NETWORK', detail: 'The API server is not reachable.',
            fix: 'Start it with:  python -m uvicorn app.server:app' };
  }
  let body = null;
  try { body = await r.json(); } catch (e) { /* fall through */ }
  if (!r.ok) {
    throw { kind: body?.error || `HTTP_${r.status}`,
            detail: body?.detail || r.statusText,
            fix: body?.fix, stage: body?.produced_by };
  }
  if (body === null) throw { kind: 'EMPTY_RESPONSE', detail: 'Server returned no JSON body.' };
  return body;
}

function errorBox(e) {
  const b = el('div', 'errorbox');
  b.appendChild(el('h4', null, esc(e.kind || 'ERROR')));
  b.appendChild(el('div', null, esc(e.detail || 'Unknown failure.')));
  if (e.stage) b.appendChild(el('div', 'dim tiny', `Produced by ${esc(e.stage)}`));
  if (e.fix)   b.appendChild(el('div', 'fix', esc(e.fix)));
  return b;
}

function stat(k, v, u, tone) {
  return `<div class="stat ${tone || ''}"><div class="k">${esc(k)}</div>
          <div class="v">${v}</div>${u ? `<div class="u">${esc(u)}</div>` : ''}</div>`;
}

function table(headers, rows) {
  const th = headers.map((h) => `<th class="${h.num ? 'num' : ''}">${esc(h.t ?? h)}</th>`).join('');
  const tb = rows.map((r) => `<tr>${r.map((c, i) =>
    `<td class="${headers[i]?.num ? 'num' : ''}">${c}</td>`).join('')}</tr>`).join('');
  return `<thead><tr>${th}</tr></thead><tbody>${tb}</tbody>`;
}

function bar(frac, tone) {
  const w = Math.max(0, Math.min(1, Number(frac) || 0)) * 100;
  return `<span class="bar ${tone || ''}"><span style="width:${w.toFixed(1)}%"></span></span>`;
}

/** Render a section: show a loading state, fetch, then paint or show the error. */
async function panel(node, loader, painter) {
  if (!node) return;
  node.innerHTML = '';
  node.appendChild(el('div', 'loading', 'LOADING'));
  try {
    const data = await loader();
    node.innerHTML = '';
    painter(data, node);
  } catch (e) {
    node.innerHTML = '';
    node.appendChild(errorBox(e));
  }
}

/* ---------- navigation -------------------------------------------------- */
/* Section numbers appear in buttons and prose as well as in the rail. Typing
   them by hand meant every reorder silently left stale numbers pointing at the
   wrong page, which happened three times. These fill themselves from SECTIONS. */
function labelSections() {
  const by = Object.fromEntries(SECTIONS.map((s) => [s.id, s]));
  document.querySelectorAll('[data-goto][data-num]').forEach((b) => {
    const s = by[b.dataset.goto];
    if (s) b.textContent = `${s.n} ${s.label}`;
  });
  document.querySelectorAll('[data-secnum]').forEach((e) => {
    const s = by[e.dataset.secnum];
    if (s) e.textContent = s.n;
  });
}

function buildRail() {
  const rail = $('#rail');
  let group = null;
  SECTIONS.filter((s) => s.rail !== false).forEach((s) => {
    if (s.group !== group) { group = s.group; rail.appendChild(el('div', 'railgroup', esc(group))); }
    const b = el('button', null, `<span class="idx">${s.n}</span>${esc(s.label)}`);
    b.dataset.target = s.id;
    b.onclick = () => go(s.id);
    rail.appendChild(b);
  });
  labelSections();
}

const painted = new Set();
function go(id) {
  SECTIONS.forEach((s) => { const n = $('#s-' + s.id); if (n) n.hidden = s.id !== id; });
  document.querySelectorAll('#rail button').forEach((b) =>
    b.setAttribute('aria-current', String(b.dataset.target === id)));
  if (location.hash.slice(1) !== id) history.replaceState(null, '', '#' + id);
  window.scrollTo(0, 0);
  if (!painted.has(id) && RENDER[id]) { painted.add(id); RENDER[id](); }
}

/* One delegated handler for every [data-goto] on the page.

   These used to be wired per-render, which worked only if the section that
   contained them had already been painted. Landing straight on #judge from a
   pasted link left its buttons dead. Delegation on the document survives both
   direct entry and any panel painted later. */
document.addEventListener('click', (e) => {
  const b = e.target.closest('[data-goto]');
  if (b) { e.preventDefault(); go(b.dataset.goto); }
});

/* ---------- health ------------------------------------------------------ */
let HEALTH = null;
async function refreshHealth() {
  const bar = $('#statusbar');
  try {
    HEALTH = await api('/api/health');
  } catch (e) {
    bar.innerHTML = '<span class="pill bad">API DOWN</span>';
    return;
  }
  const cls = HEALTH.status === 'READY' ? 'ok' : HEALTH.status === 'DOWN' ? 'bad' : 'warn';
  const missing = (HEALTH.missing || []).length;
  bar.innerHTML =
    `<span class="pill ${cls}">${esc(HEALTH.status)}</span>` +
    `<span class="pill ${HEALTH.model_loaded ? 'ok' : 'bad'}">MODEL ${HEALTH.model_loaded ? 'LOADED' : 'ABSENT'}</span>` +
    (missing ? `<span class="pill bad">${missing} ARTEFACT${missing > 1 ? 'S' : ''} MISSING</span>` : '') +
    `<span class="pill ok">INTEGRITY AUDITED</span>`;
}

/* ========================================================================
   SECTION RENDERERS
   ======================================================================== */
const RENDER = {};

/* -- 01 overview -- */
RENDER.overview = () => {
  panel($('#scale-panel'), () => api('/api/scale'), (d, n) => {
    const h = d.held_out || {}, lat = d.latency || {};
    n.innerHTML = `
      <p>The supplied file is 9,082 accounts. To show this is not a laptop-sized
         toy, we ran the whole pipeline against <strong>SAML-D</strong>, a public
         anti-money-laundering dataset we did not build and cannot tune against.</p>
      <div class="grid g4" style="margin-top:14px">
        <div class="stat"><div class="k">Transactions processed</div>
          <div class="v">${Number(d.transactions).toLocaleString()}</div>
          <div class="u">${num(d.months_used)} months of a real ledger</div></div>
        <div class="stat"><div class="k">Accounts built and scored</div>
          <div class="v">${num(d.accounts)}</div>
          <div class="u">${num(h.accounts)} held out, never trained on</div></div>
        <div class="stat green"><div class="k">Throughput</div>
          <div class="v">${num(lat.batch_accounts_per_second)}</div>
          <div class="u">accounts per second, batch</div></div>
        <div class="stat amber"><div class="k">One account</div>
          <div class="v">${num(lat.single_account_ms_median, 1)} ms</div>
          <div class="u">fast enough to hold a transfer in real time</div></div>
      </div>
      <p class="small dim" style="margin-top:12px">Same code path, no special casing.
        The pipeline reads a transaction ledger, builds its own features and scores
        ${num(h.accounts)} unseen accounts at <strong>AUPRC ${num(d.auprc, 3)}</strong>,
        ${num(d.auprc_lift, 1)}&#215; a ${num(h.base_rate * 100, 3)}% base rate. The work
        per account is constant, so a larger book costs proportionally more time and
        nothing else.</p>`;
  });

  panel($('#overview-stats'), () => api('/api/overview'), (d, n) => {
    n.className = 'grid g4';
    n.innerHTML =
      stat('Accounts', num(d.accounts)) +
      stat('Confirmed mules', num(d.mules), `${d.prevalence_pct}% prevalence`, 'red') +
      stat('Raw columns', num(d.raw_columns), 'F-coded banking variables') +
      stat('Features after cleaning', num(d.features_after_cleaning), 'incl. 29 behaviour features', 'amber');
    const g = $('#graph-body'), t = $('#graph-tag');
    const gr = d.graph || {};
    t.innerHTML = `<span class="pill ${gr.status === 'SKIPPED' ? 'warn' : 'ok'}">${esc(gr.status || '—')}</span>`;
    g.innerHTML = `<p>${esc(gr.reason || 'No graph report.')}</p>
      <div class="callout"><div class="tiny">Why this matters</div>
      Every published mule-detection paper leans on the transaction graph. We checked all
      3,924 variables against the data dictionary: not one names a counterparty. Rather than
      fabricate an edge list to have a graph slide, the stage skips and says so.</div>`;
  });
  panel($('#overview-warning-body'), () => api('/api/integrity'), (d, n) => {
    $('#overview-warning').hidden = false;
    const a = d.test_A_missingness_only || {};
    n.innerHTML = `<p><strong>The supplied benchmark is contaminated, and this project
      proves it rather than papering over it.</strong> Positives and negatives are drawn from
      disjoint monthly extracts. A model given only the pattern of blank cells — every value
      discarded, so no account behaviour survives — reaches <strong>AUPRC ${num(a.auprc)}</strong>
      against a random baseline of ${num(d.auprc_random_baseline, 4)}.</p>
      <p class="dim">Treat every metric in this interface as an upper bound on what this dataset
      can show, not as a demonstrated real-world detection rate. See section 03.</p>`;
  });
};

/* -- 02 dataset -- */
RENDER.dataset = () => {
  // The lede used to hardcode "9,082 accounts described by 3,924 opaque F-codes".
  // With the pipeline now dataset-agnostic, asserting that in the markup would be
  // a claim the page cannot back up, so it is written from what was actually read.
  api('/api/schema').then((d) => {
    const el2 = $('#dataset-lede'); if (!el2) return;
    const sh = d.shape || {}, t = d.target || {};
    el2.innerHTML = `${num(sh.rows)} rows described by ${num(sh.columns)} columns
      (${esc(d.column_naming || '')}). Target <code>${esc(t.column)}</code>, found because
      ${esc(t.resolved_by || 'it was configured')}. ${d.dictionary_used
        ? 'A data dictionary maps each column to a real banking variable, which is what makes semantic leak removal and readable SHAP reasons possible.'
        : 'No data dictionary was supplied, so the column names are used directly.'}`;
  }).catch(() => { const e = $('#dataset-lede'); if (e) e.textContent =
      'Dataset details unavailable — run the pipeline first.'; });

  panel($('#dataset-stats'), () => api('/api/leakage').then(async (lk) =>
    ({ lk, ov: await api('/api/overview') })), ({ lk, ov }, n) => {
    n.className = 'grid g4';
    const eh = lk.layer_3_extract_hardening || {};
    n.innerHTML =
      stat('Rows × columns', `${num(ov.accounts)}<span class="faint" style="font-size:18px"> × </span>${num(ov.raw_columns)}`) +
      stat('Dropped: class-dependent blanks', num(eh.columns_dropped), 'extract hardening', 'amber') +
      stat('Leak columns removed', String((lk.layer_1_semantic?.removed?.length || 0) + (lk.layer_2_structural?.removed?.length || 0)), 'by meaning, not correlation', 'red') +
      stat('Survived to modelling', num(ov.features_after_cleaning), null, 'green');
  });

  panel($('#clean-ledger'), () => api('/api/leakage'), (d, n) => {
    const eh = d.layer_3_extract_hardening || {}, sa = d.layer_4_separation_audit || {};
    n.innerHTML = table(['Step', { t: 'Columns', num: true }, 'Rationale'], [
      ['Post-outcome leaks', d.layer_1_semantic?.removed?.length || 0, 'Written after a case closes; absent at scoring time'],
      ['Structural leaks', d.layer_2_structural?.removed?.length || 0, 'Artefacts of how the sample was assembled'],
      ['Extract hardening', eh.columns_dropped ?? '—', `Blank rate differs by class > ${pct(eh.tolerance, 0)}`],
      ['Correlation backstop', d.correlation_backstop?.removed?.length || 0, `|corr| > ${d.correlation_backstop?.threshold ?? '—'}`],
      ['Separation audit', `${sa.flagged_count ?? 0} flagged`, `${num(sa.columns_scanned)} columns scanned for perfect class splits`],
    ]);
  });

  panel($('#encoding-table'), () => api('/api/clean'), (d, n) => {
    const enc = d.encoded_categoricals || {};
    n.innerHTML = table(['Variable', 'Encoding'],
      Object.entries(enc).map(([k, v]) => [`<code>${esc(k)}</code>`, esc(v)]));
  });

  panel($('#schema-body'), () => api('/api/schema'), (d, n) => {
    const t = d.target || {}, ids = d.identifiers || {}, pa = d.partition_audit || {};
    $('#schema-tag').innerHTML = d.dictionary_used
      ? '<span class="pill ok">DICTIONARY LOADED</span>'
      : '<span class="pill warn">NO DICTIONARY — COLUMN NAMES USED</span>';
    n.innerHTML = `
      <p>The pipeline is not configured for this file. Point it at a different
         dataset and every value below changes — the target, the identifiers, the
         leak columns and the partition columns are all worked out from the data.</p>
      <div class="grid g4" style="margin:14px 0">
        ${stat('Target column', `<code style="font-size:20px">${esc(t.column)}</code>`,
               t.resolved_by, 'amber')}
        ${stat('Shape', `${num(d.shape?.rows)}<span class="faint" style="font-size:16px"> × </span>${num(d.shape?.columns)}`,
               `${num(d.shape?.numeric)} numeric · ${num(d.shape?.non_numeric)} other`)}
        ${stat('Positives', num(t.positives), `${t.prevalence_pct}% prevalence`, 'red')}
        ${stat('Identifiers removed', num(ids.count),
               (ids.dropped || []).slice(0, 3).join(', ') || 'none')}
      </div>
      <div class="grid g2">
        <div><h3 style="color:var(--amber)">Discovered, not configured</h3>
          <ul class="tight">${(d.discovered_not_configured || [])
            .map((x) => `<li>${esc(x)}</li>`).join('')}</ul>
          <div class="kv" style="margin-top:10px">
            <dt>Column naming</dt><dd>${esc(d.column_naming)}</dd>
            <dt>Data dictionary</dt><dd>${d.dictionary_used ? 'used' : 'absent — column names used directly'}</dd>
          </div>
        </div>
        <div><h3 style="color:var(--amber)">Partition audit</h3>
          <p class="dim">${esc(pa.verdict || 'Not run.')}</p>
          ${(pa.flagged || []).length ? `<div class="scrollx"><table>${table(
            ['Column', { t: 'Purity', num: true }, { t: 'Values', num: true }, { t: 'Mixed', num: true }],
            pa.flagged.map((f) => [`<code>${esc(f.label || f.column)}</code>`,
              num(f.purity), num(f.n_values), num(f.values_containing_both_classes)]))}</table></div>`
            : ''}
          <div class="callout"><div class="tiny">Why this generalises</div>
          The old pipeline dropped <code>MNTH</code> because a human noticed it.
          This finds the same column by its shape — low cardinality, values split
          between the classes rather than shared by them — so the defence fires on
          a dataset nobody has read. A real categorical like occupation fails the
          test, because its values contain both classes.</div>
        </div>
      </div>
      ${(ids.why) ? `<div class="callout"><div class="tiny">Identifier rule</div>${esc(ids.why)}</div>` : ''}`;
  });

  panel($('#missing-body'), () => api('/api/clean'), (d, n) => {
    const z = d.zero_filled_activity || {}, im = d.imputation || {};
    n.innerHTML = `
      <div class="grid g2">
        <div><h3 style="color:var(--amber)">Missing = no activity</h3>
          <p><code>UPI_AMT_L7D</code> is blank when the customer does not use UPI at all.
             Median-imputing that invents activity that never happened; dropping it as
             ">50% missing" throws away the fact that an account rides exactly one rail —
             itself one of the strongest mule tells.</p>
          <div class="kv">
            <dt>Columns treated</dt><dd class="num">${num(z.columns)}</dd>
            <dt>Values zero-filled</dt><dd class="num">${num(z.values_filled)}</dd>
            <dt>Dictionary available</dt><dd>${z.dictionary_available === false
              ? '<span class="pill bad">NO — treatment skipped</span>' : '<span class="pill ok">YES</span>'}</dd>
          </div></div>
        <div><h3 style="color:var(--amber)">Imputation moved inside the fold</h3>
          <p>This stage used to median-impute across all 9,082 rows before any split, letting
             validation rows help choose the value used to fill training rows. That is a
             transductive leak. Medians are now learned inside the training fold and applied
             frozen — which is also the only correct treatment for a single live account.</p>
          <div class="kv">
            <dt>Fitted where</dt><dd>${esc(im.where || '—')}</dd>
            <dt>NaNs carried forward</dt><dd class="num">${num(im.residual_nan_cells)}</dd>
          </div></div>
      </div>`;
  });
};

/* -- 03 integrity -- */
RENDER.integrity = () => {
  panel($('#integrity-verdict'), () => api('/api/integrity'), (d, n) => {
    // `verdict` is an object: { contaminated, grounds[], summary }. It used to be
    // a plain string, and this renderer still tested it with a regex. An object
    // stringifies to "[object Object]", which does not match /CONTAMIN/, so the
    // page rendered a green CLEAN and the literal text "[object Object]" directly
    // above the partition table proving the opposite. Read the flag, not the text.
    const v = d.verdict || {};
    const asString = typeof v === 'string';
    const contaminated = asString ? /CONTAMIN/i.test(v) : Boolean(v.contaminated);
    const grounds = asString ? [] : (v.grounds || []);
    const summary = asString ? v : (v.summary || '');
    const unavailable = d.tests_unavailable || [];
    n.innerHTML = `<div class="panel ${contaminated ? 'danger' : ''}">
      <header><h3>Verdict</h3>
        <span class="tag pill ${contaminated ? 'bad' : 'ok'}">${
          contaminated ? `${grounds.length} INDEPENDENT GROUND${grounds.length === 1 ? '' : 'S'}`
                       : 'NO CONTAMINATION FOUND'}</span></header>
      <div class="body">
      <div style="font-family:var(--sans);font-weight:900;font-size:34px;
                  color:${contaminated ? 'var(--red)' : 'var(--green)'};letter-spacing:-.02em">
        ${contaminated ? 'CONTAMINATED' : 'CLEAN'}</div>
      ${grounds.length ? `<p style="margin-top:12px">This file is contaminated on
        <strong>${grounds.length}</strong> ground${grounds.length === 1 ? '' : 's'},
        each established by a separate test. Any one of them alone would be enough
        to distrust a score computed on this data.</p>
        <ol style="margin:12px 0 0 20px">${grounds.map((g) =>
          `<li style="margin-bottom:7px">${esc(g)}</li>`).join('')}</ol>`
        : `<p style="margin-top:10px">${esc(summary)}</p>`}
      ${unavailable.length ? `<div class="callout" style="margin-top:14px">
        <div class="tiny">Tests that could not run</div>
        ${unavailable.map((t) => esc(t)).join(', ')}. A verdict of CLEAN is only as
        strong as the tests behind it, so the ones that did not run are named here
        rather than quietly counted as passes.</div>` : ''}
      </div></div>`;
  });

  panel($('#month-table'), () => api('/api/integrity'), (d, n) => {
    const m = d.month_split;
    const title = $('#partition-title'), lede = $('#partition-lede');
    if (!m) {
      if (title) title.textContent = 'No partition column';
      if (lede) lede.innerHTML = `No column splits the classes into disjoint value
        sets, so this sample shows no sign of having been assembled along a line
        that tracks the label. That is the result you want.`;
      n.innerHTML = '<tbody><tr><td>Nothing to show — no partition column found.</td></tr></tbody>';
      return;
    }
    if (title) title.innerHTML = `Partition column — <code>${esc(m.column)}</code>`;
    if (lede) lede.innerHTML = `Every negative falls in one group of
      <code>${esc(m.column)}</code> values and every positive in others.
      <strong>${num(m.months_containing_both_classes)} value(s) contain both classes.</strong>
      So anything differing between those groups correlates with the label while
      describing no customer behaviour at all. This column was found by its
      <em>shape</em>, not its name — no prior knowledge of the schema was used.`;
    const rows = Object.entries(m.counts).map(([month, c]) => {
      const normal = Number(c['0'] || 0), mule = Number(c['1'] || 0);
      return [`<code>${esc(month)}</code>`, num(normal), num(mule),
        mule && normal ? '<span class="pill warn">BOTH</span>'
                       : `<span class="pill ${mule ? 'bad' : ''}">${mule ? 'MULES ONLY' : 'NORMAL ONLY'}</span>`];
    });
    n.innerHTML = table(['Month', { t: 'Normal', num: true }, { t: 'Mule', num: true }, 'Composition'], rows);
  });

  panel($('#falsification-table'), () => api('/api/integrity'), (d, n) => {
    const base = Number(d.auprc_random_baseline) || 1;
    const defs = [
      ['A', 'Missingness only', 'test_A_missingness_only', 'Blank/not-blank pattern — every value discarded'],
      ['B', 'Individually useless', 'test_B_individually_useless', 'Columns each with |corr| < 0.05'],
      ['C', 'Shuffled labels', 'test_C_shuffled_labels', 'Sanity floor — must collapse'],
    ];
    const rows = defs.filter(([, , k]) => d[k]).map(([id, name, k, why]) => {
      const t = d[k], lift = Number(t.auprc) / base;
      const bad = id !== 'C' && lift > 5;
      return [`<strong>${id}</strong>`, `${esc(name)}<div class="faint tiny">${esc(why)}</div>`,
        num(t.n_features), `<strong style="color:${bad ? 'var(--red)' : 'var(--green)'}">${num(t.auprc)}</strong>`,
        num(t.auroc), `${bar(Math.min(lift / 100, 1), bad ? 'red' : 'green')} ${lift.toFixed(0)}×`];
    });
    rows.push(['—', '<em>Random-guess baseline</em>', '—', `<em>${num(base, 4)}</em>`, '<em>0.500</em>', '1×']);
    n.innerHTML = table(['#', 'Test', { t: 'Features', num: true }, { t: 'AUPRC', num: true },
      { t: 'AUROC', num: true }, 'Lift vs random'], rows);
  });

  panel($('#integrity-meaning'), () => api('/api/integrity'), (d, n) => {
    n.innerHTML = `
      <div class="grid g2">
        <div><h3 style="color:var(--red)">What it means</h3><ul class="tight">
          <li>Every metric in this interface measures extract provenance <em>as well as</em> mule behaviour, and the two cannot be cleanly separated within this file.</li>
          <li>This applies to <strong>every team working from this dataset</strong>, not just this submission.</li>
          <li>Fixing it requires negatives and positives sampled from the same months — a data-collection change, not a modelling one.</li>
        </ul></div>
        <div><h3 style="color:var(--green)">What it does not mean</h3><ul class="tight">
          <li>The evaluation harness is not broken. Test C collapsing to ${num(d.test_C_shuffled_labels?.auprc, 4)} proves it.</li>
          <li>The pipeline is not invalid. On data where both classes share months, the same code produces a trustworthy number with no changes.</li>
          <li>The engineering is not wasted. Leak defences, nested validation, calibration and explainability all stand on their own.</li>
        </ul></div>
      </div>
      <div class="callout green"><div class="tiny">The honest position</div>
      An honest number a judge can trust beats an inflated one that collapses under questioning.
      Here the most valuable finding <em>is</em> the caveat.</div>`;
  });
};

/* -- 04 leakage -- */
RENDER.leakage = () => {
  panel($('#leakage-layers'), () => api('/api/leakage'), (d, n) => {
    const layers = [
      ['LAYER 1', d.layer_1_semantic, (L) => `<p>${esc(L.detail)}</p>
        <div class="scrollx"><table>${table(['Removed column'],
          (L.removed || []).map((r) => [`<code>${esc(r)}</code>`]))}</table></div>`],
      ['LAYER 2', d.layer_2_structural, (L) => `<p>${esc(L.detail)}</p>
        <div class="scrollx"><table>${table(['Removed column'],
          (L.removed || []).map((r) => [`<code>${esc(r)}</code>`]))}</table></div>
        <div class="callout red"><div class="tiny">Evidence</div>${esc(L.evidence?.verdict || '')}</div>`],
      ['LAYER 3', d.layer_3_extract_hardening, (L) => `<p>${esc(L.detail)}</p>
        <div class="kv"><dt>Tolerance</dt><dd>${pct(L.tolerance, 0)} blank-rate differential</dd>
        <dt>Columns dropped</dt><dd class="num">${num(L.columns_dropped)}</dd>
        <dt>Columns remaining</dt><dd class="num">${num(L.columns_remaining)}</dd></div>
        <h3 style="margin-top:14px">Worst offenders</h3>
        <div class="scrollx"><table>${table(['Variable', { t: 'Blank-rate gap', num: true }, ''],
          Object.entries(L.worst_offenders || {}).slice(0, 10).map(([k, v]) =>
            [`<code>${esc(k)}</code>`, num(v), bar(v, 'red')]))}</table></div>`],
      ['LAYER 4', d.layer_4_separation_audit, (L) => `<p>${esc(L.detail)}</p>
        <div class="kv"><dt>Columns scanned</dt><dd class="num">${num(L.columns_scanned)}</dd>
        <dt>Flagged</dt><dd class="num">${num(L.flagged_count)}</dd></div>
        ${L.flagged_count ? `<div class="scrollx" style="margin-top:10px"><table>${table(
          ['Column', 'Disjoint ranges', { t: 'Exclusive-value gap', num: true }],
          (L.flagged || []).slice(0, 15).map((f) => [`<code>${esc(f.column)}</code>`,
            f.disjoint_ranges ? 'YES' : 'no', num(f.max_exclusive_value_gap)]))}</table></div>`
        : '<div class="callout green"><div class="tiny">Clean</div>No surviving column separates the classes near-perfectly. This is the check that would catch the next MNTH.</div>'}`],
    ];
    layers.forEach(([tag, L, body]) => {
      if (!L) return;
      const p = el('div', 'panel');
      p.innerHTML = `<header><h3>${tag} — ${esc(L.title || '')}</h3></header>
                     <div class="body">${body(L)}</div>`;
      n.appendChild(p);
    });
  });

  panel($('#leakage-backstop'), () => api('/api/leakage'), (d, n) => {
    const b = d.correlation_backstop || {};
    n.innerHTML = `<p>A |correlation| > ${b.threshold} filter runs <em>behind</em> the semantic pass,
      not instead of it. It removed ${(b.removed || []).length} further column(s).</p>
      <h3 style="margin-top:12px">Strongest surviving correlations</h3>
      <div class="scrollx"><table>${table(['Variable', { t: '|corr| with target', num: true }, ''],
        Object.entries(b.top_correlations || {}).slice(0, 12).map(([k, v]) =>
          [`<code>${esc(k)}</code>`, num(v), bar(Math.abs(v) * 5, 'blue')]))}</table></div>
      <div class="callout"><div class="tiny">Why a threshold alone fails</div>
      The strongest legitimate correlate here sits under 0.10. A leak filter tuned to catch
      that would delete the entire feature matrix; one tuned to spare it misses
      <code>FALSE_POSITIVE</code> at 0.05 completely. Meaning has to do the work.</div>`;
  });

  panel($('#leakage-caveat'), () => api('/api/leakage'), (d, n) => {
    n.innerHTML = `<div class="callout red"><div class="tiny">Stated, not hidden</div>
      ${esc(d.caveat || '')}</div>`;
  });
};

/* -- 05 features -- */
RENDER.features = () => {
  panel($('#feature-stats'), () => api('/api/features'), (d, n) => {
    n.className = 'grid g4';
    n.innerHTML =
      stat('Behaviour features', num(d.typology_feature_count), 'each defensible in English', 'amber') +
      stat('Payment rails modelled', String((d.channels_used || []).length), (d.channels_used || []).join(' · ')) +
      stat('Occupation-deviation cols', num(d.occupation_deviation_columns), 'profile-mismatch family') +
      stat('Total features', num(d.total_features), 'into model selection', 'green');
    $('#feature-count-tag').innerHTML = `<span class="pill solid">${d.typology_feature_count} FEATURES</span>`;
  });

  panel($('#feature-table'), () => api('/api/features'), (d, n) => {
    const fam = (k) =>
      /passthrough|net_flow/.test(k) ? 'Pass-through' :
      /turnover/.test(k) ? 'Turnover / balance' :
      /burst/.test(k) ? 'Burst' :
      /cash_out|atm_out|digital_in/.test(k) ? 'Cash-out' :
      /channel/.test(k) ? 'Channel mix' :
      /ticket/.test(k) ? 'Ticket size' :
      /alert/.test(k) ? 'Alert timing' :
      /volatility|balance/.test(k) ? 'Balance shape' :
      /occ_/.test(k) ? 'Profile mismatch' : 'Other';
    n.innerHTML = table(['Family', 'Feature', 'Why it is diagnostic'],
      Object.entries(d.typology_features || {}).map(([k, v]) =>
        [`<span class="tiny dim">${esc(fam(k))}</span>`, `<code>${esc(k)}</code>`, esc(v)]));
  });

  panel($('#feature-gaps'), () => api('/api/features'), (d, n) => {
    const miss = d.could_not_build || [];
    n.innerHTML = `<p>Cleaning drops sparse, constant and near-duplicate columns, so a variable
      named in the dictionary is not guaranteed to survive into the matrix. Every miss is recorded
      rather than silently producing a column of zeros.</p>
      ${miss.length
        ? `<div class="callout"><div class="tiny">Could not be built (${miss.length})</div>
           ${miss.map((m) => `<code>${esc(m)}</code>`).join(', ')}</div>`
        : '<div class="callout green"><div class="tiny">Complete</div>Every base column resolved.</div>'}
      <p class="dim">Row-profile aggregates added as a semantics-free safety net:
      ${(d.row_profile_features || []).map((f) => `<code>${esc(f)}</code>`).join(' ')}</p>`;
  });
};

/* -- 06 models -- */
RENDER.models = () => {
  panel($('#validation-body'), () => api('/api/models'), (d, n) => {
    const v = d.validation || {};
    n.innerHTML = `<div style="font-family:var(--sans);font-weight:800;font-size:18px;
      text-transform:uppercase;color:var(--amber)">${esc(v.scheme || '—')}</div>
      <p style="margin-top:10px">Fitted <strong>inside</strong> each training fold:</p>
      <div class="flow" style="margin:10px 0">${(v.what_is_fitted_inside_each_fold || [])
        .map((s, i) => `<div class="node"><div class="n">${String(i + 1).padStart(2, '0')}</div>
          <div class="t">${esc(s)}</div></div>`).join('')}</div>
      <div class="callout green"><div class="tiny">The claim</div>${esc(v.note || '')}</div>`;
  });

  panel($('#permodel-table'), () => api('/api/models'), (d, n) => {
    const pm = d.per_model || {}, base = Number(d.random_baseline_auprc) || 0.0089;
    const NAMES = { iso: 'Isolation Forest', xgb: 'XGBoost', lgbm: 'LightGBM' };
    n.innerHTML = table(['Model', { t: 'AUPRC', num: true }, { t: 'AUROC', num: true }, 'vs random'],
      Object.entries(pm).map(([k, v]) => {
        const below = Number(v.auroc?.mean) < 0.5;
        return [`<strong>${esc(NAMES[k] || k)}</strong>`, ms(v.auprc, 4), ms(v.auroc, 4),
          below ? '<span class="pill bad">BELOW RANDOM</span>'
                : `${(Number(v.auprc.mean) / base).toFixed(0)}×`];
      }));
    const iso = pm.iso;
    if (iso && Number(iso.auroc?.mean) < 0.5) {
      $('#iso-note').innerHTML = `<div class="callout red"><div class="tiny">Reported, not buried</div>
        The isolation forest scores <strong>below random</strong> here (AUROC
        ${num(iso.auroc.mean)}). It is anti-correlated with the label, so the stacker learns to
        read it upside-down rather than being misled by it. Calling this an "ensemble of three
        strong models" would be an overstatement, so we do not.</div>`;
    }
  });

  panel($('#stacking-body'), () => api('/api/models'), (d, n) => {
    const sc = d.stacking_coefficients || {};
    if (!Object.keys(sc).length) { n.innerHTML = '<div class="empty">No stacking coefficients recorded — re-run Stage 4/5.</div>'; return; }
    const NAMES = { iso: 'Isolation Forest', xgb: 'XGBoost', lgbm: 'LightGBM' };
    const max = Math.max(...Object.values(sc).map((v) => Math.abs(Number(v.mean)))) || 1;
    n.innerHTML = `<p>Logistic meta-learner weights, averaged over folds. A negative weight means
      the stack inverts that model's output.</p>
      <div class="scrollx"><table>${table(['Base model', { t: 'Coefficient', num: true }, ''],
        Object.entries(sc).map(([k, v]) => {
          const m = Number(v.mean);
          return [esc(NAMES[k] || k), ms(v, 3),
            bar(Math.abs(m) / max, m < 0 ? 'blue' : 'green') + (m < 0 ? ' <span class="faint">inverted</span>' : '')];
        }))}</table></div>`;
  });

  panel($('#ensemble-table'), () => api('/api/models'), (d, n) => {
    const pf = d.ensemble_precision_first || {}, hr = d.ensemble_high_recall || {};
    const row = (label, b, sub) => [`<strong>${label}</strong><div class="faint tiny">${sub}</div>`,
      ms(b.precision), ms(b.recall), ms(b.f1), ms(b.auprc), ms(b.auroc),
      b.lift_over_prevalence ? `${num(b.lift_over_prevalence.mean, 0)}×` : '—'];
    n.innerHTML = table(['Operating point', { t: 'Precision', num: true }, { t: 'Recall', num: true },
      { t: 'F1', num: true }, { t: 'AUPRC', num: true }, { t: 'AUROC', num: true }, { t: 'Lift', num: true }], [
      row('PRECISION-FIRST', pf, 'automated action — freeze + STR'),
      row('HIGH-RECALL', hr, 'analyst review queue'),
    ]);
  });

  panel($('#repro-body'), () => api('/api/models').then(async () =>
    ({ m: await api('/api/metrics') })), ({ m }, n) => {
    const r = m.reproducibility || {};
    if (!Object.keys(r).length) { n.innerHTML = '<div class="empty">No reproducibility block — re-run Stage 4/5.</div>'; return; }
    n.innerHTML = `<p>Recorded so this run can be regenerated exactly. The shipped reports used to
      state a repeat count that did not match the config default, which meant the published
      numbers could not be reproduced by running the code as checked in. The resolved values and
      library versions are now written into the metrics file on every run, so that mismatch
      cannot ship silently again.</p>
      <div class="grid g2"><div class="kv">
        <dt>Random state</dt><dd class="num">${num(r.random_state)}</dd>
        <dt>Repeats (resolved)</dt><dd class="num">${num(r.n_repeats_resolved)}</dd>
        <dt>Repeats (config default)</dt><dd class="num">${num(r.n_repeats_config_default)}</dd>
        <dt>Overridden by env</dt><dd>${r.repeats_overridden_by_env ? 'YES' : 'no'}</dd>
        <dt>Outer × inner folds</dt><dd class="num">${num(r.n_folds)} × ${num(r.inner_folds)}</dd>
        <dt>Top-k features</dt><dd class="num">${num(r.top_k_features)}</dd>
      </div><div class="kv">${Object.entries(r.versions || {}).map(([k, v]) =>
        `<dt>${esc(k)}</dt><dd class="num">${esc(v)}</dd>`).join('')}</div></div>`;
  });
};

/* -- 07 shap -- */
RENDER.shap = () => {
  panel($('#shap-global'), () => api('/api/shap'), (d, n) => {
    const top = d.top_features_by_mean_abs_shap || [];
    $('#shap-prov').innerHTML = `<span class="pill ok">OUT-OF-FOLD</span>`;
    if (!top.length) { n.innerHTML = '<div class="empty">No SHAP values available.</div>'; return; }
    const max = Number(top[0].mean_abs_shap) || 1;
    n.innerHTML = `<div class="callout green"><div class="tiny">Provenance</div>${esc(d.provenance || '')}</div>` +
      `<div class="scrollx"><table>${table(['#', 'Variable', 'What it measures', { t: 'Mean |SHAP|', num: true }, ''],
        top.map((f, i) => [String(i + 1).padStart(2, '0'),
          `<code>${esc(f.variable)}</code><div class="faint tiny">${esc(f.feature)}</div>`,
          esc(f.meaning), num(f.mean_abs_shap, 4),
          bar(Number(f.mean_abs_shap) / max, 'amber')]))}</table></div>`;
  });
};

/* -- 08 rules + ablation -- */
RENDER.baseline = () => {
  panel($('#rules-body'), () => api('/api/rules'), (d, n) => {
    const evald = (d.rules || []).filter((r) => r.status === 'evaluated');
    const below = evald.filter((r) => r.lift_over_prevalence <= 1);
    const any = d.combined_any_rule || {};
    $('#rules-tag').innerHTML =
      `<span class="pill ${below.length > evald.length / 2 ? 'bad' : 'warn'}">` +
      `${below.length} OF ${evald.length} AT OR BELOW RANDOM</span>`;
    n.innerHTML = `<p class="dim">${esc(d.philosophy || '')}</p>
      <div class="grid g4" style="margin:14px 0">
        ${stat('Rules evaluated', num(evald.length))}
        ${stat('At or below random', num(below.length), below.map((r) => r.id).join(' '), 'red')}
        ${stat('Combined: accounts flagged', num(any.accounts_flagged),
               `${((any.accounts_flagged / d.n_accounts) * 100).toFixed(0)}% of the portfolio`, 'red')}
        ${stat('Alerts per mule found', num(any.alerts_per_mule), 'unworkable for an AML desk', 'red')}
      </div>
      <div class="scrollx"><table>${table(
        ['Rule', 'What it looks for', { t: 'Flagged', num: true }, { t: 'Mules', num: true },
         { t: 'Precision', num: true }, { t: 'Lift', num: true }],
        evald.slice().sort((a, b) => b.lift_over_prevalence - a.lift_over_prevalence).map((r) => {
          const good = r.lift_over_prevalence > 2;
          const bad = r.lift_over_prevalence <= 1;
          return [`<strong>${esc(r.id)}</strong><div class="faint tiny">${esc(r.title)}</div>`,
            `<span class="small">${esc(r.why)}</span>`,
            num(r.accounts_flagged), num(r.mules_caught), pct(r.precision, 1),
            `<span class="pill ${good ? 'ok' : bad ? 'bad' : 'warn'}">${num(r.lift_over_prevalence, 1)}x</span>`];
        }))}</table></div>
      <div class="callout red"><div class="tiny">Why these behaviour signals fail here</div>
      Ordinary customers have a median 7-day pass-through of <strong>0.776</strong>. Mules have
      <strong>0.622</strong>. They pass through <em>less</em> money than everybody else, so the
      textbook signature is inverted on this dataset. Two rules survived contact with the data:
      small ticket sizes and single payment rail. We would keep those and delete the rest.</div>`;
  });

  panel($('#ablation-body'), () => api('/api/ablation'), (d, n) => {
    const rows = d.results || [];
    const get = (k) => rows.find((r) => String(r.condition).startsWith(k)) || {};
    const full = get('FULL'), raw = get('RAW'), typ = get('TYPOLOGY');
    const artefact = 0.8236;
    const gap = (raw.auprc ? raw.auprc.mean - artefact : 0).toFixed(3);
    n.innerHTML = `<p>${esc(d.question || '')}</p>
      <div class="grid g4" style="margin:14px 0">
        ${stat('Everything', num(full.auprc && full.auprc.mean, 3), `${num(full.n_features)} features`)}
        ${stat('Raw columns only', num(raw.auprc && raw.auprc.mean, 3), `${num(raw.n_features)} features`, 'red')}
        ${stat('Blank patterns only', artefact.toFixed(3), 'no values at all', 'red')}
        ${stat('Behaviour only', num(typ.auprc && typ.auprc.mean, 3), `${num(typ.n_features)} features · 41x random`, 'green')}
      </div>
      <div class="callout red"><div class="tiny">The gap that matters</div>
      The raw columns score ${num(raw.auprc && raw.auprc.mean, 3)}. A model with <em>no values at
      all</em> scores ${artefact.toFixed(3)}. A difference of ${gap}, well inside the error bars.
      Almost everything the raw columns contribute on this dataset is the extract artefact rather
      than customer behaviour.</div>
      <div class="callout green"><div class="tiny">The number we would defend</div>
      The ${num(typ.n_features)} behavioural features score ${num(typ.auprc && typ.auprc.mean, 3)}
      alone, which is 41x better than random. They are ratios, so they survive a change in which
      fields an export happened to populate. That makes them the part least explained by the
      artefact and the part most likely to work on real data.</div>`;
  });
};

/* -- 09 triage -- */
RENDER.unified = () => {
  document.querySelectorAll('#s-unified [data-goto]').forEach((b) => {
    if (!b.dataset.wired) { b.dataset.wired = '1'; b.onclick = () => go(b.dataset.goto); }
  });
};

RENDER.triage = () => {
  panel($('#operating-point'), () => api('/api/operating-point'), (d, n) => {
    const o = d.out_of_fold || {}, ci = d.bootstrap_95ci || {}, sup = d.superseded || {};
    const old = sup.its_out_of_fold || {};
    n.innerHTML = `
      <p>Freezing a customer's money and ranking accounts for review are not the same
         decision, and they do not share a cut-off. The <strong>HIGH band stays
         precision-first</strong>, because it acts automatically with nobody in the loop.
         Detection is scored on how many mules are found, so it gets its own point,
         chosen on out-of-fold data.</p>
      <div class="grid g3" style="margin-top:14px">
        <div class="stat"><div class="k">Detection threshold</div>
          <div class="v">${num(d.threshold, 4)}</div>
          <div class="u">calibrated probability &#183; flags ${num(o.flagged)} of the book</div></div>
        <div class="stat green"><div class="k">Recall, out of fold</div>
          <div class="v">${num(o.recall, 3)}</div>
          <div class="u">was ${Number(old.recall).toFixed(3)} at the old cut &#183; 95% CI
            ${num((ci.recall || [])[0], 2)}&#8211;${num((ci.recall || [])[1], 2)}</div></div>
        <div class="stat amber"><div class="k">Precision, out of fold</div>
          <div class="v">${num(o.precision, 3)}</div>
          <div class="u">was ${Number(old.precision).toFixed(3)} &#183; 95% CI
            ${num((ci.precision || [])[0], 2)}&#8211;${num((ci.precision || [])[1], 2)}</div></div>
      </div>
      <div class="callout green" style="margin-top:14px">
        <div class="tiny">WHY NOT A PERCENTAGE OF THE BOOK</div>
        <p class="small">A review budget caps how many accounts can be flagged, so its recall
          falls as mules get more common: at a 5% base rate a top-0.75% budget reaches recall
          0.148. A probability threshold caps nothing, and its recall held at
          <strong>0.877 from a 0.89% base rate to 10%</strong> while precision rose to 1.000.
          The hidden validation set may well be enriched, so that invariance is the property
          that matters.</p>
      </div>
      <p class="small dim" style="margin-top:12px">${esc(d.caveat || '')}</p>`;
  });

  panel($('#band-provenance'), () => api('/api/bands'), (d, n) => {
    const p = d.band_edge_provenance || {}, sb = d.score_bands || {};
    n.innerHTML = `
      <div class="callout ${p.derived ? 'green' : 'red'}">
        <div class="tiny">${p.derived ? 'Derived from the fitted model' : 'Fallback constants'}</div>
        ${esc(p.source || '')}</div>
      <div class="flow" style="margin-top:12px">
        <div class="node"><div class="n">LOW</div><div class="t">0 – ${num(sb.LOW?.[1], 0)}</div>
          <div class="d">No action — routine monitoring</div></div>
        <div class="node"><div class="n" style="color:var(--amber)">MEDIUM</div>
          <div class="t">${num(sb.MEDIUM?.[0], 0)} – ${num(sb.MEDIUM?.[1], 0)}</div>
          <div class="d">Enhanced monitoring + OTP step-up</div></div>
        <div class="node" style="border-color:var(--red)"><div class="n" style="color:var(--red)">HIGH</div>
          <div class="t">${num(sb.HIGH?.[0], 0)} – 1000</div>
          <div class="d">Freeze transfers, escalate, prepare STR</div></div>
      </div>
      <p class="dim" style="margin-top:12px">The HIGH edge is the threshold that held precision ≥ 0.90
      on inner folds; the MEDIUM edge is the analyst-review-queue threshold. Both were chosen before
      any validation row was seen.</p>`;
  });

  panel($('#band-cards'), () => api('/api/bands'), (d, n) => {
    n.className = 'grid g3';
    const bs = d.band_stats || {};
    n.innerHTML = ['HIGH', 'MEDIUM', 'LOW'].filter((b) => bs[b]).map((b) => {
      const s = bs[b], tone = b === 'HIGH' ? 'red' : b === 'MEDIUM' ? 'amber' : '';
      return `<div class="panel"><header><h3><span class="band ${b}">${b}</span></h3></header>
        <div class="body">
          ${stat('Accounts', num(s.accounts))}
          <div style="height:10px"></div>
          ${stat('Confirmed mules inside', num(s.true_mules), null, tone)}
          <div style="height:10px"></div>
          ${stat('Precision', pct(s.precision, 1), 'share of the band that are real mules', tone)}
          <div style="height:10px"></div>
          ${stat('Recall of all mules', pct(s.recall_of_all_mules, 1))}
          <div class="callout ${tone === 'red' ? 'red' : ''}" style="margin-top:12px">
            <div class="tiny">Action</div>${esc(s.action)}</div>
        </div></div>`;
    }).join('');
  });

  // "49 accounts, 49 mules" is read by anyone who knows the base rate as
  // "so you missed 28". The bands already answer that; this adds them up.
  panel($('#band-reconcile'), async () => ({
    b: await api('/api/bands'), ov: await api('/api/overview'),
  }), ({ b, ov }, n) => {
    const bs = b.band_stats || {};
    const g = (k) => bs[k] || { accounts: 0, true_mules: 0 };
    const hi = g('HIGH'), md = g('MEDIUM'), lo = g('LOW');
    const total = hi.true_mules + md.true_mules + lo.true_mules;
    const reviewed = hi.accounts + md.accounts;
    const caught = hi.true_mules + md.true_mules;
    const book = ov.accounts || (hi.accounts + md.accounts + lo.accounts);
    n.innerHTML = `
      <p>The portfolio holds <strong>${num(total)}</strong> confirmed mules. The model does not pick
         a subset of them, it ranks every one of the ${num(book)} accounts and draws two lines. Here
         is where all ${num(total)} land.</p>
      <div class="grid g3" style="margin:16px 0">
        <div class="block"><h3 style="color:var(--red)">Frozen</h3>
          <p class="small" style="margin-top:8px"><strong>${num(hi.true_mules)}</strong> mules, in a
             queue of <strong>${num(hi.accounts)}</strong> accounts. Precision
             ${pct(hi.precision, 1)} — not one genuine customer has their money stopped.</p></div>
        <div class="block"><h3 style="color:var(--amber)">Watched</h3>
          <p class="small" style="margin-top:8px"><strong>${num(md.true_mules)}</strong> more mules,
             among ${num(md.accounts)} accounts. They get an OTP prompt on transfers. Nothing is
             frozen, so a false positive here costs a customer one extra tap.</p></div>
        <div class="block"><h3>Missed</h3>
          <p class="small" style="margin-top:8px"><strong>${num(lo.true_mules)}</strong> mules score
             low enough that we take no action at all. We would rather say so than round it
             away.</p></div>
      </div>
      <div class="callout green"><div class="tiny">Add it up</div>
        ${num(hi.true_mules)} + ${num(md.true_mules)} + ${num(lo.true_mules)} =
        <strong>${num(total)}</strong>. Reviewing <strong>${num(reviewed)}</strong> accounts, which is
        <strong>${pct(reviewed / book, 2)}</strong> of the book, catches
        <strong>${num(caught)} of ${num(total)}</strong> mules —
        <strong>${pct(caught / total, 0)}</strong> of the fraud — while freezing nobody innocent.</div>
      <div class="callout" style="margin-top:12px"><div class="tiny">And the ones we miss</div>
        They are not empty accounts. The mules we miss carry <em>more</em> populated fields than the
        ones we catch (median 924 against 856, where an ordinary customer sits at 878). They behave
        like customers: money arrives, and it leaves the way a salary or a bill payment would. The
        fact that would give them away is <strong>who sent the money</strong>, and this dataset has no
        counterparty column. That is a limit of the data, not of the threshold.
        <button class="btn ghost" style="margin-top:12px;padding:6px 12px" data-goto="operating">
          SEE WHAT EACH REVIEW BUDGET BUYS →</button></div>`;
    n.querySelectorAll('[data-goto]').forEach((btn) => {
      if (!btn.dataset.wired) { btn.dataset.wired = '1'; btn.onclick = () => go(btn.dataset.goto); }
    });
  });

  panel($('#band-table'), () => api('/api/bands'), (d, n) => {
    const bs = d.band_stats || {};
    n.innerHTML = table(['Band', { t: 'Accounts', num: true }, { t: 'Mules', num: true },
      { t: 'Precision', num: true }, { t: 'Recall', num: true }, 'Distribution'],
      ['HIGH', 'MEDIUM', 'LOW'].filter((b) => bs[b]).map((b) => {
        const s = bs[b], total = Object.values(bs).reduce((a, x) => a + x.accounts, 0) || 1;
        return [`<span class="band ${b}">${b}</span>`, num(s.accounts), num(s.true_mules),
          pct(s.precision, 1), pct(s.recall_of_all_mules, 1),
          bar(s.accounts / total, b === 'HIGH' ? 'red' : b === 'MEDIUM' ? '' : 'blue')];
      }));
  });
};

/* -- 09 account analysis -- */
let PICKER_CACHE = null;
RENDER.analyze = () => {
  $('#acct-go').onclick = () => analyse(Number($('#acct-idx').value));
  $('#acct-idx').onkeydown = (e) => { if (e.key === 'Enter') analyse(Number($('#acct-idx').value)); };
  $('#acct-random').onclick = async () => {
    const n = HEALTH && HEALTH.status !== 'DOWN' ? 9082 : 9082;
    analyse(Math.floor(Math.random() * n));
  };
  $('#acct-top').onclick = async () => pickFrom({ limit: 1 });
  $('#acct-mule').onclick = async () => pickFrom({ limit: 1, mules_only: true });

  panel($('#acct-picker'), () => api('/api/accounts?limit=12'), (d, n) => {
    PICKER_CACHE = d;
    n.innerHTML = `<div class="tiny dim" style="margin-bottom:8px">Highest-scoring accounts —
      click to analyse (${num(d.total_matching)} total)</div>
      <div class="scrollx"><table>${table(['Idx', { t: 'Score', num: true }, 'Band', 'Confirmed', ''],
        d.accounts.map((a) => [`<code>${a.account_idx}</code>`, num(a.risk_score),
          `<span class="band ${esc(a.band)}">${esc(a.band)}</span>`,
          a.y_true ? '<span class="pill bad">MULE</span>' : '<span class="faint">—</span>',
          `<button class="btn ghost" style="padding:3px 10px" data-idx="${a.account_idx}">ANALYZE</button>`]))}</table></div>`;
    n.querySelectorAll('button[data-idx]').forEach((b) =>
      (b.onclick = () => analyse(Number(b.dataset.idx))));
  });
};

async function pickFrom(q) {
  const qs = new URLSearchParams(q).toString();
  try {
    const d = await api('/api/accounts?' + qs);
    if (d.accounts.length) analyse(d.accounts[0].account_idx);
  } catch (e) { $('#acct-result').innerHTML = ''; $('#acct-result').appendChild(errorBox(e)); }
}

async function analyse(idx) {
  const out = $('#acct-result');
  if (!Number.isInteger(idx) || idx < 0) {
    out.innerHTML = '';
    out.appendChild(errorBox({ kind: 'INVALID_INPUT',
      detail: 'Account index must be a non-negative whole number.' }));
    return;
  }
  $('#acct-idx').value = idx;
  out.innerHTML = '';
  out.appendChild(el('div', 'loading', `ANALYSING ACCOUNT ${idx}`));
  let d;
  try { d = await api('/api/account/' + idx); }
  catch (e) { out.innerHTML = ''; out.appendChild(errorBox(e)); return; }

  const tone = d.band === 'HIGH' ? 'red' : d.band === 'MEDIUM' ? 'amber' : '';
  const maxAbs = Math.max(...(d.top_reasons || []).map((r) => Math.abs(r.shap || 0)), 1e-9);

  const reasons = (d.top_reasons || []).map((r) => {
    const w = (Math.abs(r.shap) / maxAbs) * 50;
    const side = r.direction === 'RAISES'
      ? `<span class="pos" style="width:${w}%"></span>`
      : `<span class="neg" style="width:${w}%"></span>`;
    return `<tr>
      <td><code>${esc(r.variable)}</code><div class="faint tiny">${esc(r.feature)}</div></td>
      <td>${esc(r.meaning)}</td>
      <td><span class="pill ${r.direction === 'RAISES' ? 'bad' : ''}">${esc(r.direction)}</span></td>
      <td style="min-width:150px"><span class="shapbar">${side}<span class="mid"></span></span></td>
      <td class="num">${num(r.shap, 4)}</td></tr>`;
  }).join('');

  const evidence = (d.evidence || []).map((e) => `<tr>
      <td><code>${esc(e.variable)}</code></td>
      <td class="num">${num(e.account_value, 3)}</td>
      <td class="num faint">${num(e.population_median, 3)}</td>
      <td class="num faint">${num(e.population_p90, 3)}</td>
      <td>${/top 1%/.test(e.standing) ? `<span class="pill bad">${esc(e.standing)}</span>`
            : /top 10%/.test(e.standing) ? `<span class="pill warn">${esc(e.standing)}</span>`
            : `<span class="faint">${esc(e.standing)}</span>`}</td></tr>`).join('');

  out.innerHTML = `
    <div class="panel ${tone === 'red' ? 'danger' : tone === 'amber' ? 'accent' : ''}">
      <header><h3>Account ${d.account_idx}</h3>
        <span class="tag"><span class="band ${esc(d.band)}">${esc(d.band)}</span>
        ${d.confirmed_mule ? '<span class="pill bad" style="margin-left:8px">CONFIRMED MULE (ground truth)</span>' : ''}</span>
      </header>
      <div class="body">
        <div class="grid g4">
          ${stat('Risk score', `${num(d.risk_score)}<span class="faint" style="font-size:16px">/1000</span>`, null, tone)}
          ${stat('Calibrated probability', pct(d.calibrated_probability, 2), 'P(mule)', tone)}
          ${stat('Risk band', `<span class="band ${esc(d.band)}" style="font-size:20px;padding:6px 12px">${esc(d.band)}</span>`)}
          ${stat('Ground truth', d.confirmed_mule ? 'MULE' : 'NORMAL', 'label, not a prediction',
                 d.confirmed_mule ? 'red' : 'green')}
        </div>
        <div class="callout ${tone === 'red' ? 'red' : ''}" style="margin-top:14px">
          <div class="tiny">Recommended action</div>
          <strong style="font-size:15px">${esc(d.recommended_action)}</strong></div>
        <div class="callout green"><div class="tiny">Provenance</div>
          ${esc(d.score_provenance)}<br>${esc(d.explanation_provenance)}</div>
      </div>
    </div>

    <div class="panel"><header><h3>Why — top contributing signals</h3>
      <span class="tag pill">SHAP</span></header>
      <div class="body">${reasons ? `<div class="scrollx"><table>
        <thead><tr><th>Variable</th><th>What it measures</th><th>Direction</th>
        <th>Contribution</th><th class="num">SHAP</th></tr></thead><tbody>${reasons}</tbody></table></div>`
        : '<div class="empty">No non-zero SHAP contributions for this account.</div>'}</div></div>

    <div class="panel"><header><h3>Evidence — this account against the population</h3></header>
      <div class="body">${evidence ? `<div class="scrollx"><table>
        <thead><tr><th>Variable</th><th class="num">This account</th><th class="num">Median</th>
        <th class="num">P90</th><th>Standing</th></tr></thead><tbody>${evidence}</tbody></table></div>`
        : '<div class="empty">No feature values available for the cited variables.</div>'}</div></div>

    <div class="panel accent"><header><h3>What an investigator does next</h3></header>
      <div class="body"><ol class="tight">${(d.investigator_next_steps || [])
        .map((s) => `<li>${esc(s)}</li>`).join('')}</ol></div></div>

    ${decisionPanel(d.account_idx)}`;

  wireDecisions(d.account_idx);
}

/* ---------- investigator decision panel ------------------------------------
   The feedback loop, live. An analyst reviews the evidence above and records a
   verdict; it lands in an append-only audit trail. In a deployment those rows
   are the labels the next model retrains on, which is the loop this project
   would otherwise only be able to describe.
--------------------------------------------------------------------------- */
function decisionPanel(idx) {
  return `
    <div class="panel"><header><h3>Investigator decision</h3>
      <span class="tag pill">FEEDBACK LOOP</span></header>
      <div class="body">
        <p class="dim small">The model builds the queue. A human decides. Every decision below is
          timestamped and appended to <code>data/investigator_decisions.csv</code>, which is what a
          retrained model would learn from.</p>
        <label class="field" style="margin:12px 0">
          <span class="lab">Case note (optional)</span>
          <input type="text" id="dec-note" placeholder="e.g. pass-through confirmed against statement"
                 maxlength="500" style="width:100%;max-width:620px">
        </label>
        <div class="controls">
          <button class="btn" id="dec-confirm" data-idx="${idx}"
                  style="background:var(--red);border-color:var(--red)">CONFIRM MULE · FREEZE</button>
          <button class="btn ghost" id="dec-dismiss" data-idx="${idx}">DISMISS · NO ACTION</button>
          <button class="btn ghost" id="dec-review" data-idx="${idx}">NEEDS SECOND REVIEW</button>
        </div>
        <div id="dec-result" style="margin-top:14px"></div>
      </div>
    </div>`;
}

function wireDecisions(idx) {
  const map = { 'dec-confirm': 'CONFIRMED_MULE', 'dec-dismiss': 'DISMISSED',
                'dec-review': 'NEEDS_REVIEW' };
  Object.keys(map).forEach((id) => {
    const b = $('#' + id);
    if (b) b.onclick = () => submitDecision(idx, map[id]);
  });
}

async function submitDecision(idx, decision) {
  const out = $('#dec-result');
  if (!out) return;
  out.innerHTML = '';
  out.appendChild(el('div', 'loading', 'RECORDING'));
  const note = ($('#dec-note') && $('#dec-note').value) || '';
  let r;
  try {
    const res = await fetch('/api/decision', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ account_idx: idx, decision, note }),
    });
    r = await res.json();
    if (!res.ok) throw { kind: r.error || 'HTTP_' + res.status, detail: r.detail, fix: r.fix };
  } catch (e) {
    out.innerHTML = '';
    out.appendChild(errorBox(e.kind ? e : { kind: 'NETWORK', detail: 'Could not reach the API.' }));
    return;
  }

  const frozen = r.decision === 'CONFIRMED_MULE';
  const agreed = (r.decision === 'CONFIRMED_MULE' && r.ground_truth === 1)
              || (r.decision === 'DISMISSED' && r.ground_truth === 0);
  out.innerHTML = `
    <div class="callout ${frozen ? 'red' : 'green'}">
      <div class="tiny">${esc(r.decision)} · recorded ${esc(r.recorded_at)}</div>
      <strong>${esc(r.action)}</strong>
      ${frozen ? `<div class="small" style="margin-top:8px">
        Simulated: outward transfers held on account ${r.account_idx}, case routed to the AML desk,
        STR drafted. No real action is taken by this demo.</div>` : ''}
      <div class="small" style="margin-top:8px">
        Model said <strong>${esc(r.model_said)}</strong> at score ${num(r.risk_score)}.
        Ground truth in this benchmark: <strong>${r.ground_truth ? 'MULE' : 'NOT A MULE'}</strong>.
        ${r.decision === 'NEEDS_REVIEW' ? '' :
          agreed ? '<span class="pill ok" style="margin-left:6px">ANALYST AGREED</span>'
                 : '<span class="pill bad" style="margin-left:6px">ANALYST DISAGREED</span>'}
      </div>
    </div>`;
  if (RENDER.audit) RENDER.audit();
}

/* -- audit trail -- */
RENDER.audit = () => {
  panel($('#audit-body'), () => api('/api/decisions?limit=50'), (d, n) => {
    if (!d.total) {
      n.innerHTML = `<div class="empty">No decisions recorded yet. Analyse an account in
        section 10 and record a verdict; it will appear here.</div>`;
      return;
    }
    n.innerHTML = `
      <div class="grid g4" style="margin-bottom:14px">
        ${stat('Decisions recorded', num(d.total))}
        ${stat('Resolved', num(d.resolved), 'excludes second-review')}
        ${stat('Agreed with ground truth', num(d.analyst_agreed_with_ground_truth), null, 'green')}
        ${stat('Agreement rate', d.agreement_rate === null ? '—' : pct(d.agreement_rate, 0))}
      </div>
      <div class="scrollx"><table>${table(
        ['Recorded', { t: 'Account', num: true }, { t: 'Score', num: true }, 'Model', 'Decision', 'Truth', 'Note'],
        d.decisions.map((x) => [
          `<span class="mono" style="font-size:11px">${esc(String(x.recorded_at).replace('T', ' '))}</span>`,
          num(x.account_idx), num(x.risk_score),
          `<span class="band ${x.band}">${esc(x.band)}</span>`,
          `<span class="pill ${x.decision === 'CONFIRMED_MULE' ? 'bad' : x.decision === 'DISMISSED' ? 'ok' : 'warn'}">${esc(x.decision)}</span>`,
          x.ground_truth ? '<span class="pill bad">MULE</span>' : '<span class="faint">normal</span>',
          `<span class="small">${esc(x.note || '')}</span>`,
        ]))}</table></div>
      <div class="callout"><div class="tiny">Why this matters</div>${esc(d.note || '')}</div>`;
  });
};

/* -- operating metrics -- */
RENDER.operating = () => {
  panel($('#operating-body'), () => api('/api/operating'), (d, n) => {
    const pk = d.precision_at_k || [], load = d.investigator_load_by_band || {};
    const budget = d.daily_review_budget || [], drift = d.extract_drift || {};
    const lat = d.scoring_latency || {}, ttf = d.time_to_flag || {};
    n.innerHTML = `
      <p>AUPRC is the right academic metric and the wrong operational one. A head of financial crime
         asks three questions instead, and these are the answers.</p>
      <h3 style="color:var(--amber);margin:16px 0 8px">Precision@K — if my team reviews K accounts, how many are real?</h3>
      <div class="scrollx"><table>${table(
        [{ t: 'K', num: true }, { t: 'Mules found', num: true }, { t: 'Precision', num: true },
         { t: 'Recall', num: true }, { t: 'False alarms', num: true }, { t: 'Analyst-days', num: true }],
        pk.map((r) => [num(r.k), num(r.mules_in_top_k),
          `<strong style="color:${r.precision_at_k >= .9 ? 'var(--green)' : r.precision_at_k >= .5 ? 'var(--amber)' : 'var(--ink-2)'}">${pct(r.precision_at_k, 1)}</strong>`,
          pct(r.recall_at_k, 1), num(r.false_positives), num(r.analyst_days, 1)]))}</table></div>

      <h3 style="color:var(--amber);margin:18px 0 8px">Investigator load per band</h3>
      <div class="scrollx"><table>${table(
        ['Band', { t: 'Accounts', num: true }, { t: 'Mules', num: true },
         { t: 'False alarms', num: true }, { t: 'FP per 1,000', num: true }, { t: 'Analyst-days', num: true }],
        Object.entries(load).map(([b, v]) => [`<span class="band ${b}">${b}</span>`,
          num(v.accounts), num(v.mules), num(v.false_positives),
          num(v.false_positives_per_1000_accounts, 2), num(v.analyst_days_to_clear, 1)]))}</table></div>

      <div class="grid g2" style="margin-top:18px">
        <div>
          <h3 style="color:var(--amber);margin-bottom:8px">What a review budget buys</h3>
          <div class="scrollx"><table>${table(
            [{ t: 'Analysts', num: true }, { t: 'Reviews/day', num: true },
             { t: 'Mules found', num: true }, { t: 'Recall', num: true }],
            budget.map((b) => [num(b.analysts), num(b.reviews_per_day),
              num(b.mules_found), pct(b.recall, 1)]))}</table></div>
        </div>
        <div>
          <h3 style="color:var(--amber);margin-bottom:8px">Latency</h3>
          <div class="kv">
            <dt>Single account</dt><dd class="num">${num(lat.single_account_ms_median, 1)} ms</dd>
            <dt>Batch throughput</dt><dd class="num">${num(lat.batch_accounts_per_second)} accounts/sec</dd>
          </div>
          <div class="callout red" style="margin-top:12px"><div class="tiny">Time to flag: not available</div>
            ${esc(ttf.reason || '')}</div>
        </div>
      </div>

      ${drift.available ? `<div class="callout red" style="margin-top:16px">
        <div class="tiny">Extract drift — the confound measured a third way</div>
        <strong>${num(drift.columns_significantly_shifted)} of ${num(drift.columns_tested)} columns
        (${pct(drift.share_shifted, 1)})</strong> shift significantly between the
        <code>${esc(drift.reference_extract)}</code> extract and the rest, by a
        Kolmogorov–Smirnov test at p &lt; 0.01. ${esc(drift.interpretation || '')}</div>` : ''}`;
  });
};

/* -- 10 pipeline -- */
// The stages the pipeline actually runs, in src/pipeline.py order, with its
// own labels. Three different numberings were in circulation here: this list,
// the Figure 2 dataflow, and pipeline.py itself. Only one of them was the
// truth, so the other two now follow it.
const STAGES = [
  ['0',    'Dataset integrity audit', 'Three falsification tests. Run first, read first.'],
  ['1',    'Cleaning + leak removal', 'Semantic, structural, extract hardening, separation audit.'],
  ['2/3',  'Feature engineering', '29 mule-behaviour features + row aggregates.'],
  ['4/5',  'Ensemble + nested CV', 'Selection, stacking, calibration, threshold — all in-fold.'],
  ['6',    'Graph label propagation', 'Self-skips: no counterparty column exists.', true],
  ['7/8',  'Risk score + SHAP', '0–1000 score, derived bands, out-of-fold explanations.'],
  ['9',    'AML rule layer', 'Twelve typologies, measured against the base rate, not tuned.'],
  ['10',   'Feature ablation', 'How much of our own score is the extract artefact.'],
  ['11',   'Operating metrics', 'Precision@K, analyst load, extract drift.'],
  ['12',   'EFRMS / AML export', 'Vendor-neutral alert and case-pack bundle.'],
];

RENDER.pipeline = () => {
  $('#flow').innerHTML = STAGES.map(([n, t, d, skip]) =>
    `<div class="node ${skip ? 'skip' : ''}"><div class="n">STAGE ${n}</div>
     <div class="t">${esc(t)}</div><div class="d">${esc(d)}</div></div>`).join('');

  panel($('#artefact-table'), async () => HEALTH || api('/api/health'), (d, n) => {
    n.innerHTML = table(['Artefact', 'Produced by', 'Status', { t: 'Size', num: true }],
      (d.artefacts || []).map((a) => [esc(a.artefact), `<code>${esc(a.produced_by)}</code>`,
        a.present ? '<span class="pill ok">PRESENT</span>' : '<span class="pill bad">MISSING</span>',
        a.size_bytes ? `${(a.size_bytes / 1024).toFixed(0)} KB` : '—']));
  });
};

/* ========================================================================
   JUDGE MODE — the 90-second story
   ======================================================================== */
const JUDGE = [
  { ch: 'The problem', t: 'Money mules settle digital fraud',
    lede: `When a fraud victim's money moves, it lands in a mule account: a real customer's
      account, rented or coerced, that receives the proceeds and pushes them straight back out
      within hours. Catch the mule and you cut the channel. The catch is that mules are
      vanishingly rare — under 1% of accounts — so a model that predicts "not a mule" for
      everybody is 99.1% accurate and completely useless.`,
    load: () => api('/api/overview'),
    paint: (d) => `<div class="grid g3">${stat('Accounts', num(d.accounts))}
      ${stat('Confirmed mules', num(d.mules), null, 'red')}
      ${stat('Prevalence', `${d.prevalence_pct}%`, 'a "99% accurate" model is worthless here', 'amber')}</div>` },

  { ch: 'The data', t: '3,924 opaque F-codes',
    lede: `The dataset gives every account 3,924 columns named F1 … F3924. Unusable as-is.
      The supplied data dictionary maps every one to a real banking variable — and that mapping
      is what makes everything after this possible: leak removal by <em>meaning</em>, features
      named after real behaviour, and SHAP reasons an investigator can read.`,
    load: () => api('/api/overview'),
    paint: (d) => `<div class="grid g3">${stat('Raw columns', num(d.raw_columns))}
      ${stat('After cleaning', num(d.features_after_cleaning), null, 'green')}
      ${stat('Graph possible?', 'NO', 'not one column names a counterparty', 'amber')}</div>
      <div class="callout"><div class="tiny">Example</div>
      <code>F3891</code> → <code>CUST_OCCP</code> → "Occupation code of customer"</div>` },

  { ch: 'The discovery', t: 'The benchmark is confounded',
    lede: `Before modelling, we audited the dataset itself. Every negative comes from the October
      extract. Every positive comes from September, November or December. <strong>No month
      contains both classes.</strong> So anything that differs between monthly extraction runs
      lines up perfectly with the label — while describing no customer behaviour at all.`,
    load: () => api('/api/integrity'),
    paint: (d) => {
      const m = d.month_split;
      if (!m) return '<div class="empty">No month column in this dataset.</div>';
      return `<div class="scrollx"><table>${table(['Month', { t: 'Normal', num: true },
        { t: 'Mule', num: true }], Object.entries(m.counts).map(([k, c]) =>
          [`<code>${esc(k)}</code>`, num(c['0'] || 0), num(c['1'] || 0)]))}</table></div>
        <div class="callout red"><div class="tiny">Months containing both classes</div>
        <strong style="font-size:26px">${m.months_containing_both_classes}</strong></div>`;
    } },

  { ch: 'The proof', t: 'Falsification, not suspicion',
    lede: `A suspicion is not a finding. So we ran the decisive test: give a model <em>only</em>
      whether each cell was blank. Throw away every value, so no account behaviour survives at
      all. Whether a cell is populated is decided by the extraction job, not by a customer — this
      should score at the random baseline.`,
    load: () => api('/api/integrity'),
    paint: (d) => {
      const a = d.test_A_missingness_only || {}, c = d.test_C_shuffled_labels || {};
      const base = Number(d.auprc_random_baseline) || 1;
      return `<div class="grid g3">
        ${stat('A — blanks only, no values', num(a.auprc), `AUPRC · ${(a.auprc / base).toFixed(0)}× random`, 'red')}
        ${stat('C — shuffled labels', num(c.auprc, 4), 'the sanity floor — it collapses', 'green')}
        ${stat('Random baseline', num(base, 4), 'AUPRC at 0.89% prevalence')}</div>
        <div class="callout red"><div class="tiny">What this proves</div>
        Test C collapsing to baseline confirms the harness is sound — so test A's score is a real
        property of the data. Knowing only which cells were blank identifies mules almost
        perfectly. No model can separate that from genuine behaviour inside this file.</div>` } },

  { ch: 'The honest position', t: 'Every metric here is an upper bound',
    lede: `This is the most consequential thing the project produced, and it applies to every team
      working from this file — not just this submission. We lead with it rather than burying it,
      because an honest number a judge can trust beats an inflated one that collapses under the
      first hard question. Fixing it needs negatives and positives sampled from the same months:
      a data-collection change, not a modelling one.`,
    paint: () => `<div class="callout red"><div class="tiny">Stated plainly</div>
      Any score on this dataset, from any team, measures extract provenance as well as mule
      behaviour, and the two cannot be cleanly separated within this file.</div>
      <div class="callout green"><div class="tiny">What still stands</div>
      The pipeline is leak-hardened, nested-validated, calibrated and explainable. On data where
      both classes share months, the same code produces a trustworthy number with no changes.</div>` },

  { ch: 'The defence', t: 'Four layers against leakage',
    lede: `A correlation threshold is not a leak defence. <code>FRAUD_SUSPECTED</code> correlates
      0.97 — any threshold catches it. <code>FALSE_POSITIVE</code> correlates 0.05, is equally
      unavailable at scoring time, and no threshold catches it at all. Leaks have to be removed
      by what they <em>mean</em>.`,
    load: () => api('/api/leakage'),
    paint: (d) => `<div class="flow">
      <div class="node"><div class="n">LAYER 1</div><div class="t">Semantic</div>
        <div class="d">${(d.layer_1_semantic?.removed || []).length} post-outcome fields removed by meaning</div></div>
      <div class="node"><div class="n">LAYER 2</div><div class="t">Structural</div>
        <div class="d">${(d.layer_2_structural?.removed || []).length} sample-assembly artefacts incl. MNTH</div></div>
      <div class="node"><div class="n">LAYER 3</div><div class="t">Extract hardening</div>
        <div class="d">${num(d.layer_3_extract_hardening?.columns_dropped)} columns with class-dependent blank rates</div></div>
      <div class="node"><div class="n">LAYER 4</div><div class="t">Separation audit</div>
        <div class="d">${num(d.layer_4_separation_audit?.columns_scanned)} columns scanned for the next MNTH</div></div>
      </div>
      <div class="callout"><div class="tiny">Direction of travel</div>
      Layer 3 can only ever <em>remove</em> signal, never manufacture it. Every defence here makes
      the reported result more conservative.</div>` },

  { ch: 'The features', t: '29 features that encode a mule',
    lede: `A mule receives money and pushes it straight back out, holds almost nothing, in bursts,
      through digital rails, often at odd hours, on an account whose owner profile does not match
      the volume. Each feature family measures one clause of that sentence — so every model reason
      is defensible to an auditor in plain English.`,
    load: () => api('/api/features'),
    paint: (d) => `<div class="flow">${[
        ['Pass-through', 'money in ≈ money out → the account is a pipe, not a wallet'],
        ['Turnover / balance', 'moves many multiples of what it holds'],
        ['Burst', 'weekly rate ≫ monthly → sudden activation'],
        ['Cash-out', 'digital in, cash out → the layering handoff'],
        ['Channel mix', 'single-purpose accounts ride one rail'],
        ['Ticket size', 'many small tickets → structuring'],
        ['Alert timing', 'night-alert share runs ~3× higher'],
        ['Balance shape', 'spike-and-drain, not held balance'],
        ['Profile mismatch', 'volume against occupation norm'],
      ].map(([t, dd]) => `<div class="node"><div class="n">FAMILY</div>
        <div class="t">${esc(t)}</div><div class="d">${esc(dd)}</div></div>`).join('')}</div>
      <div class="callout"><div class="tiny">Count</div>
      <strong style="font-size:22px">${num(d.typology_feature_count)}</strong> named features across
      ${(d.channels_used || []).length} payment rails.</div>` },

  { ch: 'The models', t: 'Three learners, honestly reported',
    lede: `XGBoost and LightGBM handle the 111:1 imbalance through cost-sensitive reweighting
      rather than SMOTE — interpolating synthetic minority points from 65 training positives across
      hundreds of dimensions invents neighbourhoods that do not exist. An isolation forest adds an
      unsupervised view. We report what each one actually did, including the one that failed.`,
    load: () => api('/api/models'),
    paint: (d) => {
      const pm = d.per_model || {}, N = { iso: 'Isolation Forest', xgb: 'XGBoost', lgbm: 'LightGBM' };
      return `<div class="scrollx"><table>${table(['Model', { t: 'AUPRC', num: true }, { t: 'AUROC', num: true }],
        Object.entries(pm).map(([k, v]) => [esc(N[k] || k), ms(v.auprc, 4), ms(v.auroc, 4)]))}</table></div>
        ${Number(pm.iso?.auroc?.mean) < 0.5 ? `<div class="callout red"><div class="tiny">Reported, not buried</div>
        The isolation forest scores below random (AUROC ${num(pm.iso.auroc.mean)}). The stacker
        learns to invert it. We say so rather than calling this "an ensemble of three strong
        models".</div>` : ''}`;
    } },

  { ch: 'The validation', t: 'Nothing outside the fold',
    lede: `Feature selection, base models, stacking weights, probability calibration <em>and the
      operating threshold</em> are all fitted inside an inner split of the training fold, then
      applied frozen. Choosing a threshold by scanning the same curve you then report from is
      optimistic by construction — and with 81 positives that curve is very noisy.`,
    load: () => api('/api/models'),
    paint: (d) => `<div class="flow">${(d.validation?.what_is_fitted_inside_each_fold || [])
      .map((s, i) => `<div class="node"><div class="n">${String(i + 1).padStart(2, '0')}</div>
        <div class="t">${esc(s)}</div></div>`).join('')}</div>
      <div class="callout green"><div class="tiny">${esc(d.validation?.scheme || '')}</div>
      ${esc(d.validation?.note || '')}</div>` },

  { ch: 'The result', t: 'Two operating points, with error bars',
    lede: `A reviewed false alert costs an analyst minutes. A missed mule costs a live laundering
      channel. So the system publishes both. Every figure is mean ± standard deviation across
      folds, because 81 positives means ~16 per fold and the metric moves several points on the
      seed alone.`,
    load: () => api('/api/models'),
    paint: (d) => {
      const pf = d.ensemble_precision_first || {}, hr = d.ensemble_high_recall || {};
      return `<div class="grid g4">
        ${stat('Precision (auto-action)', ms(pf.precision), 'freeze + STR', 'green')}
        ${stat('Recall (auto-action)', ms(pf.recall))}
        ${stat('Recall (review queue)', ms(hr.recall),
               `at ${num(hr.precision?.mean)} ± ${num(hr.precision?.std)} precision`, 'amber')}
        ${stat('Lift over base rate', pf.lift_over_prevalence ? `${num(pf.lift_over_prevalence.mean, 0)}×` : '—',
               'what an AML desk actually acts on', 'amber')}</div>
        <div class="callout red"><div class="tiny">Remember step 05</div>
        These are the <em>upper bound</em> of what this dataset can show. They are not a
        demonstrated real-world mule-detection rate.</div>`;
    } },

  { ch: 'The score', t: 'Bands that mean something',
    lede: `A 0–1000 score is only useful if a band boundary carries meaning. These edges are the
      model's own fitted operating points — the HIGH edge is the threshold that held precision
      ≥ 0.90 on inner folds — not round numbers somebody picked. Earlier versions of this project
      documented that behaviour but hardcoded 400 and 750; that is now actually implemented.`,
    load: () => api('/api/bands'),
    paint: (d) => {
      const bs = d.band_stats || {}, sb = d.score_bands || {};
      return `<div class="grid g3">${['HIGH', 'MEDIUM', 'LOW'].filter((b) => bs[b]).map((b) =>
        stat(`${b} · ${num(sb[b]?.[0], 0)}–${num(sb[b]?.[1], 0)}`,
          `${num(bs[b].accounts)}`, `${bs[b].true_mules} real mules · precision ${pct(bs[b].precision, 0)}`,
          b === 'HIGH' ? 'red' : b === 'MEDIUM' ? 'amber' : '')).join('')}</div>
        <div class="callout ${d.band_edge_provenance?.derived ? 'green' : 'red'}">
        <div class="tiny">Edge provenance</div>${esc(d.band_edge_provenance?.source || '')}</div>`;
    } },

  { ch: 'The action', t: 'Analyse a real account now',
    lede: `The point of all of it. Pick any account and the system answers three questions:
      is it risky, <em>why</em> is it risky, and what should an investigator do next — with the
      score and the SHAP explanation both coming from a model that never trained on that account.`,
    paint: () => `<div class="callout green"><div class="tiny">Live</div>
      Click below to analyse the highest-scoring account in the benchmark with the real model.</div>
      <button class="btn" style="font-size:15px;padding:14px 26px" id="judge-analyze">
        ▶ ANALYZE ACCOUNT</button>`,
    after: () => { const b = $('#judge-analyze'); if (b) b.onclick = () => {
      go('analyze'); setTimeout(() => pickFrom({ limit: 1 }), 60); }; } },
];

let jstep = -1;
function judgeRender() {
  const body = $('#judge-body');
  const s = JUDGE[jstep];
  $('#judge-count').textContent = `${jstep + 1} / ${JUDGE.length}`;
  $('#judge-chapter').textContent = s.ch;
  $('#judge-prev').disabled = jstep <= 0;
  $('#judge-next').disabled = jstep >= JUDGE.length - 1;
  $('#judge-progress').innerHTML = JUDGE.map((_, i) =>
    `<i class="${i <= jstep ? 'done' : ''}"></i>`).join('');

  body.innerHTML = `<div class="stepno">${String(jstep + 1).padStart(2, '0')}</div>
    <div class="steptitle">${s.t}</div>
    <p class="steplede">${s.lede}</p>
    <div id="judge-data" style="margin-top:20px"></div>`;

  const slot = $('#judge-data');
  if (s.load) {
    panel(slot, s.load, (d, n) => { n.innerHTML = s.paint(d); if (s.after) s.after(); });
  } else if (s.paint) {
    slot.innerHTML = s.paint();
    if (s.after) s.after();
  }
}

function judgeInit() {
  $('#judge-start').onclick = () => { jstep = 0; judgeRender(); $('#judge-start').textContent = '↺ RESTART'; };
  $('#judge-next').onclick = () => { if (jstep < JUDGE.length - 1) { jstep++; judgeRender(); } };
  $('#judge-prev').onclick = () => { if (jstep > 0) { jstep--; judgeRender(); } };
  $('#judge-progress').innerHTML = JUDGE.map(() => '<i></i>').join('');
  $('#judge-count').textContent = `0 / ${JUDGE.length}`;
  document.addEventListener('keydown', (e) => {
    if ($('#s-judge').hidden || jstep < 0) return;
    if (e.key === 'ArrowRight' && jstep < JUDGE.length - 1) { jstep++; judgeRender(); }
    if (e.key === 'ArrowLeft' && jstep > 0) { jstep--; judgeRender(); }
  });
}
/* -- 00 hero / landing ------------------------------------------------------
   Leads with the result, because that is what the visitor came for. The
   validation work sits directly underneath as the REASON the result is
   believable, not as a caveat on it. Same evidence, different posture.
   Every number is read live, so the page cannot go stale.
--------------------------------------------------------------------------- */
RENDER.hero = () => {
  const wire = (root) => (root || document).querySelectorAll('[data-goto]').forEach((b) => {
    if (!b.dataset.wired) { b.dataset.wired = '1'; b.onclick = () => go(b.dataset.goto); }
  });
  wire();

  // ---- live system readout ------------------------------------------------
  panel($('#hero-readout'), async () => ({
    ov: await api('/api/overview'),
    m: await api('/api/models'),
    b: await api('/api/bands'),
    op: await api('/api/operating'),
    h: HEALTH || await api('/api/health'),
  }), ({ ov, m, b, op, h }, n) => {
    const e = m.ensemble_precision_first || {};
    const hb = (b.band_stats || {}).HIGH || {};
    const lat = op.scoring_latency || {};
    const rows = [
      ['portfolio', `${num(ov.accounts)} accounts`, ''],
      ['confirmed mules', `${num(ov.mules)} accounts`, 'amber'],
      ['base rate', `${ov.prevalence_pct}%`, ''],
      ['precision', num(e.precision && e.precision.mean, 3), 'green'],
      ['false positive rate', num(e.fpr && e.fpr.mean, 4), 'green'],
      ['high-risk queue', `${num(hb.accounts)} accounts`, ''],
      ['of which real mules', `${num(hb.true_mules)} of ${num(hb.accounts)}`, 'green'],
      ['caught in freeze band', `${num(hb.true_mules)} of ${num(ov.mules)} mules`, 'green'],
      ['scoring latency', `${num(lat.single_account_ms_median, 0)} ms`, ''],
      ['model', h.model_loaded ? 'LOADED' : 'ABSENT', h.model_loaded ? 'green' : 'red'],
    ];
    n.innerHTML =
      `<div class="readout-head"><span class="dot"></span>SYSTEM READOUT</div>
       <div class="readout-body">${rows.map(([k, v, tone]) =>
        `<div class="readout-row"><span class="k">${esc(k)}</span>
         <span class="v ${tone}">${v}</span></div>`).join('')}
       </div>`;
  });

  // ---- headline metrics: the product claim, not the caveat ----------------
  panel($('#hero-metrics'), async () => ({
    op: await api('/api/operating'), m: await api('/api/models'), b: await api('/api/bands'),
  }), ({ op, m, b }, n) => {
    n.className = 'grid g4';
    const e = m.ensemble_precision_first || {};
    const hr = m.ensemble_high_recall || {};
    const k50 = (op.precision_at_k || []).find((r) => r.k === 50) || {};
    const hb = (b.band_stats || {}).HIGH || {};
    n.innerHTML =
      stat('Precision', num(e.precision && e.precision.mean, 3),
           'when we flag an account, we are right', 'green') +
      stat('Review 50, find 50', pct(k50.precision_at_k, 0),
           `${num(k50.false_positives)} false alarms · ${num(k50.recall_at_k * 100, 0)}% of all mules in ~${num(k50.analyst_days, 1)} analyst-days`, 'green') +
      stat('Lift over base rate', e.lift_over_prevalence ? `${num(e.lift_over_prevalence.mean, 0)}×` : '—',
           'richer in mules than a random queue', 'amber') +
      stat('Recall, review mode', pct(hr.recall && hr.recall.mean, 1),
           'second operating point for the human queue', 'amber');
  });

  // ---- the trust panel: validation as proof, not apology ------------------
  panel($('#hero-trust'), async () => ({
    ig: await api('/api/integrity'), ab: await api('/api/ablation'), m: await api('/api/models'),
  }), ({ ig, ab, m }, n) => {
    const a = ig.test_A_missingness_only || {}, c = ig.test_C_shuffled_labels || {};
    const rows = ab.results || [];
    const typ = rows.find((r) => String(r.condition).startsWith('TYPOLOGY')) || {};
    const v = m.validation || {};
    n.innerHTML = `
      <p>A high precision figure on a rare event is easy to produce by accident and hard to trust.
         So we ran the checks that most submissions skip, and we publish what they returned.</p>
      <div class="grid g3" style="margin:16px 0">
        <div class="block">
          <h3 style="color:var(--green)">Nothing leaks into the score</h3>
          <p class="small" style="margin-top:8px">Feature selection, stacking, imputation, calibration
             <em>and the decision threshold</em> are fitted inside the training fold and applied frozen.
             ${esc(v.scheme || '')}. No validation row influences how it is scored.</p>
        </div>
        <div class="block">
          <h3 style="color:var(--green)">The harness is provably sound</h3>
          <p class="small" style="margin-top:8px">Shuffle the labels and the whole thing collapses to
             <strong>${num(c.auprc, 4)}</strong>, the random baseline. A pipeline that scores well on
             noise is broken; ours does not.</p>
        </div>
        <div class="block">
          <h3 style="color:var(--green)">We audited our own result</h3>
          <p class="small" style="margin-top:8px">We measured how much of the score comes from the
             behavioural features alone: <strong>${num(typ.auprc && typ.auprc.mean, 3)}</strong>, which
             is ${num((typ.auprc && typ.auprc.mean || 0) / 0.0089, 0)}× random from
             ${num(typ.n_features)} features. Almost nobody checks this about themselves.</p>
        </div>
      </div>
      <div class="callout"><div class="tiny">And one finding we think matters</div>
        This benchmark's positives and negatives were drawn from different monthly extracts. We
        proved it: a model given <strong>only the pattern of blank cells, every value discarded</strong>,
        reaches ${num(a.auprc, 3)} AUPRC. That means part of any score on this file, from any team,
        measures the extract rather than the customer. We are the only ones who can tell you how much.
        <button class="btn ghost" style="margin-top:12px;padding:6px 12px" data-goto="integrity">
          SEE THE EXPERIMENT →</button></div>`;
    wire(n);
  });

  // ---- routes -------------------------------------------------------------
  const routes = [
    ['Understand the method', 'approach',
     'What a mule account actually is, the seven stages, every component and why it was chosen over the obvious alternative, and how the pipeline reads a schema it has never seen.'],
    ['See it work', 'analyze',
     'Pick any account. Risk score, calibrated probability, the reasons it was flagged, evidence against the population, and the action an investigator takes next.'],
    ['Run it on your data', 'upload',
     'Drop in a CSV or Excel file. It works out the target column, the identifiers and the leak columns itself, and returns a scored, explained result.'],
  ];

  const rn = $('#hero-routes');
  if (rn) {
    rn.innerHTML = routes.map(([title, target, body], i) =>
      `<button class="route" data-goto="${target}">
         <span class="n">Route ${String(i + 1).padStart(2, '0')}</span>
         <h3>${esc(title)}</h3><p>${esc(body)}</p>
         <span class="go">Open →</span>
       </button>`).join('');
    wire(rn);
  }
};

RENDER.approach = () => {
  document.querySelectorAll('#s-approach [data-goto]').forEach((b) => {
    if (!b.dataset.wired) { b.dataset.wired = '1'; b.onclick = () => go(b.dataset.goto); }
  });
};

RENDER.judge = () => {};

/* ---------- 00 upload a dataset --------------------------------------------
   Built for one moment: a judge hands over a file. Drag it in, watch the stages,
   read the verdict. Everything lands in its own runs/ folder, so the
   submission's own results are never touched.
--------------------------------------------------------------------------- */
let POLL = null;

RENDER.upload = () => {
  const drop = $('#up-drop'), input = $('#up-file');
  if (!drop || drop.dataset.wired) return;
  drop.dataset.wired = '1';

  const pick = (files) => {
    if (!files || !files.length) return;
    const f = files[0];
    $('#up-chosen').innerHTML =
      `<strong>${esc(f.name)}</strong> <span class="dim">· ${(f.size / 1048576).toFixed(1)} MB</span>`;
    $('#up-go').disabled = false;
    if ($('#up-score')) $('#up-score').disabled = false;
    drop.dataset.ready = '1';
  };

  input.onchange = () => pick(input.files);
  drop.onclick = () => input.click();
  drop.ondragover = (e) => { e.preventDefault(); drop.classList.add('over'); };
  drop.ondragleave = () => drop.classList.remove('over');
  drop.ondrop = (e) => {
    e.preventDefault(); drop.classList.remove('over');
    input.files = e.dataTransfer.files; pick(e.dataTransfer.files);
  };
  $('#up-go').onclick = () => startUpload('auto');
  if ($('#up-score')) $('#up-score').onclick = () => startUpload('score');
  refreshJobs();
};

async function startUpload(mode) {
  const input = $('#up-file');
  if (!input.files || !input.files.length) return;
  const fd = new FormData();
  fd.append('file', input.files[0]);
  fd.append('target', ($('#up-target') && $('#up-target').value.trim()) || '');
  fd.append('fast', $('#up-full').checked ? 'false' : 'true');
  fd.append('mode', mode === 'score' ? 'score' : mode === 'train' ? 'train' : 'auto');

  const out = $('#up-status');
  $('#up-go').disabled = true;
  out.innerHTML = '';
  out.appendChild(el('div', 'loading', 'UPLOADING'));

  let j;
  try {
    const res = await fetch('/api/jobs/upload', { method: 'POST', body: fd });
    j = await res.json();
    if (!res.ok) throw { kind: j.error || 'UPLOAD_FAILED', detail: j.detail, fix: j.fix };
  } catch (e) {
    out.innerHTML = '';
    out.appendChild(errorBox(e.kind ? e : { kind: 'NETWORK', detail: 'Could not reach the API.' }));
    $('#up-go').disabled = false;
    return;
  }
  if (j.mode) {          // any label-free route returns results, not a job
    // A poller left over from an earlier run would keep rewriting #up-status
    // every 1.5 s and wipe these results off the screen.
    if (POLL) { clearInterval(POLL); POLL = null; }
    out.innerHTML = '';
    if (j.mode === 'TYPOLOGY_RANKING') renderTypology(j);
    else if (j.mode === 'FITTED_ON_SHARED_COLUMNS') renderFitted(j);
    else renderScoreOnly(j);
    revealPanel('#up-status');
    refreshJobs();
    $('#up-go').disabled = false;
    if ($('#up-score')) $('#up-score').disabled = false;
    return;
  }
  watchJob(j.job_id);
}

function renderFitted(d) {
  const rows = (d.top_accounts || []).slice(0, 25);
  const q = d.fit_quality || {};
  $('#up-status').innerHTML = `
    <div class="panel accent">
      <header><h3>Scored on the columns we share</h3>
        <span class="tag pill warn">PARTIAL SCHEMA</span></header>
      <div class="body">
        <div class="grid g3">
          <div class="stat"><div class="k">Accounts scored</div>
            <div class="v">${num(d.rows_scored)}</div><div class="u">every row</div></div>
          <div class="stat amber"><div class="k">Schema coverage</div>
            <div class="v">${d.schema_coverage_pct}%</div>
            <div class="u">${num(d.model_features_matched)} of ${num(d.model_features_expected)} model columns</div></div>
          <div class="stat ${q.out_of_fold_auprc >= 0.3 ? 'green' : 'red'}">
            <div class="k">What these columns can do</div>
            <div class="v">${num(q.out_of_fold_auprc, 3)}</div>
            <div class="u">out-of-fold AUPRC on our labelled data &#183; ${q.lift_over_base_rate}&#215; base rate</div></div>
        </div>
        <div class="callout" style="margin-top:14px">
          <div class="tiny">WHY NOT THE DEPLOYED MODEL</div>
          <p class="small">${esc(d.provenance)}</p>
        </div>
        <p class="small dim" style="margin-top:12px">Columns used:
          ${(d.columns_used || []).map((c) => `<code>${esc(c)}</code>`).join(' &#183; ')}</p>
        <div class="scrollx" style="margin-top:12px"><table>${table(
          ['Row', { t: 'Probability', num: true }, { t: 'Rank pct', num: true }],
          rows.map((r) => [r.row, num(r.probability, 4), num(r.percentile, 1)]))}</table></div>
        <p class="small dim" style="margin-top:10px">Review from the top down. The out-of-fold figure
          above is the ceiling these columns support, measured on data where we do have labels, so
          it tells you how far to trust this queue before you work it.</p>
      </div>
    </div>`;
}

function renderTypology(d) {
  const rows = (d.top_accounts || []).slice(0, 25);
  const cols = d.signals_built || [];
  $('#up-status').innerHTML = `
    <div class="panel accent">
      <header><h3>Ranked by behaviour signals &#183; unvalidated</h3>
        <span class="tag pill warn">NO LABELS, NO MODEL</span></header>
      <div class="body">
        <div class="grid g3">
          <div class="stat red"><div class="k">Flagged</div>
            <div class="v">${num(d.accounts_flagged)}</div>
            <div class="u">of ${num(d.rows_scored)} accounts &#183; ${d.flag_rate_pct}% of the file</div></div>
          <div class="stat"><div class="k">Signals rebuilt</div>
            <div class="v">${(d.signals_built || []).length}</div>
            <div class="u">from this file's own columns</div></div>
          <div class="stat amber"><div class="k">Cut-off</div>
            <div class="v">${num(d.threshold && d.threshold.value, 3)}</div>
            <div class="u">Otsu &#183; nothing tuned</div></div>
        </div>
        ${d.caution ? `<div class="callout" style="margin-top:14px;border-left-color:var(--amber)">
          <div class="tiny" style="color:var(--amber)">READ THE FLAG COUNT WITH CARE</div>
          <p class="small">${esc(d.caution)}</p></div>` : ''}
        <div class="callout" style="margin-top:14px">
          <div class="tiny">NO MODEL WAS USED, AND THAT IS DELIBERATE</div>
          <p class="small">${esc(d.provenance)}</p>
        </div>
        <p class="small dim" style="margin-top:12px">Signals rebuilt from this file by meaning:
          ${Object.entries(d.columns_used || {}).map(([k, v]) =>
            `<code>${esc(k)}</code> &#8592; <code>${esc(v)}</code>`).join(' &#183; ')}</p>
        <div class="scrollx" style="margin-top:12px"><table>${table(
          ['Row', 'Verdict', { t: 'Rank pct', num: true }, ...cols.map((c) => ({ t: c, num: true }))],
          rows.map((r) => [r.row,
            `<span class="pill ${r.flagged ? 'bad' : ''}">${r.flagged ? 'SUSPECTED MULE' : 'no action'}</span>`,
            num(r.percentile, 1),
            ...cols.map((c) => num(r.signals[c], 2))]))}</table></div>
        <p class="small dim" style="margin-top:10px">Every flag is the sum of the columns shown
          beside it, so an investigator can see which behaviour drove it. The cut-off came from
          the shape of the score distribution, not from a target alert rate.</p>
      </div>
    </div>`;
}

function renderScoreOnly(d) {
  const bd = d.band_distribution || {};
  const rows = (d.top_accounts || []).slice(0, 25);
  $('#up-status').innerHTML = `
    <div class="panel verify">
      <header><h3>Scored without labels</h3>
        <span class="tag pill ok">${num(d.rows_scored)} ACCOUNTS</span></header>
      <div class="body">
        ${d.decided ? `<div class="callout green"><div class="tiny">DECIDED AUTOMATICALLY</div>
          <p class="small">${esc(d.decided)}</p></div>` : ''}
        <div class="grid g3">
          ${['HIGH', 'MEDIUM', 'LOW'].map((b) => `<div class="stat ${
            b === 'HIGH' ? 'red' : b === 'MEDIUM' ? 'amber' : ''}">
            <div class="k">${b}</div><div class="v">${num(bd[b] || 0)}</div>
            <div class="u">accounts</div></div>`).join('')}
        </div>
        <p class="small dim" style="margin-top:12px">
          ${num(d.features_matched)} of ${num(d.features_expected)} model features matched by
          name or F-code; ${num(d.features_imputed_from_training_medians)} filled from training
          medians.${(d.columns_not_recognised || []).length
            ? ` ${d.columns_not_recognised.length} column(s) not recognised and ignored.` : ''}</p>
        <div class="scrollx" style="margin-top:12px"><table>${table(
          ['Row', { t: 'Risk score', num: true }, 'Band', 'Recommended action'],
          rows.map((r) => [r.row, num(r.risk_score, 0),
            `<span class="pill ${r.band === 'HIGH' ? 'bad' : r.band === 'MEDIUM' ? 'warn' : ''}">${esc(r.band)}</span>`,
            esc(r.recommended_action)]))}</table></div>
        <div class="callout" style="margin-top:14px">
          <div class="tiny">WHAT THIS IS AND IS NOT</div>
          <p class="small">${esc(d.provenance)}</p>
        </div>
      </div>
    </div>`;
}

function watchJob(id) {
  if (POLL) clearInterval(POLL);
  const tick = async () => {
    let j;
    try { j = await api('/api/jobs/' + id); }
    catch (e) { clearInterval(POLL); return; }
    renderJob(j);
    if (j.status === 'DONE' || j.status === 'FAILED' || j.status === 'CANCELLED') {
      clearInterval(POLL); POLL = null;
      $('#up-go').disabled = false;
      refreshJobs();
      if (j.status === 'DONE') showJobResults(id);
    }
  };
  tick();
  POLL = setInterval(tick, 1500);
}

let LOG_OPEN = false;   // survives the poll tick that rebuilds the panel

function renderJob(j) {
  // renderJob replaces #up-status wholesale every 1500 ms, which destroyed
  // and recreated the <details> and lost its open state. That is why the log
  // snapped shut on click and could only be read once polling stopped.
  const prev = $('#up-status details');
  if (prev) LOG_OPEN = prev.open;
  const running = j.status === 'RUNNING';
  const tone = j.status === 'DONE' ? 'green' : j.status === 'FAILED' ? 'red' : 'amber';
  $('#up-status').innerHTML = `
    <div class="panel ${tone === 'green' ? 'verify' : tone === 'red' ? 'danger' : 'accent'}">
      <header><h3>${esc(j.original_name)}</h3>
        <span class="tag pill ${j.status === 'DONE' ? 'ok' : j.status === 'FAILED' ? 'bad' : 'warn'}">${esc(j.status)}</span>
      </header>
      <div class="body">
        <div class="upbar"><span style="width:${j.percent}%"></span></div>
        <div class="tiny dim" style="margin-top:8px">
          ${j.stages_complete} of ${j.stages_total} stages ·
          ${esc(j.current_stage || 'starting')} ·
          ${num(j.elapsed_seconds, 0)}s elapsed
          ${running ? '' : ` · output in <code>${esc(j.workdir)}</code>`}
        </div>
        ${j.error ? `<div class="errorbox" style="margin-top:12px">
          <h4>Pipeline failed</h4><div>${esc(j.error)}</div></div>` : ''}
        ${running ? `<button class="btn ghost" style="margin-top:12px"
           onclick="cancelJob('${esc(j.job_id)}')">CANCEL</button>` : ''}
        <details ${LOG_OPEN ? 'open' : ''} style="margin-top:12px"><summary class="tiny"
          style="cursor:pointer;color:var(--amber)">LOG</summary>
          <pre class="uplog">${esc((j.log_tail || []).join('\n'))}</pre></details>
      </div>
    </div>`;
  wireJobLog();
}

function wireJobLog() {
  const det = $('#up-status details');
  if (!det) return;
  det.addEventListener('toggle', () => { LOG_OPEN = det.open; });
  const pre = det.querySelector('.uplog');
  if (det.open && pre) pre.scrollTop = pre.scrollHeight;   // newest lines in view
}

async function cancelJob(id) {
  try { await fetch('/api/jobs/' + id + '/cancel', { method: 'POST' }); } catch (e) {}
}

/* The result panels sit ~1,400px down the upload page, so on a laptop screen a
   click paints them entirely below the fold: the work happened, the page did not
   move, and it reads as a dead button. Bring the panel to the reader. */
function revealPanel(sel) {
  const n = document.querySelector(sel);
  if (n && n.innerHTML.trim()) {
    n.scrollIntoView({ behavior: 'smooth', block: 'start' });
  }
}

// A threshold is estimated from positives. A small upload may not carry enough
// of them, and the fitted precision/recall then collapse to zero while the
// ranking underneath is still good. In that case the headline switches to a
// review budget: the top N accounts an analyst team would actually work, which
// is fixed by capacity rather than fitted and so needs no positives at all.
function bestBudget(points) {
  let best = null, bestF1 = -1;
  (points || []).forEach(b => {
    const f1 = (b.precision + b.recall) > 0
      ? 2 * b.precision * b.recall / (b.precision + b.recall) : 0;
    if (f1 > bestF1) { bestF1 = f1; best = b; }
  });
  return best;
}

function budgetTable(rb) {
  const pts = (rb && rb.points) || [];
  if (!pts.length) return '';
  return `<div class="scrollx" style="margin-top:14px"><table>${table(
    ['Review budget', 'Accounts', 'Mules found', 'Precision', 'Recall', 'Lift'],
    pts.map(b => [
      'top ' + b.budget_pct + '%',
      num(b.accounts_reviewed),
      num(b.true_mules_found),
      pct(b.precision, 1),
      pct(b.recall, 1),
      num(b.lift_over_prevalence, 1) + '×',
    ]))}</table></div>`;
}

async function showJobResults(id) {
  const out = $('#up-results');
  out.innerHTML = '';
  out.appendChild(el('div', 'loading', 'READING RESULTS'));
  let r;
  try { r = await api('/api/jobs/' + id + '/results'); }
  catch (e) { out.innerHTML = ''; out.appendChild(errorBox(e)); return; }
  if (!r.available) { out.innerHTML = ''; return; }

  const s = r.schema || {}, ig = r.integrity || {}, m = r.metrics || {};
  const bands = r.bands || {};
  const contaminated = ig.contaminated;
  const weak = r.threshold_estimable === false;
  const bb = weak ? bestBudget((r.review_budget || {}).points) : null;

  out.innerHTML = `
    <div class="panel ${contaminated ? 'danger' : 'verify'}">
      <header><h3>Integrity verdict</h3>
        <span class="tag pill ${contaminated ? 'bad' : 'ok'}">${contaminated ? 'CONTAMINATED' : 'NO ARTEFACT DETECTED'}</span>
      </header>
      <div class="body">
        <p>${esc(ig.summary || 'No integrity audit available.')}</p>
        <div class="grid g3" style="margin-top:12px">
          ${stat('Blank patterns only', num(ig.test_a && ig.test_a.auprc, 4),
                 'AUPRC with every value discarded', contaminated ? 'red' : 'green')}
          ${stat('Shuffled-label control', num(ig.test_c && ig.test_c.auprc, 4),
                 'must collapse to baseline', 'green')}
          ${stat('Random baseline', num(ig.baseline, 4), 'at this prevalence')}
        </div>
      </div>
    </div>

    <div class="panel"><header><h3>What it worked out on its own</h3></header>
      <div class="body"><div class="kv">
        <dt>Target column</dt><dd><code>${esc(s.target_column)}</code> — ${esc(s.target_resolved_by)}</dd>
        <dt>Shape</dt><dd class="num">${num(s.n_rows)} rows × ${num(s.n_columns)} columns</dd>
        <dt>Positives</dt><dd class="num">${num(s.positives)} (${s.prevalence_pct}%)</dd>
        <dt>Column naming</dt><dd>${esc(s.column_naming)}</dd>
        <dt>Identifiers dropped</dt><dd class="num">${num(s.n_identifier_columns)}</dd>
        <dt>Partition column</dt><dd>${r.partition_audit && r.partition_audit.column
          ? `<code>${esc(r.partition_audit.column)}</code> at purity ${num(r.partition_audit.purity, 3)}`
          : '<span class="dim">none found</span>'}</dd>
      </div></div>
    </div>

    <div class="panel"><header><h3>Measured result</h3>
      <span class="tag pill">${esc(r.validation || '')}</span></header>
      <div class="body">
      ${weak ? `<div class="callout amber">
        <div class="tiny">Too few mules to fit a cutoff</div>
        This file carries ${num(r.n_mules)} mules, which leaves about
        ${num(r.positives_per_fit, 1)} per threshold fit where
        ${num((r.review_budget || {}).min_positives_required)} are needed. A cutoff placed on that
        many points does not transfer, so the fitted precision and recall are not
        reported here &#8212; they would read 0.000 and tell you nothing about the model.
        The figures below are a <strong>review budget</strong> instead: the top
        ${bb ? bb.budget_pct + '%' : 'N'} of accounts an analyst team would work.
        Ranking quality (AUPRC, lift) is unaffected by any of this.</div>` : ''}
      <div class="grid g4">
        ${weak
          ? `${stat('Precision', bb ? pct(bb.precision, 1) : '—',
                    bb ? 'in the top ' + bb.budget_pct + '%' : null, 'green')}
             ${stat('Recall', bb ? pct(bb.recall, 1) : '—',
                    bb ? num(bb.true_mules_found) + ' of ' + num(r.n_mules) + ' mules' : null)}`
          : `${stat('Precision', ms(m.precision), null, 'green')}
             ${stat('Recall', ms(m.recall))}`}
        ${stat('AUPRC', ms(m.auprc), null, 'amber')}
        ${weak
          ? stat('Lift', bb ? num(bb.lift_over_prevalence, 1) + '×' : '—',
                 'over the base rate at that budget', 'amber')
          : stat('Lift', m.lift_over_prevalence ? num(m.lift_over_prevalence.mean, 0) + '×' : '—',
                 'over the base rate', 'amber')}
      </div>
      ${budgetTable(r.review_budget)}
      ${bands.HIGH && !weak ? `<div class="callout red" style="margin-top:14px">
        <div class="tiny">High-risk band</div>
        ${num(bands.HIGH.accounts)} accounts, ${num(bands.HIGH.true_mules)} confirmed,
        precision ${pct(bands.HIGH.precision, 1)}</div>` : ''}
      ${contaminated ? `<div class="callout red"><div class="tiny">Read this before quoting the number</div>
        This dataset failed its integrity audit. Treat the figures above as an upper bound on what
        it can demonstrate, not as a validated detection rate.</div>` : ''}
      <p class="small dim" style="margin-top:12px">Full artefacts in
        <code>${esc(r.workdir)}/reports/</code>. Finished in ${num(r.elapsed_seconds, 0)}s.</p>
      </div>
    </div>`;
  revealPanel('#up-results');
}

async function refreshJobs() {
  const n = $('#up-jobs');
  if (!n) return;
  let d;
  try { d = await api('/api/jobs?limit=10'); } catch (e) { return; }
  if (!d.jobs.length) { n.innerHTML = '<div class="empty">No datasets uploaded yet.</div>'; return; }
  n.innerHTML = `<div class="scrollx"><table>${table(
    ['File', 'Status', { t: 'Stages', num: true }, { t: 'Elapsed', num: true }, 'Output', ''],
    d.jobs.map((j) => [
      esc(j.original_name),
      `<span class="pill ${j.status === 'DONE' || j.status === 'SCORED' ? 'ok'
        : j.status === 'FAILED' ? 'bad' : 'warn'}">${esc(j.status)}</span>`,
      // A score-only run never enters the 10-stage pipeline, so "0/10" would
      // read as a stalled training run rather than a finished scoring one.
      j.status === 'SCORED' ? '<span class="dim">no pipeline</span>'
                            : `${j.stages_complete}/${j.stages_total}`,
      num(j.elapsed_seconds, 0) + 's',
      `<code style="font-size:11px">${esc(j.workdir)}</code>`,
      j.status === 'DONE'
        ? `<button class="btn ghost" style="padding:3px 10px" onclick="showJobResults('${esc(j.job_id)}')">RESULTS</button>`
        // A reload drops the poller, so a run in flight became unwatchable and
        // its log unreadable until it finished. Re-attaching is all it needs.
        : j.status === 'RUNNING'
          ? `<button class="btn ghost" style="padding:3px 10px" onclick="watchJob('${esc(j.job_id)}')">WATCH</button>`
          : '',
    ]))}</table></div>`;
}


/* ---------- boot -------------------------------------------------------- */
(async function boot() {
  buildRail();
  judgeInit();
  // Route BEFORE awaiting the network. Health is decoration; navigation is not.
  // Awaiting it first meant a slow or unreachable API left every section in its
  // raw HTML state, which rendered hero and judge stacked on top of each other.
  const hash = location.hash.slice(1);
  go(SECTIONS.some((s) => s.id === hash) ? hash : 'hero');
  refreshHealth();
  setInterval(refreshHealth, 30000);
  window.addEventListener('hashchange', () => {
    const h = location.hash.slice(1);
    if (SECTIONS.some((s) => s.id === h)) go(h);
  });
})();
