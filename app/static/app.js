/* Paperless AI Classifier dashboard. Vanilla JS, no build step. */
'use strict';

// ----------------------------------------------------------------------------- helpers
const $ = (sel, root = document) => root.querySelector(sel);
const $$ = (sel, root = document) => Array.from(root.querySelectorAll(sel));
const esc = (v) => v === undefined || v === null ? '' : String(v)
  .replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;').replace(/'/g, '&#39;');
const fmtTime = (iso) => iso ? new Date(iso).toLocaleString() : '';
const fmtShort = (iso) => iso ? `<span title="${esc(fmtTime(iso))}">${esc(new Date(iso).toLocaleString(undefined, { day: '2-digit', month: '2-digit', hour: '2-digit', minute: '2-digit' }))}</span>` : '';
const fmtSec = (s) => (s === null || s === undefined) ? '' : (s >= 60 ? `${Math.floor(s / 60)}m ${Math.round(s % 60)}s` : `${Number(s).toFixed(1)}s`);
const pct = (v) => (v === null || v === undefined) ? '' : Math.round(v * 100) + '%';
const tags = (arr) => (arr || []).map(t => `<span class="tag">${esc(t)}</span>`).join('') || '<span class="muted">–</span>';

async function api(path, opts = {}) {
  const init = { method: opts.method || 'GET', headers: {} };
  if (opts.body !== undefined) { init.headers['Content-Type'] = 'application/json'; init.body = JSON.stringify(opts.body); }
  const r = await fetch(path, init);
  let data = null;
  try { data = await r.json(); } catch (e) { /* no body */ }
  if (!r.ok) throw new Error((data && (data.detail || data.error)) || `${r.status} ${r.statusText}`);
  return data;
}

function toast(msg, kind = '') {
  const el = document.createElement('div');
  el.className = kind; el.textContent = msg;
  $('#toast').appendChild(el);
  setTimeout(() => el.remove(), 4000);
}
const guard = (fn) => async (...a) => { try { await fn(...a); } catch (e) { toast(e.message, 'bad'); console.error(e); } };

function openModal(html) { $('#modal-body').innerHTML = html; $('#modal').classList.remove('hidden'); }
function closeModal() { $('#modal').classList.add('hidden'); $('#modal-body').innerHTML = ''; }
$('#modal-close').onclick = closeModal;
$('#modal').addEventListener('click', (e) => { if (e.target.id === 'modal') closeModal(); });

// ----------------------------------------------------------------------------- tabs
let activeTab = 'overview';
$$('#tabs button').forEach(b => b.onclick = () => showTab(b.dataset.tab));
function showTab(name) {
  activeTab = name;
  $$('#tabs button').forEach(b => b.classList.toggle('active', b.dataset.tab === name));
  $$('.tab').forEach(t => t.classList.toggle('active', t.id === `tab-${name}`));
  ({ overview: refreshRecent, activity: loadActivity, learning: loadLearning, config: loadConfig, tools: loadAbout, logs: loadLogs }[name] || (() => {}))().catch?.(() => {});
  location.hash = name;
}

// ----------------------------------------------------------------------------- overview / header
let names = { document_types: [], correspondents: [], tags: [] };
async function loadNames() {
  try {
    names = await api('/api/paperless/names');
    for (const [kind, id] of [['document_types', 'dl-types'], ['correspondents', 'dl-correspondents'], ['tags', 'dl-tags']]) {
      $('#' + id).innerHTML = names[kind].map(n => `<option value="${esc(n)}">`).join('');
    }
  } catch (e) { /* Paperless down: datalists stay empty */ }
}

async function refreshStatus() {
  let st;
  try { st = await api('/api/status'); } catch (e) { $('#h-paperless').className = 'dot bad'; return; }
  const s = st.stats;
  $('#tiles').innerHTML = [
    ['Completed', s.completed], ['Today', s.completed_today], ['Failed', s.failed], ['Skipped', s.skipped],
    ['Avg time', fmtSec(s.avg_processing_time)], ['Avg confidence', s.avg_confidence + '%'],
    ['Queue', st.queue_size + (st.processing ? ' + 1' : '')], ['Mappings', s.learned_mappings],
  ].map(([l, v]) => `<div class="tile"><div class="v">${esc(v)}</div><div class="l">${l}</div></div>`).join('');

  const now = $('#now');
  if (st.processing) {
    const started = st.started_at ? Math.round((Date.now() - new Date(st.started_at)) / 1000) : 0;
    now.className = 'now busy';
    now.innerHTML = `<div class="stage">${esc(st.stage || 'working')} · ${fmtSec(started)}</div>
      <div><strong>#${esc(st.current_doc)}</strong> ${esc(st.current_title || '')}</div>`;
  } else {
    now.className = 'now idle';
    const lp = st.last_processed;
    now.innerHTML = lp ? `Idle. Last: <strong>#${esc(lp.doc_id)}</strong> ${esc(lp.title)} → ${esc(lp.type)}${lp.correspondent ? ' / ' + esc(lp.correspondent) : ''} <span class="muted">(${fmtTime(lp.time)})</span>` : 'Idle';
  }
  $('#pending').innerHTML = st.pending.length
    ? 'Queued: ' + st.pending.map(id => `<span class="tag">#${id}<button data-rm="${id}" title="remove">×</button></span>`).join('')
    : '';
  $$('#pending button').forEach(b => b.onclick = guard(async () => { await api(`/api/queue/${b.dataset.rm}`, { method: 'DELETE' }); refreshStatus(); }));

  const loaded = st.models_loaded || [];
  const mine = loaded.find(m => m.name === st.model || m.name === st.model + ':latest');
  $('#model-info').innerHTML = [
    ['Configured', `<span class="mono">${esc(st.model)}</span>`],
    ['In RAM', loaded.length ? loaded.map(m => `<span class="mono">${esc(m.name)}</span> ${m.size_mb ? '(' + Math.round(m.size_mb / 1000 * 10) / 10 + ' GB)' : ''}`).join('<br>') : '<span class="muted">nothing loaded</span>'],
    ['Expires', mine && mine.expires_at ? fmtTime(mine.expires_at) : '<span class="muted">–</span>'],
    ['Processed since start', st.processed_count + (st.failed_count ? ` <span class="muted">(${st.failed_count} failed)</span>` : '')],
    ['Service started', fmtTime(st.service_started)],
  ].map(([k, v]) => `<div class="k">${k}</div><div>${v}</div>`).join('');
  $('#h-model').className = 'dot ' + (mine ? 'ok' : (loaded.length ? 'warn' : ''));
  $('#h-model').title = mine ? 'model loaded' : 'model not in RAM (loads on first request)';
  const v = st.version || {};
  $('#version-chip').textContent = `v${v.version || '?'}${v.commit ? ' · ' + v.commit : ''}`;
}

async function refreshHealth() {
  try {
    const h = await api('/api/health');
    $('#h-paperless').className = 'dot ' + (h.paperless === 'healthy' ? 'ok' : 'bad'); $('#h-paperless').title = 'Paperless: ' + h.paperless;
    $('#h-ollama').className = 'dot ' + (h.ollama === 'healthy' ? 'ok' : 'bad'); $('#h-ollama').title = 'Ollama: ' + h.ollama;
  } catch (e) { $('#h-paperless').className = $('#h-ollama').className = 'dot bad'; }
}

async function refreshRecent() {
  const data = await api('/api/audit?limit=10');
  $('#recent tbody').innerHTML = data.logs.map(l => `<tr>
    <td class="nowrap muted c-time">${fmtShort(l.timestamp)}</td>
    <td class="c-doc"><strong>#${l.document_id}</strong><span class="title" title="${esc(l.document_title)}">${esc(l.document_title)}</span></td>
    <td data-label="Type">${esc(l.document_type || '')}</td><td data-label="From">${esc(l.correspondent || '')}</td><td data-label="Tags">${tags(l.tags)}</td>
    <td class="hide-sm">${pct(l.confidence)}</td><td class="c-status"><span class="status ${esc(l.status)}">${esc(l.status)}</span></td>
  </tr>`).join('') || '<tr><td class="muted">Nothing processed yet</td></tr>';
}

$('#classify-btn').onclick = guard(async () => {
  const id = parseInt($('#classify-id').value, 10); if (!id) return;
  const r = await api('/api/classify', { method: 'POST', body: { document_id: id } });
  toast(`#${id}: ${r.status}`, 'ok'); $('#classify-id').value = ''; refreshStatus();
});
$('#reanalyze-btn').onclick = guard(async () => {
  const id = parseInt($('#classify-id').value, 10); if (!id) return;
  const r = await api(`/api/reanalyze/${id}`, { method: 'POST' });
  toast(`#${id}: ${r.status} (re-analyze)`, 'ok'); $('#classify-id').value = ''; refreshStatus();
});
$('#clear-queue-btn').onclick = guard(async () => {
  if (!confirm('Remove all queued documents?')) return;
  const r = await api('/api/queue/clear', { method: 'DELETE' }); toast(`Removed ${r.removed}`, 'ok'); refreshStatus();
});
$('#unload-btn').onclick = guard(async () => { await api('/api/ollama/unload', { method: 'POST' }); toast('Unload requested', 'ok'); setTimeout(refreshStatus, 1000); });

// ----------------------------------------------------------------------------- activity
const act = { offset: 0, limit: 50, total: 0 };
async function loadActivity() {
  const status = $('#act-status').value, q = $('#act-q').value.trim();
  const data = await api(`/api/audit?limit=${act.limit}&offset=${act.offset}&status=${encodeURIComponent(status)}&q=${encodeURIComponent(q)}`);
  act.total = data.total;
  $('#activity tbody').innerHTML = data.logs.map(l => {
    const t = l.timing || {};
    const timeTitle = t.total_s !== undefined ? `load ${t.load_s}s · prompt ${t.prompt_s}s (${t.prompt_tokens} tok) · output ${t.eval_s}s (${t.eval_tokens} tok)` : '';
    return `<tr>
      <td class="nowrap muted c-time">${fmtShort(l.timestamp)}</td>
      <td class="c-doc"><strong>#${l.document_id}</strong><span class="title" title="${esc(l.document_title)}">${esc(l.document_title)}</span></td>
      <td data-label="Type">${esc(l.document_type || '')}</td><td data-label="From">${esc(l.correspondent || '')}</td><td data-label="Tags">${tags(l.tags)}</td>
      <td class="hide-sm">${pct(l.confidence)}</td><td class="nowrap hide-sm" title="${esc(timeTitle)}">${fmtSec(l.processing_time)}</td>
      <td class="c-status"><span class="status ${esc(l.status)}" title="${esc(l.error_message || '')}">${esc(l.status)}</span>${l.error_message && l.status !== 'completed' ? `<div class="muted err">${esc(l.error_message)}</div>` : ''}</td>
      <td class="actions">
        <button class="small" data-act="details" data-id="${l.id}">Details</button>
        ${l.status === 'completed' ? `<button class="small" data-act="correct" data-id="${l.id}">Correct</button>` : ''}
        <button class="small" data-act="redo" data-doc="${l.document_id}" title="Re-analyze and overwrite">Redo</button>
        <button class="small danger" data-act="del" data-id="${l.id}">×</button>
      </td></tr>`;
  }).join('') || '<tr><td colspan="9" class="muted">No entries</td></tr>';
  const from = act.total ? act.offset + 1 : 0, to = Math.min(act.offset + act.limit, act.total);
  $('#act-page').textContent = `${from}–${to} of ${act.total}`;
  $('#act-prev').disabled = act.offset === 0; $('#act-next').disabled = to >= act.total;
  $$('#activity button[data-act]').forEach(b => b.onclick = guard(async () => {
    const a = b.dataset.act;
    if (a === 'details') showDetails(parseInt(b.dataset.id, 10));
    else if (a === 'correct') showCorrect(parseInt(b.dataset.id, 10));
    else if (a === 'redo') { const r = await api(`/api/reanalyze/${b.dataset.doc}`, { method: 'POST' }); toast(`#${b.dataset.doc}: ${r.status}`, 'ok'); }
    else if (a === 'del') { await api(`/api/audit/${b.dataset.id}`, { method: 'DELETE' }); loadActivity(); }
  }));
}
$('#act-status').onchange = () => { act.offset = 0; loadActivity(); };
let qTimer; $('#act-q').oninput = () => { clearTimeout(qTimer); qTimer = setTimeout(() => { act.offset = 0; loadActivity(); }, 300); };
$('#act-prev').onclick = () => { act.offset = Math.max(0, act.offset - act.limit); loadActivity(); };
$('#act-next').onclick = () => { act.offset += act.limit; loadActivity(); };
$('#act-cleanup').onclick = guard(async () => { const r = await api('/api/audit/cleanup', { method: 'POST' }); toast(`${r.updated} marked abandoned`, 'ok'); loadActivity(); });
$('#act-del-failed').onclick = guard(async () => { if (confirm('Delete all failed entries?')) { const r = await api('/api/audit/status/failed', { method: 'DELETE' }); toast(`${r.deleted} deleted`, 'ok'); loadActivity(); } });
$('#act-del-skipped').onclick = guard(async () => { if (confirm('Delete all skipped entries?')) { const r = await api('/api/audit/status/skipped', { method: 'DELETE' }); toast(`${r.deleted} deleted`, 'ok'); loadActivity(); } });

function kvTable(obj) {
  return `<div class="kv">${Object.entries(obj).map(([k, v]) => `<div class="k">${esc(k)}</div><div>${v}</div>`).join('')}</div>`;
}

async function showDetails(id) {
  const l = await api(`/api/audit/${id}`);
  let raw = {};
  try { raw = typeof l.ai_raw === 'string' ? JSON.parse(l.ai_raw) : (l.ai_raw || {}); } catch (e) { raw = { answer: l.ai_raw }; }
  const t = l.timing || {};
  openModal(`<h2>#${l.document_id} ${esc(l.document_title)}</h2>
    ${kvTable({
      Status: `<span class="status ${esc(l.status)}">${esc(l.status)}</span> ${esc(l.error_message || '')}`,
      Result: `${esc(l.document_type || '')} / ${esc(l.correspondent || '')} ${tags(l.tags)}`,
      Confidence: pct(l.confidence), Model: esc(l.model || ''),
      Timing: t.total_s !== undefined ? `total ${t.total_s}s = load ${t.load_s}s + prompt ${t.prompt_s}s (${t.prompt_tokens} tok, ${l.pages_sent || 0} page images) + output ${t.eval_s}s (${t.eval_tokens} tok)` : fmtSec(l.processing_time),
      'Written to Paperless': (l.changes || []).length ? esc(l.changes.join(', ')) : (l.auto_approved ? 'yes' : '<span class="muted">nothing</span>'),
      Explanation: esc(l.explanation || ''),
    })}
    ${raw.model_output ? `<details><summary>Model output before normalization</summary><pre class="block">${esc(JSON.stringify(raw.model_output, null, 2))}</pre></details>` : ''}
    ${raw.normalized && raw.normalized.length ? `<details open><summary>Normalization</summary><pre class="block">${esc(raw.normalized.join('\n'))}</pre></details>` : ''}
    ${raw.answer ? `<details><summary>Raw answer</summary><pre class="block">${esc(raw.answer)}</pre></details>` : ''}`);
}

async function showCorrect(id) {
  const l = await api(`/api/audit/${id}`);
  openModal(`<h2>Correct #${l.document_id} <small class="muted">${esc(l.document_title)}</small></h2><div class="muted">Loading Paperless data…</div>`);
  let pd = null;
  try { pd = await api(`/api/paperless/document/${l.document_id}`); } catch (e) { /* document may be gone */ }
  const aiTags = l.tags || [];
  openModal(`<h2>Correct #${l.document_id} <small class="muted">${esc(l.document_title)}</small></h2>
    <table class="cmp"><thead><tr><th style="width:18%">Field</th><th style="width:24%">Model said</th><th style="width:24%">Paperless now</th><th>Your correction</th><th style="width:70px">Write</th></tr></thead><tbody>
      <tr><td class="f">Type</td><td data-label="Model said">${esc(l.document_type || '')}</td><td data-label="Paperless now">${pd ? esc(pd.document_type || '–') : '?'}</td>
          <td data-label="Your correction"><input id="c-type" list="dl-types" value="${esc(pd && pd.document_type || l.document_type || '')}"></td><td class="w"><label><input type="checkbox" id="c-type-sync" checked> <span class="sm-only">write to Paperless</span></label></td></tr>
      <tr><td class="f">Correspondent</td><td data-label="Model said">${esc(l.correspondent || '')}</td><td data-label="Paperless now">${pd ? esc(pd.correspondent || '–') : '?'}</td>
          <td data-label="Your correction"><input id="c-corr" list="dl-correspondents" value="${esc(pd && pd.correspondent || l.correspondent || '')}"></td><td class="w"><label><input type="checkbox" id="c-corr-sync" checked> <span class="sm-only">write to Paperless</span></label></td></tr>
      <tr><td class="f">Tags</td><td data-label="Model said">${tags(aiTags)}</td><td data-label="Paperless now">${pd ? tags(pd.tags) : '?'}</td>
          <td data-label="Your correction"><input id="c-tags" list="dl-tags" value="${esc((pd ? pd.tags : aiTags).join(', '))}" placeholder="comma separated"></td><td class="w"><label><input type="checkbox" id="c-tags-sync" checked> <span class="sm-only">write to Paperless</span></label></td></tr>
    </tbody></table>
    <p class="muted">Changed type and correspondent become term mappings (applied to future answers). The tag list, when written, replaces the document's tags.</p>
    <div class="row"><button class="primary" id="c-save">Save</button><button id="c-cancel">Cancel</button></div>`);
  $('#c-cancel').onclick = closeModal;
  $('#c-save').onclick = guard(async () => {
    const userType = $('#c-type').value.trim(), userCorr = $('#c-corr').value.trim();
    const userTags = $('#c-tags').value.split(',').map(t => t.trim()).filter(Boolean);
    const corrections = {
      document_type: { ai: l.document_type || '', user: userType, sync: $('#c-type-sync').checked && !!userType },
      correspondent: { ai: l.correspondent || '', user: userCorr, sync: $('#c-corr-sync').checked && !!userCorr },
      tags: { ai: aiTags, user: userTags, sync: $('#c-tags-sync').checked },
    };
    const r = await api('/api/learn/correct', { method: 'POST', body: { audit_id: id, corrections } });
    toast(`Saved. Learned: ${r.learned.length}, written: ${r.paperless_updated.join(', ') || 'nothing'}`, 'ok');
    closeModal(); loadNames();
  });
}

// ----------------------------------------------------------------------------- learning
async function loadLearning() {
  const [m, e] = await Promise.all([api('/api/mappings'), api('/api/examples')]);
  $('#mappings tbody').innerHTML = m.mappings.map(x => `<tr><td class="muted">${esc(x.term_type)}</td><td>${esc(x.ai_term)}</td><td><strong>${esc(x.approved_term)}</strong></td><td>${x.times_used}×</td>
    <td><button class="small danger" data-id="${x.id}">×</button></td></tr>`).join('') || '<tr><td colspan="5" class="muted">No mappings yet. Correct a classification in Activity to create one.</td></tr>';
  $$('#mappings button').forEach(b => b.onclick = guard(async () => { await api(`/api/mappings/${b.dataset.id}`, { method: 'DELETE' }); loadLearning(); }));
  $('#examples tbody').innerHTML = e.examples.map(x => `<tr><td><strong>#${esc(x.document_id)}</strong><span class="title">${esc(x.document_title)}</span></td><td>${esc(x.document_type)}</td><td>${esc(x.correspondent || '')}</td><td>${tags(x.tags)}</td>
    <td><button class="small danger" data-id="${x.id}">×</button></td></tr>`).join('') || '<tr><td colspan="5" class="muted">No examples yet</td></tr>';
  $$('#examples button').forEach(b => b.onclick = guard(async () => { await api(`/api/examples/${b.dataset.id}`, { method: 'DELETE' }); loadLearning(); }));
}
$('#map-add').onclick = guard(async () => {
  await api('/api/mappings', { method: 'POST', body: { term_type: $('#map-type').value, ai_term: $('#map-ai').value, approved_term: $('#map-user').value } });
  $('#map-ai').value = $('#map-user').value = ''; loadLearning();
});
$('#ex-clear').onclick = guard(async () => { if (confirm('Delete all examples?')) { await api('/api/learn/examples', { method: 'DELETE' }); loadLearning(); } });
$('#learn-reset').onclick = guard(async () => { if (confirm('Delete all mappings and examples?')) { await api('/api/learn/reset', { method: 'DELETE' }); loadLearning(); } });

// ----------------------------------------------------------------------------- config
let cfg = { values: {}, schema: [] };
async function loadConfig() {
  cfg = await api('/api/config');
  let models = [];
  try { models = (await api('/api/ollama/models')).models; } catch (e) { /* offline */ }
  $('#dl-models').innerHTML = models.map(m => `<option value="${esc(m)}">`).join('');
  const groups = [...new Set(cfg.schema.map(f => f.group))];
  $('#config-form').innerHTML = groups.map(g => `<div class="group"><h3>${esc(g)}</h3>${cfg.schema.filter(f => f.group === g).map(renderField).join('')}</div>`).join('');
  $$('#config-form [data-key]').forEach(el => el.addEventListener('input', markDirty));
  markDirty();
}
function renderField(f) {
  const v = cfg.values[f.key];
  const val = (v === null || v === undefined) ? '' : String(v);
  let input;
  if (f.type === 'bool') input = `<select data-key="${f.key}"><option value="true" ${val === 'true' || v === true ? 'selected' : ''}>true</option><option value="false" ${val === 'false' || v === false ? 'selected' : ''}>false</option></select>`;
  else if (f.type === 'choice') input = `<select data-key="${f.key}">${f.options.map(o => `<option ${o === val ? 'selected' : ''}>${esc(o)}</option>`).join('')}</select>`;
  else if (f.type === 'secret') input = `<input data-key="${f.key}" type="password" value="${esc(val)}" autocomplete="off">`;
  else if (f.key === 'OLLAMA_MODEL') input = `<input data-key="${f.key}" list="dl-models" value="${esc(val)}">`;
  else input = `<input data-key="${f.key}" value="${esc(val)}" placeholder="${f.type.endsWith('_opt') ? 'model default' : ''}">`;
  return `<div class="field" id="f-${f.key}"><label>${esc(f.label)}<small class="mono">${f.key}${f.restart ? ' · restart' : ''}</small></label>${input}${f.help ? `<div class="help">${esc(f.help)}</div>` : ''}</div>`;
}
function changedValues() {
  const out = {};
  $$('#config-form [data-key]').forEach(el => {
    const k = el.dataset.key, orig = cfg.values[k];
    const origStr = (orig === null || orig === undefined) ? '' : (typeof orig === 'boolean' ? String(orig) : String(orig));
    if (el.value !== origStr) out[k] = el.value;
    $('#f-' + k).classList.toggle('changed', el.value !== origStr);
  });
  return out;
}
function markDirty() { const n = Object.keys(changedValues()).length; $('#config-dirty').textContent = n ? `${n} unsaved change${n > 1 ? 's' : ''}` : 'No unsaved changes'; }
$('#config-save').onclick = guard(async () => {
  const values = changedValues();
  if (!Object.keys(values).length) return toast('Nothing changed');
  const r = await api('/api/config', { method: 'POST', body: { values } });
  toast(`Saved ${r.updated.length} setting(s)` + (r.restart_required ? '' : ', applied'), 'ok');
  $('#restart-banner').classList.toggle('hidden', !r.restart_required);
  loadConfig();
});
$('#config-reload').onclick = () => loadConfig();
$('#restart-btn').onclick = guard(async () => { if (confirm('Restart the service now?')) await restartAndWait(); });
async function restartAndWait() {
  await api('/api/restart', { method: 'POST' }).catch(() => {});
  toast('Restarting…');
  await new Promise(r => setTimeout(r, 2500));
  for (let i = 0; i < 40; i++) {
    try { await api('/api/health'); location.reload(); return; } catch (e) { await new Promise(r => setTimeout(r, 1500)); }
  }
  toast('Service did not come back yet. Check the logs.', 'bad');
}

// ----------------------------------------------------------------------------- tools
$('#dry-btn').onclick = guard(async () => {
  const id = parseInt($('#dry-id').value, 10); if (!id) return;
  $('#dry-btn').disabled = true; $('#dry-state').textContent = 'running… (watch Overview for the stage)'; $('#dry-result').innerHTML = '';
  try {
    const o = await api(`/api/dryrun/${id}`, { method: 'POST' });
    const t = o.timing || {};
    $('#dry-result').innerHTML = `
      ${kvTable({
        Document: `#${o.doc_id} ${esc(o.title)}`,
        Outcome: o.ok ? '<span class="status completed">ok</span>' : `<span class="status failed">failed</span> ${esc(o.reason)}`,
        Input: `${o.pages_sent} page image(s), ${o.image_kb} KB${o.ocr_chars ? `, ${o.ocr_chars} chars OCR text` : ''}`,
        Timing: t.total_s !== undefined ? `total ${t.total_s}s = load ${t.load_s}s + prompt ${t.prompt_s}s (${t.prompt_tokens} tok) + output ${t.eval_s}s (${t.eval_tokens} tok)` : '',
        Result: o.ok ? `${esc(o.result.document_type)} / ${esc(o.result.correspondent || 'keine')} ${tags(o.result.tags)} · conf ${pct(o.result.confidence)}` : '',
      })}
      ${o.ok && o.result.normalized && o.result.normalized.length ? `<details open><summary>Normalization</summary><pre class="block">${esc(o.result.normalized.join('\n'))}</pre></details>` : ''}
      ${o.model_output && Object.keys(o.model_output).length ? `<details><summary>Model output before normalization</summary><pre class="block">${esc(JSON.stringify(o.model_output, null, 2))}</pre></details>` : ''}
      <details><summary>Raw answer</summary><pre class="block">${esc(o.raw_answer || '(empty)')}</pre></details>
      ${o.thinking ? `<details><summary>Thinking</summary><pre class="block">${esc(o.thinking)}</pre></details>` : ''}
      <details><summary>Prompt (${(o.prompt || '').length} chars)</summary><pre class="block">${esc(o.prompt)}</pre></details>`;
  } finally { $('#dry-btn').disabled = false; $('#dry-state').textContent = ''; }
});

async function loadAbout() {
  const v = await api('/api/version');
  $('#about').innerHTML = kvTable({
    Version: esc(v.version), Commit: v.commit ? `<span class="mono">${esc(v.commit)}</span> ${esc(v.commit_date || '')} ${esc(v.subject || '')}` : '<span class="muted">not a git checkout</span>',
    Branch: esc(v.branch || ''), Python: esc(v.python),
  });
}
$('#update-btn').onclick = guard(async () => {
  if (!confirm('Pull the latest version, install requirements and restart the service?')) return;
  await api('/api/update', { method: 'POST' });
  const pre = $('#update-log'); pre.classList.remove('hidden'); pre.textContent = 'started…';
  let gone = false;
  for (let i = 0; i < 120; i++) {
    await new Promise(r => setTimeout(r, 1500));
    try {
      const l = await api('/api/update/log');
      pre.textContent = l.lines.join('\n'); pre.scrollTop = pre.scrollHeight;
      if (gone) { toast('Service is back', 'ok'); setTimeout(() => location.reload(), 1500); return; }
      if (l.lines.some(x => /service running|not restarted|failed/i.test(x)) && !gone) {
        if (/service running|not restarted/i.test(pre.textContent)) { setTimeout(() => location.reload(), 2000); return; }
      }
    } catch (e) { gone = true; pre.textContent += '\n(restarting…)'; }
  }
});

// ----------------------------------------------------------------------------- logs
async function loadLogs() {
  const level = $('#log-level').value, lines = $('#log-lines').value;
  const d = await api(`/api/logs?lines=${lines}&level=${encodeURIComponent(level)}`);
  const view = $('#log-view');
  const stick = view.scrollTop + view.clientHeight >= view.scrollHeight - 30;
  view.innerHTML = d.lines.map(l => {
    const cls = l.includes('| ERROR |') ? 'e' : l.includes('| WARNING |') ? 'w' : l.includes('| DEBUG |') ? 'd' : (/updated|Vision success|conf \d/.test(l) ? 'ok' : '');
    return `<span class="${cls}">${esc(l)}</span>`;
  }).join('\n');
  if (stick) view.scrollTop = view.scrollHeight;
}
$('#log-refresh').onclick = loadLogs; $('#log-level').onchange = loadLogs; $('#log-lines').onchange = loadLogs;

// ----------------------------------------------------------------------------- boot
async function tick() {
  await refreshStatus();
  if (activeTab === 'overview') refreshRecent().catch(() => {});
  if (activeTab === 'logs' && $('#log-auto').checked) loadLogs().catch(() => {});
}
refreshHealth(); loadNames(); tick();
setInterval(tick, 3000);
setInterval(refreshHealth, 15000);
setInterval(loadNames, 120000);
if (location.hash && $(`#tab-${location.hash.slice(1)}`)) showTab(location.hash.slice(1));
