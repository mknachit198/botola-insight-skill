'use strict';
const $ = (s, el = document) => el.querySelector(s);
const esc = s => String(s ?? '').replace(/[&<>"']/g, c => ({'&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;'}[c]));
const api = async (url, opts = {}) => {
  const r = await fetch(url, opts);
  const j = await r.json().catch(() => ({}));
  if (!r.ok) throw new Error(j.error || ('HTTP ' + r.status));
  return j;
};
const post = (url, body) => api(url, {method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(body || {})});
const KIND_FR = {identity: 'Identité', roster: 'Composition', team_colors: 'Couleurs des équipes', clock_read: 'Chrono illisible',
  event_review: 'Action / ralenti', banner_check: 'Bandeau mal lu', calibration: 'Calibration du terrain', conflict: 'Conflit'};
const STAGE_FR = {identify: 'Identification', import_prior: 'Réponses existantes', clock_templates: 'Apprentissage du chrono', scan: 'Balayage chrono/score',
  timeline: 'Chronologie', analyze: 'Détection + suivi', teams: 'Équipes', identity: 'Identités', events: 'Événements', calibration: 'Calibration', session_stats: 'Statistiques'};
const STATUS_FR = {created: 'créée', running: 'en cours', waiting_human: 'en pause : réponse attendue', done: 'terminée', stopped: 'arrêtée', error: 'erreur'};
const EV_FR = {yellow_card: 'carton jaune', red_card_possible: 'carton rouge ?', substitution_out: 'remplacement — sortie(s)', substitution_in: 'remplacement — entrée(s)', banner_unclassified: 'bandeau non classé'};
const SEG_FR = {live: 'direct', stoppage: 'temps additionnel', live_banner: 'direct (bandeau)', live_interpolated: 'direct (chrono interpolé)',
  no_scoreboard: 'sans tableau (ralenti…)', discontinuity: 'coupure', frozen: 'image figée', unreadable: 'chrono illisible'};

// ------------------------------------------------------------------ onglets
document.querySelectorAll('#tabs button').forEach(b => b.onclick = () => showTab(b.dataset.tab));
function showTab(t) {
  document.querySelectorAll('#tabs button').forEach(x => x.classList.toggle('on', x.dataset.tab === t));
  document.querySelectorAll('.tab').forEach(x => x.classList.toggle('on', x.id === t));
  if (t === 'questions') loadQuestions();
  if (t === 'joueurs') loadStats();
  if (t === 'chronologie') loadTimeline();
  if (t === 'exports') loadExports();
}

// ------------------------------------------------------------------ traitement
let STATE = null;
async function refresh() {
  try { STATE = await api('/api/state'); } catch (e) { return; }
  $('#match-title').textContent = STATE.match.title || STATE.match.key;
  $('#q-badge').textContent = STATE.pending || '';
  const pb = $('#pending-box');
  pb.innerHTML = `<p><b>${STATE.pending}</b> question(s) en attente dont <b>${STATE.pending_blocking}</b> bloquante(s).</p>` +
    (STATE.pending ? '<button id="go-q" class="primary">Ouvrir le parcours</button>' : '<p class="muted">Rien à répondre pour l\'instant.</p>');
  if ($('#go-q')) $('#go-q').onclick = () => showTab('questions');
  const er = $('#runner-error'); er.hidden = !STATE.runner_error; er.textContent = STATE.runner_error || '';
  const box = $('#sessions');
  box.innerHTML = '';
  for (const s of STATE.sessions) {
    const el = $('#tpl-session').content.firstElementChild.cloneNode(true);
    $('.s-title', el).textContent = `Partie ${s.period ?? '?'} — ${s.video}` + (s.known_video ? '' : ' (fichier non référencé dans la configuration)');
    const st = $('.s-status', el); st.textContent = s.running ? 'en cours' : (STATUS_FR[s.status] || s.status || '—'); st.classList.add(s.status || '');
    const p = s.progress;
    $('.bar', el).style.width = p ? Math.round(100 * (p.fraction || 0)) + '%' : (s.status === 'done' ? '100%' : '0%');
    $('.s-prog', el).textContent = p ? `${STAGE_FR[p.stage] || p.stage} — ${Math.round(100 * (p.fraction || 0))} %` +
      (p.speed_x ? ` — vitesse mesurée ×${p.speed_x.toFixed(2)} du temps réel` : '') : '';
    const ol = $('.stages', el);
    for (const g of s.stages) {
      const li = document.createElement('li'); li.className = g.status;
      li.textContent = STAGE_FR[g.name] || g.name; li.title = JSON.stringify(g, null, 1);
      ol.appendChild(li);
    }
    if (s.waiting) $('.s-wait', el).innerHTML = `<div class="note">⏸ En pause : ${esc(s.waiting.message)} <a href="#" class="goq">Répondre</a></div>`;
    if (s.error) $('.s-wait', el).innerHTML += `<div class="error">${esc(s.error)}</div>`;
    el.querySelectorAll('.actions button').forEach(b => b.onclick = async () => { await post(`/api/sessions/${s.id}/${b.dataset.a}`); refresh(); });
    const gq = $('.goq', el); if (gq) gq.onclick = e => { e.preventDefault(); showTab('questions'); };
    box.appendChild(el);
  }
}
$('#f-path').onsubmit = async e => {
  e.preventDefault();
  const fd = Object.fromEntries(new FormData(e.target).entries());
  try { await post('/api/sessions', fd); refresh(); } catch (err) { alert(err.message); }
};
$('#f-upload').onsubmit = e => {
  e.preventDefault();
  const xhr = new XMLHttpRequest(); xhr.open('POST', '/api/sessions');
  xhr.upload.onprogress = ev => $('#up-prog').textContent = ev.lengthComputable ? Math.round(100 * ev.loaded / ev.total) + ' %' : '';
  xhr.onload = () => { $('#up-prog').textContent = xhr.status === 200 ? 'téléversé' : xhr.responseText; refresh(); };
  xhr.send(new FormData(e.target));
};
setInterval(() => { if ($('#traitement').classList.contains('on') || !STATE) refresh(); else api('/api/state').then(s => { STATE = s; $('#q-badge').textContent = s.pending || ''; }); }, 2500);
refresh();

// ------------------------------------------------------------------ questions
let QS = [], QI = 0, ROSTER = [];
for (const [k, v] of Object.entries(KIND_FR)) $('#q-kind').insertAdjacentHTML('beforeend', `<option value="${k}">${v}</option>`);
$('#q-kind').onchange = $('#q-status').onchange = () => loadQuestions();
async function loadQuestions() {
  const j = await api(`/api/questions?status=${$('#q-status').value}&kind=${$('#q-kind').value}`);
  QS = j.items; ROSTER = j.roster; QI = Math.min(QI, Math.max(0, QS.length - 1));
  renderQ();
}
function media(rel) { return rel ? '/media/' + rel.split('/').map(encodeURIComponent).join('/') : ''; }
function renderQ() {
  const P = $('#q-panel');
  if (!QS.length) { P.innerHTML = '<div class="card"><p>Aucune question dans cette sélection. Le traitement peut continuer.</p></div>'; return; }
  const t = QS[QI];
  const ev = t.evidence || {};
  const hasClip = ev.clip_window || ev.v2_clip || t.kind === 'event_review' || t.kind === 'banner_check' || t.kind === 'conflict';
  let left = '';
  if (t.kind === 'calibration') left = `<div class="calib"><img id="cal-img" src="${media(ev.image)}"><canvas id="cal-cv"></canvas></div>`;
  else {
    if (ev.image) left += `<img src="${media(ev.image)}" alt="image justificative">`;
    if (ev.crop) left += `<img class="crop" src="${media(ev.crop)}" alt="agrandissement">`;
    if (hasClip) left += `<p><button id="load-clip">Voir l'extrait vidéo</button></p><div id="clip"></div>`;
  }
  const r = t.response;
  P.innerHTML = `<div class="card">
    <div class="qnav"><button id="qp">◀</button><span class="muted">${QI + 1} / ${QS.length} — ${KIND_FR[t.kind] || t.kind}${t.blocking ? ' — <b>bloquante</b>' : ''}</span><button id="qn">▶</button></div>
    <div class="q"><div>${left}</div><div>
      <h2>${esc(t.question || t.reason)}</h2>
      <div class="meta"><span>Raison</span><span>${esc(t.reason)}</span>
        <span>Chrono match</span><span>${esc(t.match_clock || 'inconnu')} <span class="muted">(${esc(t.clock_status || '—')})</span></span>
        <span>Temps fichier</span><span>${t.time != null ? Number(t.time).toFixed(2) + ' s (partie ' + (t.period ?? '?') + ')' : '—'}</span>
        ${t.segment ? `<span>Portée</span><span>tronçon ${t.segment[0].toFixed(1)}–${t.segment[1].toFixed(1)} s seulement</span>` : ''}
        ${r ? `<span>Réponse actuelle</span><span><b>${esc(r.label || r.value)}</b> ${r.player ? '· joueur ' + esc(r.player) : ''} <span class="muted">${esc(r.origin)} ${esc(r.reviewedAt)}</span></span>` : ''}
      </div>
      <div id="q-form"></div>
    </div></div></div>`;
  $('#qp').onclick = () => { QI = Math.max(0, QI - 1); renderQ(); };
  $('#qn').onclick = () => { QI = Math.min(QS.length - 1, QI + 1); renderQ(); };
  if ($('#load-clip')) $('#load-clip').onclick = () => { $('#clip').innerHTML = `<video src="/clip/${encodeURIComponent(t.id)}" controls autoplay muted playsinline></video>`; };
  renderForm(t);
}
function renderForm(t) {
  const F = $('#q-form');
  let sel = {value: null, label: ''};
  const choiceList = (t.kind === 'identity' || t.kind === 'roster') ? ((t.choices && t.choices.length) ? t.choices : ROSTER) : (t.choices || []);
  let html = '';
  if (t.kind === 'clock_read') html += `<label>Chrono lu (MM:SS)<input id="free" placeholder="27:41"></label>`;
  else if (t.kind !== 'calibration') html += '<div class="choices">' + choiceList.map((c, i) => `<button data-i="${i}">${esc(c.label)}</button>`).join('') + '</div>';
  if (t.player_choice) html += `<label>Joueur du Raja concerné<select id="player"><option value="">— aucun / adverse / inconnu —</option>${ROSTER.filter(c => c.jersey_number != null).map(c => `<option value="${esc(c.value)}">${esc(c.label)}</option>`).join('')}</select></label>
    <label>Issue (facultatif)<input id="outcome" placeholder="ex. capté, corner, hors cadre"></label>`;
  if (t.kind === 'calibration') html += `<label>Repère à placer<select id="lm">${(t.choices || []).map(c => `<option value="${esc(c.value)}">${esc(c.label)}</option>`).join('')}</select></label>
    <p class="muted">Choisissez un repère puis cliquez sur l'image. Au moins 4 points ; 5 ou plus permettent un contrôle « en laisser un ».</p><ol id="pts"></ol><button id="undo">Annuler le dernier point</button>`;
  html += `<label>Note (facultatif)<textarea id="note" rows="2"></textarea></label>
    <div class="actions"><button class="primary" id="ok">Enregistrer</button><button id="unk">Illisible / impossible à dire</button><button id="skip">Passer</button></div>`;
  F.innerHTML = html;
  F.querySelectorAll('.choices button').forEach(b => b.onclick = () => {
    F.querySelectorAll('.choices button').forEach(x => x.classList.remove('sel')); b.classList.add('sel');
    sel = choiceList[+b.dataset.i];
  });
  let points = [];
  if (t.kind === 'calibration') setupCalib(points);
  const send = async status => {
    const body = {status, note: $('#note').value};
    if (status === 'resolved') {
      if (t.kind === 'clock_read') { body.value = $('#free').value.trim(); body.label = body.value; }
      else if (t.kind === 'calibration') { if (points.length < 4) return alert('Au moins 4 repères.'); body.value = 'points'; body.points = points; body.label = points.length + ' repères'; }
      else { if (sel.value == null) return alert('Choisissez une réponse.'); body.value = sel.value; body.label = sel.label; }
      if (t.player_choice) { body.player = $('#player').value || null; body.outcome = $('#outcome').value; }
    }
    await post(`/api/questions/${encodeURIComponent(t.id)}/answer`, body);
    QS.splice(QI, 1); if (QI >= QS.length) QI = Math.max(0, QS.length - 1);
    if ($('#q-status').value !== 'pending') await loadQuestions(); else renderQ();
    refresh();
  };
  $('#ok').onclick = () => send('resolved');
  $('#unk').onclick = () => send('unknown');
  $('#skip').onclick = () => send('skipped');
}
function setupCalib(points) {
  const img = $('#cal-img'), cv = $('#cal-cv');
  const draw = () => {
    cv.width = img.clientWidth; cv.height = img.clientHeight; const c = cv.getContext('2d');
    c.clearRect(0, 0, cv.width, cv.height); c.font = '12px sans-serif';
    points.forEach((p, i) => { c.fillStyle = '#ff0'; c.beginPath(); c.arc(p.x * cv.width, p.y * cv.height, 4, 0, 7); c.fill(); c.fillText(String(i + 1), p.x * cv.width + 6, p.y * cv.height - 6); });
    $('#pts').innerHTML = points.map(p => `<li>${esc(p.landmark)} (${p.x.toFixed(3)}, ${p.y.toFixed(3)})</li>`).join('');
  };
  img.onload = draw; if (img.complete) draw();
  cv.onclick = e => { const r = cv.getBoundingClientRect(); points.push({landmark: $('#lm').value, x: (e.clientX - r.left) / r.width, y: (e.clientY - r.top) / r.height}); draw(); };
  $('#undo').onclick = () => { points.pop(); draw(); };
}

// ------------------------------------------------------------------ joueurs
const COLS = [['visible_wide_s', 'Visible large (s)'], ['visible_close_s', 'Gros plan (s)'], ['ball_near_s', 'Proche ballon (s)'],
  ['distance_observed_m', 'Distance obs. (m)'], ['vmax_kmh', 'Vmax (km/h)'], ['goals', 'Buts'], ['shots', 'Tirs'], ['shots_on_target', 'Cadrés'],
  ['passes_completed', 'Passes réussies'], ['shot_assists', 'Passes avant tir'], ['saves', 'Arrêts'], ['keeper_claims', 'Prises'],
  ['keeper_punches', 'Poings'], ['tackles', 'Tacles'], ['duels', 'Duels'], ['fouls_won', 'Fautes obt.'], ['yellow_cards', 'Jaunes'], ['sub_out_banner', 'Sortie (bandeau)'], ['sub_in_banner', 'Entrée (bandeau)'], ['xg', 'xG'], ['xa', 'xA'],
  ['pass_candidates', 'Passes candidates']];
let STATS = null;
$('#refresh-stats').onclick = () => loadStats(true);
async function loadStats(refreshIt) {
  STATS = await api('/api/stats' + (refreshIt ? '?refresh=1' : ''));
  const T = $('#t-players');
  if (!STATS.players || !STATS.players.length) { T.innerHTML = '<tr><td>Pas encore de statistiques : terminez au moins une partie.</td></tr>'; return; }
  const cov = STATS.coverage || {};
  $('#cov').innerHTML = '<div class="cov">' + (cov.sessions || []).map(s => `<div class="k">Partie ${s.period}<b>${Math.round(s.live_seconds)} s</b>de direct analysé · ballon localisé ${s.ball_located_fraction == null ? '—' : Math.round(100 * s.ball_located_fraction) + ' %'} · images attribuées ${s.attributed_frames ?? '—'} / ${s.raja_track_frames ?? '—'}</div>`).join('') +
    `<div class="k">Calibration<b>${cov.calibration_available ? 'partielle' : 'aucune'}</b>distances seulement sur plans calibrés</div></div>
    <p class="legend">Fiabilité : <span class="rel élevée"></span> élevée <span class="rel moyenne"></span> moyenne <span class="rel faible"></span> faible — règles dans la méthode de chaque valeur.</p>`;
  const used = COLS.filter(([k]) => STATS.players.some(p => p.metrics[k]));
  T.innerHTML = '<thead><tr><th>N°</th><th>Joueur</th>' + used.map(([, l]) => `<th>${l}</th>`).join('') + '</tr></thead><tbody>' +
    STATS.players.map(p => `<tr class="pl" data-n="${p.jersey_number}"><td>${p.jersey_number}</td><td>${esc(p.player_name)}</td>` +
      used.map(([k]) => { const m = p.metrics[k]; return m && m.value != null ? `<td class="num" title="${esc(m.method + ' — ' + m.reliability_reason)}">${esc(m.value)}<span class="rel ${esc(m.reliability)}"></span></td>` : '<td class="num muted">—</td>'; }).join('') + '</tr>').join('') + '</tbody>';
  T.querySelectorAll('tr.pl').forEach(tr => tr.onclick = () => detail(+tr.dataset.n));
}
function detail(n) {
  const p = STATS.players.find(x => x.jersey_number === n);
  const D = $('#player-detail');
  D.innerHTML = `<div class="card detail"><h2>${p.jersey_number} · ${esc(p.player_name)}</h2>
    <p class="muted">Composition : ${esc(p.participation_status)} — entrée ${p.declared_minutes.entered ?? '—'}′, sortie ${p.declared_minutes.left ?? '—'}′ (déclaré, non vérifié).</p>
    <h3>Mesures</h3>${Object.values(p.metrics).map(m => `<div class="ev"><b>${esc(m.label)}</b> : ${esc(m.value)} ${esc(m.unit === 'nb' ? '' : m.unit)} <span class="rel ${esc(m.reliability)}"></span> ${esc(m.reliability)} · <span class="muted">${esc(m.scope)}</span><br><span class="muted">${esc(m.method)} ${esc(m.reliability_reason)}</span>${(m.evidence || []).length ? '<br>' + m.evidence.map(e => esc((e.clock || '') + ' ' + (e.source || '') + ' ' + (e.clip || ''))).join(' · ') : ''}</div>`).join('') || '<p class="muted">Aucune mesure attribuée.</p>'}
    <h3>Preuves de suivi</h3>${(p.evidence || []).slice(0, 80).map(e => `<div class="ev">${esc(e.kind)} · partie ${e.period} · ${esc(e.clock_from || e.clock || '?')}${e.clock_to ? '–' + esc(e.clock_to) : ''} · ${esc(e.source || e.status || '')} ${e.reliability ? '· ' + esc(e.reliability) : ''}</div>`).join('') || '<p class="muted">—</p>'}</div>`;
  D.scrollIntoView({behavior: 'smooth'});
}

// ------------------------------------------------------------------ chronologie
async function loadTimeline() {
  const box = $('#tl'); box.innerHTML = '';
  if (!STATE) await refresh();
  for (const s of STATE.sessions) {
    let j; try { j = await api('/api/timeline/' + s.id); } catch (e) { continue; }
    if (!j.timeline) continue;
    const D = j.duration || j.timeline.duration_s;
    const segs = j.timeline.segments;
    const bar = segs.map(g => `<div class="k-${g.kind}" style="left:${100 * g.file_start / D}%;width:${Math.max(0.15, 100 * (g.file_end - g.file_start) / D)}%" title="${esc(SEG_FR[g.kind] || g.kind)} ${g.file_start.toFixed(1)}–${g.file_end.toFixed(1)} s"></div>`).join('');
    const disc = segs.filter(g => ['discontinuity', 'frozen'].includes(g.kind));
    const ev = j.events || {};
    box.insertAdjacentHTML('beforeend', `<div class="card"><h2>Partie ${s.period} — ${esc(s.video)}</h2>
      <div class="tlbar">${bar}</div><p class="legend">${Object.entries(SEG_FR).map(([k, v]) => `<span class="rel k-${k}" style="border-radius:2px"></span> ${v}`).join(' &nbsp; ')}</p>
      <h3>Ruptures</h3>${(j.timeline.hard_cuts || []).map(c => `<div class="ev">coupure franche à ${c.file_t.toFixed(1)} s · ${esc(c.clock_before)} → ${esc(c.clock_after)} · ${esc(c.note)}${c.match_seconds_missing ? ' (' + c.match_seconds_missing + ' s)' : ''}</div>`).join('')}${disc.map(g => `<div class="ev">${SEG_FR[g.kind]} (intervalle) : fichier ${g.file_start.toFixed(1)}–${g.file_end.toFixed(1)} s · ${esc(g.clock_before || '?')} → ${esc(g.clock_after || '?')} ${g.match_seconds_missing ? '· ' + g.match_seconds_missing + ' s de match non diffusées' : ''} ${esc(g.note || '')}</div>`).join('') || ''}
      <h3>Bandeaux et score lus automatiquement</h3>${(ev.banners || []).filter(b => b.duplicate_of == null).map(b => `<div class="ev">${esc(b.match_clock_banner_start || '?')} <span class="muted">(${esc(b.clock_status)})</span> · ${esc(EV_FR[b.type] || b.type)} · ${esc(b.team_tag || '')} · ${b.players.map(p => esc((p.jersey_number ?? p.jersey_number_read ?? '?') + ' ' + (p.player_name || p.name_read || ''))).join(', ')} ${b.needs_review ? '<b>à vérifier</b>' : ''}</div>`).join('') || '<p class="muted">—</p>'}
      ${(ev.goals || []).map(g => `<div class="ev">Score ${esc(g.score_after.join('-'))} lu à ${esc(g.match_clock_score_update || '?')} (${esc(g.team)})</div>`).join('')}</div>`);
  }
}

// ------------------------------------------------------------------ exports
async function loadExports() {
  const j = await api('/api/exports');
  $('#exp').innerHTML = `<h2>Fichiers produits</h2><ul>${j.files.map(f => `<li><a href="/download/${encodeURIComponent(f)}">${esc(f)}</a></li>`).join('')}</ul>
    <p class="muted">stats_par_joueur.csv (séparateur « ; », vide = inconnu), evenements.csv, stats_joueurs.json (complet, avec méthodes et preuves),
    import_portail_*.json (format de l'API WordPress Raja Insight).</p>
    <h2>Envoi au portail WordPress</h2>${j.portal_configured ? `<label>ID WordPress du match<input id="wpid" type="number"></label>
      <button id="push-a">Envoyer les métriques (analysis)</button> <span id="push-r" class="muted"></span>` :
      '<p class="muted">Non configuré : définir RI_WP_BASE_URL (https), RI_WP_USER et RI_WP_APP_PASSWORD avant de lancer l\'interface, ou utiliser engine_client.py avec les fichiers ci-dessus.</p>'}`;
  if ($('#push-a')) $('#push-a').onclick = async () => {
    try { const r = await post('/api/portal/analysis', {match_id: $('#wpid').value}); $('#push-r').textContent = JSON.stringify(r); }
    catch (e) { $('#push-r').textContent = e.message; }
  };
}
