// FRC Vision Scorer - Frontend Logic

class VisionScorer {
    constructor() {
        this.jobId = null;
        this.ws = null;
        this.canvas = document.getElementById('videoCanvas');
        this.ctx = this.canvas.getContext('2d');

        // Event/match state
        this.eventData = null;
        this.selectedMatch = null;
        this.inferenceResults = null;  // Final results from inference

        // UI elements
        this.eventSection = document.getElementById('eventSection');
        this.matchSetupSection = document.getElementById('matchSetupSection');
        this.videoSection = document.getElementById('videoSection');
        this.resultsSection = document.getElementById('resultsSection');
        this.statusBar = document.getElementById('statusBar');
        this.statusText = document.getElementById('statusText');
        this.progressBar = document.getElementById('progressBar');
        this.degradationBanner = document.getElementById('degradationBanner');
        this.degradationTitle = document.getElementById('degradationTitle');
        this.degradationList = document.getElementById('degradationList');
        this.statusLog = document.getElementById('statusLog');
        this.recoveryActions = document.getElementById('recoveryActions');
        this.degradationWarnings = [];
        this.lastWsMessageAt = 0;
        this.stallTimer = null;
        this.trackingPanel = document.getElementById('trackingPanel');
        this.trackingCards = document.getElementById('trackingCards');
        this.hudStartTime = document.getElementById('hudStartTime');
        this.hudElapsed = document.getElementById('hudElapsed');
        this.runStartedAt = null;
        this.elapsedTimer = null;
        this.viewCards = {};
        this.inferenceControls = document.getElementById('inferenceControls');
        this.startInferenceBtn = document.getElementById('startInferenceBtn');
        this.pauseInferenceBtn = document.getElementById('pauseInferenceBtn');
        this.resumeInferenceBtn = document.getElementById('resumeInferenceBtn');
        this.stopInferenceBtn = document.getElementById('stopInferenceBtn');

        // HUD elements
        this.hudFrame = document.getElementById('hudFrame');
        this.hudFps = document.getElementById('hudFps');
        this.hudReader = document.getElementById('hudReader');
        this.scoreBlue = document.getElementById('scoreBlue');
        this.scoreRed = document.getElementById('scoreRed');

        // Robot tables
        this.blueRobotsTable = document.getElementById('blueRobotsTable');
        this.redRobotsTable = document.getElementById('redRobotsTable');
        this.vlmCrops = document.getElementById('vlmCrops');
        this.latestVlmCrops = new Map();

        // Alliance score headers
        this.blueAllianceScore = document.getElementById('blueAllianceScore');
        this.redAllianceScore = document.getElementById('redAllianceScore');

        // Timeline
        this.timelineContainer = document.getElementById('timelineContainer');

        // Header job pill (both modes)
        this.jobPill = document.getElementById('jobPill');
        this._jobPollTimer = null;

        // Diagnostic viewer (debug-only)
        this.diagnosticToggleBtn = document.getElementById('diagnosticToggleBtn');
        this.diagnosticPanel = document.getElementById('diagnosticPanel');
        this.diagnosticFilter = document.getElementById('diagnosticFilter');
        this.diagnosticReloadBtn = document.getElementById('diagnosticReloadBtn');
        this.diagnosticStatus = document.getElementById('diagnosticStatus');
        this.diagnosticLines = document.getElementById('diagnosticLines');
        this._diagnosticRawLines = [];
        this._diagnosticTotalLines = 0;

        this.replayDiagnosticToggleBtn = document.getElementById('replayDiagnosticToggleBtn');
        this.replayDiagnosticPanel = document.getElementById('replayDiagnosticPanel');
        this.replayDiagnosticFilter = document.getElementById('replayDiagnosticFilter');
        this.replayDiagnosticReloadBtn = document.getElementById('replayDiagnosticReloadBtn');
        this.replayDiagnosticStatus = document.getElementById('replayDiagnosticStatus');
        this.replayDiagnosticLines = document.getElementById('replayDiagnosticLines');
        this._replayDiagnosticRawLines = [];
        this._replayDiagnosticTotalLines = 0;

        this.init();
    }

    init() {
        // UI mode (standard | debug). Standard is the default; the mode is a
        // pure CSS gate (see style.css) so WebSocket data keeps flowing into
        // hidden nodes and toggling works mid-run with no re-fetch.
        this.initModeToggle();

        // Event form
        document.getElementById('eventForm').addEventListener('submit', (e) => this.handleEventLoad(e));

        // Match selector
        document.getElementById('matchSelect').addEventListener('change', (e) => this.handleMatchSelect(e));

        // Upload form
        document.getElementById('uploadForm').addEventListener('submit', (e) => this.handleSubmit(e));
        this.populateSam2Models();
        this.populateVlmModels();
        this.initTeamReaderControls();
        this.rejoinActiveJob();
        this.startInferenceBtn.addEventListener('click', () => this.startInference());
        this.pauseInferenceBtn.addEventListener('click', () => this.pauseInference());
        this.resumeInferenceBtn.addEventListener('click', () => this.resumeInference());
        this.stopInferenceBtn.addEventListener('click', () => this.stopInference());

        // Save results
        document.getElementById('saveResultsBtn').addEventListener('click', () => this.saveResults());
        document.getElementById('newMatchBtn').addEventListener('click', () => this.resetToMatchSelect());

        // Navigation
        document.querySelectorAll('.nav-btn').forEach(btn => {
            btn.addEventListener('click', () => {
                if (btn.dataset.view) {
                    this.switchView(btn.dataset.view);
                }
            });
        });

        // Replays player
        this.replayVideo = document.getElementById('replayVideo');
        this.replayMeta = null;
        const replayBackBtn = document.getElementById('replayBackBtn');
        if (replayBackBtn) {
            replayBackBtn.addEventListener('click', () => {
                if (this.replayVideo) this.replayVideo.pause();
                this.showReplayLibrary();
                this.loadReplays();
            });
        }
        if (this.replayVideo) {
            this.replayVideo.addEventListener('timeupdate', () => this.onReplayTimeUpdate());
        }

        // Gallery
        const galleryLoadBtn = document.getElementById('galleryLoadBtn');
        if (galleryLoadBtn) galleryLoadBtn.addEventListener('click', () => this.loadGallery());

        // Settings
        const settingsResetBtn = document.getElementById('settingsResetBtn');
        if (settingsResetBtn) settingsResetBtn.addEventListener('click', () => this.resetAllSettings());
        const settingsFilter = document.getElementById('settingsFilter');
        if (settingsFilter) settingsFilter.addEventListener('input', () => this.filterSettings(settingsFilter.value));

        // Modal
        document.getElementById('modalClose').addEventListener('click', () => this.closeModal());
        document.getElementById('modalCancelBtn').addEventListener('click', () => this.closeModal());
        document.getElementById('modalSaveBtn').addEventListener('click', () => this.saveModalChanges());

        // Deep-link: #replays/#history/#gallery/#settings opens that view
        const hashView = window.location.hash.replace('#', '');
        if (['replays', 'history', 'gallery', 'settings'].includes(hashView)) {
            this.switchView(hashView);
        }

        // Check if we have an active event
        this.checkExistingEvent();

        // Header job pill (both modes)
        this.initJobPill();

        // Diagnostic viewer (debug-only)
        this.initDiagnosticViewer();
    }

    // --- UI mode (Standard / Debug) ---

    initModeToggle() {
        let mode = 'standard';
        try {
            const saved = localStorage.getItem('uiMode');
            if (saved === 'debug' || saved === 'standard') mode = saved;
        } catch (e) { /* localStorage unavailable — fall back to default */ }
        this.setMode(mode, false);
        const toggle = document.getElementById('modeToggle');
        if (toggle) {
            toggle.querySelectorAll('.mode-toggle-btn').forEach(btn => {
                btn.addEventListener('click', () => this.setMode(btn.dataset.mode, true));
            });
        }
    }

    setMode(mode, persist) {
        mode = mode === 'debug' ? 'debug' : 'standard';
        document.body.dataset.mode = mode;
        if (persist) {
            try { localStorage.setItem('uiMode', mode); } catch (e) { /* ignore */ }
        }
        // A debug-only view can be active when switching to standard; the CSS
        // hides its nav tab but the view div itself is toggled by switchView,
        // so kick the user back to the scorer.
        if (mode === 'standard') {
            const active = document.querySelector('.nav-btn.active');
            const activeView = active && active.dataset.view;
            if (activeView === 'gallery' || activeView === 'settings') {
                this.switchView('scorer');
            }
        }
    }

    // --- Navigation ---

    switchView(view, params) {
        document.querySelectorAll('.nav-btn').forEach(b => b.classList.remove('active'));
        document.querySelector(`.nav-btn[data-view="${view}"]`).classList.add('active');

        document.getElementById('scorerView').style.display = view === 'scorer' ? 'block' : 'none';
        document.getElementById('historyView').style.display = view === 'history' ? 'block' : 'none';
        document.getElementById('replaysView').style.display = view === 'replays' ? 'block' : 'none';
        document.getElementById('galleryView').style.display = view === 'gallery' ? 'block' : 'none';
        document.getElementById('settingsView').style.display = view === 'settings' ? 'block' : 'none';
        document.getElementById('calibrationView').style.display = view === 'calibration' ? 'block' : 'none';

        if (view === 'history') {
            this.loadHistory();
        } else if (view === 'replays') {
            // Leaving any open player when navigating back to the library
            if (this.replayVideo) this.replayVideo.pause();
            this.showReplayLibrary();
            this.loadReplays();
        } else if (view === 'gallery') {
            this.loadGallery();
        } else if (view === 'settings') {
            this.loadSettings();
        } else if (view === 'calibration') {
            // Lazy-load: don't let calibrate.html start grabbing frames /
            // hitting APIs until the tab is actually opened. A non-empty
            // params query string forces a reload (deep-link recovery flow
            // wants that specific match/camera targeted); otherwise leave an
            // already-loaded iframe alone so in-progress state isn't reset.
            const iframe = document.getElementById('calibrationFrame');
            if (params) {
                iframe.src = '/static/calibrate.html' + params;
            } else if (!iframe.src) {
                iframe.src = '/static/calibrate.html';
            }
        }

        this.renderJobPill();
    }

    // --- Settings (runtime knob overrides) ---

    async loadSettings() {
        const list = document.getElementById('settingsList');
        list.innerHTML = '<p class="empty-state">Loading…</p>';
        try {
            const resp = await fetch('/api/settings');
            if (!resp.ok) { list.innerHTML = '<p class="empty-state">Failed to load settings</p>'; return; }
            const { knobs } = await resp.json();
            list.innerHTML = '';
            // Group by section, preserving first-appearance order from the registry.
            const sections = new Map();
            knobs.forEach(k => {
                const name = k.section || 'misc';
                if (!sections.has(name)) sections.set(name, []);
                sections.get(name).push(k);
            });
            sections.forEach((rows, section) => {
                const header = document.createElement('div');
                header.className = 'settings-section-header';
                header.dataset.section = section;
                header.innerHTML = `<span class="settings-section-name">${section}</span>` +
                    `<span class="settings-section-count">${rows.length}</span>`;
                list.appendChild(header);
                rows.forEach(k => list.appendChild(this.settingRow(k)));
            });
            const filterInput = document.getElementById('settingsFilter');
            if (filterInput) this.filterSettings(filterInput.value);
        } catch (e) {
            list.innerHTML = '<p class="empty-state">Failed to load settings</p>';
        }
    }

    filterSettings(query) {
        const list = document.getElementById('settingsList');
        if (!list) return;
        const q = query.trim().toLowerCase();
        let currentHeader = null;
        let currentHeaderVisibleCount = 0;
        const flushHeader = () => {
            if (currentHeader) currentHeader.style.display = currentHeaderVisibleCount > 0 ? '' : 'none';
        };
        Array.from(list.children).forEach(el => {
            if (el.classList.contains('settings-section-header')) {
                flushHeader();
                currentHeader = el;
                currentHeaderVisibleCount = 0;
                return;
            }
            if (!el.classList.contains('setting-row')) return;
            const name = (el.dataset.name || '').toLowerCase();
            const env = (el.querySelector('.setting-env')?.textContent || '').toLowerCase();
            const visible = !q || name.includes(q) || env.includes(q);
            el.style.display = visible ? '' : 'none';
            if (visible) currentHeaderVisibleCount++;
        });
        flushHeader();
    }

    settingControl(k) {
        if (k.type === 'bool') {
            const cb = document.createElement('input');
            cb.type = 'checkbox';
            cb.checked = !!k.effective;
            cb.addEventListener('change', () => this.saveSetting(k.name, cb.checked));
            return cb;
        }
        if (k.choices) {
            const sel = document.createElement('select');
            k.choices.forEach(c => {
                const opt = document.createElement('option');
                opt.value = c; opt.textContent = c;
                if (c === k.effective) opt.selected = true;
                sel.appendChild(opt);
            });
            sel.addEventListener('change', () => this.saveSetting(k.name, sel.value));
            return sel;
        }
        const inp = document.createElement('input');
        if (k.type === 'int' || k.type === 'float') {
            inp.type = 'number';
            if (k.min != null) inp.min = k.min;
            if (k.max != null) inp.max = k.max;
            if (k.type === 'float') inp.step = 'any';
        } else {
            inp.type = 'text';
        }
        inp.value = k.effective;
        // Commit on change (Enter/blur), not per keystroke.
        inp.addEventListener('change', () => this.saveSetting(k.name, inp.value));
        return inp;
    }

    settingRow(k) {
        const row = document.createElement('div');
        row.className = 'setting-row';
        row.dataset.name = k.name;

        const label = document.createElement('div');
        label.className = 'setting-label';
        const nameEl = document.createElement('span');
        nameEl.className = 'setting-name';
        nameEl.textContent = k.name;
        const envEl = document.createElement('span');
        envEl.className = 'setting-env';
        envEl.textContent = k.env;
        label.appendChild(nameEl);
        label.appendChild(envEl);

        const control = document.createElement('div');
        control.className = 'setting-control';
        control.appendChild(this.settingControl(k));

        const meta = document.createElement('div');
        meta.className = 'setting-meta';
        const badge = document.createElement('span');
        badge.className = `setting-source setting-source-${k.source}`;
        badge.textContent = k.source === 'runtime'
            ? `overridden (launch: ${k.launch_value})` : k.source;
        meta.appendChild(badge);
        if (k.source === 'runtime') {
            const reset = document.createElement('button');
            reset.className = 'setting-reset';
            reset.textContent = 'reset';
            reset.title = `Back to launch value (${k.launch_value})`;
            reset.addEventListener('click', () => this.saveSetting(k.name, null));
            meta.appendChild(reset);
        }
        const err = document.createElement('span');
        err.className = 'setting-error';
        meta.appendChild(err);

        row.appendChild(label);
        row.appendChild(control);
        row.appendChild(meta);
        return row;
    }

    async saveSetting(name, value) {
        const row = document.querySelector(`.setting-row[data-name="${name}"]`);
        const errEl = row ? row.querySelector('.setting-error') : null;
        if (errEl) errEl.textContent = '';
        try {
            const resp = await fetch('/api/settings', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ name, value }),
            });
            const body = await resp.json();
            if (!resp.ok) {
                if (errEl) errEl.textContent = body.error || `HTTP ${resp.status}`;
                return;
            }
            if (row) {
                row.replaceWith(this.settingRow(body.knob));
                // The fresh row ignores any active filter; re-apply it so the
                // saved row doesn't pop back into a filtered-out section (and
                // section header visible-counts stay correct).
                const filterInput = document.getElementById('settingsFilter');
                if (filterInput) this.filterSettings(filterInput.value);
            }
        } catch (e) {
            if (errEl) errEl.textContent = 'request failed';
        }
    }

    async resetAllSettings() {
        try {
            await fetch('/api/settings/reset', { method: 'POST' });
        } finally {
            this.loadSettings();
        }
    }

    // --- Gallery Manager ---

    async loadGallery() {
        const key = document.getElementById('galleryEventKey').value.trim();
        const grid = document.getElementById('galleryGrid');
        if (!key) { grid.innerHTML = '<p class="empty-state">Enter an event key</p>'; return; }
        try {
            const resp = await fetch(`/api/gallery?event_key=${encodeURIComponent(key)}`);
            if (!resp.ok) { grid.innerHTML = '<p class="empty-state">Failed to load gallery</p>'; return; }
            const data = await resp.json();
            const ex = data.exemplars || [];
            if (ex.length === 0) { grid.innerHTML = '<p class="empty-state">No exemplars</p>'; return; }

            // Group exemplars by team number so the manager reads as "this team's
            // appearances", and sort numerically (blank/unknown team last).
            const groups = new Map();
            for (const e of ex) {
                const t = (e.team || '').trim() || '—';
                if (!groups.has(t)) groups.set(t, []);
                groups.get(t).push(e);
            }
            const teams = [...groups.keys()].sort((a, b) => {
                if (a === '—') return 1;
                if (b === '—') return -1;
                return (parseInt(a, 10) || 1e9) - (parseInt(b, 10) || 1e9);
            });

            // One collapsible accordion row per team — scales to the 30-75 teams
            // an event accumulates. Rows are collapsed by default; expand to edit.
            const rows = teams.map(team => {
                const items = groups.get(team);
                const color = items[0].alliance_color;
                const verified = items.filter(e => e.verified).length;
                const thumb = (items.find(e => e.image) || {}).image;
                const cards = items.map(e => `
                    <div class="review-robot" data-id="${e.id}">
                        <div class="review-imgs">${e.image ? `<img class="review-crop" src="${this.escapeHtml(e.image)}">` : '<span class="empty-state">no image</span>'}</div>
                        <div class="review-meta">
                            ${this.galleryTeamField(e)}
                            <select class="g-color"><option value="blue"${e.alliance_color==='blue'?' selected':''}>blue</option><option value="red"${e.alliance_color==='red'?' selected':''}>red</option></select>
                            <label class="g-verify-label"><input type="checkbox" class="g-verified"${e.verified?' checked':''}> verified</label>
                            <div class="g-actions">
                                <button class="g-save" data-id="${e.id}">Save</button>
                                <button class="g-del" data-id="${e.id}" title="Delete exemplar">✕</button>
                            </div>
                        </div>
                    </div>`).join('');
                return `
                    <section class="gallery-group team-${this.escapeHtml(color || 'unknown')}" data-team="${this.escapeHtml(team)}">
                        <button class="gallery-group-head" type="button" aria-expanded="false">
                            <span class="gallery-chevron">▸</span>
                            ${thumb ? `<img class="gallery-head-thumb" src="${this.escapeHtml(thumb)}">` : '<span class="gallery-head-thumb gallery-head-thumb--empty"></span>'}
                            <span class="gallery-team-num">${this.escapeHtml(team)}</span>
                            <span class="gallery-team-count" data-count><span class="gallery-count-n">${items.length}</span> exemplar${items.length === 1 ? '' : 's'}${verified ? ` · ${verified} verified` : ''}</span>
                        </button>
                        <div class="review-grid gallery-body" hidden>${cards}</div>
                    </section>`;
            }).join('');
            grid.innerHTML = `
                <div class="gallery-toolbar">
                    <button class="btn-small" id="galleryExpandAll">Expand all</button>
                    <button class="btn-small" id="galleryCollapseAll">Collapse all</button>
                    <span class="gallery-toolbar-count">${teams.length} team${teams.length === 1 ? '' : 's'}</span>
                    <button class="btn-small gallery-clear-all" id="galleryClearAll">Clear all</button>
                </div>
                ${rows}`;

            grid.querySelectorAll('.gallery-group-head').forEach(h => h.addEventListener('click', () => this.toggleGalleryGroup(h)));
            grid.querySelector('#galleryExpandAll').addEventListener('click', () => this.setAllGalleryGroups(true));
            grid.querySelector('#galleryCollapseAll').addEventListener('click', () => this.setAllGalleryGroups(false));
            grid.querySelector('#galleryClearAll').addEventListener('click', () => this.clearGallery());
            grid.querySelectorAll('.g-save').forEach(b => b.addEventListener('click', () => this.saveGalleryExemplar(b)));
            grid.querySelectorAll('.g-del').forEach(b => b.addEventListener('click', () => this.deleteGalleryExemplar(b)));
        } catch (e) {
            grid.innerHTML = '<p class="empty-state">Failed to load gallery</p>';
        }
    }

    toggleGalleryGroup(head, force) {
        const section = head.closest('.gallery-group');
        const body = section.querySelector('.gallery-body');
        const open = force === undefined ? body.hidden : force;
        body.hidden = !open;
        section.classList.toggle('open', open);
        head.setAttribute('aria-expanded', String(open));
    }

    setAllGalleryGroups(open) {
        document.querySelectorAll('#galleryGrid .gallery-group-head')
            .forEach(h => this.toggleGalleryGroup(h, open));
    }

    // Team field: a dropdown of the 6 robots from the match this exemplar was
    // pulled from. Falls back to a free text input for legacy exemplars that
    // predate source-match tracking (no roster available).
    galleryTeamField(e) {
        const roster = (e.match_teams || []).map(String);
        if (roster.length === 0) {
            return `<input class="g-team" type="text" value="${this.escapeHtml(e.team || '')}">`;
        }
        const opts = roster.slice();
        const cur = String(e.team || '');
        if (cur && !opts.includes(cur)) opts.unshift(cur);  // never lose the current label
        return `<select class="g-team">${opts.map(t =>
            `<option value="${this.escapeHtml(t)}"${t === cur ? ' selected' : ''}>${this.escapeHtml(t)}</option>`
        ).join('')}</select>`;
    }

    async saveGalleryExemplar(btn) {
        const wrap = btn.closest('.review-robot');
        const id = Number(btn.dataset.id);
        await fetch(`/api/gallery/${id}`, {
            method: 'POST', headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                team: wrap.querySelector('.g-team').value.trim(),
                alliance_color: wrap.querySelector('.g-color').value,
                verified: wrap.querySelector('.g-verified').checked,
            }),
        });
        await this.loadGallery();
    }

    async deleteGalleryExemplar(btn) {
        // No confirmation: exemplars are expected to be pruned frequently, and a
        // re-run of the match regenerates them. Optimistically drop the card so
        // bulk cleanup feels instant, then sync with the server.
        const id = Number(btn.dataset.id);
        const card = btn.closest('.review-robot');
        const section = btn.closest('.gallery-group');
        if (card) card.remove();
        // Keep the accordion header count honest; drop the team row if now empty.
        if (section) {
            const remaining = section.querySelectorAll('.review-robot').length;
            if (remaining === 0) {
                section.remove();
            } else {
                const n = section.querySelector('.gallery-count-n');
                if (n) n.textContent = remaining;
            }
        }
        await fetch(`/api/gallery/${id}`, { method: 'DELETE' });
    }

    async clearGallery() {
        // Wipe every exemplar for the loaded event. Confirmed because it's the
        // whole gallery at once (unlike single-card deletes); the gallery
        // regenerates on the next run, so this is a troubleshooting reset.
        const key = document.getElementById('galleryEventKey').value.trim();
        if (!key) return;
        const grid = document.getElementById('galleryGrid');
        const n = grid.querySelectorAll('.review-robot').length;
        if (!confirm(`Delete all ${n} gallery exemplar${n === 1 ? '' : 's'} for event "${key}"? They regenerate on the next run.`)) return;
        try {
            const resp = await fetch(`/api/gallery?event_key=${encodeURIComponent(key)}`, { method: 'DELETE' });
            if (!resp.ok) { alert('Failed to clear gallery'); return; }
            this.loadGallery();   // reload -> "No exemplars"
        } catch (e) {
            alert('Failed to clear gallery');
        }
    }

    // --- Event Loading ---

    async checkExistingEvent() {
        try {
            const resp = await fetch('/api/event/current');
            if (resp.ok) {
                const data = await resp.json();
                if (data.error) return; // No event loaded
                this.eventData = data;
                // Always go to match setup — calibration happens per-match before inference
                this.showMatchSetup();
            }
        } catch (e) {
            // No event loaded, show event form
        }
    }

    async handleEventLoad(e) {
        e.preventDefault();
        const url = document.getElementById('eventUrl').value.trim();
        if (!url) return;

        const statusEl = document.getElementById('eventStatus');
        statusEl.style.display = 'block';
        statusEl.textContent = 'Loading event data from The Blue Alliance...';
        statusEl.className = 'event-status loading';

        try {
            const resp = await fetch('/api/event/load', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ url }),
            });
            const data = await resp.json();

            if (data.error) {
                statusEl.textContent = `Error: ${data.error}`;
                statusEl.className = 'event-status error';
                return;
            }

            this.eventData = data;
            statusEl.style.display = 'none';
            
            // Always go to match setup — calibration happens per-match before inference
            this.showMatchSetup();
        } catch (err) {
            statusEl.textContent = `Error: ${err.message}`;
            statusEl.className = 'event-status error';
        }
    }

    // --- OCR Region Setup ---

    showOcrSetup(videoUrl) {
        this.eventSection.style.display = 'none';
        this.matchSetupSection.style.display = 'none';
        document.getElementById('ocrSetupSection').style.display = 'block';
        
        const banner = document.getElementById('ocrEventBanner');
        banner.innerHTML = `<strong>${this.eventData.event_name}</strong> — Scoreboard Calibration`;
        
        // Bind events (use onclick to avoid stacking listeners)
        document.getElementById('ocrMarkRedHub').onclick = () => this.startHubMark('red');
        document.getElementById('ocrMarkBlueHub').onclick = () => this.startHubMark('blue');
        document.getElementById('ocrDrawRed').onclick = () => this.startOcrDraw('red');
        document.getElementById('ocrDrawBlue').onclick = () => this.startOcrDraw('blue');
        document.getElementById('ocrDrawClock').onclick = () => this.startOcrDraw('clock');
        document.getElementById('ocrSaveBtn').onclick = () => this.saveOcrRegions();
        document.getElementById('ocrSkipBtn').onclick = () => {
            document.getElementById('ocrSetupSection').style.display = 'none';
            if (this._postCalibrationTarget === 'inference') {
                this._postCalibrationTarget = null;
                this._showInferenceView();
            } else {
                this.showMatchSetup();
            }
        };
        
        this.ocrRegions = { red_center: null, blue_center: null, match_clock: null };
        this.hubPositions = { blue: null, red: null };
        this.ocrDrawMode = null;  // 'red', 'blue' for score regions
        this.hubMarkMode = null;  // 'red', 'blue' for hub clicks
        this.ocrFrameWidth = 0;
        this.ocrFrameHeight = 0;
        
        // Grab a frame — use the specific match video URL if provided
        this._calibrationVideoUrl = videoUrl || null;
        this.grabOcrFrame();
    }

    async grabOcrFrame() {
        const loading = document.getElementById('ocrLoading');
        const status = document.getElementById('ocrStatus');
        loading.style.display = 'block';
        status.textContent = '';
        
        try {
            // Use specific match video URL if available, otherwise backend picks one
            const grabBody = { timestamp: 60 };
            if (this._calibrationVideoUrl) {
                grabBody.url = this._calibrationVideoUrl;
            }
            const resp = await fetch('/api/grab-frame', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(grabBody)
            });
            const data = await resp.json();
            
            if (data.error) {
                loading.style.display = 'none';
                status.textContent = `Error: ${data.error}`;
                return;
            }
            
            this.ocrFrameWidth = data.width;
            this.ocrFrameHeight = data.height;
            
            loading.style.display = 'none';
            const container = document.getElementById('ocrCanvasContainer');
            container.style.display = 'block';
            
            const canvas = document.getElementById('ocrCanvas');
            const img = new Image();
            img.onload = () => {
                // Scale canvas to fit container (max 960px wide)
                const maxWidth = Math.min(960, container.clientWidth - 20);
                const scale = maxWidth / img.width;
                canvas.width = maxWidth;
                canvas.height = img.height * scale;
                this.ocrScale = scale;
                this.ocrImage = img;
                
                const ctx = canvas.getContext('2d');
                ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
                
                status.textContent = 'Tap a button below, then draw a box around that score on the image.';
                document.getElementById('ocrDrawControls').style.display = 'flex';
            };
            img.onerror = () => {
                loading.style.display = 'none';
                status.textContent = 'Error: Failed to load frame image. Try refreshing the page.';
            };
            img.src = 'data:image/jpeg;base64,' + data.frame;
            
            // Set up drawing handlers (mouse + touch)
            canvas.onmousedown = (e) => this.ocrMouseDown(e);
            canvas.onmousemove = (e) => this.ocrMouseMove(e);
            canvas.onmouseup = (e) => this.ocrMouseUp(e);
            
            // Touch support for mobile
            canvas.ontouchstart = (e) => { e.preventDefault(); this.ocrMouseDown(e.touches[0]); };
            canvas.ontouchmove = (e) => { e.preventDefault(); this.ocrMouseMove(e.touches[0]); };
            canvas.ontouchend = (e) => { e.preventDefault(); this.ocrMouseUp(e.changedTouches[0]); };
            
        } catch (err) {
            console.error('grabOcrFrame error:', err);
            loading.style.display = 'none';
            status.textContent = `Error: ${err.message}`;
        }
    }

    startHubMark(alliance) {
        this.hubMarkMode = alliance;
        this.ocrDrawMode = null;
        const canvas = document.getElementById('ocrCanvas');
        canvas.style.cursor = 'crosshair';
        document.getElementById('ocrStatus').textContent = 
            `Click on the ${alliance.toUpperCase()} hub on the field.`;
        
        // Highlight active button
        document.getElementById('ocrMarkRedHub').classList.toggle('active', alliance === 'red');
        document.getElementById('ocrMarkBlueHub').classList.toggle('active', alliance === 'blue');
        document.getElementById('ocrDrawRed').classList.remove('active');
        document.getElementById('ocrDrawBlue').classList.remove('active');
        document.getElementById('ocrDrawClock').classList.remove('active');
    }

    startOcrDraw(kind) {
        this.ocrDrawMode = kind;
        this.hubMarkMode = null;
        const canvas = document.getElementById('ocrCanvas');
        canvas.style.cursor = 'crosshair';
        document.getElementById('ocrStatus').textContent = 
            kind === 'clock'
                ? 'Draw a box around the match clock.'
                : `Draw a box around the ${kind.toUpperCase()} alliance score number.`;
        
        // Highlight active button
        document.getElementById('ocrDrawRed').classList.toggle('active', kind === 'red');
        document.getElementById('ocrDrawBlue').classList.toggle('active', kind === 'blue');
        document.getElementById('ocrDrawClock').classList.toggle('active', kind === 'clock');
        document.getElementById('ocrMarkRedHub').classList.remove('active');
        document.getElementById('ocrMarkBlueHub').classList.remove('active');
    }

    ocrMouseDown(e) {
        const canvas = document.getElementById('ocrCanvas');
        const rect = canvas.getBoundingClientRect();
        const cx = e.clientX - rect.left;
        const cy = e.clientY - rect.top;
        
        // Hub click-to-place mode
        if (this.hubMarkMode) {
            const scale = this.ocrScale;
            const rx = cx / scale;
            const ry = cy / scale;
            this.hubPositions[this.hubMarkMode] = [Math.round(rx), Math.round(ry)];
            
            this.hubMarkMode = null;
            canvas.style.cursor = 'default';
            document.getElementById('ocrMarkRedHub').classList.remove('active');
            document.getElementById('ocrMarkBlueHub').classList.remove('active');
            
            this.redrawOcrCanvas();
            this.checkSetupComplete();
            return;
        }
        
        // Score region draw mode
        if (!this.ocrDrawMode) return;
        this.ocrDragStart = { x: cx, y: cy };
        this.ocrDragging = true;
    }

    ocrMouseMove(e) {
        if (!this.ocrDragging || !this.ocrDrawMode) return;
        const canvas = document.getElementById('ocrCanvas');
        const rect = canvas.getBoundingClientRect();
        const x = e.clientX - rect.left;
        const y = e.clientY - rect.top;
        
        // Redraw image and all regions
        this.redrawOcrCanvas();
        
        // Draw current selection
        const ctx = canvas.getContext('2d');
        ctx.strokeStyle = this.ocrDrawMode === 'red' ? '#e74c3c'
            : this.ocrDrawMode === 'blue' ? '#3498db' : '#f1c40f';
        ctx.lineWidth = 2;
        ctx.setLineDash([5, 3]);
        ctx.strokeRect(this.ocrDragStart.x, this.ocrDragStart.y, 
                       x - this.ocrDragStart.x, y - this.ocrDragStart.y);
        ctx.setLineDash([]);
    }

    ocrMouseUp(e) {
        if (!this.ocrDragging || !this.ocrDrawMode) return;
        this.ocrDragging = false;
        
        const canvas = document.getElementById('ocrCanvas');
        const rect = canvas.getBoundingClientRect();
        const x = e.clientX - rect.left;
        const y = e.clientY - rect.top;
        
        // Calculate region in original frame coordinates
        const scale = this.ocrScale;
        const rx = Math.min(this.ocrDragStart.x, x) / scale;
        const ry = Math.min(this.ocrDragStart.y, y) / scale;
        const rw = Math.abs(x - this.ocrDragStart.x) / scale;
        const rh = Math.abs(y - this.ocrDragStart.y) / scale;
        
        if (rw < 5 || rh < 5) return; // Too small
        
        const key = this.ocrDrawMode === 'clock'
            ? 'match_clock'
            : this.ocrDrawMode + '_center';
        this.ocrRegions[key] = [Math.round(rx), Math.round(ry), Math.round(rw), Math.round(rh)];
        
        this.ocrDrawMode = null;
        canvas.style.cursor = 'default';
        document.getElementById('ocrDrawRed').classList.remove('active');
        document.getElementById('ocrDrawBlue').classList.remove('active');
        document.getElementById('ocrDrawClock').classList.remove('active');
        
        this.redrawOcrCanvas();
        this.checkSetupComplete();
    }

    checkSetupComplete() {
        const hasHubs = this.hubPositions.blue && this.hubPositions.red;
        const hasRegions = this.ocrRegions.red_center && this.ocrRegions.blue_center;
        const canSave = hasHubs && hasRegions;
        document.getElementById('ocrSaveBtn').disabled = !canSave;
        
        if (canSave) {
            document.getElementById('ocrStatus').textContent = 'All set. Click Save & Continue.';
        } else {
            const missing = [];
            if (!this.hubPositions.blue) missing.push('BLUE hub');
            if (!this.hubPositions.red) missing.push('RED hub');
            if (!this.ocrRegions.red_center) missing.push('RED score region');
            if (!this.ocrRegions.blue_center) missing.push('BLUE score region');
            document.getElementById('ocrStatus').textContent = `Still need: ${missing.join(', ')}`;
        }
    }

    redrawOcrCanvas() {
        const canvas = document.getElementById('ocrCanvas');
        const ctx = canvas.getContext('2d');
        ctx.drawImage(this.ocrImage, 0, 0, canvas.width, canvas.height);
        const s = this.ocrScale;
        
        // Draw hub positions (circles with crosshairs)
        for (const [alliance, pos] of Object.entries(this.hubPositions)) {
            if (!pos) continue;
            const color = alliance === 'red' ? '#e74c3c' : '#3498db';
            const cx = pos[0] * s;
            const cy = pos[1] * s;
            
            ctx.strokeStyle = color;
            ctx.lineWidth = 3;
            ctx.beginPath();
            ctx.arc(cx, cy, 20, 0, Math.PI * 2);
            ctx.stroke();
            
            // Crosshair
            ctx.beginPath();
            ctx.moveTo(cx - 12, cy); ctx.lineTo(cx + 12, cy);
            ctx.moveTo(cx, cy - 12); ctx.lineTo(cx, cy + 12);
            ctx.stroke();
            
            ctx.fillStyle = color;
            ctx.font = 'bold 12px sans-serif';
            ctx.fillText(`${alliance.toUpperCase()} HUB`, cx - 25, cy - 25);
        }
        
        // Draw saved score regions
        for (const [key, region] of Object.entries(this.ocrRegions)) {
            if (!region) continue;
            const color = key.startsWith('red') ? '#e74c3c'
                : key.startsWith('blue') ? '#3498db' : '#f1c40f';
            const label = key.startsWith('red') ? 'RED SCORE'
                : key.startsWith('blue') ? 'BLUE SCORE' : 'MATCH CLOCK';
            const [rx, ry, rw, rh] = region;
            
            ctx.strokeStyle = color;
            ctx.lineWidth = 3;
            ctx.strokeRect(rx * s, ry * s, rw * s, rh * s);
            
            ctx.fillStyle = color;
            ctx.font = 'bold 12px sans-serif';
            ctx.fillText(label, rx * s, ry * s - 4);
        }
    }

    async saveOcrRegions() {
        try {
            // Save OCR score regions
            const ocrResp = await fetch('/api/ocr-regions', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(this.ocrRegions)
            });
            const ocrData = await ocrResp.json();
            
            // Save hub positions
            const hubResp = await fetch('/api/hub-positions', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(this.hubPositions)
            });
            const hubData = await hubResp.json();
            
            if (ocrData.status === 'ok' && hubData.status === 'ok') {
                document.getElementById('ocrSetupSection').style.display = 'none';
                if (this._postCalibrationTarget === 'inference') {
                    this._postCalibrationTarget = null;
                    this._showInferenceView();
                } else {
                    this.showMatchSetup();
                }
            }
        } catch (err) {
            document.getElementById('ocrStatus').textContent = `Error saving: ${err.message}`;
        }
    }

    showMatchSetup() {
        this.eventSection.style.display = 'none';
        this.matchSetupSection.style.display = 'block';

        // Event banner
        const banner = document.getElementById('eventBanner');
        banner.innerHTML = `
            <strong>${this.eventData.event_name}</strong>
            <span class="event-meta">${this.eventData.team_count || Object.keys(this.eventData.teams).length} teams &middot; ${this.eventData.match_count || this.eventData.matches.length} matches</span>
            <button class="btn-small" id="recalibrateBtn">Recalibrate Scoreboard</button>
            <button class="btn-small" id="changeEventBtn">Change Event</button>
        `;
        document.getElementById('recalibrateBtn').addEventListener('click', () => {
            this._postCalibrationTarget = null;
            this.matchSetupSection.style.display = 'none';
            this.showOcrSetup();
        });
        document.getElementById('changeEventBtn').addEventListener('click', () => {
            this.eventData = null;
            this.matchSetupSection.style.display = 'none';
            this.eventSection.style.display = 'block';
        });

        // Populate match dropdown — grouped by comp level, split by day within a level
        const select = document.getElementById('matchSelect');
        select.innerHTML = '<option value="">-- Choose a match --</option>';

        const tz = this.eventData.timezone;
        const calendarDate = (match) => {
            if (!match.time) return null;
            const opts = { year: 'numeric', month: '2-digit', day: '2-digit' };
            if (tz) opts.timeZone = tz;
            // en-CA locale formats as YYYY-MM-DD, so dates sort chronologically as strings
            return new Intl.DateTimeFormat('en-CA', opts).format(new Date(match.time * 1000));
        };
        const prettyDate = (match) => {
            const opts = { weekday: 'short', month: 'short', day: 'numeric' };
            if (tz) opts.timeZone = tz;
            return new Intl.DateTimeFormat('en-US', opts).format(new Date(match.time * 1000));
        };

        // Number the event's distinct match days chronologically: Day 1, Day 2, ...
        const dayNumbers = new Map();
        const allDates = [...new Set(this.eventData.matches.map(calendarDate).filter(Boolean))].sort();
        for (const d of allDates) dayNumbers.set(d, dayNumbers.size + 1);

        let currentGroupKey = null;
        let currentGroup = null;
        for (const match of this.eventData.matches) {
            const levelLabel = { qm: 'Qualifications', sf: 'Semifinals', f: 'Finals' }[match.comp_level] || match.comp_level;
            const date = calendarDate(match);
            const groupKey = `${match.comp_level}|${date || ''}`;
            if (groupKey !== currentGroupKey) {
                currentGroup = document.createElement('optgroup');
                currentGroup.label = date
                    ? `${levelLabel} — Day ${dayNumbers.get(date)} (${prettyDate(match)})`
                    : levelLabel;
                select.appendChild(currentGroup);
                currentGroupKey = groupKey;
            }

            const opt = document.createElement('option');
            opt.value = match.key;

            const blueTeams = match.blue_teams.join(', ');
            const redTeams = match.red_teams.join(', ');
            const scoreStr = match.blue_score != null ? ` [${match.blue_score}-${match.red_score}]` : '';
            opt.textContent = `${match.label} — Blue: ${blueTeams} vs Red: ${redTeams}${scoreStr}`;
            opt.dataset.matchData = JSON.stringify(match);

            currentGroup.appendChild(opt);
        }
    }

    handleMatchSelect(e) {
        const matchKey = e.target.value;
        const teamsDiv = document.getElementById('matchTeams');
        const uploadArea = document.getElementById('uploadArea');

        if (!matchKey) {
            teamsDiv.style.display = 'none';
            uploadArea.style.display = 'none';
            this.selectedMatch = null;
            return;
        }

        // Find match data
        this.selectedMatch = this.eventData.matches.find(m => m.key === matchKey);
        if (!this.selectedMatch) return;

        // Show teams
        teamsDiv.style.display = 'flex';
        teamsDiv.innerHTML = `
            <div class="match-alliance blue">
                <h4>Blue Alliance</h4>
                <div class="team-list">
                    ${this.selectedMatch.blue_teams.map((t, i) => {
                        const team = this.eventData.teams[t];
                        const name = team ? team.name : `Team ${t}`;
                        return `<span class="team-chip blue">${t} <small>${name}</small></span>`;
                    }).join('')}
                </div>
            </div>
            <div class="match-alliance red">
                <h4>Red Alliance</h4>
                <div class="team-list">
                    ${this.selectedMatch.red_teams.map((t, i) => {
                        const team = this.eventData.teams[t];
                        const name = team ? team.name : `Team ${t}`;
                        return `<span class="team-chip red">${t} <small>${name}</small></span>`;
                    }).join('')}
                </div>
            </div>
        `;

        // Pre-fill YouTube URL if TBA has a video
        if (this.selectedMatch.video_key) {
            document.getElementById('youtubeUrl').value = `https://www.youtube.com/watch?v=${this.selectedMatch.video_key}`;
        } else {
            document.getElementById('youtubeUrl').value = '';
        }

        uploadArea.style.display = 'block';
    }

    // --- Video Inference (existing logic) ---

    async populateSam2Models() {
        const select = document.getElementById('sam2ModelSize');
        const backendSelect = document.getElementById('trackingBackend');
        if (!select || !backendSelect) return;
        try {
            const resp = await fetch('/api/sam2_models');
            const data = await resp.json();
            backendSelect.replaceChildren();
            for (const backend of data.backends || []) {
                const opt = document.createElement('option');
                opt.value = backend.id;
                opt.textContent = `${backend.label} (${backend.detail})`;
                if (backend.id === data.default_backend) opt.selected = true;
                backendSelect.appendChild(opt);
            }
            select.replaceChildren();
            for (const m of data.models) {
                const opt = document.createElement('option');
                opt.value = m.size;
                const tag = m.available ? '' : '  — needs download';
                opt.textContent = `${m.label} (${m.detail})${tag}`;
                opt.disabled = !m.available;
                if (m.size === data.default && m.available) opt.selected = true;
                select.appendChild(opt);
            }
            // If the env default isn't downloaded, fall back to the first available size.
            if (select.selectedOptions.length === 0 || select.selectedOptions[0].disabled) {
                const firstAvailable = Array.from(select.options).find((o) => !o.disabled);
                if (firstAvailable) firstAvailable.selected = true;
            }
            // Team-number reader is server-chosen (vllm default with
            // in-process fallback); no UI selection remains.
            // Show only the controls that apply to the selected backend.
            const sam3Group = document.getElementById('sam3ChunkGroup');
            const sam2Groups = [
                document.getElementById('sam2ModelGroup'),
                document.getElementById('sam2VramGroup'),
            ];
            const syncModelControls = () => {
                const isSam2 = backendSelect.value === 'sam2';
                select.disabled = !isSam2;
                if (sam3Group) sam3Group.style.display = isSam2 ? 'none' : '';
                for (const g of sam2Groups) {
                    if (g) g.style.display = isSam2 ? '' : 'none';
                }
            };
            backendSelect.addEventListener('change', syncModelControls);
            syncModelControls();
        } catch (err) {
            console.error('Failed to load tracker model list:', err);
        }
    }

    async populateVlmModels() {
        const select = document.getElementById('vlmModel');
        if (!select) return;
        try {
            const resp = await fetch('/api/vlm_models');
            const data = await resp.json();
            select.replaceChildren();
            for (const m of data.models || []) {
                const opt = document.createElement('option');
                opt.value = m.key;
                // needs_download models stay selectable (vLLM fetches on first
                // serve); annotate the size hint when we have one.
                let tag = '';
                if (m.status === 'needs_download') {
                    const gb = m.approx_gb ? ` (~${m.approx_gb} GB)` : '';
                    tag = ` — needs download${gb}`;
                }
                opt.textContent = `${m.label}${tag}`;
                if (m.notes) opt.title = m.notes;
                if (m.key === data.default) opt.selected = true;
                select.appendChild(opt);
            }
            this._defaultVlmModel = data.default;
        } catch (err) {
            console.error('Failed to load VLM model list:', err);
        }
    }

    initTeamReaderControls() {
        // Debug-only reader backend picker. Clef runs in the cloud, so the
        // vLLM model picker doesn't apply to it; show the Clef model instead.
        const reader = document.getElementById('teamReader');
        if (!reader) return;
        const sync = () => {
            const isClef = reader.value === 'clef';
            const clefGroup = document.getElementById('clefModelGroup');
            const vlmGroup = document.getElementById('vlmModelGroup');
            if (clefGroup) clefGroup.style.display = isClef ? '' : 'none';
            if (vlmGroup) vlmGroup.style.display = isClef ? 'none' : '';
        };
        reader.addEventListener('change', sync);
        sync();
    }

    async handleSubmit(e) {
        e.preventDefault();

        const videoFile = document.getElementById('videoFile').files[0];
        const youtubeUrl = document.getElementById('youtubeUrl').value.trim();

        if (!videoFile && !youtubeUrl) {
            alert('Please select a video file or enter a YouTube URL');
            return;
        }

        const formData = new FormData();
        if (videoFile) {
            formData.append('video_file', videoFile);
        } else if (youtubeUrl) {
            formData.append('youtube_url', youtubeUrl);
        }
        // team_reader is only sent (via config_json below) when the debug
        // picker selects Clef; otherwise the server default (vllm via
        // start_all.sh) applies, with automatic in-process fallback.
        const vramBudget = parseInt(document.getElementById('sam2MaxGpuFrames')?.value, 10);
        if (Number.isFinite(vramBudget) && vramBudget > 0) {
            formData.append('sam2_max_gpu_frames', vramBudget);
        }
        const trackingBackend = document.getElementById('trackingBackend')?.value;
        if (trackingBackend) {
            formData.append('tracking_backend', trackingBackend);
        }
        const modelSize = document.getElementById('sam2ModelSize')?.value;
        if (modelSize) {
            formData.append('model_size', modelSize);
        }
        // Accumulate per-job knobs into one config_json object (validated
        // server-side against the JobConfig registry).
        const config = {};
        const chunkFrames = parseInt(document.getElementById('sam3ChunkFrames')?.value, 10);
        if (trackingBackend === 'sam3' && Number.isFinite(chunkFrames) && chunkFrames > 0) {
            config.sam3_chunk_frames = chunkFrames;
        }
        if (document.getElementById('autoPeriodOnly')?.checked) {
            config.auto_period_only = true;
        }
        if (document.getElementById('fieldWireframeOverlay')?.checked) {
            config.field_wireframe_overlay = true;
        }
        const teamReader = document.getElementById('teamReader')?.value;
        if (teamReader === 'clef') {
            config.team_reader = 'clef';
            const clefModel = document.getElementById('clefModel')?.value;
            if (clefModel && clefModel !== 'clef-flash') {
                config.clef_model = clefModel;
            }
        } else {
            // Bumper reader model: only send the override when it differs from the
            // server default (mirrors how tracking_backend / model_size are sent).
            const vlmModel = document.getElementById('vlmModel')?.value;
            if (vlmModel && vlmModel !== this._defaultVlmModel) {
                config.vlm_model = vlmModel;
            }
        }
        if (Object.keys(config).length > 0) {
            formData.append('config_json', JSON.stringify(config));
        }
        if (this.selectedMatch?.key) {
            formData.append('match_key', this.selectedMatch.key);
        }

        this.setStatus('Uploading...', 0);

        try {
            const response = await fetch('/api/infer', {
                method: 'POST',
                body: formData
            });

            const data = await response.json();

            if (data.error) {
                this.setStatus(`Error: ${data.error}`, 0);
                return;
            }

            this.jobId = data.job_id;
            this.refreshDiagnosticButtonVisibility();

            // Store the match video URL for calibration frame grab
            this._pendingMatchVideoUrl = youtubeUrl || null;

            // Always show OCR calibration before starting inference
            this._postCalibrationTarget = 'inference';
            this.showOcrSetup(this._pendingMatchVideoUrl);
        } catch (error) {
            this.setStatus(`Error: ${error.message}`, 0);
        }
    }

    async rejoinActiveJob() {
        // A page reload loses the job id; if a match is still running
        // server-side, rejoin it (the ws reattach path streams without
        // re-running the job).
        try {
            const resp = await fetch('/api/job/active');
            const data = await resp.json();
            if (!data.job_id) return;
            this.jobId = data.job_id;
            this.refreshDiagnosticButtonVisibility();
            this.jobFinished = false;
            this.videoSection.style.display = 'grid';
            this.setInferenceControlState('running');
            this.resetRunIndicators();
            // Truthful clocks: the run started before this page existed.
            if (data.started_at) {
                this.runStartedAt = data.started_at * 1000;
                if (this.hudStartTime) {
                    this.hudStartTime.textContent = new Date(this.runStartedAt)
                        .toLocaleTimeString([], { hour12: false });
                }
            }
            this.appendStatusLog(`Rejoined running job ${data.job_id.slice(0, 8)}…`);
            this.connectWebSocket();
        } catch (e) {
            // No active job or server unreachable — normal page load.
        }
    }

    // --- Header job pill (both modes) ---

    initJobPill() {
        if (!this.jobPill) return;
        this.jobPill.addEventListener('click', () => this.onJobPillClick());

        const poll = () => this.pollActiveJob();
        poll();
        this._jobPollTimer = setInterval(() => {
            if (document.hidden) return;
            poll();
        }, 5000);

        document.addEventListener('visibilitychange', () => {
            if (!document.hidden) poll();
        });
    }

    async pollActiveJob() {
        try {
            const resp = await fetch('/api/job/active');
            const data = await resp.json();
            this._lastActiveJob = data && data.job_id ? data : null;
        } catch (e) {
            this._lastActiveJob = null;
        }
        this.renderJobPill();
    }

    renderJobPill() {
        if (!this.jobPill) return;
        const data = this._lastActiveJob;
        if (!data || !data.job_id) {
            this.jobPill.style.display = 'none';
            return;
        }
        const activeNav = document.querySelector('.nav-btn.active');
        const activeView = activeNav && activeNav.dataset.view;
        const show = activeView !== 'scorer' || this.jobId !== data.job_id;
        if (!show) {
            this.jobPill.style.display = 'none';
            return;
        }
        // No markup here — textContent is enough and needs no HTML-escaping.
        this.jobPill.textContent = `⏳ ${data.match_key || 'job'} — ${data.status || ''}`;
        this.jobPill.style.display = 'inline-flex';
    }

    onJobPillClick() {
        const data = this._lastActiveJob;
        this.switchView('scorer');
        if (data && data.job_id && this.jobId == null) {
            this.rejoinActiveJob();
        }
        this.renderJobPill();
    }

    _showInferenceView() {
        this.matchSetupSection.style.display = 'none';
        this.videoSection.style.display = 'grid';
        this.setStatus('Ready to start inference', 0);
        this.setInferenceControlState('start');
    }

    setInferenceControlState(state) {
        const show = (el, visible) => { el.style.display = visible ? 'inline-flex' : 'none'; };

        if (state === 'hidden') {
            this.inferenceControls.style.display = 'none';
            show(this.startInferenceBtn, false);
            show(this.pauseInferenceBtn, false);
            show(this.resumeInferenceBtn, false);
            show(this.stopInferenceBtn, false);
            return;
        }

        this.inferenceControls.style.display = 'flex';
        show(this.startInferenceBtn, state === 'start');
        show(this.pauseInferenceBtn, state === 'running');
        show(this.resumeInferenceBtn, state === 'paused');
        show(this.stopInferenceBtn, state === 'running' || state === 'paused');
    }

    startInference() {
        if (!this.jobId) return;
        this.jobFinished = false;
        this.resetRunIndicators();
        this.setInferenceControlState('running');
        this.connectWebSocket();
    }

    async pauseInference() {
        if (!this.jobId) return;
        try {
            const resp = await fetch(`/api/job/${this.jobId}/pause`, { method: 'POST' });
            const data = await resp.json();
            if (!resp.ok || data.error) {
                this.setStatus(`Error: ${data.error || 'Failed to pause job'}`, parseFloat(this.progressBar.style.width) || 0);
                return;
            }
            this.setInferenceControlState('paused');
        } catch (err) {
            this.setStatus(`Error: ${err.message}`, parseFloat(this.progressBar.style.width) || 0);
        }
    }

    async resumeInference() {
        if (!this.jobId) return;
        try {
            const resp = await fetch(`/api/job/${this.jobId}/resume`, { method: 'POST' });
            const data = await resp.json();
            if (!resp.ok || data.error) {
                this.setStatus(`Error: ${data.error || 'Failed to resume job'}`, parseFloat(this.progressBar.style.width) || 0);
                return;
            }
            this.setInferenceControlState('running');
            this.setStatus('Processing...', parseFloat(this.progressBar.style.width) || 0);
        } catch (err) {
            this.setStatus(`Error: ${err.message}`, parseFloat(this.progressBar.style.width) || 0);
        }
    }

    async stopInference() {
        if (!this.jobId) return;
        try {
            const resp = await fetch(`/api/job/${this.jobId}/stop`, { method: 'POST' });
            const data = await resp.json();
            if (!resp.ok || data.error) {
                this.setStatus(`Error: ${data.error || 'Failed to stop job'}`, parseFloat(this.progressBar.style.width) || 0);
                return;
            }
            this.setInferenceControlState('hidden');
            this.setStatus('Stopping inference...', parseFloat(this.progressBar.style.width) || 0);
        } catch (err) {
            this.setStatus(`Error: ${err.message}`, parseFloat(this.progressBar.style.width) || 0);
        }
    }

    connectWebSocket() {
        const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        const wsUrl = `${protocol}//${window.location.host}/ws/infer/${this.jobId}`;

        this.setStatus('Connecting...', 0);
        this.ws = new WebSocket(wsUrl);

        this.ws.onopen = () => {
            this.setStatus('Processing...', 0);
            // Reset scores at the start of each new inference
            this.scoreBlue.textContent = '0';
            this.scoreRed.textContent = '0';
            if (this.blueAllianceScore) this.blueAllianceScore.textContent = '0';
            if (this.redAllianceScore) this.redAllianceScore.textContent = '0';
            this.hudFrame.textContent = '0 / 0';
            this.hudFps.textContent = '0';
            if (this.hudReader) this.hudReader.textContent = 'local waiting';

            if (this.selectedMatch) {
                const blueTeams = this.selectedMatch.blue_teams || [];
                const redTeams = this.selectedMatch.red_teams || [];
                this.ws.send(JSON.stringify({
                    type: 'robot_identifications',
                    match_teams: [...blueTeams, ...redTeams],
                    blue_teams: blueTeams,
                    red_teams: redTeams
                }));
            }
        };

        this.ws.onmessage = (event) => {
            this.lastWsMessageAt = Date.now();
            const data = JSON.parse(event.data);
            this.handleWebSocketMessage(data);
        };

        this.ws.onerror = (error) => {
            // onclose always follows; reconnection is handled there.
        };

        // The server supports re-attaching to a running job (streaming only,
        // no re-run), so a dropped connection is recoverable: keep retrying
        // until the job reports a terminal state.
        this.ws.onclose = () => {
            if (this.jobFinished || !this.jobId) return;
            const currentProgress = parseFloat(this.progressBar.style.width) || 0;
            this.setStatus('Connection lost — reconnecting…', currentProgress);
            setTimeout(() => {
                if (!this.jobFinished && this.jobId) this.connectWebSocket();
            }, 3000);
        };
    }

    handleWebSocketMessage(data) {
        switch (data.type) {
            case 'frame':
                this.updateFrame(data);
                break;
            case 'status':
                this.handleStatus(data);
                break;
            case 'complete':
                this.handleComplete(data);
                break;
            case 'error':
                this.handleError(data);
                break;
            case 'degradation_warning':
                this.addDegradationWarning(`${data.component}: ${data.message}`);
                break;
            case 'tracking_view_seed':
                this.updateTrackingCard(data.view, (card) => {
                    if (data.image) card.img.src = 'data:image/jpeg;base64,' + data.image;
                    card.meta.textContent = `${data.robots} robot${data.robots === 1 ? '' : 's'} seeded`;
                });
                break;
            case 'tracking_progress':
                this.updateTrackingCard(data.view, (card) => {
                    const pct = data.total ? (data.current / data.total) * 100 : 0;
                    card.bar.style.width = `${pct.toFixed(1)}%`;
                    card.frames.textContent = `${data.current} / ${data.total} frames`;
                });
                break;
        }
    }

    updateTrackingCard(viewName, apply) {
        if (!this.trackingCards) return;
        if (this.trackingPanel && !this.jobFinished &&
                this.trackingPanel.style.display === 'none' && !this.overlayStreaming) {
            this.trackingPanel.style.display = 'block';
        }
        let card = this.viewCards[viewName];
        if (!card) {
            const root = document.createElement('div');
            root.className = 'tracking-card';
            const title = document.createElement('div');
            title.className = 'tracking-card-title';
            title.textContent = viewName;
            const img = document.createElement('img');
            img.className = 'tracking-card-img';
            img.alt = `${viewName} seed frame`;
            const meta = document.createElement('div');
            meta.className = 'tracking-card-meta';
            meta.textContent = 'waiting…';
            const barWrap = document.createElement('div');
            barWrap.className = 'tracking-card-barwrap';
            const bar = document.createElement('div');
            bar.className = 'tracking-card-bar';
            barWrap.appendChild(bar);
            const frames = document.createElement('div');
            frames.className = 'tracking-card-frames';
            frames.textContent = '0 / 0 frames';
            root.append(title, img, meta, barWrap, frames);
            this.trackingCards.appendChild(root);
            card = this.viewCards[viewName] = { root, img, meta, bar, frames };
        }
        apply(card);
    }

    addDegradationWarning(text) {
        this.degradationWarnings.push(text);
        if (this.degradationList) {
            const li = document.createElement('li');
            li.textContent = text;
            this.degradationList.appendChild(li);
        }
        if (this.degradationTitle) {
            this.degradationTitle.textContent =
                `⚠ ${this.degradationWarnings.length} warning${this.degradationWarnings.length > 1 ? 's' : ''} — results may be incomplete`;
        }
        if (this.degradationBanner) this.degradationBanner.style.display = 'block';
        this.appendStatusLog(`⚠ ${text}`);
    }

    resetRunIndicators() {
        this.degradationWarnings = [];
        if (this.degradationList) this.degradationList.replaceChildren();
        if (this.degradationBanner) this.degradationBanner.style.display = 'none';
        if (this.statusLog) {
            this.statusLog.replaceChildren();
            this.statusLog.style.display = 'block';
        }
        if (this.recoveryActions) {
            this.recoveryActions.replaceChildren();
            this.recoveryActions.style.display = 'none';
        }
        // Run clocks + tracking panel.
        this.runStartedAt = Date.now();
        this.overlayStreaming = false;
        if (this.hudStartTime) {
            this.hudStartTime.textContent =
                new Date(this.runStartedAt).toLocaleTimeString([], { hour12: false });
        }
        if (this.elapsedTimer) clearInterval(this.elapsedTimer);
        this.elapsedTimer = setInterval(() => {
            if (!this.hudElapsed || this.runStartedAt === null) return;
            const s = Math.floor((Date.now() - this.runStartedAt) / 1000);
            const mm = Math.floor(s / 60);
            const ss = String(s % 60).padStart(2, '0');
            this.hudElapsed.textContent = mm >= 60
                ? `${Math.floor(mm / 60)}:${String(mm % 60).padStart(2, '0')}:${ss}`
                : `${mm}:${ss}`;
            if (this.jobFinished) clearInterval(this.elapsedTimer);
        }, 1000);
        this.viewCards = {};
        if (this.trackingCards) this.trackingCards.replaceChildren();
        if (this.trackingPanel) this.trackingPanel.style.display = 'none';
        // Clear the previous run's outputs so a re-run (retry) doesn't leak
        // stale crops/rows/events into the new run.
        this.clearRunOutputs();
        this.lastWsMessageAt = Date.now();
        if (this.stallTimer) clearInterval(this.stallTimer);
        // Stall watchdog: tracking can be quiet for a while, but minutes of
        // total silence means the job is hung — say so instead of letting a
        // frozen frame impersonate progress.
        this.stallTimer = setInterval(() => {
            if (this.jobFinished) { clearInterval(this.stallTimer); return; }
            const quiet = Math.round((Date.now() - this.lastWsMessageAt) / 1000);
            if (quiet >= 120) {
                const currentProgress = parseFloat(this.progressBar.style.width) || 0;
                // User-facing message stays mode-agnostic; the debug diagnostics
                // link goes into the status log (which is itself debug-only).
                this.setStatus(`No updates for ${quiet}s — job may be stalled; ` +
                               `results so far are preserved.`,
                               currentProgress);
                this.appendDebugTasksLink();
            }
        }, 15000);
    }

    // Debug-only diagnostics link, appended to the status log. Built with DOM
    // APIs (like appendCalibrationLink) so it opens /api/debug/tasks in a new tab.
    appendDebugTasksLink() {
        if (!this.statusLog) return;
        // One diagnostics link per stall — drop any prior one.
        this.statusLog.querySelectorAll('.debug-tasks-row').forEach((n) => n.remove());
        const row = document.createElement('div');
        row.className = 'debug-tasks-row';
        const ts = new Date().toLocaleTimeString([], { hour12: false });
        row.appendChild(document.createTextNode(`${ts}  job may be stalled — `));
        const a = document.createElement('a');
        a.href = '/api/debug/tasks';
        a.target = '_blank';
        a.rel = 'noopener';
        a.textContent = 'inspect running tasks →';
        row.appendChild(a);
        this.statusLog.appendChild(row);
        this.statusLog.scrollTop = this.statusLog.scrollHeight;
    }

    appendStatusLog(text) {
        if (!this.statusLog) return;
        const last = this.statusLog.lastElementChild;
        if (last && last.dataset.msg === text) return;  // collapse repeats
        const row = document.createElement('div');
        row.dataset.msg = text;
        const ts = new Date().toLocaleTimeString([], { hour12: false });
        row.textContent = `${ts}  ${text}`;
        this.statusLog.appendChild(row);
        while (this.statusLog.childElementCount > 50) {
            this.statusLog.removeChild(this.statusLog.firstElementChild);
        }
        this.statusLog.scrollTop = this.statusLog.scrollHeight;
    }

    handleStatus(data) {
        if (data.status === 'completed' || data.status === 'error') {
            // Terminal state reported on re-attach: stop the reconnect loop.
            this.jobFinished = true;
        }
        if (data.status === 'paused') {
            const currentProgress = parseFloat(this.progressBar.style.width) || 0;
            this.setStatus(`Paused at frame ${data.frame_num}`, currentProgress);
        } else if (data.message) {
            const currentProgress = data.progress !== undefined ? data.progress : (parseFloat(this.progressBar.style.width) || 0);
            this.setStatus(data.message, currentProgress);
            this.appendStatusLog(data.message);
        }
    }

    updateFrame(data) {
        // First overlay frame = analysis phase has begun: swap the tracking
        // panel out for the live annotated video feed.
        if (!this.overlayStreaming) {
            this.overlayStreaming = true;
            if (this.trackingPanel) this.trackingPanel.style.display = 'none';
        }
        const progress = (data.frame_num / data.total_frames) * 100;
        this.setStatus(`Processing frame ${data.frame_num} of ${data.total_frames}`, progress);

        this.hudFrame.textContent = `${data.frame_num} / ${data.total_frames}`;
        this.hudFps.textContent = data.fps.toFixed(1);
        const hudBackend = document.getElementById('hudBackend');
        if (hudBackend && data.backend) {
            hudBackend.textContent = data.backend === 'sam3' ? 'SAM 3' : 'SAM 2.1';
            hudBackend.classList.toggle('hud-backend-sam3', data.backend === 'sam3');
        }
        if (this.hudReader && data.vlm_status) {
            const status = data.vlm_status;
            const provider = status.provider || 'local';
            const calls = status.calls || 0;
            const sent = status.last_sent || 0;
            const candidates = status.last_candidates || 0;
            const unconfirmed = status.unconfirmed ?? 0;
            const interval = Number.isFinite(status.last_interval_sec)
                ? `${status.last_interval_sec.toFixed(2)}s`
                : '—';
            const error = status.last_error ? ` ${status.last_error}` : '';
            const resultCount = Array.isArray(status.last_results)
                ? status.last_results.filter(r => r.team_number).length
                : 0;
            this.hudReader.textContent = `${provider} c${calls} ${resultCount}/${sent} reads ${candidates} crops u${unconfirmed} ${interval}${error}`;
        }
        this.scoreBlue.textContent = data.scores.blue;
        this.scoreRed.textContent = data.scores.red;

        if (this.blueAllianceScore) this.blueAllianceScore.textContent = data.scores.blue;
        if (this.redAllianceScore) this.redAllianceScore.textContent = data.scores.red;

        if (data.frame_data) {
            this.drawFrame(data.frame_data);
        }

        this.updateRobotTables(data.robots);
        this.updateVlmCrops(data.ocr_crops);
        this.updateTimeline(data.scoring_events);
        this.updateTeamsSummary(data.robots);
    }

    // Standard-mode HUD: confirmed team identities over the known roster.
    // Roster is 6 when both alliances are populated, else the count of
    // distinct slots we've actually seen so the ratio stays honest early on.
    updateTeamsSummary(robots) {
        const el = document.getElementById('hudTeams');
        if (!el) return;
        const list = Array.isArray(robots) ? robots : [];
        const confirmed = list.filter(r => r.confirmed).length;
        const blueSlots = list.some(r => r.alliance === 'blue');
        const redSlots = list.some(r => r.alliance === 'red');
        const roster = (blueSlots && redSlots)
            ? 6
            : list.filter(r => typeof r.slot === 'string' && /^[BR][1-3]$/.test(r.slot)).length;
        el.textContent = `${confirmed} / ${roster || 6}`;
    }

    drawFrame(frameDataBase64) {
        const img = new Image();
        img.onload = () => {
            this.canvas.width = img.width;
            this.canvas.height = img.height;
            this.ctx.drawImage(img, 0, 0);
        };
        img.src = 'data:image/jpeg;base64,' + frameDataBase64;
    }

    updateRobotTables(robots, blueTable, redTable) {
        const blueEl = blueTable || this.blueRobotsTable;
        const redEl = redTable || this.redRobotsTable;
        const blueRobots = robots.filter(r => r.alliance === 'blue');
        const redRobots = robots.filter(r => r.alliance === 'red');
        this.updateRobotTable(blueEl, blueRobots, 'blue');
        this.updateRobotTable(redEl, redRobots, 'red');
    }

    updateRobotTable(tableBody, robots, alliance) {
        const prefix = alliance === 'blue' ? 'B' : 'R';
        const bySlot = new Map(
            robots
                .filter(r => typeof r.slot === 'string' && r.slot.match(/^[BR][1-3]$/))
                .map(r => [r.slot, r])
        );
        const rows = [1, 2, 3].map(index => {
            const label = `${prefix}${index}`;
            const robot = bySlot.get(label);
            const goals = robot && robot.scores != null ? robot.scores : 0;
            // Confirmed identities render solid with a check; a leading-but-
            // unconfirmed candidate renders dimmed with a "?" so operators can
            // tell a locked read from a guess at a glance.
            let teamCell = '—';
            if (robot && robot.team_number) {
                const num = this.escapeHtml(robot.team_number);
                teamCell = robot.confirmed
                    ? `<span class="team-confirmed">${num} ✓</span>`
                    : `<span class="team-tentative">${num}</span>`;
            } else if (robot && robot.team_candidate) {
                teamCell = `<span class="team-candidate">${this.escapeHtml(robot.team_candidate)}?</span>`;
            }
            return `
                <tr>
                    <td>${label}</td>
                    <td>${teamCell}</td>
                    <td class="goal-count">${goals}</td>
                </tr>
            `;
        }).join('');

        tableBody.innerHTML = rows;
    }

    updateVlmCrops(crops) {
        if (!this.vlmCrops) return;

        if (Array.isArray(crops)) {
            crops.forEach(crop => {
                const key = crop.slot || `R${crop.robot_id}`;
                this.latestVlmCrops.set(key, {
                    ...crop,
                    slot: key,
                    updatedAt: new Date()
                });
            });
        }

        if (this.latestVlmCrops.size === 0) {
            this.vlmCrops.innerHTML = '<p class="empty-state">No VLM crops sent yet</p>';
            return;
        }

        const entries = [...this.latestVlmCrops.values()]
            .sort((a, b) => this.slotSortValue(a.slot) - this.slotSortValue(b.slot));
        this.vlmCrops.innerHTML = entries.map(crop => {
            const raw = crop.raw_read || '—';
            const matched = crop.matched || 'no match';
            const resultClass = crop.matched ? 'matched' : 'unmatched';
            const confidence = Number.isFinite(crop.confidence)
                ? `${Math.round(crop.confidence * 100)}%`
                : '—';
            const viaGallery = crop.via === 'gallery';
            const galleryBadge = viaGallery
                ? `<span class="appearance-badge">appearance ${Number.isFinite(crop.gallery_sim) ? Math.round(crop.gallery_sim * 100) + '%' : ''}</span>`
                : '';
            const subtext = viaGallery ? `recovered by appearance` : `${confidence} confidence`;
            const error = crop.error ? `<div class="vlm-error">${this.escapeHtml(crop.error)}</div>` : '';
            return `
                <div class="vlm-crop ${crop.alliance || 'unknown'}">
                    <img src="data:image/jpeg;base64,${crop.crop}" alt="${this.escapeHtml(crop.slot)} bumper crop">
                    <div class="vlm-crop-meta">
                        <div class="vlm-crop-title">
                            <span>${this.escapeHtml(crop.slot)}</span>
                            <span>${this.escapeHtml(crop.source || 'cam')}</span>
                        </div>
                        <div class="vlm-result ${resultClass}">
                            ${this.escapeHtml(crop.slot)} ${this.escapeHtml(raw)} &gt; ${this.escapeHtml(matched)} ${galleryBadge}
                        </div>
                        <div class="vlm-subtext">${subtext}</div>
                        ${error}
                    </div>
                </div>
            `;
        }).join('');
    }

    slotSortValue(slot) {
        const text = String(slot || '');
        const match = text.match(/([BR])(\d+)/);
        if (!match) return 999;
        const base = match[1] === 'B' ? 0 : 100;
        return base + Number(match[2]);
    }

    updateTimeline(events, options = {}) {
        const container = options.container || this.timelineContainer;
        const onSeek = options.onSeek || null;
        const activeIndex = options.activeIndex;

        if (!events || events.length === 0) return;

        if (container.querySelector('.empty-state')) {
            container.innerHTML = '';
        }

        // Live path renders newest-first; clickable (replay) path keeps chronological
        // order so timestamps line up with the scrubber.
        const indexed = events.map((event, i) => ({ event, i }));
        const ordered = onSeek ? indexed : [...indexed].reverse();

        const eventElements = ordered.map(({ event, i }) => {
            // Replay events: robot_label/slot; live events: team_number/robot_id.
            const teamNumber = event.team_number || event.robot_label || event.slot
                || (event.robot_id != null ? `R${event.robot_id}` : '—');
            // Live events carry `timestamp` + `frame`; replay events carry `t` + `frame_num`.
            const seconds = event.timestamp != null ? event.timestamp : event.t;
            const frame = event.frame != null ? event.frame : event.frame_num;
            const timestamp = this.formatTimestamp(seconds || 0);
            const clickAttrs = onSeek ? ` data-event-idx="${i}" role="button" tabindex="0"` : '';
            const activeClass = (activeIndex != null && i === activeIndex) ? ' active' : '';
            const confidence = event.confidence != null ? event.confidence : '';

            return `
                <div class="timeline-event ${event.alliance}${onSeek ? ' clickable' : ''}${activeClass}"${clickAttrs}>
                    <div class="timeline-event-info">
                        <div class="timeline-event-time">
                            <span class="debug-only">Frame ${frame} | </span>${timestamp}
                        </div>
                        <div class="timeline-event-details">
                            <span class="timeline-event-alliance ${event.alliance}">
                                ${event.alliance.toUpperCase()}
                            </span>
                            scored by <strong>${teamNumber}</strong>
                        </div>
                    </div>
                    ${confidence !== '' ? `<span class="confidence-badge ${confidence}">${confidence}</span>` : ''}
                </div>
            `;
        }).join('');

        container.innerHTML = eventElements;

        if (onSeek) {
            container.querySelectorAll('.timeline-event.clickable').forEach(el => {
                const idx = parseInt(el.dataset.eventIdx, 10);
                const seek = () => {
                    const ev = events[idx];
                    if (ev) onSeek(ev.t != null ? ev.t : ev.timestamp);
                };
                el.addEventListener('click', seek);
                el.addEventListener('keydown', (e) => {
                    if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); seek(); }
                });
            });
        }
    }

    formatTimestamp(seconds) {
        const mins = Math.floor(seconds / 60);
        const secs = Math.floor(seconds % 60);
        return `${mins}:${secs.toString().padStart(2, '0')}`;
    }

    escapeHtml(value) {
        return String(value ?? '').replace(/[&<>"']/g, ch => ({
            '&': '&amp;',
            '<': '&lt;',
            '>': '&gt;',
            '"': '&quot;',
            "'": '&#39;'
        }[ch]));
    }

    formatDuration(sec) {
        if (sec == null || isNaN(sec)) return '—';
        sec = Math.round(sec);
        if (sec < 60) return `${sec}s`;
        const m = Math.floor(sec / 60), s = sec % 60;
        if (m < 60) return `${m}m ${String(s).padStart(2, '0')}s`;
        const h = Math.floor(m / 60);
        return `${h}h ${String(m % 60).padStart(2, '0')}m`;
    }

    // Render a processing-time breakdown from a timings object (seconds per phase).
    renderTimings(timings) {
        if (!timings || timings.total == null) return '<p class="empty-state">No timing data</p>';
        const labels = {
            model_load: 'Model load', frame_export: 'Frame export',
            sam2_tracking: 'Tracker propagation', analysis_loop: 'Analysis loop',
            bumper_vlm: '↳ Bumper VLM',
        };
        const topLevel = ['model_load', 'frame_export', 'sam2_tracking', 'analysis_loop'];
        const total = timings.total;
        const known = topLevel.reduce((a, k) => a + (timings[k] || 0), 0);
        const rowHtml = (label, val, sub) => {
            const pct = total ? Math.max(0, Math.min(100, Math.round((val / total) * 100))) : 0;
            return `<div class="timing-row${sub ? ' sub' : ''}">
                <span class="timing-label">${label}</span>
                <span class="timing-bar"><span style="width:${pct}%"></span></span>
                <span class="timing-val">${this.formatDuration(val)}</span>
            </div>`;
        };
        let rows = '';
        for (const k of ['model_load', 'frame_export', 'sam2_tracking', 'analysis_loop', 'bumper_vlm']) {
            if (timings[k] != null) rows += rowHtml(labels[k], timings[k], k === 'bumper_vlm');
        }
        const other = Math.max(0, total - known);
        if (other > 0.05) rows += rowHtml('Setup / other', other, false);
        rows += `<div class="timing-row total">
            <span class="timing-label">Total</span><span class="timing-bar"></span>
            <span class="timing-val">${this.formatDuration(total)}</span>
        </div>`;
        return rows;
    }

    handleComplete(data) {
        this.jobFinished = true;
        if (this.stallTimer) clearInterval(this.stallTimer);
        // The completion payload carries the authoritative warning list
        // (covers warnings sent while this client was disconnected).
        (data.degradation_warnings || []).forEach((w) => {
            const text = typeof w === 'string' ? w : `${w.component}: ${w.message}`;
            if (!this.degradationWarnings.includes(text)) this.addDegradationWarning(text);
        });
        if (this.degradationWarnings.length > 0) {
            this.setStatus(
                `Complete with ${this.degradationWarnings.length} warning(s) — review before trusting results`,
                100);
            this.appendStatusLog('Job finished WITH warnings.');
        } else {
            this.setStatus('Processing complete!', 100);
            this.appendStatusLog('Job finished cleanly.');
        }
        this.inferenceResults = data;
        this.setInferenceControlState('hidden');

        if (data.robots) this.updateRobotTables(data.robots);
        if (data.scoring_events) this.updateTimeline(data.scoring_events);

        // Show results section for editing after a brief delay
        setTimeout(() => {
            this.showResultsEditor(data);
        }, 1500);
    }

    handleError(data) {
        this.jobFinished = true;
        if (this.stallTimer) clearInterval(this.stallTimer);
        this.setInferenceControlState('hidden');
        this.setStatus(`Error: ${data.error}`, 0);
        this.appendStatusLog(`✖ Job failed: ${data.error}`);
        // A calibration-drift abort carries needs_calibration; surface a clickable
        // jump to the calibrate page (approval panel / manual flow) plus a button to
        // continue the analysis once the calibration is fixed — no re-upload.
        if (data.needs_calibration) {
            this.appendCalibrationLink(data.needs_calibration);
            this.appendContinueAnalysisButton(this.jobId);
        }
    }

    appendContinueAnalysisButton(previousJobId) {
        if (!this.recoveryActions || !previousJobId) return;
        this.recoveryActions.querySelectorAll('.continue-analysis-row').forEach((n) => n.remove());
        const row = document.createElement('div');
        row.className = 'continue-analysis-row';
        row.style.marginTop = '4px';
        const btn = document.createElement('button');
        btn.className = 'btn-small';
        btn.style.color = '#80ffb4';
        btn.textContent = '▶ Continue analysis — re-check calibration & resume';
        btn.title = 'Re-runs this match with the fixed calibration, reusing the '
                  + 'already-uploaded video (no need to start over).';
        btn.addEventListener('click', () => { btn.disabled = true; this.retryAnalysis(previousJobId); });
        row.appendChild(btn);
        this.recoveryActions.appendChild(row);
        this.recoveryActions.style.display = 'block';
    }

    async retryAnalysis(previousJobId) {
        if (!previousJobId) return;
        this.appendStatusLog('Re-checking calibration and continuing analysis…');
        try {
            const resp = await fetch(`/api/job/${previousJobId}/retry`, { method: 'POST' });
            const data = await resp.json();
            if (!resp.ok || data.error) {
                this.setStatus(`Error: ${data.error || 'Could not continue analysis'}`, 0);
                return;
            }
            // Fresh job id; drive it exactly like a new run (the websocket start path
            // runs process_video, which re-scans calibration before tracking).
            this.jobId = data.job_id;
            this.refreshDiagnosticButtonVisibility();
            this.jobFinished = false;
            this.resetRunIndicators();
            this.setInferenceControlState('running');
            this.connectWebSocket();
        } catch (e) {
            this.setStatus(`Error: ${e.message}`, 0);
        }
    }

    appendCalibrationLink(nc) {
        if (!this.recoveryActions) return;
        // Only one calibration link per failure — drop any prior one.
        this.recoveryActions.querySelectorAll('.calib-link-row').forEach((n) => n.remove());
        const params = new URLSearchParams();
        if (nc.match_key) params.set('match', nc.match_key);
        if (nc.camera) params.set('camera', nc.camera);
        const qs = params.toString();
        const row = document.createElement('div');
        row.className = 'calib-link-row';
        const a = document.createElement('a');
        a.href = '/calibrate' + (qs ? `?${qs}` : '');
        a.rel = 'noopener';
        a.style.color = '#80ffb4';
        a.style.fontWeight = 'bold';
        a.textContent = nc.autofix_pending
            ? `🔧 Review the auto-fix for ${nc.camera || 'main'} — approve or reject →`
            : `📐 Recalibrate ${nc.camera || 'main'} on the calibration page →`;
        a.addEventListener('click', (event) => {
            event.preventDefault();
            this.switchView('calibration', qs ? `?${qs}` : '');
        });
        row.appendChild(a);
        this.recoveryActions.appendChild(row);
        this.recoveryActions.style.display = 'block';
    }

    setStatus(text, progress) {
        this.statusText.textContent = text;
        this.progressBar.style.width = `${progress}%`;

        if (progress > 0 && progress < 100) {
            this.statusText.classList.add('processing');
        } else {
            this.statusText.classList.remove('processing');
        }
    }

    // --- Post-Inference Results Editor ---

    showResultsEditor(data) {
        this.resultsSection.style.display = 'block';
        this.resultsSection.scrollIntoView({ behavior: 'smooth' });

        // Processing-time breakdown
        const ptCard = document.getElementById('processingTimeCard');
        if (ptCard) {
            if (data && data.timings && data.timings.total != null) {
                document.getElementById('processingTimeBreakdown').innerHTML =
                    this.renderTimings(data.timings);
                ptCard.style.display = 'block';
            } else {
                ptCard.style.display = 'none';
            }
        }

        // Show TBA ground truth if available
        const gtCard = document.getElementById('groundTruthCard');
        if (this.selectedMatch) {
            const blueHub = this.selectedMatch.blue_hub;
            const redHub = this.selectedMatch.red_hub;

            if (blueHub && blueHub.total != null) {
                document.getElementById('gtBlueTotal').textContent = blueHub.total;
                document.getElementById('gtRedTotal').textContent = redHub.total;
                gtCard.style.display = 'block';
            } else {
                gtCard.style.display = 'none';
            }
        } else {
            gtCard.style.display = 'none';
        }

        // Build editable robot tables
        this.buildEditTable('blue', data);
        this.buildEditTable('red', data);
        this.updateEditTotals();

        // Offer a "Watch Replay" shortcut for the match we just analyzed.
        // The replay id is the same as the inference job_id.
        this.setupReplayShortcut();
    }

    setupReplayShortcut() {
        const actions = document.querySelector('#resultsSection .save-actions');
        if (!actions || !this.jobId) return;
        let btn = document.getElementById('watchReplayBtn');
        if (!btn) {
            btn = document.createElement('button');
            btn.id = 'watchReplayBtn';
            btn.className = 'btn-secondary';
            btn.textContent = 'Watch Replay';
            // Insert right after the save button so it reads Save / Replay / New
            const saveBtn = document.getElementById('saveResultsBtn');
            actions.insertBefore(btn, saveBtn ? saveBtn.nextSibling : null);
        }
        const jobId = this.jobId;
        btn.onclick = () => this.openReplay(jobId);
    }

    buildEditTable(alliance, data) {
        const tableBody = document.getElementById(`edit${alliance.charAt(0).toUpperCase() + alliance.slice(1)}Table`);
        const robots = (data.robots || []).filter(r => r.alliance === alliance);

        // Get TBA team numbers for pre-fill
        const tbaTeams = this.selectedMatch ?
            (alliance === 'blue' ? this.selectedMatch.blue_teams : this.selectedMatch.red_teams) : [];

        // Ensure we have 3 rows (3 robots per alliance)
        const rows = [];
        for (let i = 0; i < 3; i++) {
            const robot = robots[i];
            const tbaTeam = tbaTeams[i] || '';
            const predicted = robot ? robot.scores : 0;
            // Use TBA team number as default, fall back to detected
            const teamNum = tbaTeam || (robot ? (robot.team_number || '') : '');

            rows.push(`
                <tr data-position="${i + 1}" data-alliance="${alliance}" data-robot-id="${robot ? robot.id : ''}">
                    <td>${i + 1}</td>
                    <td><input type="text" class="edit-team-input" value="${teamNum}" placeholder="Team #" data-field="team_number"></td>
                    <td class="predicted-count">${predicted}</td>
                    <td><input type="number" class="edit-count-input" value="" placeholder="${predicted}" min="0" data-field="corrected_fuel_count" data-predicted="${predicted}"></td>
                </tr>
            `);
        }

        tableBody.innerHTML = rows.join('');

        // Listen for changes to update totals
        tableBody.querySelectorAll('.edit-count-input').forEach(input => {
            input.addEventListener('input', () => this.updateEditTotals());
        });
    }

    updateEditTotals() {
        for (const alliance of ['Blue', 'Red']) {
            const tableBody = document.getElementById(`edit${alliance}Table`);
            let predTotal = 0;
            let corrTotal = 0;
            let hasCorrected = false;

            tableBody.querySelectorAll('tr').forEach(row => {
                const predCell = row.querySelector('.predicted-count');
                const corrInput = row.querySelector('.edit-count-input');
                if (predCell) predTotal += parseInt(predCell.textContent) || 0;
                if (corrInput && corrInput.value !== '') {
                    corrTotal += parseInt(corrInput.value) || 0;
                    hasCorrected = true;
                } else if (corrInput) {
                    corrTotal += parseInt(corrInput.dataset.predicted) || 0;
                }
            });

            document.getElementById(`edit${alliance}PredTotal`).textContent = predTotal;
            document.getElementById(`edit${alliance}CorrTotal`).textContent = hasCorrected ? corrTotal : '—';
        }
    }

    async saveResults() {
        const saveBtn = document.getElementById('saveResultsBtn');
        const saveStatus = document.getElementById('saveStatus');
        saveBtn.disabled = true;
        saveStatus.textContent = 'Saving...';

        try {
            // Collect robot data from edit tables
            const robots = [];
            for (const alliance of ['blue', 'red']) {
                const allianceCap = alliance.charAt(0).toUpperCase() + alliance.slice(1);
                const tableBody = document.getElementById(`edit${allianceCap}Table`);

                tableBody.querySelectorAll('tr').forEach(async (row) => {
                    const position = parseInt(row.dataset.position);
                    const teamInput = row.querySelector('[data-field="team_number"]');
                    const corrInput = row.querySelector('[data-field="corrected_fuel_count"]');
                    const predicted = parseInt(row.querySelector('.predicted-count').textContent) || 0;

                    robots.push({
                        alliance,
                        position,
                        team_number: teamInput.value.trim(),
                        predicted_fuel_count: predicted,
                        corrected_fuel_count: corrInput.value !== '' ? parseInt(corrInput.value) : null,
                    });
                });
            }

            // Calculate predicted totals
            const bluePred = robots.filter(r => r.alliance === 'blue')
                .reduce((s, r) => s + r.predicted_fuel_count, 0);
            const redPred = robots.filter(r => r.alliance === 'red')
                .reduce((s, r) => s + r.predicted_fuel_count, 0);

            // Get video source
            const videoFile = document.getElementById('videoFile').files[0];
            const youtubeUrl = document.getElementById('youtubeUrl').value.trim();
            const videoSource = videoFile ? videoFile.name : youtubeUrl;

            // First, get the match from DB to get robot IDs
            const matchKey = this.selectedMatch ? this.selectedMatch.key : null;

            if (!matchKey) {
                saveStatus.textContent = 'Error: No match selected';
                saveBtn.disabled = false;
                return;
            }

            // Fetch DB match to get robot IDs
            const matchResp = await fetch(`/api/event/match/${matchKey}`);
            const matchData = await matchResp.json();

            // Map our edit data to DB robot IDs
            const robotUpdates = robots.map(r => {
                const dbRobot = (matchData.robots || []).find(
                    db => db.alliance === r.alliance && db.position === r.position
                );
                return {
                    id: dbRobot ? dbRobot.id : null,
                    team_number: r.team_number,
                    predicted_fuel_count: r.predicted_fuel_count,
                    corrected_fuel_count: r.corrected_fuel_count,
                };
            }).filter(r => r.id);

            const payload = {
                match_key: matchKey,
                blue_predicted_score: bluePred,
                red_predicted_score: redPred,
                video_source: videoSource,
                notes: document.getElementById('matchNotes').value.trim(),
                robots: robotUpdates,
            };

            const resp = await fetch('/api/match/save', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload),
            });

            if (resp.ok) {
                saveStatus.textContent = 'Saved successfully!';
                saveStatus.className = 'save-status success';
            } else {
                const err = await resp.json();
                saveStatus.textContent = `Error: ${err.error || 'Save failed'}`;
                saveStatus.className = 'save-status error';
            }
        } catch (err) {
            saveStatus.textContent = `Error: ${err.message}`;
            saveStatus.className = 'save-status error';
        }

        saveBtn.disabled = false;
    }

    resetToMatchSelect() {
        if (this.ws) {
            this.ws.close();
            this.ws = null;
        }
        this.videoSection.style.display = 'none';
        this.resultsSection.style.display = 'none';
        this.matchSetupSection.style.display = 'block';
        this.setInferenceControlState('hidden');

        // Reset state
        this.inferenceResults = null;
        this.jobId = null;
        this.refreshDiagnosticButtonVisibility();
        if (this.diagnosticPanel) this.diagnosticPanel.style.display = 'none';

        // Reset UI
        document.getElementById('matchSelect').value = '';
        document.getElementById('matchTeams').style.display = 'none';
        document.getElementById('uploadArea').style.display = 'none';
        document.getElementById('videoFile').value = '';
        document.getElementById('youtubeUrl').value = '';
        document.getElementById('matchNotes').value = '';
        document.getElementById('saveStatus').textContent = '';

        // Reset score displays
        this.scoreBlue.textContent = '0';
        this.scoreRed.textContent = '0';
        if (this.blueAllianceScore) this.blueAllianceScore.textContent = '0';
        if (this.redAllianceScore) this.redAllianceScore.textContent = '0';

        // Robot tables, VLM crops, timeline — shared with resetRunIndicators.
        this.clearRunOutputs();

        this.setStatus('Ready', 0);
    }

    // Restore the live-output panels to their empty states. Shared by
    // resetToMatchSelect (full flow reset) and resetRunIndicators (per-run
    // reset) so a re-run doesn't leak the previous run's crops/rows/events.
    clearRunOutputs() {
        if (this.blueRobotsTable) {
            this.blueRobotsTable.innerHTML =
                '<tr><td colspan="3" class="empty-state">Waiting for detection...</td></tr>';
        }
        if (this.redRobotsTable) {
            this.redRobotsTable.innerHTML =
                '<tr><td colspan="3" class="empty-state">Waiting for detection...</td></tr>';
        }
        this.latestVlmCrops.clear();
        if (this.vlmCrops) {
            this.vlmCrops.innerHTML = '<p class="empty-state">No VLM crops sent yet</p>';
        }
        if (this.timelineContainer) {
            this.timelineContainer.innerHTML = '<p class="empty-state">No events yet</p>';
        }
    }

    // --- Match History ---

    async loadHistory() {
        const container = document.getElementById('historyContent');
        const filterBar = document.getElementById('historyFilterBar');
        const filterSelect = document.getElementById('historyEventFilter');
        container.innerHTML = '<p class="empty-state">Loading...</p>';

        try {
            const resp = await fetch('/api/history/all');
            const events = await resp.json();

            if (!events || events.length === 0) {
                filterBar.style.display = 'none';
                container.innerHTML = '<p class="empty-state">No match data yet. Load an event and analyze some matches to see history here.</p>';
                return;
            }

            // Store events for filtering
            this._historyEvents = events;

            // Build filter dropdown
            filterBar.style.display = 'flex';
            filterSelect.innerHTML = '<option value="all">All Events</option>';
            for (const event of events) {
                const opt = document.createElement('option');
                opt.value = event.event_key;
                opt.textContent = `${event.event_name} (${event.year})`;
                filterSelect.appendChild(opt);
            }

            // Default to current event if loaded
            if (this.eventData && this.eventData.event_key) {
                const currentKey = this.eventData.event_key;
                const hasMatch = events.some(e => e.event_key === currentKey);
                if (hasMatch) {
                    filterSelect.value = currentKey;
                }
            }

            // Bind filter change (use onchange to avoid stacking)
            filterSelect.onchange = () => this.renderHistory();

            this.renderHistory();
        } catch (err) {
            container.innerHTML = `<p class="empty-state">Error loading history: ${err.message}</p>`;
        }
    }

    renderHistory() {
        const container = document.getElementById('historyContent');
        const filterValue = document.getElementById('historyEventFilter').value;
        const events = this._historyEvents || [];

        const filtered = filterValue === 'all'
            ? events
            : events.filter(e => e.event_key === filterValue);

        if (filtered.length === 0) {
            container.innerHTML = '<p class="empty-state">No matches found for this event.</p>';
            return;
        }

        let html = '';
        for (const event of filtered) {
            html += `<div class="history-event">
                    <h3>${this.escapeHtml(event.event_name)} <span class="event-year">${this.escapeHtml(event.year)}</span></h3>
                    <table class="history-table">
                        <thead>
                            <tr>
                                <th>Match</th>
                                <th>Blue Teams</th>
                                <th>Red Teams</th>
                                <th>TBA Score</th>
                                <th>Predicted</th>
                                <th>Status</th>
                                <th></th>
                            </tr>
                        </thead>
                        <tbody>`;

            for (const match of event.matches) {
                const blueTeams = match.robots
                    .filter(r => r.alliance === 'blue')
                    .sort((a, b) => a.position - b.position)
                    .map(r => r.team_number)
                    .join(', ');
                const redTeams = match.robots
                    .filter(r => r.alliance === 'red')
                    .sort((a, b) => a.position - b.position)
                    .map(r => r.team_number)
                    .join(', ');

                const tbaScore = match.tba_blue_score != null ?
                    `${match.tba_blue_score}-${match.tba_red_score}` : '—';

                const predScore = match.video_source ?
                    `${match.blue_predicted_score || 0}-${match.red_predicted_score || 0}` : '—';

                const hasVideo = match.video_source ? 'scored' : 'pending';
                const hasCorrected = match.robots.some(r => r.corrected_fuel_count != null);
                const statusLabel = hasVideo === 'scored' ?
                    (hasCorrected ? 'Corrected' : 'Predicted') : 'Not scored';
                const statusClass = hasVideo === 'scored' ?
                    (hasCorrected ? 'status-corrected' : 'status-predicted') : 'status-pending';

                // Match label
                const levelMap = { qm: 'Q', sf: 'SF', f: 'F' };
                const matchLabel = `${levelMap[match.comp_level] || match.comp_level}${match.match_number}`;

                html += `
                        <tr>
                            <td><strong>${this.escapeHtml(matchLabel)}</strong></td>
                            <td class="team-cell blue">${this.escapeHtml(blueTeams)}</td>
                            <td class="team-cell red">${this.escapeHtml(redTeams)}</td>
                            <td>${tbaScore}</td>
                            <td>${predScore}</td>
                            <td><span class="status-badge ${statusClass}">${statusLabel}</span></td>
                            <td><button class="btn-small" onclick="scorer.openMatchDetail('${match.tba_match_key}')">View</button></td>
                        </tr>`;
            }

            html += `</tbody></table></div>`;
        }

        container.innerHTML = html;
    }

    // --- Match Replays Library ---

    showReplayLibrary() {
        document.getElementById('replaysLibrary').style.display = 'block';
        document.getElementById('replayPlayer').style.display = 'none';
    }

    showReplayPlayerView() {
        document.getElementById('replaysLibrary').style.display = 'none';
        document.getElementById('replayPlayer').style.display = 'block';
    }

    async loadReplays() {
        const grid = document.getElementById('replaysGrid');
        grid.innerHTML = '<p class="empty-state">Loading replays...</p>';

        try {
            const resp = await fetch('/api/replays');
            const replays = await resp.json();

            if (!Array.isArray(replays) || replays.length === 0) {
                grid.innerHTML = '<p class="empty-state">No replays yet. Analyze a match to record one.</p>';
                return;
            }

            grid.innerHTML = replays.map(r => this.renderReplayCard(r)).join('');

            grid.querySelectorAll('[data-replay-action="play"]').forEach(btn => {
                btn.addEventListener('click', () => this.openReplay(btn.dataset.replayId));
            });
            grid.querySelectorAll('[data-replay-action="delete"]').forEach(btn => {
                btn.addEventListener('click', () => this.deleteReplay(btn.dataset.replayId));
            });
        } catch (err) {
            grid.innerHTML = `<p class="empty-state">Error loading replays: ${this.escapeHtml(err.message)}</p>`;
        }
    }

    // "qm 30" -> "Qualification 30", "sf 2" -> "Playoff 2", "f 1" -> "Final 1".
    // Falls back to the raw match_key, then the stored label (job GUID).
    formatMatchTitle(r) {
        const names = { qm: 'Qualification', qf: 'Quarterfinal', sf: 'Playoff', f: 'Final' };
        if (r.comp_level && r.match_number != null) {
            const level = names[r.comp_level] || r.comp_level.toUpperCase();
            const num = r.comp_level === 'sf' ? (r.set_number ?? r.match_number) : r.match_number;
            const match = `${level} ${num}`;
            return r.event_name ? `${r.event_name} · ${match}` : match;
        }
        return r.match_key || r.label || `Replay ${r.id}`;
    }

    renderReplayCard(r) {
        const id = this.escapeHtml(r.id);
        const label = this.escapeHtml(this.formatMatchTitle(r));
        const blueTeams = (r.blue_teams || []).filter(Boolean).join(', ') || '—';
        const redTeams = (r.red_teams || []).filter(Boolean).join(', ') || '—';
        const blueScore = r.blue_score != null ? r.blue_score : '—';
        const redScore = r.red_score != null ? r.red_score : '—';
        const date = r.created_at ? new Date(r.created_at).toLocaleString() : '';
        const duration = r.duration_sec != null ? `${Math.round(r.duration_sec)}s` : '';
        const procTime = r.processing_sec != null ? this.formatDuration(r.processing_sec) : null;
        const partialBadge = r.status === 'partial'
            ? '<span class="replay-badge partial">PARTIAL</span>' : '';
        const thumb = r.thumb_url
            ? `<img class="replay-thumb" src="${this.escapeHtml(r.thumb_url)}" alt="${label} thumbnail" loading="lazy">`
            : '<div class="replay-thumb placeholder">No preview</div>';

        return `
            <div class="replay-card">
                <div class="replay-thumb-wrap">
                    ${thumb}
                    ${partialBadge}
                </div>
                <div class="replay-card-body">
                    <div class="replay-card-title">${label}</div>
                    <div class="replay-card-score">
                        <span class="blue">${blueScore}</span>
                        <span class="dash">-</span>
                        <span class="red">${redScore}</span>
                    </div>
                    <div class="replay-card-teams">
                        <span class="blue">${this.escapeHtml(blueTeams)}</span>
                        <span class="vs">vs</span>
                        <span class="red">${this.escapeHtml(redTeams)}</span>
                    </div>
                    <div class="replay-card-meta">${this.escapeHtml(date)}${duration ? ` &middot; ${duration}` : ''}</div>
                    ${procTime ? `<div class="replay-card-proctime">⏱ Processed in ${procTime}</div>` : ''}
                    <div class="replay-card-actions">
                        <button class="btn-small" data-replay-action="play" data-replay-id="${id}">Replay</button>
                        <button class="btn-small replay-delete" data-replay-action="delete" data-replay-id="${id}">Delete</button>
                    </div>
                </div>
            </div>
        `;
    }

    async deleteReplay(id) {
        if (!confirm('Delete this replay? This cannot be undone.')) return;
        try {
            const resp = await fetch(`/api/replays/${id}`, { method: 'DELETE' });
            if (!resp.ok) {
                const err = await resp.json().catch(() => ({}));
                alert(`Delete failed: ${err.error || resp.statusText}`);
                return;
            }
            this.loadReplays();
        } catch (err) {
            alert(`Delete failed: ${err.message}`);
        }
    }

    // --- Diagnostic viewer (debug-only) ---

    initDiagnosticViewer() {
        const MAX_LINES = 2000;
        this._diagnosticMaxLines = MAX_LINES;
        this.refreshDiagnosticButtonVisibility();

        // Scorer-view instance, keyed off the live job id.
        if (this.diagnosticToggleBtn) {
            this.diagnosticToggleBtn.addEventListener('click', () => {
                const showing = this.diagnosticPanel.style.display !== 'none';
                if (showing) {
                    this.diagnosticPanel.style.display = 'none';
                } else {
                    this.diagnosticPanel.style.display = 'block';
                    this.loadDiagnostic(this.jobId, {
                        panel: this.diagnosticPanel,
                        statusEl: this.diagnosticStatus,
                        linesEl: this.diagnosticLines,
                        rawKey: '_diagnosticRawLines',
                        totalKey: '_diagnosticTotalLines',
                        filterValue: this.diagnosticFilter ? this.diagnosticFilter.value : '',
                    });
                }
            });
        }
        if (this.diagnosticReloadBtn) {
            this.diagnosticReloadBtn.addEventListener('click', () => {
                this.loadDiagnostic(this.jobId, {
                    panel: this.diagnosticPanel,
                    statusEl: this.diagnosticStatus,
                    linesEl: this.diagnosticLines,
                    rawKey: '_diagnosticRawLines',
                    totalKey: '_diagnosticTotalLines',
                    filterValue: this.diagnosticFilter ? this.diagnosticFilter.value : '',
                });
            });
        }
        if (this.diagnosticFilter) {
            this.diagnosticFilter.addEventListener('input', () => {
                this.renderDiagnosticLines(this._diagnosticRawLines, this._diagnosticTotalLines,
                    this.diagnosticStatus, this.diagnosticLines, this.diagnosticFilter.value);
            });
        }

        // Replay-player instance, keyed off the replay/job id from openReplay().
        if (this.replayDiagnosticToggleBtn) {
            this.replayDiagnosticToggleBtn.addEventListener('click', () => {
                const showing = this.replayDiagnosticPanel.style.display !== 'none';
                if (showing) {
                    this.replayDiagnosticPanel.style.display = 'none';
                } else {
                    this.replayDiagnosticPanel.style.display = 'block';
                    this.loadDiagnostic(this._replayDiagnosticJobId, {
                        panel: this.replayDiagnosticPanel,
                        statusEl: this.replayDiagnosticStatus,
                        linesEl: this.replayDiagnosticLines,
                        rawKey: '_replayDiagnosticRawLines',
                        totalKey: '_replayDiagnosticTotalLines',
                        filterValue: this.replayDiagnosticFilter ? this.replayDiagnosticFilter.value : '',
                    });
                }
            });
        }
        if (this.replayDiagnosticReloadBtn) {
            this.replayDiagnosticReloadBtn.addEventListener('click', () => {
                this.loadDiagnostic(this._replayDiagnosticJobId, {
                    panel: this.replayDiagnosticPanel,
                    statusEl: this.replayDiagnosticStatus,
                    linesEl: this.replayDiagnosticLines,
                    rawKey: '_replayDiagnosticRawLines',
                    totalKey: '_replayDiagnosticTotalLines',
                    filterValue: this.replayDiagnosticFilter ? this.replayDiagnosticFilter.value : '',
                });
            });
        }
        if (this.replayDiagnosticFilter) {
            this.replayDiagnosticFilter.addEventListener('input', () => {
                this.renderDiagnosticLines(this._replayDiagnosticRawLines, this._replayDiagnosticTotalLines,
                    this.replayDiagnosticStatus, this.replayDiagnosticLines, this.replayDiagnosticFilter.value);
            });
        }
    }

    // Reveal the scorer-view "Diagnostics" button once a job id exists (call
    // whenever this.jobId is assigned: handleSubmit, rejoinActiveJob, etc).
    refreshDiagnosticButtonVisibility() {
        if (this.diagnosticToggleBtn) {
            this.diagnosticToggleBtn.style.display = this.jobId ? 'inline-flex' : 'none';
        }
    }

    async loadDiagnostic(key, target) {
        const { statusEl, linesEl, rawKey, totalKey } = target;
        if (!key) {
            if (statusEl) statusEl.textContent = 'no diagnostic data';
            if (linesEl) linesEl.textContent = '';
            this[rawKey] = [];
            this[totalKey] = 0;
            return;
        }
        if (statusEl) statusEl.textContent = 'Loading…';
        try {
            const resp = await fetch(`/api/diagnostic/${encodeURIComponent(key)}`);
            if (resp.status === 404) {
                if (statusEl) statusEl.textContent = 'no diagnostic data';
                if (linesEl) linesEl.textContent = '';
                this[rawKey] = [];
                this[totalKey] = 0;
                return;
            }
            if (!resp.ok) {
                if (statusEl) statusEl.textContent = `Failed to load diagnostics (${resp.status})`;
                return;
            }
            const text = await resp.text();
            const allLines = text.split('\n').filter(line => line.length > 0);
            const total = allLines.length;
            const kept = allLines.slice(-this._diagnosticMaxLines);
            this[rawKey] = kept;
            this[totalKey] = total;
            this.renderDiagnosticLines(kept, total, statusEl, linesEl, target.filterValue || '');
        } catch (e) {
            if (statusEl) statusEl.textContent = `Failed to load diagnostics: ${e.message}`;
        }
    }

    renderDiagnosticLines(kept, total, statusEl, linesEl, filterValue) {
        if (!linesEl) return;
        const filter = (filterValue || '').trim();
        const filtered = filter ? kept.filter(line => line.includes(filter)) : kept;
        if (statusEl) {
            const capNote = total > this._diagnosticMaxLines
                ? `showing last ${this._diagnosticMaxLines} of ~${total}`
                : `${total} line${total === 1 ? '' : 's'}`;
            const filterNote = filter ? ` (${filtered.length} match filter)` : '';
            statusEl.textContent = capNote + filterNote;
        }
        // escapeHtml per line — diagnostic content is untrusted job output.
        linesEl.innerHTML = filtered.map(line => this.escapeHtml(line)).join('\n');
    }

    // --- Match Replay Player ---

    async openReplay(id) {
        // Switch to the replays view if invoked from elsewhere (e.g. results screen)
        this.switchToReplaysView();
        this.showReplayPlayerView();

        const labelEl = document.getElementById('replayPlayerLabel');
        labelEl.textContent = 'Loading…';

        // Reset player UI
        this._replayActiveEventIdx = -1;
        document.getElementById('replayScoreBlue').textContent = '0';
        document.getElementById('replayScoreRed').textContent = '0';
        document.getElementById('replayBlueAllianceScore').textContent = '0';
        document.getElementById('replayRedAllianceScore').textContent = '0';
        document.getElementById('replayHudFrame').textContent = '0 / 0';
        document.getElementById('replayHudTime').textContent = '0:00';
        document.getElementById('replayBlueRobotsTable').innerHTML =
            '<tr><td colspan="3" class="empty-state">No robots</td></tr>';
        document.getElementById('replayRedRobotsTable').innerHTML =
            '<tr><td colspan="3" class="empty-state">No robots</td></tr>';
        document.getElementById('replayTimelineContainer').innerHTML =
            '<p class="empty-state">No events</p>';

        // Diagnostic viewer tracks the replay/job id being viewed; collapse and
        // clear any previously loaded lines so switching replays can't leak data.
        this._replayDiagnosticJobId = id;
        this._replayDiagnosticRawLines = [];
        this._replayDiagnosticTotalLines = 0;
        if (this.replayDiagnosticPanel) this.replayDiagnosticPanel.style.display = 'none';
        if (this.replayDiagnosticLines) this.replayDiagnosticLines.textContent = '';
        if (this.replayDiagnosticStatus) this.replayDiagnosticStatus.textContent = '';
        if (this.replayDiagnosticFilter) this.replayDiagnosticFilter.value = '';

        try {
            const resp = await fetch(`/api/replays/${id}`);
            if (!resp.ok) {
                labelEl.textContent = 'Failed to load replay metadata';
                return;
            }
            const meta = await resp.json();
            this.replayMeta = meta;

            const summary = meta.summary || {};
            const info = meta.replay_info || {};
            labelEl.textContent = info.event_name
                ? this.formatMatchTitle(info)
                : (summary.match_label || `Replay ${id}`);
            // Job GUID kept visible (small) for troubleshooting
            const idEl = document.getElementById('replayPlayerId');
            if (idEl) idEl.textContent = id;

            // Processing-time breakdown for this match
            const ptCard = document.getElementById('replayTimingCard');
            if (ptCard) {
                if (summary.timings && summary.timings.total != null) {
                    document.getElementById('replayTimingBreakdown').innerHTML =
                        this.renderTimings(summary.timings);
                    ptCard.style.display = 'block';
                } else {
                    ptCard.style.display = 'none';
                }
            }

            // Point the <video> at the range-aware stream (native scrub/seek/speed)
            this.replayVideo.src = `/api/replays/${id}/video`;
            this.replayVideo.load();

            // Wire the export/download buttons (server sends Content-Disposition: attachment)
            const dlVideo = document.getElementById('replayDownloadVideo');
            const dlData = document.getElementById('replayDownloadData');
            if (dlVideo) dlVideo.href = `/api/replays/${id}/download/video`;
            if (dlData) dlData.href = `/api/replays/${id}/download/data`;

            // Render the full clickable scoring-event timeline up front
            const events = meta.events || [];
            this.updateTimeline(events, {
                container: document.getElementById('replayTimelineContainer'),
                onSeek: (t) => {
                    if (Number.isFinite(t)) {
                        this.replayVideo.currentTime = t;
                        this.replayVideo.play().catch(() => {});
                    }
                },
            });

            // Prime the displays from frame 0
            this.renderReplayFrame(0);

            this.loadReplayIdentities(id);
        } catch (err) {
            labelEl.textContent = `Error: ${err.message}`;
        }
    }

    switchToReplaysView() {
        document.querySelectorAll('.nav-btn').forEach(b => b.classList.remove('active'));
        const navBtn = document.querySelector('.nav-btn[data-view="replays"]');
        if (navBtn) navBtn.classList.add('active');
        document.getElementById('scorerView').style.display = 'none';
        document.getElementById('historyView').style.display = 'none';
        document.getElementById('replaysView').style.display = 'block';
    }

    onReplayTimeUpdate() {
        if (!this.replayMeta) return;
        const summary = this.replayMeta.summary || {};
        const fps = summary.fps || 30;
        const frames = this.replayMeta.frames || [];
        if (frames.length === 0) return;

        let idx = Math.round(this.replayVideo.currentTime * fps);
        idx = Math.max(0, Math.min(frames.length - 1, idx));
        this.renderReplayFrame(idx);
    }

    // Whole-match label confidence: a slot whose label holds less than this
    // share of the slot's team evidence is flagged "low confidence — check".
    static get LOW_LABEL_SHARE() { return 0.6; }

    // "3128 · 62%  (alt 6773 · 21%)" line + low-confidence badge + a
    // "Use <runner-up>" button for one review row. Empty when the job ran
    // without the whole-match resolve (no share stored).
    reviewConfidenceHtml(r) {
        const pct = (v) => `${Math.round(Number(v) * 100)}%`;
        const share = r.label_share;
        const alt = r.runner_up_team;
        if (share == null && alt == null) return '';
        const team = this.escapeHtml(r.team_number || '—');
        // No share: an unlabelled slot, or a label corrected by hand to a
        // team other than the runner-up (its evidence share is unknown).
        const main = share != null ? `${team} · ${pct(share)}` : `${team} · —`;
        const altTxt = alt != null
            ? ` <span class="review-alt">(alt ${this.escapeHtml(alt)}`
              + `${r.runner_up_share != null ? ' · ' + pct(r.runner_up_share) : ''})</span>`
            : '';
        const low = share != null && Number(share) < VisionScorer.LOW_LABEL_SHARE;
        const badge = low && r.team_number
            ? '<span class="low-conf-badge" title="The label holds a small share of this slot\'s read evidence">low confidence — check</span>'
            : '';
        const useBtn = alt != null
            ? `<button class="review-use-alt" data-slot="${r.slot}" data-team="${this.escapeHtml(alt)}"
                       title="Correct this slot to the runner-up team">Use ${this.escapeHtml(alt)}</button>`
            : '';
        return `<div class="review-conf${low ? ' low' : ''}">
                    <span class="review-share">${main}</span>${altTxt}${badge}${useBtn}
                </div>`;
    }

    async loadReplayIdentities(id) {
        this._replayId = id;
        const replayId = id;
        const card = document.getElementById('replayReviewCard');
        const grid = document.getElementById('replayReviewGrid');
        if (!card || !grid) return;
        try {
            const resp = await fetch(`/api/replays/${id}/identities`);
            if (!resp.ok) { card.style.display = 'none'; return; }
            const data = await resp.json();
            const robots = data.robots || [];
            if (robots.length === 0) { card.style.display = 'none'; return; }
            grid.innerHTML = robots.map(r => {
                const imgs = (r.crops || []).map(src =>
                    `<img class="review-crop" src="${src}" alt="${this.escapeHtml(r.slot_label || '')} crop">`).join('')
                    || '<span class="empty-state">no image</span>';
                const badge = r.source === 'gallery' ? '<span class="appearance-badge">appearance</span>' : '';
                return `
                <div class="review-robot" data-slot="${r.slot}">
                    <div class="review-imgs">${imgs}</div>
                    <div class="review-meta">
                        <strong>${this.escapeHtml(r.slot_label || ('slot ' + r.slot))}</strong> ${badge}
                        <input class="review-team" type="text" value="${this.escapeHtml(r.team_number || '')}"
                               placeholder="team #">
                        <button class="review-save" data-slot="${r.slot}">Save</button>
                        <span class="review-status"></span>
                    </div>
                    ${this.reviewConfidenceHtml(r)}
                </div>`;
            }).join('');
            grid.querySelectorAll('.review-save').forEach(btn => {
                btn.addEventListener('click', () => this.saveIdentityCorrection(btn, replayId));
            });
            grid.querySelectorAll('.review-use-alt').forEach(btn => {
                btn.addEventListener('click', async () => {
                    // Same correct endpoint as Save, with the runner-up team;
                    // reload so the swapped share/runner-up line re-renders.
                    const wrap = btn.closest('.review-robot');
                    wrap.querySelector('.review-team').value = btn.dataset.team;
                    const ok = await this.saveIdentityCorrection(
                        wrap.querySelector('.review-save'), replayId);
                    if (ok) this.loadReplayIdentities(replayId);
                });
            });
            card.style.display = 'block';
        } catch (e) {
            card.style.display = 'none';
        }
    }

    async saveIdentityCorrection(btn, replayId) {
        const wrap = btn.closest('.review-robot');
        const slot = Number(btn.dataset.slot);
        const team = wrap.querySelector('.review-team').value.trim();
        const status = wrap.querySelector('.review-status');
        status.textContent = '…';
        try {
            const resp = await fetch(`/api/replays/${replayId}/correct`, {
                method: 'POST', headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ slot, team_number: team }),
            });
            const data = await resp.json().catch(() => ({}));
            status.textContent = resp.ok
                ? `✓ verified (${data.verified_exemplars || 0})` : `✗ ${data.error || ('HTTP ' + resp.status)}`;
            return resp.ok;
        } catch (e) {
            status.textContent = '✗ ' + e.message;
            return false;
        }
    }

    renderReplayFrame(idx) {
        if (!this.replayMeta) return;
        const frames = this.replayMeta.frames || [];
        const frame = frames[idx];
        if (!frame) return;

        // Score display
        const scores = frame.scores || {};
        const blue = scores.blue != null ? scores.blue : 0;
        const red = scores.red != null ? scores.red : 0;
        document.getElementById('replayScoreBlue').textContent = blue;
        document.getElementById('replayScoreRed').textContent = red;
        document.getElementById('replayBlueAllianceScore').textContent = blue;
        document.getElementById('replayRedAllianceScore').textContent = red;

        // HUD
        const total = frames.length;
        document.getElementById('replayHudFrame').textContent =
            `${frame.i != null ? frame.i : idx} / ${total}`;
        document.getElementById('replayHudTime').textContent =
            this.formatTimestamp(frame.t != null ? frame.t : this.replayVideo.currentTime);

        // Robot tables — map replay robot fields to the shape updateRobotTables expects.
        // Replay robots carry {slot, alliance, team_number, confirmed, field_pos}; no per-robot
        // goal count, so `scores` is left absent (table renders '—'). `position` is null here.
        const robots = (frame.robots || []).map(r => ({
            slot: r.slot,
            alliance: r.alliance,
            team_number: r.team_number,
            confirmed: r.confirmed,
            position: null,
        }));
        this.updateRobotTables(
            robots,
            document.getElementById('replayBlueRobotsTable'),
            document.getElementById('replayRedRobotsTable'),
        );

        // Highlight current scoring event (largest event.t <= currentTime)
        this.highlightReplayEvent();
    }

    highlightReplayEvent() {
        if (!this.replayMeta) return;
        const events = this.replayMeta.events || [];
        if (events.length === 0) return;

        const now = this.replayVideo.currentTime;
        let activeIdx = -1;
        for (let i = 0; i < events.length; i++) {
            if ((events[i].t || 0) <= now) activeIdx = i;
            else break;
        }
        if (activeIdx === this._replayActiveEventIdx) return;
        this._replayActiveEventIdx = activeIdx;

        const container = document.getElementById('replayTimelineContainer');
        container.querySelectorAll('.timeline-event').forEach(el => {
            const idx = parseInt(el.dataset.eventIdx, 10);
            el.classList.toggle('active', idx === activeIdx);
        });
        const activeEl = container.querySelector(`.timeline-event[data-event-idx="${activeIdx}"]`);
        if (activeEl) activeEl.scrollIntoView({ block: 'nearest' });
    }

    // --- Match Detail Modal ---

    async openMatchDetail(matchKey) {
        const modal = document.getElementById('matchDetailModal');
        const body = document.getElementById('modalBody');
        const title = document.getElementById('modalTitle');

        body.innerHTML = '<p>Loading...</p>';
        modal.style.display = 'flex';
        modal.dataset.matchKey = matchKey;

        try {
            const resp = await fetch(`/api/event/match/${matchKey}`);
            const match = await resp.json();

            title.textContent = matchKey;

            const blueRobots = match.robots.filter(r => r.alliance === 'blue').sort((a, b) => a.position - b.position);
            const redRobots = match.robots.filter(r => r.alliance === 'red').sort((a, b) => a.position - b.position);

            body.innerHTML = `
                <div class="modal-scores">
                    <div class="modal-score-group">
                        <h4>TBA Ground Truth</h4>
                        <div class="modal-score-row">
                            <span class="blue">Blue: ${match.tba_blue_hub_total != null ? match.tba_blue_hub_total : '—'} fuel</span>
                            <span class="red">Red: ${match.tba_red_hub_total != null ? match.tba_red_hub_total : '—'} fuel</span>
                        </div>
                    </div>
                    <div class="modal-score-group">
                        <h4>Predicted</h4>
                        <div class="modal-score-row">
                            <span class="blue">Blue: ${match.blue_predicted_score || 0}</span>
                            <span class="red">Red: ${match.red_predicted_score || 0}</span>
                        </div>
                    </div>
                </div>

                <div class="modal-robots">
                    <div class="modal-alliance blue">
                        <h4>Blue Alliance</h4>
                        <table class="edit-table">
                            <thead><tr><th>Pos</th><th>Team</th><th>Predicted</th><th>Corrected</th></tr></thead>
                            <tbody>
                                ${blueRobots.map(r => `
                                    <tr data-robot-id="${r.id}">
                                        <td>${r.position}</td>
                                        <td><input type="text" class="edit-team-input modal-edit" value="${r.team_number}" data-field="team_number"></td>
                                        <td>${r.predicted_fuel_count}</td>
                                        <td><input type="number" class="edit-count-input modal-edit" value="${r.corrected_fuel_count != null ? r.corrected_fuel_count : ''}" placeholder="${r.predicted_fuel_count}" min="0" data-field="corrected_fuel_count"></td>
                                    </tr>
                                `).join('')}
                            </tbody>
                        </table>
                    </div>
                    <div class="modal-alliance red">
                        <h4>Red Alliance</h4>
                        <table class="edit-table">
                            <thead><tr><th>Pos</th><th>Team</th><th>Predicted</th><th>Corrected</th></tr></thead>
                            <tbody>
                                ${redRobots.map(r => `
                                    <tr data-robot-id="${r.id}">
                                        <td>${r.position}</td>
                                        <td><input type="text" class="edit-team-input modal-edit" value="${r.team_number}" data-field="team_number"></td>
                                        <td>${r.predicted_fuel_count}</td>
                                        <td><input type="number" class="edit-count-input modal-edit" value="${r.corrected_fuel_count != null ? r.corrected_fuel_count : ''}" placeholder="${r.predicted_fuel_count}" min="0" data-field="corrected_fuel_count"></td>
                                    </tr>
                                `).join('')}
                            </tbody>
                        </table>
                    </div>
                </div>

                <div class="modal-notes">
                    <label>Notes:</label>
                    <textarea class="modal-edit" id="modalNotes" rows="2">${match.notes || ''}</textarea>
                </div>
            `;
        } catch (err) {
            body.innerHTML = `<p class="error">Error: ${err.message}</p>`;
        }
    }

    async saveModalChanges() {
        const matchKey = document.getElementById('matchDetailModal').dataset.matchKey;
        const body = document.getElementById('modalBody');

        const robots = [];
        body.querySelectorAll('tr[data-robot-id]').forEach(row => {
            const robotId = parseInt(row.dataset.robotId);
            const teamInput = row.querySelector('[data-field="team_number"]');
            const corrInput = row.querySelector('[data-field="corrected_fuel_count"]');

            robots.push({
                id: robotId,
                team_number: teamInput.value.trim(),
                corrected_fuel_count: corrInput.value !== '' ? parseInt(corrInput.value) : null,
            });
        });

        const notesEl = document.getElementById('modalNotes');

        try {
            const resp = await fetch('/api/match/save', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    match_key: matchKey,
                    notes: notesEl ? notesEl.value.trim() : null,
                    robots,
                }),
            });

            if (resp.ok) {
                this.closeModal();
                this.loadHistory();  // Refresh
            } else {
                const err = await resp.json();
                alert(`Save failed: ${err.error || 'Unknown error'}`);
            }
        } catch (err) {
            alert(`Save failed: ${err.message}`);
        }
    }

    closeModal() {
        document.getElementById('matchDetailModal').style.display = 'none';
    }
}

// Initialize on page load
let scorer;
document.addEventListener('DOMContentLoaded', () => {
    scorer = new VisionScorer();
});
