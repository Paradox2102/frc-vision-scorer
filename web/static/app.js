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

        this.init();
    }

    init() {
        // Event form
        document.getElementById('eventForm').addEventListener('submit', (e) => this.handleEventLoad(e));

        // Match selector
        document.getElementById('matchSelect').addEventListener('change', (e) => this.handleMatchSelect(e));

        // Upload form
        document.getElementById('uploadForm').addEventListener('submit', (e) => this.handleSubmit(e));
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
            this.replayVideo.addEventListener('loadedmetadata', () => this.drawReplayFieldMap([]));
        }

        // Modal
        document.getElementById('modalClose').addEventListener('click', () => this.closeModal());
        document.getElementById('modalCancelBtn').addEventListener('click', () => this.closeModal());
        document.getElementById('modalSaveBtn').addEventListener('click', () => this.saveModalChanges());

        // Check if we have an active event
        this.checkExistingEvent();
    }

    // --- Navigation ---

    switchView(view) {
        document.querySelectorAll('.nav-btn').forEach(b => b.classList.remove('active'));
        document.querySelector(`.nav-btn[data-view="${view}"]`).classList.add('active');

        document.getElementById('scorerView').style.display = view === 'scorer' ? 'block' : 'none';
        document.getElementById('historyView').style.display = view === 'history' ? 'block' : 'none';
        document.getElementById('replaysView').style.display = view === 'replays' ? 'block' : 'none';

        if (view === 'history') {
            this.loadHistory();
        } else if (view === 'replays') {
            // Leaving any open player when navigating back to the library
            if (this.replayVideo) this.replayVideo.pause();
            this.showReplayLibrary();
            this.loadReplays();
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
        
        this.ocrRegions = { red_center: null, blue_center: null };
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
    }

    startOcrDraw(alliance) {
        this.ocrDrawMode = alliance;
        this.hubMarkMode = null;
        const canvas = document.getElementById('ocrCanvas');
        canvas.style.cursor = 'crosshair';
        document.getElementById('ocrStatus').textContent = 
            `Draw a box around the ${alliance.toUpperCase()} alliance score number.`;
        
        // Highlight active button
        document.getElementById('ocrDrawRed').classList.toggle('active', alliance === 'red');
        document.getElementById('ocrDrawBlue').classList.toggle('active', alliance === 'blue');
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
        ctx.strokeStyle = this.ocrDrawMode === 'red' ? '#e74c3c' : '#3498db';
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
        
        const key = this.ocrDrawMode + '_center';
        this.ocrRegions[key] = [Math.round(rx), Math.round(ry), Math.round(rw), Math.round(rh)];
        
        this.ocrDrawMode = null;
        canvas.style.cursor = 'default';
        document.getElementById('ocrDrawRed').classList.remove('active');
        document.getElementById('ocrDrawBlue').classList.remove('active');
        
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
            const color = key.startsWith('red') ? '#e74c3c' : '#3498db';
            const label = key.startsWith('red') ? 'RED SCORE' : 'BLUE SCORE';
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

        // Populate match dropdown
        const select = document.getElementById('matchSelect');
        select.innerHTML = '<option value="">-- Choose a match --</option>';

        let currentLevel = '';
        for (const match of this.eventData.matches) {
            const levelLabel = { qm: 'Qualifications', sf: 'Semifinals', f: 'Finals' }[match.comp_level] || match.comp_level;
            if (match.comp_level !== currentLevel) {
                const optgroup = document.createElement('optgroup');
                optgroup.label = levelLabel;
                select.appendChild(optgroup);
                currentLevel = match.comp_level;
            }

            const opt = document.createElement('option');
            opt.value = match.key;

            const blueTeams = match.blue_teams.join(', ');
            const redTeams = match.red_teams.join(', ');
            const scoreStr = match.blue_score != null ? ` [${match.blue_score}-${match.red_score}]` : '';
            opt.textContent = `${match.label} — Blue: ${blueTeams} vs Red: ${redTeams}${scoreStr}`;
            opt.dataset.matchData = JSON.stringify(match);

            // Append to last optgroup
            const groups = select.querySelectorAll('optgroup');
            if (groups.length > 0) {
                groups[groups.length - 1].appendChild(opt);
            } else {
                select.appendChild(opt);
            }
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
        const teamReader = document.querySelector('input[name="teamReader"]:checked')?.value || 'local';
        formData.append('team_reader', teamReader);
        const vramBudget = parseInt(document.getElementById('sam2MaxGpuFrames')?.value, 10);
        if (Number.isFinite(vramBudget) && vramBudget > 0) {
            formData.append('sam2_max_gpu_frames', vramBudget);
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

            // Store the match video URL for calibration frame grab
            this._pendingMatchVideoUrl = youtubeUrl || null;

            // Always show OCR calibration before starting inference
            this._postCalibrationTarget = 'inference';
            this.showOcrSetup(this._pendingMatchVideoUrl);
        } catch (error) {
            this.setStatus(`Error: ${error.message}`, 0);
        }
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
            const data = JSON.parse(event.data);
            this.handleWebSocketMessage(data);
        };

        this.ws.onerror = (error) => {
            this.setStatus('Connection error', 0);
        };

        this.ws.onclose = () => {};
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
        }
    }

    handleStatus(data) {
        if (data.status === 'paused') {
            const currentProgress = parseFloat(this.progressBar.style.width) || 0;
            this.setStatus(`Paused at frame ${data.frame_num}`, currentProgress);
        } else if (data.message) {
            const currentProgress = data.progress !== undefined ? data.progress : (parseFloat(this.progressBar.style.width) || 0);
            this.setStatus(data.message, currentProgress);
        }
    }

    updateFrame(data) {
        const progress = (data.frame_num / data.total_frames) * 100;
        this.setStatus(`Processing frame ${data.frame_num} of ${data.total_frames}`, progress);

        this.hudFrame.textContent = `${data.frame_num} / ${data.total_frames}`;
        this.hudFps.textContent = data.fps.toFixed(1);
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
        if (robots.length === 0) {
            tableBody.innerHTML = '<tr><td colspan="3" class="empty-state">No robots detected</td></tr>';
            return;
        }

        robots.sort((a, b) => (b.scores || 0) - (a.scores || 0));

        const rows = robots.map(robot => {
            const label = robot.slot || (robot.id != null ? `Robot ${robot.id}` : '—');
            const goals = robot.scores != null ? robot.scores : '—';
            return `
                <tr>
                    <td>${label}</td>
                    <td>${robot.team_number || '—'}</td>
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
                            ${this.escapeHtml(crop.slot)} ${this.escapeHtml(raw)} &gt; ${this.escapeHtml(matched)}
                        </div>
                        <div class="vlm-subtext">${confidence} confidence</div>
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
                            Frame ${frame} | ${timestamp}
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
            sam2_tracking: 'SAM2 tracking', analysis_loop: 'Analysis loop',
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
        this.setStatus('Processing complete!', 100);
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
        this.setInferenceControlState('hidden');
        this.setStatus(`Error: ${data.error}`, 0);
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

        // Reset robot tables
        this.blueRobotsTable.innerHTML = '<tr><td colspan="3" class="empty-state">Waiting for detection...</td></tr>';
        this.redRobotsTable.innerHTML = '<tr><td colspan="3" class="empty-state">Waiting for detection...</td></tr>';
        if (this.vlmCrops) this.vlmCrops.innerHTML = '<p class="empty-state">No VLM crops sent yet</p>';
        this.latestVlmCrops.clear();
        this.timelineContainer.innerHTML = '<p class="empty-state">No events yet</p>';

        this.setStatus('Ready', 0);
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
                    <h3>${event.event_name} <span class="event-year">${event.year}</span></h3>
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
                            <td><strong>${matchLabel}</strong></td>
                            <td class="team-cell blue">${blueTeams}</td>
                            <td class="team-cell red">${redTeams}</td>
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

    renderReplayCard(r) {
        const id = this.escapeHtml(r.id);
        const label = this.escapeHtml(r.label || `Replay ${r.id}`);
        const blueTeams = (r.blue_teams || []).join(', ') || '—';
        const redTeams = (r.red_teams || []).join(', ') || '—';
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

        try {
            const resp = await fetch(`/api/replays/${id}`);
            if (!resp.ok) {
                labelEl.textContent = 'Failed to load replay metadata';
                return;
            }
            const meta = await resp.json();
            this.replayMeta = meta;

            const summary = meta.summary || {};
            labelEl.textContent = summary.match_label || `Replay ${id}`;

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

        // Client-side field map
        this.drawReplayFieldMap(frame.robots || []);
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

    drawReplayFieldMap(robots) {
        const canvas = document.getElementById('replayFieldMap');
        if (!canvas) return;
        const ctx = canvas.getContext('2d');
        const W = canvas.width;
        const H = canvas.height;

        // 2026 REBUILT field dimensions (metres).
        // Source: data/field/2026-rebuilt-welded.json -> field.length / field.width,
        // loaded by web/field/field_model.py (FieldModel.length_m / width_m).
        const FIELD_LENGTH_M = 16.541;
        const FIELD_WIDTH_M = 8.069;

        // Fit the field rectangle inside the canvas with a small margin, preserving aspect.
        const margin = 12;
        const availW = W - margin * 2;
        const availH = H - margin * 2;
        const scale = Math.min(availW / FIELD_LENGTH_M, availH / FIELD_WIDTH_M);
        const fieldW = FIELD_LENGTH_M * scale;
        const fieldH = FIELD_WIDTH_M * scale;
        const ox = (W - fieldW) / 2;
        const oy = (H - fieldH) / 2;

        ctx.clearRect(0, 0, W, H);

        // Blue (left) / red (right) halves
        ctx.fillStyle = 'rgba(59, 130, 246, 0.12)';
        ctx.fillRect(ox, oy, fieldW / 2, fieldH);
        ctx.fillStyle = 'rgba(239, 68, 68, 0.12)';
        ctx.fillRect(ox + fieldW / 2, oy, fieldW / 2, fieldH);

        // Field border
        ctx.strokeStyle = '#2a3550';
        ctx.lineWidth = 2;
        ctx.strokeRect(ox, oy, fieldW, fieldH);

        // Center line
        ctx.beginPath();
        ctx.moveTo(ox + fieldW / 2, oy);
        ctx.lineTo(ox + fieldW / 2, oy + fieldH);
        ctx.stroke();

        // Plot robots
        for (const r of robots) {
            const pos = r.field_pos;
            if (!Array.isArray(pos) || pos.length < 2) continue;
            const [x, y] = pos;
            if (!Number.isFinite(x) || !Number.isFinite(y)) continue;
            const px = ox + (x / FIELD_LENGTH_M) * fieldW;
            const py = oy + (y / FIELD_WIDTH_M) * fieldH;
            ctx.beginPath();
            ctx.arc(px, py, 7, 0, Math.PI * 2);
            ctx.fillStyle = r.alliance === 'red' ? '#ef4444' : '#3b82f6';
            ctx.fill();
            ctx.lineWidth = 1.5;
            ctx.strokeStyle = 'rgba(0,0,0,0.4)';
            ctx.stroke();
            if (r.team_number) {
                ctx.fillStyle = '#e8eaf0';
                ctx.font = 'bold 10px sans-serif';
                ctx.textAlign = 'center';
                ctx.fillText(String(r.team_number), px, py - 10);
            }
        }
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
