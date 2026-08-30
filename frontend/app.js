// YouTube Video Intelligence & Executive Summary Studio - Frontend Engine

document.addEventListener('DOMContentLoaded', () => {
    // =========================================================================
    // 1. 3D Animated Interactive Background Engine (Canvas 3D Mesh)
    // =========================================================================
    const canvas = document.getElementById('bg3dCanvas');
    if (canvas) {
        const ctx = canvas.getContext('2d');
        let width = (canvas.width = window.innerWidth);
        let height = (canvas.height = window.innerHeight);

        window.addEventListener('resize', () => {
            width = canvas.width = window.innerWidth;
            height = canvas.height = window.innerHeight;
        });

        // 3D Particles & Constellation Nodes
        const NUM_PARTICLES = 85;
        const FOCAL_LENGTH = 400;
        const particles = [];

        for (let i = 0; i < NUM_PARTICLES; i++) {
            particles.push({
                x: (Math.random() - 0.5) * width * 1.5,
                y: (Math.random() - 0.5) * height * 1.5,
                z: Math.random() * 800 - 200,
                vx: (Math.random() - 0.5) * 0.35,
                vy: (Math.random() - 0.5) * 0.35,
                vz: (Math.random() - 0.5) * 0.45,
                radius: Math.random() * 2 + 1,
                color: Math.random() > 0.5 ? 'rgba(0, 242, 254,' : 'rgba(139, 92, 246,'
            });
        }

        // Mouse Parallax
        let targetRotX = 0;
        let targetRotY = 0;
        let rotX = 0;
        let rotY = 0;

        window.addEventListener('mousemove', (e) => {
            const normX = (e.clientX / width) * 2 - 1;
            const normY = (e.clientY / height) * 2 - 1;
            targetRotY = normX * 0.25;
            targetRotX = -normY * 0.25;
        });

        function rotateX3D(p, angle) {
            const cos = Math.cos(angle);
            const sin = Math.sin(angle);
            const y = p.y * cos - p.z * sin;
            const z = p.z * cos + p.y * sin;
            return { x: p.x, y: y, z: z };
        }

        function rotateY3D(p, angle) {
            const cos = Math.cos(angle);
            const sin = Math.sin(angle);
            const x = p.x * cos + p.z * sin;
            const z = p.z * cos - p.x * sin;
            return { x: x, y: p.y, z: z };
        }

        function render3DBackground() {
            ctx.clearRect(0, 0, width, height);

            rotX += (targetRotX - rotX) * 0.05;
            rotY += (targetRotY - rotY) * 0.05;

            const cx = width / 2;
            const cy = height / 2;
            const projected = [];

            for (let i = 0; i < particles.length; i++) {
                const p = particles[i];
                p.x += p.vx;
                p.y += p.vy;
                p.z += p.vz;

                if (p.x < -width) p.x = width;
                if (p.x > width) p.x = -width;
                if (p.y < -height) p.y = height;
                if (p.y > height) p.y = -height;
                if (p.z < -200) p.z = 600;
                if (p.z > 600) p.z = -200;

                let rotated = rotateX3D(p, rotX);
                rotated = rotateY3D(rotated, rotY);

                const depth = rotated.z + FOCAL_LENGTH;
                if (depth > 10) {
                    const scale = FOCAL_LENGTH / depth;
                    const px = rotated.x * scale + cx;
                    const py = rotated.y * scale + cy;
                    const alpha = Math.min(1, Math.max(0.1, (1 - rotated.z / 600) * 0.7));

                    projected.push({ x: px, y: py, scale: scale, alpha: alpha, color: p.color, radius: p.radius });

                    ctx.beginPath();
                    ctx.arc(px, py, p.radius * scale, 0, Math.PI * 2);
                    ctx.fillStyle = `${p.color} ${alpha})`;
                    ctx.fill();
                } else {
                    projected.push(null);
                }
            }

            // Draw Constellation Lines
            ctx.lineWidth = 0.6;
            for (let i = 0; i < projected.length; i++) {
                const p1 = projected[i];
                if (!p1) continue;

                for (let j = i + 1; j < projected.length; j++) {
                    const p2 = projected[j];
                    if (!p2) continue;

                    const dx = p1.x - p2.x;
                    const dy = p1.y - p2.y;
                    const dist = Math.sqrt(dx * dx + dy * dy);

                    if (dist < 130) {
                        const lineAlpha = (1 - dist / 130) * Math.min(p1.alpha, p2.alpha) * 0.4;
                        ctx.beginPath();
                        ctx.moveTo(p1.x, p1.y);
                        ctx.lineTo(p2.x, p2.y);
                        ctx.strokeStyle = `rgba(0, 242, 254, ${lineAlpha})`;
                        ctx.stroke();
                    }
                }
            }

            requestAnimationFrame(render3DBackground);
        }

        render3DBackground();
    }

    // =========================================================================
    // 2. DOM Elements & State
    // =========================================================================
    const extractForm = document.getElementById('extractForm');
    const youtubeUrlInput = document.getElementById('youtubeUrl');
    const urlInputContainer = document.querySelector('.url-input-container');
    const urlFeedback = document.getElementById('urlFeedback');
    const submitBtn = document.getElementById('submitBtn');
    const btnText = submitBtn.querySelector('.btn-text');
    const btnSpinner = submitBtn.querySelector('.btn-spinner');

    const targetLanguage = document.getElementById('targetLanguage');
    const customFocus = document.getElementById('customFocus');
    const requireHumanReview = document.getElementById('requireHumanReview');

    // Holographic Percentile Engine Elements
    const pipelineSection = document.getElementById('pipelineSection');
    const stopPipelineBtn = document.getElementById('stopPipelineBtn');
    const ringProgress = document.getElementById('ringProgress');
    const progressPercentage = document.getElementById('progressPercentage');
    const progressBar = document.getElementById('progressBar');
    const progressBarGlow = document.getElementById('progressBarGlow');
    const barHeadPip = document.getElementById('barHeadPip');
    const stageStatusText = document.getElementById('stageStatusText');
    const executionTimer = document.getElementById('executionTimer');
    const jobStatusBadge = document.getElementById('jobStatusBadge');

    // Milestone Chips
    const msUrl = document.getElementById('msUrl');
    const msTranscript = document.getElementById('msTranscript');
    const msAgents = document.getElementById('msAgents');
    const msTranslate = document.getElementById('msTranslate');
    const msHitl = document.getElementById('msHitl');
    const msPdf = document.getElementById('msPdf');

    // Supervisor Review Elements
    const hitlSection = document.getElementById('hitlSection');
    const hitlReasonText = document.getElementById('hitlReasonText');
    const hitlReviewerName = document.getElementById('hitlReviewerName');
    const hitlSummaryText = document.getElementById('hitlSummaryText');
    const hitlComments = document.getElementById('hitlComments');
    const hitlApproveBtn = document.getElementById('hitlApproveBtn');
    const hitlEditBtn = document.getElementById('hitlEditBtn');
    const hitlRejectBtn = document.getElementById('hitlRejectBtn');

    // Results Studio Elements
    const resultsSection = document.getElementById('resultsSection');
    const videoThumb = document.getElementById('videoThumb');
    const videoTitle = document.getElementById('videoTitle');
    const videoChannel = document.getElementById('videoChannel');
    const videoDuration = document.getElementById('videoDuration');
    const videoLang = document.getElementById('videoLang');
    const videoWords = document.getElementById('videoWords');
    const videoHitlStatus = document.getElementById('videoHitlStatus');
    const videoLink = document.getElementById('videoLink');
    const downloadPdfBtn = document.getElementById('downloadPdfBtn');
    const viewTranscriptBtn = document.getElementById('viewTranscriptBtn');

    // Document Sheet View Elements
    const docVideoTitle = document.getElementById('docVideoTitle');
    const docChannel = document.getElementById('docChannel');
    const docDuration = document.getElementById('docDuration');
    const docDate = document.getElementById('docDate');
    const docSafetySeal = document.getElementById('docSafetySeal');
    const docSummaryList = document.getElementById('docSummaryList');
    const docActionsList = document.getElementById('docActionsList');

    // Toolbar Action Buttons
    const openPdfNewTabBtn = document.getElementById('openPdfNewTabBtn');
    const downloadPdfInlineBtn = document.getElementById('downloadPdfInlineBtn');
    const bottomDownloadPdfBtn = document.getElementById('bottomDownloadPdfBtn');

    // Safety & Summary Elements
    const safetyBadge = document.getElementById('safetyBadge');
    const safetyAssessment = document.getElementById('safetyAssessment');
    const safetyGrid = document.getElementById('safetyGrid');

    const lineCountBadge = document.getElementById('lineCountBadge');
    const copySummaryBtn = document.getElementById('copySummaryBtn');
    const summaryContainer = document.getElementById('summaryContainer');
    const actionItemsCount = document.getElementById('actionItemsCount');
    const actionItemsContainer = document.getElementById('actionItemsContainer');

    // Raw Transcript Modal Elements
    const transcriptModal = document.getElementById('transcriptModal');
    const modalOverlay = document.getElementById('modalOverlay');
    const closeModalBtn = document.getElementById('closeModalBtn');
    const rawTranscriptText = document.getElementById('rawTranscriptText');

    let currentJobId = null;
    let currentJobData = null;
    let eventSource = null;
    let timerInterval = null;
    let startTime = null;

    // =========================================================================
    // 3. Strict YouTube URL Validation
    // =========================================================================
    const YT_REGEX = /^(https?:\/\/)?(www\.|m\.|music\.)?(youtube\.com\/(watch\?v=|shorts\/|embed\/|v\/)|youtu\.be\/)([a-zA-Z0-9_-]{11})(\S*)?$/;

    function validateUrlInput() {
        const val = youtubeUrlInput.value.trim();
        if (!val) {
            urlInputContainer.classList.remove('valid-url', 'invalid-url');
            urlFeedback.textContent = '';
            urlFeedback.className = 'url-feedback';
            return false;
        }

        if (YT_REGEX.test(val)) {
            urlInputContainer.classList.add('valid-url');
            urlInputContainer.classList.remove('invalid-url');
            urlFeedback.textContent = '✓ Verified YouTube Link';
            urlFeedback.className = 'url-feedback success';
            return true;
        } else {
            urlInputContainer.classList.add('invalid-url');
            urlInputContainer.classList.remove('valid-url');
            urlFeedback.textContent = '✗ Please paste a valid YouTube video link (e.g. youtube.com or youtu.be)';
            urlFeedback.className = 'url-feedback error';
            return false;
        }
    }

    youtubeUrlInput.addEventListener('input', validateUrlInput);

    // Preset Sample Video Chips
    document.querySelectorAll('.preset-chip').forEach(chip => {
        chip.addEventListener('click', () => {
            youtubeUrlInput.value = chip.getAttribute('data-url');
            validateUrlInput();
            youtubeUrlInput.focus();
        });
    });

    // =========================================================================
    // 4. Health Check Diagnostic
    // =========================================================================
    async function checkHealth() {
        try {
            const res = await fetch('/api/health');
            if (res.ok) {
                document.getElementById('healthText').innerHTML = 'AI Engine Ready';
            }
        } catch (e) {
            console.warn('Health check warning:', e);
        }
    }
    checkHealth();

    // =========================================================================
    // 5. Form Submission & Agent Dispatch
    // =========================================================================
    extractForm.addEventListener('submit', async (e) => {
        e.preventDefault();
        const url = youtubeUrlInput.value.trim();

        if (!validateUrlInput()) {
            youtubeUrlInput.focus();
            return;
        }

        // Reset UI State
        resultsSection.classList.add('hidden');
        hitlSection.classList.add('hidden');
        pipelineSection.classList.remove('hidden');
        submitBtn.disabled = true;
        btnSpinner.classList.remove('hidden');
        btnText.textContent = 'Analyzing Video...';

        // Reset progress bar styling
        progressBar.style.background = 'linear-gradient(90deg, var(--cyan-primary) 0%, var(--purple-primary) 50%, var(--emerald-primary) 100%)';
        ringProgress.style.stroke = 'var(--cyan-primary)';

        // Start Live Timer
        startTime = Date.now();
        if (timerInterval) clearInterval(timerInterval);
        timerInterval = setInterval(() => {
            const elapsed = ((Date.now() - startTime) / 1000).toFixed(1);
            executionTimer.textContent = `Time: ${elapsed}s`;
        }, 100);

        try {
            const response = await fetch('/api/extract', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    url: url,
                    target_language: targetLanguage.value,
                    custom_focus: customFocus.value.trim() || null,
                    require_human_review: requireHumanReview.checked
                })
            });

            if (!response.ok) {
                const err = await response.json();
                throw new Error(err.detail || 'Analysis initialization failed.');
            }

            const data = await response.json();
            currentJobId = data.job_id;

            if (data.cached && data.result) {
                updatePercentileHUD({ progress_percent: 100, status: 'COMPLETED', current_stage: 'Analysis retrieved from cache' });
                setTimeout(() => {
                    renderResults(data.result);
                    resetSubmitButton();
                }, 300);
            } else {
                listenToSSEStream(currentJobId);
            }
        } catch (err) {
            alert('Error starting analysis: ' + err.message);
            resetSubmitButton();
            pipelineSection.classList.add('hidden');
        }
    });

    // =========================================================================
    // 6. Stop Pipeline Execution (With User Confirmation Dialog)
    // =========================================================================
    if (stopPipelineBtn) {
        stopPipelineBtn.addEventListener('click', async () => {
            if (!currentJobId) {
                pipelineSection.classList.add('hidden');
                resetSubmitButton();
                return;
            }

            // Ask user confirmation before stopping
            const userConfirmed = window.confirm("Are you sure you want to stop this analysis?");
            if (!userConfirmed) {
                return; // User clicked Cancel, continue execution
            }

            // User gave permission to stop
            try {
                if (eventSource) eventSource.close();
                if (timerInterval) clearInterval(timerInterval);

                await fetch(`/api/cancel/${currentJobId}`, { method: 'POST' });

                stageStatusText.textContent = "Analysis stopped by user.";
                jobStatusBadge.textContent = "CANCELLED";
                jobStatusBadge.className = "hud-status-badge status-cancelled";
                progressBar.style.background = "var(--rose-danger)";
                progressBarGlow.style.background = "var(--rose-danger)";
                ringProgress.style.stroke = "var(--rose-danger)";

                resetSubmitButton();
            } catch (err) {
                console.error("Error cancelling job:", err);
                resetSubmitButton();
            }
        });
    }

    // =========================================================================
    // 7. Real-Time SSE Stream Listener & Polling Fallback
    // =========================================================================
    function listenToSSEStream(jobId) {
        if (eventSource) eventSource.close();

        eventSource = new EventSource(`/api/stream/${jobId}`);

        eventSource.onmessage = (event) => {
            try {
                const job = JSON.parse(event.data);
                currentJobData = job;
                updatePercentileHUD(job);

                if (job.status === 'COMPLETED') {
                    eventSource.close();
                    if (timerInterval) clearInterval(timerInterval);
                    renderResults(job);
                    resetSubmitButton();
                } else if (job.status === 'WAITING_HUMAN_REVIEW') {
                    eventSource.close();
                    if (timerInterval) clearInterval(timerInterval);
                    renderHITLDrawer(job);
                } else if (job.status === 'CANCELLED') {
                    eventSource.close();
                    if (timerInterval) clearInterval(timerInterval);
                    stageStatusText.textContent = "Analysis stopped by user.";
                    jobStatusBadge.textContent = "CANCELLED";
                    jobStatusBadge.className = "hud-status-badge status-cancelled";
                    resetSubmitButton();
                } else if (job.status === 'FAILED' || job.status === 'REJECTED') {
                    eventSource.close();
                    if (timerInterval) clearInterval(timerInterval);
                    alert(`Analysis halted: ${job.error_message || 'Content could not be processed'}`);
                    resetSubmitButton();
                }
            } catch (e) {
                console.error('Error parsing SSE event:', e);
            }
        };

        eventSource.onerror = () => {
            eventSource.close();
            startPollingFallback(jobId);
        };
    }

    function startPollingFallback(jobId) {
        const interval = setInterval(async () => {
            try {
                const res = await fetch(`/api/status/${jobId}`);
                if (res.ok) {
                    const job = await res.json();
                    currentJobData = job;
                    updatePercentileHUD(job);

                    if (job.status === 'COMPLETED') {
                        clearInterval(interval);
                        if (timerInterval) clearInterval(timerInterval);
                        renderResults(job);
                        resetSubmitButton();
                    } else if (job.status === 'WAITING_HUMAN_REVIEW') {
                        clearInterval(interval);
                        if (timerInterval) clearInterval(timerInterval);
                        renderHITLDrawer(job);
                    } else if (job.status === 'CANCELLED') {
                        clearInterval(interval);
                        if (timerInterval) clearInterval(timerInterval);
                        resetSubmitButton();
                    } else if (job.status === 'FAILED' || job.status === 'REJECTED') {
                        clearInterval(interval);
                        if (timerInterval) clearInterval(timerInterval);
                        resetSubmitButton();
                    }
                }
            } catch (e) {
                console.warn('Polling error:', e);
            }
        }, 1000);
    }

    function resetSubmitButton() {
        submitBtn.disabled = false;
        btnSpinner.classList.add('hidden');
        btnText.textContent = 'Extract Insights';
    }

    // =========================================================================
    // 8. Holographic Percentile HUD Engine
    // =========================================================================
    function updatePercentileHUD(job) {
        const pct = Math.min(100, Math.max(0, job.progress_percent || 5));
        
        // Linear Bar
        progressBar.style.width = pct + '%';
        progressBarGlow.style.width = pct + '%';
        barHeadPip.style.left = pct + '%';
        
        // Radial SVG Gauge
        const offset = 314.16 - (314.16 * pct / 100);
        ringProgress.style.strokeDashoffset = offset;
        progressPercentage.textContent = pct + '%';

        // Stage description
        stageStatusText.textContent = job.current_stage || 'Analyzing content...';
        jobStatusBadge.textContent = job.status === 'COMPLETED' ? 'COMPLETED' : (job.status || 'ANALYZING');
        jobStatusBadge.className = 'hud-status-badge' + (job.status === 'CANCELLED' ? ' status-cancelled' : '');

        // Milestone Micro-Chips
        const milestones = [
            { el: msUrl, threshold: 20 },
            { el: msTranscript, threshold: 40 },
            { el: msAgents, threshold: 75 },
            { el: msTranslate, threshold: 85 },
            { el: msHitl, threshold: 92 },
            { el: msPdf, threshold: 100 }
        ];

        milestones.forEach(({ el, threshold }) => {
            el.classList.remove('active', 'done');
            if (pct >= threshold) {
                el.classList.add('done');
            } else if (pct >= threshold - 15) {
                el.classList.add('active');
            }
        });
    }

    // =========================================================================
    // 9. Supervisor Review Drawer Handler
    // =========================================================================
    function renderHITLDrawer(job) {
        hitlSection.classList.remove('hidden');
        hitlReasonText.textContent = job.hitl_state?.triggered_reason || 'Please review summary lines.';
        
        if (job.summary && job.summary.lines) {
            hitlSummaryText.value = job.summary.lines.map((l, i) => `${i+1}. ${l}`).join('\n');
        }
    }

    async function sendHITLAction(action) {
        if (!currentJobId) return;

        let editedLines = null;
        if (action === 'edit') {
            const raw = hitlSummaryText.value.trim().split('\n');
            editedLines = raw.map(l => l.replace(/^\d+[\.\)]\s*/, '').trim()).filter(l => l.length > 0);
        }

        try {
            const res = await fetch(`/api/hitl/review/${currentJobId}`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    action: action,
                    reviewer_name: hitlReviewerName.value.trim() || 'Lead Reviewer',
                    comments: hitlComments.value.trim() || null,
                    edited_summary: editedLines
                })
            });

            if (!res.ok) {
                const err = await res.json();
                throw new Error(err.detail || 'Review action failed.');
            }

            const updatedJob = await res.json();
            hitlSection.classList.add('hidden');

            if (updatedJob.status === 'COMPLETED') {
                renderResults(updatedJob);
                updatePercentileHUD(updatedJob);
            } else if (updatedJob.status === 'REJECTED') {
                updatePercentileHUD(updatedJob);
                alert('Analysis was cancelled.');
            }
            resetSubmitButton();
        } catch (err) {
            alert('Review error: ' + err.message);
        }
    }

    hitlApproveBtn.addEventListener('click', () => sendHITLAction('approve'));
    hitlEditBtn.addEventListener('click', () => sendHITLAction('edit'));
    hitlRejectBtn.addEventListener('click', () => sendHITLAction('reject'));

    // =========================================================================
    // 10. Results Studio Renderer
    // =========================================================================
    function renderResults(job) {
        resultsSection.classList.remove('hidden');

        // 1. Video Metadata & Hero Banner
        if (job.metadata) {
            const title = job.metadata.title || `YouTube Video (${job.video_id})`;
            const channel = job.metadata.channel || 'Creator';
            const duration = job.metadata.duration_formatted || '00:00';
            const lang = (job.metadata.detected_language || 'EN').toUpperCase();
            const words = job.transcript_word_count || 0;

            videoThumb.src = job.metadata.thumbnail_url || `https://img.youtube.com/vi/${job.video_id}/maxresdefault.jpg`;
            videoTitle.textContent = title;
            videoChannel.textContent = channel;
            videoDuration.textContent = duration;
            videoLang.textContent = `LANG: ${lang}`;
            videoWords.textContent = `${words} words`;
            videoLink.href = job.url;

            // Populate Document Sheet Header
            docVideoTitle.textContent = title;
            docChannel.textContent = `Channel: ${channel}`;
            docDuration.textContent = `Duration: ${duration}`;
            docDate.textContent = `Generated: ${new Date().toLocaleDateString()}`;
        }

        // Status badge
        videoHitlStatus.textContent = 'STATUS: VERIFIED';

        // PDF URLs
        const pdfUrl = `/api/pdf/${job.job_id}`;
        const inlinePdfUrl = `${pdfUrl}?inline=true`;

        downloadPdfBtn.href = pdfUrl;
        downloadPdfInlineBtn.href = pdfUrl;
        bottomDownloadPdfBtn.href = pdfUrl;
        openPdfNewTabBtn.href = inlinePdfUrl;

        // 2. Safety Card & Document Seal
        if (job.safety_report) {
            const isHarmFree = job.safety_report.is_safe !== false && 
                (job.safety_report.badge_status === 'SAFE' || !job.safety_report.harm_flags_detected || job.safety_report.harm_flags_detected.length === 0);
            
            safetyBadge.textContent = isHarmFree ? 'SAFE CONTENT' : 'FLAGGED';
            safetyBadge.className = 'safety-badge ' + (isHarmFree ? 'badge-safe' : 'badge-flagged');
            safetyAssessment.textContent = job.safety_report.summary_assessment || 
                (isHarmFree ? 'Verified safe: No violence, sexual abuse, self-harm, or harassment detected.' : 'Flagged content detected.');

            docSafetySeal.textContent = isHarmFree ? '🛡️ HARM-FREE CERTIFIED' : '⚠️ REVIEWED CONTENT';

            safetyGrid.innerHTML = '';
            const categories = job.safety_report.categories || [];

            const friendlyNames = {
                'violence': 'Violence & Physical Force',
                'violence/graphic': 'Graphic Violence',
                'sexual': 'Sexual Content',
                'sexual/minors': 'Child Exploitation & Safety',
                'harassment': 'Harassment & Bullying',
                'harassment/threatening': 'Threatening Harassment',
                'hate': 'Hate Speech & Discrimination',
                'hate/threatening': 'Threatening Hate Speech',
                'self-harm': 'Self-Harm & Suicide',
                'self-harm/intent': 'Self-Harm Intent',
                'self-harm/instructions': 'Self-Harm Instructions'
            };

            categories.forEach(cat => {
                const item = document.createElement('div');
                item.className = 'safety-meter-item';
                const isFlagged = cat.flagged === true;
                
                const rawName = cat.category || cat.name || 'Safety Check';
                const displayName = friendlyNames[rawName.toLowerCase()] || 
                    rawName.replace(/[\/_]/g, ' ').replace(/\b\w/g, c => c.toUpperCase());
                
                const rawScore = typeof cat.score === 'number' ? cat.score : 
                    (typeof cat.confidence_score === 'number' ? cat.confidence_score : 0.05);
                const scorePct = Math.round(rawScore * 100);

                item.innerHTML = `
                    <div class="meter-label-row">
                        <span class="meter-name">${escapeHtml(displayName)}</span>
                        <span class="meter-status ${isFlagged ? 'flagged' : 'safe'}">${isFlagged ? 'FLAGGED' : 'PASSED'}</span>
                    </div>
                    <div class="meter-track">
                        <div class="meter-fill ${isFlagged ? 'flagged' : 'safe'}" style="width: ${isFlagged ? Math.max(scorePct, 60) : Math.max(scorePct, 15)}%;"></div>
                    </div>
                `;
                safetyGrid.appendChild(item);
            });
        }

        // 3. Grounded Executive Summary (Populates Both Cards and Document Sheet)
        if (job.summary && job.summary.lines) {
            lineCountBadge.textContent = `${job.summary.lines.length} Key Points`;
            summaryContainer.innerHTML = '';
            docSummaryList.innerHTML = '';

            job.summary.lines.forEach((line, idx) => {
                // Card View item
                const item = document.createElement('div');
                item.className = 'summary-item';
                item.innerHTML = `
                    <div class="summary-num">${idx + 1}</div>
                    <div class="summary-text">${escapeHtml(line)}</div>
                `;
                summaryContainer.appendChild(item);

                // Document Sheet View item
                const docItem = document.createElement('div');
                docItem.className = 'doc-line-row';
                docItem.innerHTML = `
                    <span class="doc-line-bullet">•</span>
                    <span>${escapeHtml(line)}</span>
                `;
                docSummaryList.appendChild(docItem);
            });
        }

        // 4. Action Items (Populates Both Cards and Document Sheet)
        const actionItems = job.action_items || [];
        actionItemsCount.textContent = `${actionItems.length} Tasks`;
        actionItemsContainer.innerHTML = '';
        docActionsList.innerHTML = '';

        if (actionItems.length === 0) {
            actionItemsContainer.innerHTML = '<p class="text-muted">No specific action items detected in this video.</p>';
            docActionsList.innerHTML = '<p style="color: var(--text-muted); font-size: 0.88rem;">No specific action items detected.</p>';
        } else {
            actionItems.forEach((item, idx) => {
                const pClass = (item.priority || 'Medium').toLowerCase();
                
                // Card View item
                const card = document.createElement('div');
                card.className = 'action-card-item';
                card.innerHTML = `
                    <div class="action-checkbox-box">
                        <input type="checkbox" class="action-checkbox">
                    </div>
                    <div class="action-body">
                        <div class="action-task-text">${escapeHtml(item.task)}</div>
                        <div class="action-meta-row">
                            <span class="priority-tag priority-${pClass}">${item.priority || 'Medium'}</span>
                            <span class="category-tag">${escapeHtml(item.category || 'Next Step')}</span>
                        </div>
                    </div>
                `;
                actionItemsContainer.appendChild(card);

                // Document Sheet View item
                const docAction = document.createElement('div');
                docAction.className = 'doc-action-row';
                docAction.innerHTML = `
                    <div class="doc-action-text">${idx + 1}. ${escapeHtml(item.task)}</div>
                    <span class="priority-tag priority-${pClass}">${item.priority || 'Medium'}</span>
                `;
                docActionsList.appendChild(docAction);
            });
        }

        // Scroll smoothly to results
        resultsSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }

    // =========================================================================
    // 11. Copy Summary to Clipboard
    // =========================================================================
    copySummaryBtn.addEventListener('click', () => {
        if (!currentJobData || !currentJobData.summary || !currentJobData.summary.lines) return;
        const text = currentJobData.summary.lines.map((l, i) => `${i + 1}. ${l}`).join('\n');
        navigator.clipboard.writeText(text).then(() => {
            const originalHTML = copySummaryBtn.innerHTML;
            copySummaryBtn.innerHTML = '<span>✓ Copied!</span>';
            setTimeout(() => { copySummaryBtn.innerHTML = originalHTML; }, 2000);
        });
    });

    // =========================================================================
    // 12. Raw Transcript Modal
    // =========================================================================
    viewTranscriptBtn.addEventListener('click', () => {
        if (!currentJobData) return;
        rawTranscriptText.textContent = currentJobData.transcript_sample || 'Full transcript loaded.';
        transcriptModal.classList.remove('hidden');
    });

    closeModalBtn.addEventListener('click', () => transcriptModal.classList.add('hidden'));
    modalOverlay.addEventListener('click', () => transcriptModal.classList.add('hidden'));

    // Helper
    function escapeHtml(str) {
        if (!str) return '';
        const div = document.createElement('div');
        div.textContent = str;
        return div.innerHTML;
    }
});
