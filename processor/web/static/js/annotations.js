/* Annotation management — inline form, color picker, current-frame highlight */

// All annotation sources use the same cyan visual language. The source is
// still shown by the AI badge; color is no longer used to distinguish source.
const ANNOTATION_COLOR = "#06B6D4";

let annotations = [];
let currentStartFrame = null;
let currentEndFrame = null;
let editingAnnotationId = null;
let _editingBaselineUpdatedAt = null;   // updated_at when editing started (conflict check)
let _editingWasCandidate = false;       // 兼容历史候选数据
let _annoSocket = null;                  // per-episode change channel (cross-device sync)
let _annoReconnectTimer = null;
let _annoReloadTimer = null;

// ── Timeline zoom(滚轮缩放 + 拖动平移 + Fit)─────────
// scale 语义 = 可见帧数 = 总帧数 ÷ scale;1× = 全片(强制 [0,total]),
// <1× = 全片压缩留边,>1× 以 centerFrame 为视窗中心;状态按批次保持。
let timelineZoom = { scale: 1.0, centerFrame: 0 };
let _panDragging = false;   // 拖拽平移中:暂停"播放跟随",避免视口被拉回
// 固定档位:最小 1×,最大 10×,线性每档 +0.5;滚轮/按钮都只在档位间切换
const ZOOM_STEPS = [1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5,
                    5.5, 6, 6.5, 7, 7.5, 8, 8.5, 9, 9.5, 10];

function _zoomTotal() {
    return Math.max(1, episodeTotalFrames || 1);
}
function _zoomVisibleRange() {
    const total = _zoomTotal();
    if (timelineZoom.scale <= 1) {
        // 1× 及以下:必须完整覆盖 [0, total],否则后半段坐标超 100%
        // 跑到轨道外(修复"1 倍看不到最后的切片")
        return { start: 0, end: total };
    }
    const half = total / (2 * timelineZoom.scale);
    return { start: timelineZoom.centerFrame - half, end: timelineZoom.centerFrame + half };
}
function _zoomFramePct(frame) {
    const r = _zoomVisibleRange();
    const span = Math.max(1, r.end - r.start);
    return Math.min(100, Math.max(0, ((frame - r.start) / span) * 100));
}
function _clampZoomCenter(scale, centerFrame) {
    const total = _zoomTotal();
    const half = total / (2 * scale);
    if (half >= total / 2) return total / 2;   // scale≤1:中心固定全片中心
    return Math.max(half, Math.min(total - half, centerFrame || total / 2));
}
function _snapScale(scale) {
    // 就近吸附到档位;低于 1 一律回 1×(不允许小于 1 倍)
    if (scale <= 1) return 1;
    for (const st of ZOOM_STEPS) {
        if (scale <= st * 1.001) return st;
    }
    return ZOOM_STEPS[ZOOM_STEPS.length - 1];
}
function _setZoom(scale, centerFrame) {
    const total = _zoomTotal();
    timelineZoom.scale = _snapScale(scale);
    timelineZoom.centerFrame = _clampZoomCenter(timelineZoom.scale, centerFrame);
    _updateZoomUI();
    renderAnnotationTimeline(annotations, total);
    // 只定位游标,不触发播放跟随(否则放大视口会被拉回蓝线)
    _positionCursor(typeof getCurrentFrame === 'function' ? getCurrentFrame() : 0);
}
function _updateZoomUI() {
    const ind = document.getElementById('zoom-indicator');
    if (ind) ind.textContent = `${timelineZoom.scale.toFixed(1)}×`;
    // 到边界时禁用对应按钮(最小 1×,最大 10×),并给视觉反馈
    const btnIn = document.getElementById('btn-zoom-in');
    const btnOut = document.getElementById('btn-zoom-out');
    if (btnIn) {
        btnIn.disabled = timelineZoom.scale >= ZOOM_STEPS[ZOOM_STEPS.length - 1];
        btnIn.style.opacity = btnIn.disabled ? '0.4' : '1';
    }
    if (btnOut) {
        btnOut.disabled = timelineZoom.scale <= 1;
        btnOut.style.opacity = btnOut.disabled ? '0.4' : '1';
    }
}


// ── Load ────────────────────────────────────────────

async function loadAnnotations(episodeId) {
    const requestEpoch = episodeId;  // capture at call time — discard if episode changed
    // Clear the old episode immediately while the new request is in flight.
    // This prevents Slice Preview and the annotation overlay from showing
    // stale data during an episode switch.
    annotations = [];
    if (typeof resetSlicePreview === 'function') resetSlicePreview(episodeId);
    renderAll();
    try {
        const signal = typeof getMediaLoadSignal === 'function'
            ? getMediaLoadSignal() : null;
        const res = await fetch(`/api/v1/episode/${episodeId}/annotations`,
            signal ? { signal } : {});
        if (requestEpoch !== currentEpisodeId) return;  // stale request — discard
        if (!res.ok) { renderAll(); return; }
        const data = await res.json();
        if (requestEpoch !== currentEpisodeId) return;  // stale response — discard
        annotations = data.annotations || [];
        renderAll();
        checkAiAnnotationEnabled(episodeId);
        connectAnnotationSocket(episodeId);
        if (typeof refreshSlicePreview === 'function') refreshSlicePreview();
    } catch (err) {
        if (err?.name === 'AbortError') return;
        if (requestEpoch !== currentEpisodeId) return;  // stale — discard
        annotations = [];
        renderAll();
    }
}

// 切换批次时立即清空标注 UI(时间轴/切片轨道/进度条 overlay/编辑态)。
// 旧批次的切片帧坐标不能残留在新批次视频上 —— 否则点击切片会按
// 旧帧号 seek,直接造成"帧不对齐"。loadAnnotations 里也有同样的清空,
// 但它要等 frames-data 返回后才执行,窗口太晚。
function clearAnnotationsNow() {
    annotations = [];
    currentStartFrame = null;
    currentEndFrame = null;
    editingAnnotationId = null;
    resetAnnotationTimelineForEpisode(
        typeof episodeTotalFrames !== 'undefined' ? episodeTotalFrames : 0,
    );
    if (typeof resetSlicePreview === 'function') {
        resetSlicePreview(typeof currentEpisodeId !== 'undefined' ? currentEpisodeId : '');
    }
    renderAll();
}

// A new episode must not inherit the previous episode's timeline viewport or
// cursor position.  The frame counter is reset by player.js, but the green
// cursor is DOM state and otherwise remains at the old percentage until the
// first successful frame callback.
function resetAnnotationTimelineForEpisode(totalFrames = 0) {
    const total = Math.max(0, Math.floor(Number(totalFrames) || 0));
    timelineZoom.scale = 1.0;
    timelineZoom.centerFrame = total > 0 ? total / 2 : 0;
    _panDragging = false;
    const cursor = document.getElementById('annotation-timeline-cursor');
    if (cursor) cursor.style.left = '0%';
    const frameLabel = document.getElementById('annotation-timeline-frame');
    if (frameLabel) frameLabel.textContent = total > 0 ? `0 / ${total}` : '0 / 0';
    _updateZoomUI();
}

// ── Cross-device sync ────────────────────────────────
// The backend broadcasts "annotations changed" over a per-episode WebSocket.
// We only reload the annotation UI (cards / timeline / overlay) — never the
// video position. REST remains the only write path; the socket is
// notification-only, so a dropped connection loses no data.

function connectAnnotationSocket(episodeId) {
    if (!window.WebSocket) return;
    if (_annoSocket && _annoSocket.episodeId === episodeId && _annoSocket.readyState <= WebSocket.OPEN) return;
    if (_annoSocket) {
        try { _annoSocket.close(); } catch (_) { /* no-op */ }
        _annoSocket = null;
    }
    const proto = location.protocol === 'https:' ? 'wss' : 'ws';
    const socket = new WebSocket(`${proto}://${location.host}/api/v1/annotations/ws?episode=${encodeURIComponent(episodeId)}`);
    socket.episodeId = episodeId;
    socket.onmessage = event => {
        try {
            const msg = JSON.parse(event.data);
            if (msg.type === 'annotations_changed' && msg.episode === currentEpisodeId) {
                // Reload unless we are mid-edit of this slice: the save itself
                // carries a version check and reloads on conflict anyway.
                if (editingAnnotationId !== msg.annotation_id) {
                    scheduleAnnotationReload(currentEpisodeId);
                }
            }
        } catch (_) { /* no-op */ }
    };
    socket.onopen = () => {
        try { socket.send(JSON.stringify({ type: 'ping' })); } catch (_) { /* no-op */ }
    };
    socket.onclose = () => {
        if (_annoSocket !== socket) return;
        _annoSocket = null;
        if (currentEpisodeId === episodeId && window.EGODATA_PAGE_MODE === 'annotation') {
            clearTimeout(_annoReconnectTimer);
            _annoReconnectTimer = setTimeout(() => {
                if (currentEpisodeId === episodeId) connectAnnotationSocket(episodeId);
            }, 1500);
        }
    };
    socket.onerror = () => { try { socket.close(); } catch (_) { /* no-op */ } };
    _annoSocket = socket;
}

function disconnectAnnotationSocket() {
    clearTimeout(_annoReconnectTimer);
    clearTimeout(_annoReloadTimer);
    _annoReconnectTimer = null;
    _annoReloadTimer = null;
    const socket = _annoSocket;
    _annoSocket = null;
    if (socket) {
        try { socket.close(); } catch (_) { /* no-op */ }
    }
}

window.disconnectAnnotationSocket = disconnectAnnotationSocket;

function scheduleAnnotationReload(episodeId) {
    clearTimeout(_annoReloadTimer);
    _annoReloadTimer = setTimeout(() => {
        if (episodeId === currentEpisodeId) loadAnnotations(episodeId);
    }, 120);
}

function sortedAnnotations() {
    return [...annotations].sort((a, b) => {
        const startDiff = (Number(a.start_frame_index) || 0) - (Number(b.start_frame_index) || 0);
        if (startDiff) return startDiff;
        const endDiff = (Number(a.end_frame_index) || 0) - (Number(b.end_frame_index) || 0);
        if (endDiff) return endDiff;
        return String(a.created_at || '').localeCompare(String(b.created_at || ''));
    });
}

function annotationColor(seg, index) {
    return ANNOTATION_COLOR;
}

function isAnnotationEditor() {
    return window.EGODATA_PAGE_MODE === 'annotation';
}

function supportsAnnotationDisplay() {
    return isAnnotationEditor() || window.EGODATA_PAGE_MODE === 'review';
}


// ── Render ──────────────────────────────────────────

function renderAll() {
    renderAnnotationList();
    renderAnnotationTimeline(annotations, episodeTotalFrames || 0);
    renderAnnotationOverlay(annotations, episodeTotalFrames || 0);
}


function showAnnotationUI() {
    const timeline = document.getElementById('annotation-timeline');
    if (!timeline) return;
    const ready = supportsAnnotationDisplay()
        && typeof episodeTotalFrames !== 'undefined'
        && episodeTotalFrames > 0;
    timeline.classList.toggle('hidden', !ready);
}


// ── Color picker ────────────────────────────────────

function buildColorPicker() {
    // Colors are assigned automatically by the API.
}

function selectColor(color, el) {
    // Kept as a no-op for compatibility with cached markup.
}


// ── Annotation list ─────────────────────────────────

function renderAnnotationList(targetId) {
    targetId = targetId || 'episode-detail-annos';
    const listEl = document.getElementById(targetId);
    if (!listEl) return;
    const countEl = document.getElementById('anno-count');
    if (countEl) countEl.textContent = annotations.length;
    const editor = isAnnotationEditor();

    if (annotations.length === 0) {
        listEl.innerHTML = `
            <div class="p-4 text-center text-gray-600 text-xs">
                <iconify-icon icon="ant-design:scissor-outlined" class="text-lg mb-1"></iconify-icon>
                <div>${t('no_slices')}</div>
                ${editor ? `<div class="text-[10px] text-gray-700 mt-1">${t('no_slices_hint')}</div>` : `<div class="text-[10px] text-gray-700 mt-1">${t('annotation_readonly')}</div>`}
            </div>`;
        return;
    }

    const currentFrame = typeof getCurrentFrame === 'function' ? getCurrentFrame() : -1;
    // 单一高亮:当前帧只属于一个段(段重叠时取开始帧最近的那段)
    const currentAnnoId = _singleCurrentAnnoId(currentFrame, null);

    listEl.innerHTML = sortedAnnotations().map((seg, i) => {
        const frames = seg.end_frame_index - seg.start_frame_index + 1;
        const duration = (frames / (episodeFps || 30)).toFixed(2);
        const isCurrent = seg.id === currentAnnoId;
        const color = annotationColor(seg, i);

        const kfHTML = (seg.keyframes || []).map(kf =>
            '<div class="text-gray-600 text-xs ml-3 mt-0.5">' +
            '<iconify-icon icon="ant-design:aim-outlined" class="icon-sm"></iconify-icon> frame ' + kf.frame_index + ': ' + escHtml(kf.event || '') +
            '</div>'
        ).join('');

        const isEditing = editor && seg.id === editingAnnotationId;
        const isCandidate = (seg.status || 'confirmed') === 'candidate';
        const isAi = seg.source === 'ai';
        return `
        <div class="annotation-card rounded-lg p-2.5 text-xs cursor-pointer transition-colors border flex gap-2 ${isEditing ? 'is-editing' : (isCurrent ? 'bg-blue-900/30 border-blue-600/60' : 'bg-gray-800/40 border-gray-700/30 hover:bg-gray-800')}${isCandidate ? ' border-cyan-700/50' : ''}"
             data-anno-id="${seg.id}"
             onclick="onAnnotationClick('${seg.id}')">
            <!-- Color strip -->
            <div class="w-1 flex-shrink-0 rounded-full" style="background:${color}"></div>
            <!-- Content -->
            <div class="flex-1 min-w-0">
            <div class="flex items-start gap-1.5 mb-1">
                    <span class="text-[10px] text-gray-500 font-mono">${i + 1}</span>
                    ${isAi ? '<iconify-icon icon="ant-design:robot-outlined" class="icon-sm text-cyan-400"></iconify-icon>' : ''}
                    <span class="annotation-card-label min-w-0 flex-1 font-medium ${isCurrent ? 'text-blue-200' : 'text-gray-200'}" title="${escHtml(seg.label || t('unnamed_slice'))}">${escHtml(seg.label || t('unnamed_slice'))}</span>
                    ${isAi ? '<span class="text-[9px] px-1 rounded bg-cyan-900/50 text-cyan-300 border border-cyan-700/50">AI</span>' : ''}
                </div>
                <div class="annotation-card-meta text-gray-400 font-mono mb-1">
                    <span>${seg.start_frame_index} - ${seg.end_frame_index}</span>
                    <span>${frames} frames · ${duration}s</span>
                </div>
                <div class="legacy-annotation-meta hidden">
                    ${seg.start_frame_index} – ${seg.end_frame_index}
                    <span class="text-gray-600 ml-1">(${frames}f · ${duration}s)</span>
                </div>
                ${kfHTML}
                <div class="flex gap-2 mt-1.5">
                    <button class="text-gray-500 hover:text-blue-400 transition-colors" title="Play segment"
                            onclick="event.stopPropagation();playSegment(${seg.start_frame_index}, ${seg.end_frame_index})">
                        <iconify-icon icon="ant-design:caret-right-outlined" class="icon-sm"></iconify-icon> Play
                    </button>
                    ${editor ? `<button class="text-gray-500 hover:text-gray-300 transition-colors" title="Edit"
                            onclick="event.stopPropagation();editAnnotation('${seg.id}')">
                        <iconify-icon icon="ant-design:edit-outlined" class="icon-sm"></iconify-icon>
                    </button>
                    <button class="text-gray-500 hover:text-red-400 transition-colors" title="Delete"
                            onclick="event.stopPropagation();deleteAnnotation('${seg.id}')">
                        <iconify-icon icon="ant-design:close-outlined" class="icon-sm"></iconify-icon>
                    </button>` : ''}
                </div>
            </div>
        </div>`;
    }).join('');
    // Normalize visible metadata after rendering so legacy stored labels or
    // old template separators cannot leak into the current UI.
    const ordered = sortedAnnotations();
    listEl.querySelectorAll('.annotation-card-meta').forEach((meta, index) => {
        const seg = ordered[index];
        if (!seg) return;
        const frames = Number(seg.end_frame_index) - Number(seg.start_frame_index) + 1;
        const duration = (frames / (episodeFps || 30)).toFixed(2);
        meta.innerHTML = `<span>${seg.start_frame_index} - ${seg.end_frame_index}</span><span>${frames} frames · ${duration}s</span>`;
    });
    updateAiControls();
}

// ═══════════════════════════════════════════════════════════
//  AI 辅助标注(多视角同步输入，成功后直接写入 confirmed)
// ═══════════════════════════════════════════════════════════

let _aiEnabled = false;
let _aiConfig = {};
let _aiBusy = false;

function updateAiControls() {
    const btn = document.getElementById('btn-ai-annotate');
    if (btn) {
        btn.classList.toggle('hidden', !_aiEnabled);
        btn.disabled = _aiBusy;
    }
}

function renderAiAnnotationStatus(data) {
    const el = document.getElementById('episode-detail-ai-status');
    if (!el) return;
    // API/provider progress is available from the workflow/task panel when
    // needed, but it should not occupy the episode detail header. The review
    // page only needs the annotation markers and the human Approve action.
    el.className = 'hidden';
    el.textContent = '';
}

async function checkAiAnnotationEnabled(episodeId) {
    try {
        const signal = typeof getMediaLoadSignal === 'function'
            ? getMediaLoadSignal() : null;
        const res = await fetch(`/api/v1/episode/${episodeId}/ai-annotation-enabled`,
            signal ? { signal } : {});
        if (episodeId !== currentEpisodeId) return;
        if (!res.ok) { _aiEnabled = false; return; }
        const data = await res.json();
        if (episodeId !== currentEpisodeId) return;
        _aiEnabled = !!data.enabled;
        _aiConfig = data.config || {};
        renderAiAnnotationStatus(data);
    } catch (_) {
        if (episodeId !== currentEpisodeId) return;
        _aiEnabled = false;
        renderAiAnnotationStatus(null);
    }
    updateAiControls();
}

// 卡片配置了多套 API 时,运行前弹窗选一套;单套/未配置直接放行。
// 返回:undefined = 用户取消;null = 不覆盖(走卡片默认);对象 = 用该套覆盖。
function _pickAiProvider() {
    const providers = Array.isArray(_aiConfig.api_providers) ? _aiConfig.api_providers : [];
    if (providers.length <= 1) return Promise.resolve(providers[0] || null);
    return new Promise((resolve) => {
        const overlay = document.createElement('div');
        overlay.className = 'fixed inset-0 flex items-center justify-center';
        overlay.style.cssText = 'position:fixed;inset:0;z-index:60;background:rgba(0,0,0,.55);';
        const box = document.createElement('div');
        box.className = 'bg-gray-900 border border-gray-700 rounded-lg shadow-2xl';
        box.style.width = '360px';
        box.innerHTML = `
            <div class="flex items-center justify-between px-4 py-3 border-b border-gray-800">
                <h3 class="text-sm font-semibold text-gray-200">Select AI API</h3>
                <button type="button" class="text-gray-500 hover:text-gray-300 text-base leading-none">✕</button>
            </div>
            <div class="p-3 space-y-2"></div>`;
        const list = box.querySelector('div.p-3');
        const done = (provider) => { overlay.remove(); resolve(provider); };
        box.querySelector('button').onclick = () => done(undefined);
        overlay.addEventListener('click', (e) => { if (e.target === overlay) done(undefined); });
        providers.forEach((p) => {
            const row = document.createElement('button');
            row.type = 'button';
            row.className = 'w-full text-left px-3 py-2 rounded border border-gray-800 bg-gray-950/60 hover:border-blue-500 hover:bg-gray-800 text-sm text-gray-200';
            row.innerHTML = `<div class="font-medium">${escHtml(p.vendor || '')} · ${escHtml(p.model || '')}</div>
                             <div class="text-[11px] text-gray-500">${escHtml(p.base_url || 'default endpoint')}</div>`;
            row.onclick = () => done(p);
            list.appendChild(row);
        });
        overlay.appendChild(box);
        document.body.appendChild(overlay);
    });
}

async function aiAnnotate() {
    if (!isAnnotationEditor() || !currentEpisodeId || _aiBusy) return;
    const provider = await _pickAiProvider();
    if (provider === undefined) return;  // 用户取消
    const btn = document.getElementById('btn-ai-annotate');
    const mode = _aiConfig.mode || 'signal_vlm';
    _aiBusy = true;
    updateAiControls();
    const setLabel = (html) => { if (btn) btn.innerHTML = html; };
    setLabel('<iconify-icon icon="ant-design:loading-outlined" class="icon-sm animate-spin"></iconify-icon> <span>AI…</span>');
    let failed = false;
    try {
        const res = await fetch(`/api/v1/episode/${currentEpisodeId}/ai-annotate`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ mode, min_confidence: _aiConfig.min_confidence || 0,
                                   lang: _aiConfig.prompt_language || 'zh',
                                   debounce_sec: _aiConfig.debounce_sec || 2.0,
                                   min_seg_sec: _aiConfig.min_seg_sec || 0.8,
                                   max_segments: _aiConfig.max_segments || 0,
                                   // 本次运行选定的 API 整套覆盖卡片默认(单套时不传)
                                   ...(provider ? { api_vendor: provider.vendor,
                                                    api_model: provider.model,
                                                    api_key: provider.key,
                                                    api_base_url: provider.base_url } : {}) }),
        });
        if (!res.ok) throw new Error('trigger failed');
        // 轮询任务状态(完成经 WS 广播刷新标注,轮询只驱动按钮文案)
        for (let i = 0; i < 240; i++) {
            await new Promise(r => setTimeout(r, 1000));
            let st = { status: 'running' };
            try {
                st = await (await fetch(`/api/v1/episode/${currentEpisodeId}/ai-annotate/status`)).json();
            } catch (_) { /* keep polling */ }
            if (st.status === 'signal_segmenting') setLabel('<iconify-icon icon="ant-design:loading-outlined" class="icon-sm animate-spin"></iconify-icon> <span>切段中…</span>');
            else if (st.status === 'vlm_analyzing') setLabel('<iconify-icon icon="ant-design:loading-outlined" class="icon-sm animate-spin"></iconify-icon> <span>VLM 分析中…</span>');
            else if (st.status === 'writing') setLabel('<iconify-icon icon="ant-design:loading-outlined" class="icon-sm animate-spin"></iconify-icon> <span>写入…</span>');
            else if (st.status === 'exporting') setLabel('<iconify-icon icon="ant-design:loading-outlined" class="icon-sm animate-spin"></iconify-icon> <span>写入数据集…</span>');
            if (st.status === 'done') break;
            if (st.status === 'failed') {
                console.warn('AI annotation failed:', st.detail);
                failed = true;
                const d = String(st.detail || '').replace(/[<>]/g, '').slice(0, 100);
                setLabel('<iconify-icon icon="ant-design:close-circle-outlined" class="icon-sm text-red-400"></iconify-icon> <span>' + (d || 'AI annotation failed') + '</span>');
                break;
            }
            if (st.status === 'interrupted') { console.warn('AI annotation interrupted:', st.detail); break; }
        }
    } catch (err) {
        console.error('AI annotation error:', err);
    }
    if (!failed) {
        setLabel('<iconify-icon icon="ant-design:robot-outlined" class="icon-sm"></iconify-icon> <span data-i18n="ai_annotate">AI Annotate</span>');
    }
    _aiBusy = false;
    updateAiControls();
    await loadAnnotations(currentEpisodeId);
}

async function aiConfirmSegment(annoId, silent) {
    if (!isAnnotationEditor()) return;
    try {
        const seg = annotations.find(a => a.id === annoId);
        if (seg) _pushUndo('update', seg);
        await fetch(`/api/v1/annotation/${annoId}`, {
            method: 'PUT',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ status: 'confirmed', updated_at: seg ? seg.updated_at : undefined }),
        });
        if (!silent) await loadAnnotations(currentEpisodeId);
    } catch (err) {
        console.error('confirm failed:', err);
    }
}

async function aiDismissSegment(annoId, silent) {
    if (!isAnnotationEditor()) return;
    // 兼容历史候选段删除；新 AI 结果不会进入 candidate 状态。
    try {
        const seg = annotations.find(a => a.id === annoId);
        if (seg) _pushUndo('delete', seg);
        await fetch(`/api/v1/annotation/${annoId}`, { method: 'DELETE' });
        if (!silent) await loadAnnotations(currentEpisodeId);
    } catch (err) {
        console.error('dismiss failed:', err);
    }
}

function onAnnotationClick(annoId) {
    const seg = annotations.find(a => a.id === annoId);
    if (!seg || typeof seekToFrame !== 'function') return;
    seekToFrame(seg.start_frame_index);
    // 同步更新卡片高亮:preferId = 被点的段,只高亮这一张(其余熄灭),
    // 不依赖播放器 seeked 事件往返,也不会因段重叠出现双高亮。
    updateCurrentAnnotation(seg.start_frame_index, seg.id);
}

function legacyRenderAnnotationTimeline(items, totalFrames) {
    if (!supportsAnnotationDisplay()) return;
    const timeline = document.getElementById('annotation-timeline');
    const track = document.getElementById('annotation-timeline-track');
    const markerTrack = document.getElementById('annotation-timeline-markers');
    if (!timeline || !track || !markerTrack) return;

    markerTrack.querySelectorAll('.annotation-timeline-segment').forEach(el => el.remove());
    if (!totalFrames) {
        timeline.classList.add('hidden');
        return;
    }

    timeline.classList.remove('hidden');
    bindFrameTimelineSeek(track, totalFrames);
    // 切片轨道同样支持放大后按住左右平移
    _attachPanDrag(markerTrack);
    // Use the same 0..last-frame coordinate system as the playback cursor.
    const denominator = Math.max(1, totalFrames - 1);
    // 多行排段:重叠的段错行显示(最多 3 行;更多时压进最后一行)。
    // 行高 18px + 上下留白,轨道高度按行数动态伸缩(52-92px)。
    const LANE_H = 18, LANE_PAD = 6, MAX_LANES = 3;
    const laneEnds = [];  // 每行最后一个段的 end_frame
    const laneItems = sortedAnnotations().map((seg, i) => {
        const start = Math.max(0, Number(seg.start_frame_index) || 0);
        const end = Math.max(start, Number(seg.end_frame_index) || start);
        let lane = laneEnds.findIndex(lastEnd => lastEnd <= start);
        if (lane === -1) {
            if (laneEnds.length < MAX_LANES) {
                lane = laneEnds.length;
                laneEnds.push(-1);
            } else {
                lane = MAX_LANES - 1;  // 封顶:压缩进最后一行
            }
        }
        laneEnds[lane] = Math.max(laneEnds[lane], end);
        return { seg, i, start, end, lane };
    });
    markerTrack.style.height = `${LANE_PAD * 2 + Math.max(1, laneEnds.length) * LANE_H}px`;
    // 播放工具条紧贴切片轨道上方(随轨道高度自动上移)
    _positionPlaybackToolbar(markerTrack.offsetHeight);
    laneItems.forEach(({ seg, i, start, end, lane }) => {
        const segment = document.createElement('button');
        const leftPercent = Math.min(100, (start / denominator) * 100);
        const rightPercent = Math.min(100, (end / denominator) * 100);
        const width = Math.max(0.7, rightPercent - leftPercent);
        segment.type = 'button';
        segment.className = 'annotation-timeline-segment' +
            ((seg.status || 'confirmed') === 'candidate' ? ' candidate' : '');
        segment.dataset.annoId = seg.id;
        segment.dataset.startFrame = start;
        segment.dataset.endFrame = end;
        // 边界自检未过 / 标签未命中词表 → 时间轴段黄色边框
        if (seg.boundary_ok === false || seg.label_matched === false) {
            segment.style.borderColor = '#f59e0b';
        }
        segment.style.left = `${leftPercent}%`;
        segment.style.width = `${Math.min(width, 100 - leftPercent)}%`;
        // 行定位(覆盖默认 top:2px/bottom:2px)
        segment.style.top = `${LANE_PAD + lane * LANE_H}px`;
        segment.style.bottom = 'auto';
        segment.style.height = `${LANE_H - 2}px`;
        segment.style.lineHeight = `${LANE_H - 2}px`;
        segment.style.fontSize = '10px';
        segment.style.padding = '0 6px';
        segment.style.background = annotationColor(seg, i);
        // 段过窄时不显示文字(悬停 tooltip 仍有完整信息)
        const label = `${i + 1} · ${seg.label || t('unnamed_slice')}`;
        segment.textContent = width >= 2 ? label : '';
        segment.title = `${seg.label || t('unnamed_slice')} · frames ${start}-${end}`;
        segment.setAttribute('aria-label', segment.title);
        segment.addEventListener('click', event => {
            event.stopPropagation();
            onAnnotationClick(seg.id);
        });
        markerTrack.appendChild(segment);
    });

    updateAnnotationTimelineCursor(
        typeof getCurrentFrame === 'function' ? getCurrentFrame() : 0,
    );
}

// Enhanced marker interaction: move a marker or resize either edge, then
// persist the new frame range through the existing annotation API.
function _clampTimelineFrame(frame, min, max) {
    return Math.max(min, Math.min(max, Math.round(Number(frame) || 0)));
}

function _setTimelineSegmentRange(segment, start, end, totalFrames) {
    const lastFrame = Math.max(0, totalFrames - 1);
    const safeStart = _clampTimelineFrame(start, 0, lastFrame);
    const safeEnd = _clampTimelineFrame(Math.max(safeStart, end), safeStart, lastFrame);
    const denominator = Math.max(1, lastFrame);
    const leftPercent = Math.min(100, (safeStart / denominator) * 100);
    const rightPercent = Math.min(100, (safeEnd / denominator) * 100);
    segment.style.left = `${leftPercent}%`;
    segment.style.width = `${Math.min(Math.max(0.7, rightPercent - leftPercent), 100 - leftPercent)}%`;
    segment.dataset.startFrame = safeStart;
    segment.dataset.endFrame = safeEnd;
    return { start: safeStart, end: safeEnd };
}

function _paintTimelineSegment(segment, index, seg, start, end) {
    // 只读显示:无拖拽/拉伸把手(边界修改走编辑表单)
    const label = seg.label || t('unnamed_slice');
    segment.innerHTML = `
        <span class="annotation-timeline-label">${index + 1} · ${escHtml(label)}</span>`;
    segment.title = `${label} · frames ${start}-${end}`;
    segment.setAttribute('aria-label', segment.title);
}
// This definition intentionally follows the original renderer so the new
// interaction is used without changing the existing annotation data flow.
/* 播放工具条贴底:紧贴切片轨道/进度条上方,随轨道高度自动上移 */
function _positionPlaybackToolbar(markersHeight) {
    const tb = document.getElementById('playback-toolbar');
    if (!tb) return;
    const controls = document.getElementById('frame-controls');
    const controlsH = (controls && !controls.classList.contains('hidden'))
        ? controls.offsetHeight : 0;
    const mH = markersHeight || 0;
    tb.style.bottom = `${mH + controlsH + 10}px`;
}

function renderAnnotationTimeline(items, totalFrames) {
    if (!supportsAnnotationDisplay()) return;
    const timeline = document.getElementById('annotation-timeline');
    const track = document.getElementById('annotation-timeline-track');
    const markerTrack = document.getElementById('annotation-timeline-markers');
    if (!timeline || !track || !markerTrack) return;

    markerTrack.querySelectorAll('.annotation-timeline-segment').forEach(el => el.remove());
    if (!totalFrames) {
        timeline.classList.add('hidden');
        _positionPlaybackToolbar(null);
        return;
    }

    timeline.classList.remove('hidden');
    bindFrameTimelineSeek(track, totalFrames);
    // 多行排段:重叠的段错行显示(最多 3 行,更多压进最后一行);
    // 轨道高度随行数动态伸缩(52-92px)
    const LANE_H = 18, LANE_PAD = 6, MAX_LANES = 3;
    const laneEnds = [];
    const laneItems = sortedAnnotations().map((seg, index) => {
        const start = Math.max(0, Number(seg.start_frame_index) || 0);
        const end = Math.max(start, Number(seg.end_frame_index) || start);
        let lane = laneEnds.findIndex(lastEnd => lastEnd <= start);
        if (lane === -1) {
            if (laneEnds.length < MAX_LANES) {
                lane = laneEnds.length;
                laneEnds.push(-1);
            } else {
                lane = MAX_LANES - 1;
            }
        }
        laneEnds[lane] = Math.max(laneEnds[lane], end);
        return { seg, index, start, end, lane };
    });
    markerTrack.style.height = `${LANE_PAD * 2 + Math.max(1, laneEnds.length) * LANE_H}px`;
    _positionPlaybackToolbar(markerTrack.offsetHeight);
    laneItems.forEach(({ seg, index, start, end, lane }) => {
        const segment = document.createElement('button');
        segment.type = 'button';
        segment.className = 'annotation-timeline-segment' +
            ((seg.status || 'confirmed') === 'candidate' ? ' candidate' : '');
        segment.dataset.annoId = seg.id;
        segment.style.background = annotationColor(seg, index);
        // 边界自检未过 / 标签未命中词表 → 黄色边框
        if (seg.boundary_ok === false || seg.label_matched === false) {
            segment.style.borderColor = '#f59e0b';
        }
        _setTimelineSegmentRange(segment, start, end, totalFrames);
        // 行定位(覆盖默认 top:2px/bottom:2px)
        segment.style.top = `${LANE_PAD + lane * LANE_H}px`;
        segment.style.bottom = 'auto';
        segment.style.height = `${LANE_H - 2}px`;
        segment.style.fontSize = '10px';
        segment.style.padding = '0 6px';
        // 缩放变换:段在视窗内的位置;完全在视窗外 → 隐藏
        const vrange = _zoomVisibleRange();
        const vspan = Math.max(1, vrange.end - vrange.start);
        if (end < vrange.start || start > vrange.end) {
            segment.style.display = 'none';
        } else {
            const vStart = Math.max(start, vrange.start);
            const vEnd = Math.min(end, vrange.end);
            segment.style.left = `${((vStart - vrange.start) / vspan) * 100}%`;
            segment.style.width = `${Math.max(0.5, ((vEnd - vStart) / vspan) * 100)}%`;
        }
        _paintTimelineSegment(segment, index, seg, start, end);
        // 段过窄 → 隐藏标签(悬停 tooltip 仍有完整信息)
        if (parseFloat(segment.style.width) < 2) {
            const lbl = segment.querySelector('.annotation-timeline-label');
            if (lbl) lbl.style.display = 'none';
        }
        segment.addEventListener('click', event => {
            event.stopPropagation();
            onAnnotationClick(seg.id);
        });
        markerTrack.appendChild(segment);
    });

    _positionCursor(typeof getCurrentFrame === 'function' ? getCurrentFrame() : 0);
}

// The playback row is shared by Review and Annotation Studio. Binding it
// from player readiness keeps click/drag seeking available on both pages,
// even when the annotation marker row is hidden.
/* 放大后按住拖动 = 左右平移视窗(1× 时无可平移空间,拖动不生效)。
   拖动结束会置 dragJustHappened 标记,click 处理器据此跳过 seek,
   区分"拖动"与"点击"。 */
function _attachPanDrag(el) {
    if (!el || el.dataset.panBound === 'true') return;
    el.dataset.panBound = 'true';
    let pan = null;
    el.addEventListener('pointerdown', event => {
        if (event.button !== 0 || timelineZoom.scale <= 1) return;
        // 按在切片段上 → 不启动平移(段只保留点击跳帧,平移走空白处)
        if (event.target.closest && event.target.closest('.annotation-timeline-segment')) return;
        _panDragging = true;   // 拖拽期间暂停播放跟随
        const rect = el.getBoundingClientRect();
        const r0 = _zoomVisibleRange();
        pan = {
            pointerId: event.pointerId,
            startX: event.clientX,
            startCenter: timelineZoom.centerFrame,
            span: Math.max(1, r0.end - r0.start),
            rect,
            moved: false,
        };
        el.setPointerCapture?.(event.pointerId);
        event.preventDefault();
    });
    el.addEventListener('pointermove', event => {
        if (!pan || event.pointerId !== pan.pointerId) return;
        event.preventDefault();
        const dx = (event.clientX - pan.startX) / Math.max(1, pan.rect.width);
        if (Math.abs(event.clientX - pan.startX) > 5) pan.moved = true;
        if (!pan.moved) return;
        // 中心钳制在 [half, total-half]:视口永不越出视频左右边界
        timelineZoom.centerFrame = _clampZoomCenter(
            timelineZoom.scale, pan.startCenter - dx * pan.span);
        renderAnnotationTimeline(annotations, episodeTotalFrames || 0);
        _positionCursor(typeof getCurrentFrame === 'function' ? getCurrentFrame() : 0);
    });
    const finish = event => {
        if (!pan || event.pointerId !== pan.pointerId) return;
        const wasDrag = pan.moved;
        pan = null;
        _panDragging = false;   // 恢复播放跟随
        // 拖动后紧随的 click 不再当"点击跳帧"
        if (wasDrag) {
            el.dataset.dragJustHappened = 'true';
            setTimeout(() => { delete el.dataset.dragJustHappened; }, 0);
        }
    };
    el.addEventListener('pointerup', finish);
    el.addEventListener('pointercancel', finish);
    el.addEventListener('lostpointercapture', finish);
}

function bindFrameTimelineSeek(track, totalFrames) {
    if (!track || track.dataset.seekBound) return;

    const frameFromPointer = event => {
        const rect = track.getBoundingClientRect();
        if (!rect.width) return 0;
        const ratio = Math.max(0, Math.min(1, (event.clientX - rect.left) / rect.width));
        // 缩放感知:按当前视窗换算帧号
        const r = _zoomVisibleRange();
        return Math.round(r.start + ratio * (r.end - r.start));
    };

    // 放大后按住拖动 = 平移视窗
    _attachPanDrag(track);

    // 点击 = 跳帧(拖动过则不触发)
    track.addEventListener('click', event => {
        if (event.button !== 0) return;
        if (track.dataset.dragJustHappened) return;
        const frame = frameFromPointer(event);
        if (typeof seekToFrame === 'function') seekToFrame(frame);
    });

    // 双击轨道 → 回到 Fit 全片
    track.addEventListener('dblclick', () => {
        _setZoom(1, _zoomTotal() / 2);
    });

    track.dataset.seekBound = 'true';
}

/* 滚轮缩放:悬停在切片轨道/进度条上滚动,以鼠标位置为中心缩放 */
function _bindTimelineZoom() {
    const markerTrack = document.getElementById('annotation-timeline-markers');
    const track = document.getElementById('annotation-timeline-track');
    [markerTrack, track].forEach(el => {
        if (!el || el.dataset.zoomBound === 'true') return;
        el.dataset.zoomBound = 'true';
        let wheelRaf = null, wheelSteps = 0, wheelAnchor = null;
        el.addEventListener('wheel', event => {
            event.preventDefault();
            // 缩放锚定 = 鼠标位置(鼠标下的帧保持不动)——剪辑软件标准做法
            const rect = el.getBoundingClientRect();
            const ratio = Math.max(0, Math.min(1,
                (event.clientX - rect.left) / Math.max(1, rect.width)));
            const r = _zoomVisibleRange();
            wheelAnchor = r.start + ratio * (r.end - r.start);
            wheelSteps += event.deltaY < 0 ? 1 : -1;
            // rAF 合并:一帧内的多次滚动只渲染一次(标注多时全量重渲染
            // 开销大,快速滚动逐格渲染会卡死)
            if (wheelRaf) return;
            wheelRaf = requestAnimationFrame(() => {
                wheelRaf = null;
                const steps = wheelSteps;
                wheelSteps = 0;
                if (!steps) return;
                const idx = ZOOM_STEPS.indexOf(timelineZoom.scale);
                const cur = idx < 0 ? 0 : idx;
                const nextIdx = Math.max(0,
                    Math.min(ZOOM_STEPS.length - 1, cur + steps));
                _setZoom(ZOOM_STEPS[nextIdx], wheelAnchor);
            });
        }, { passive: false });
    });
}

function _positionCursor(frameIndex) {
    /* 纯游标定位(无播放跟随)。缩放/渲染/平移内部只走这里,
       避免"跟随"把用户缩放/平移的视口拉回蓝线。返回归一化帧号。 */
    const isAnnotationPage = window.EGODATA_PAGE_MODE === 'annotation';
    const timeline = document.getElementById('annotation-timeline');
    const frameControls = document.getElementById('frame-controls');
    const track = document.getElementById('annotation-timeline-track');
    const cursor = document.getElementById('annotation-timeline-cursor');
    if (!track || !cursor) return null;
    if (isAnnotationPage && (!timeline || timeline.classList.contains('hidden'))) return null;
    if (!isAnnotationPage && (!frameControls || frameControls.classList.contains('hidden'))) return null;

    const total = Math.max(1, episodeTotalFrames || 1);
    const frame = Math.max(0, Math.min(total - 1, Number(frameIndex) || 0));
    cursor.style.left = `${_zoomFramePct(frame)}%`;
    const frameLabel = document.getElementById('annotation-timeline-frame');
    // Keep the UI on the source's zero-based frame index, matching the
    // parquet/video frame_index and the detail panel.
    if (frameLabel) frameLabel.textContent = `${frame} / ${total}`;
    return frame;
}

function updateAnnotationTimelineCursor(frameIndex) {
    const frame = _positionCursor(frameIndex);
    if (frame == null) return;
    const isAnnotationPage = window.EGODATA_PAGE_MODE === 'annotation';
    const markerTrack = document.getElementById('annotation-timeline-markers');
    const total = Math.max(1, episodeTotalFrames || 1);
    // 播放跟随(仅帧变化路径):放大时当前帧超出可见范围 → 视窗自动
    // 重定位;手动拖拽平移期间暂停 —— 否则视口会被拉回(回弹)。
    if (timelineZoom.scale > 1 && !_panDragging) {
        const vr = _zoomVisibleRange();
        if (frame < vr.start || frame > vr.end) {
            timelineZoom.centerFrame = _clampZoomCenter(timelineZoom.scale, frame);
            renderAnnotationTimeline(annotations, total);
            _positionCursor(frame);
        }
    }

    if (supportsAnnotationDisplay()) {
        // 时间轴段与卡片同一套单一高亮规则(段重叠时不双亮)
        const currentAnnoId = _singleCurrentAnnoId(frame, null);
        markerTrack?.querySelectorAll('.annotation-timeline-segment').forEach(segment => {
            segment.classList.toggle('is-current', segment.dataset.annoId === currentAnnoId);
        });
    }
}


// ── Inline form ─────────────────────────────────────

function showNewAnnotationForm(resetFrames = true) {
    if (!isAnnotationEditor()) return;
    editingAnnotationId = null;
    _editingBaselineUpdatedAt = null;
    if (resetFrames) {
        currentStartFrame = null;
        currentEndFrame = null;
    }
    const form = document.getElementById('anno-form');
    if (!form) return;
    const titleEl = document.getElementById('anno-form-title');
    if (form) form.classList.remove('hidden');
    if (titleEl) titleEl.textContent = t('new_annotation');
    if (resetFrames || !document.getElementById('anno-label-input').value) {
        document.getElementById('anno-label-input').value = '';
    }
    syncAnnotationFrameInputs();
    updateFormState();
    document.getElementById('anno-label-input').focus();
}


function hideAnnotationForm() {
    if (window.EGODATA_PAGE_MODE !== 'annotation') return;
    const form = document.getElementById('anno-form');
    // Annotation Studio keeps the editor mounted. Reset it for the next
    // annotation instead of hiding the whole module.
    if (form) form.classList.remove('hidden');
    editingAnnotationId = null;
    _editingBaselineUpdatedAt = null;
    _editingWasCandidate = false;
    currentStartFrame = null;
    currentEndFrame = null;
    syncAnnotationFrameInputs();
    const titleEl = document.getElementById('anno-form-title');
    if (titleEl) titleEl.textContent = t('new_annotation');
    const labelInput = document.getElementById('anno-label-input');
    if (labelInput) labelInput.value = '';
    const saveBtn = document.getElementById('btn-save-annotation');
    if (saveBtn) {
        const labelSpan = saveBtn.querySelector('[data-i18n]');
        if (labelSpan) labelSpan.textContent = t('add_slice');
    }
    updateFormState();
    renderAnnotationList();
    // Scroll anno list to top
    const listEl = document.getElementById('episode-detail-annos');
    if (listEl) listEl.scrollTop = 0;
}


function updateFormState() {
    const saveBtn = document.getElementById('btn-save-annotation');
    const preview = document.getElementById('anno-frame-preview');
    syncAnnotationFrameInputs();

    if (preview) {
        if (currentStartFrame != null && currentEndFrame != null) {
            const frames = currentEndFrame - currentStartFrame + 1;
            const duration = (frames / (episodeFps || 30)).toFixed(2);
            preview.innerHTML =
                '<span class="text-gray-500">Range: </span>' +
                '<span class="font-mono text-green-400 font-bold">' + currentStartFrame + '</span>' +
                '<span class="text-gray-600"> → </span>' +
                '<span class="font-mono text-green-400 font-bold">' + currentEndFrame + '</span>' +
                '<span class="text-gray-500 ml-2">(' + frames + 'f · ' + duration + 's)</span>';
            preview.className = 'bg-gray-900 rounded-lg px-3 py-2 text-center';
        } else if (currentStartFrame != null) {
            preview.innerHTML =
                '<span class="text-gray-500">Start: </span>' +
                '<span class="font-mono text-yellow-400 font-bold">' + currentStartFrame + '</span>' +
                '<span class="text-gray-600"> — waiting for End</span>';
            preview.className = 'bg-gray-900 rounded-lg px-3 py-2 text-center';
        } else {
            preview.innerHTML =
                '<span class="text-gray-500">Set Start / Set End using the fields or buttons above</span>';
            preview.className = 'bg-gray-900 rounded-lg px-3 py-2 text-center';
        }
    }

    if (saveBtn) {
        const canSave = currentStartFrame != null && currentEndFrame != null && currentEndFrame >= currentStartFrame;
        saveBtn.disabled = !canSave;
        saveBtn.className = canSave
            ? 'w-full bg-blue-600 hover:bg-blue-500 text-white py-2 rounded-lg text-xs font-medium transition-colors'
            : 'w-full bg-gray-700 text-gray-500 py-2 rounded-lg text-xs cursor-not-allowed font-medium';
    }
}


// ── Frame capture ────────────────────────────────────

function syncAnnotationFrameInputs() {
    const startInput = document.getElementById('anno-start-frame-input');
    const endInput = document.getElementById('anno-end-frame-input');
    if (startInput) startInput.value = currentStartFrame == null ? '' : currentStartFrame;
    if (endInput) endInput.value = currentEndFrame == null ? '' : currentEndFrame;
}

function readAnnotationFrameInputs() {
    const startInput = document.getElementById('anno-start-frame-input');
    const endInput = document.getElementById('anno-end-frame-input');
    if (startInput) currentStartFrame = startInput.value === ''
        ? null : Math.max(0, parseInt(startInput.value, 10) || 0);
    if (endInput) currentEndFrame = endInput.value === ''
        ? null : Math.max(0, parseInt(endInput.value, 10) || 0);
    updateFormState();
}

function captureStartFrame() {
    const frame = typeof getCurrentFrame === 'function' ? getCurrentFrame() : 0;
    currentStartFrame = frame;
    syncAnnotationFrameInputs();
    updateFormState();
    const form = document.getElementById('anno-form');
    if (form && form.classList.contains('hidden')) showNewAnnotationForm(false);
}

function captureEndFrame() {
    const frame = typeof getCurrentFrame === 'function' ? getCurrentFrame() : 0;
    currentEndFrame = frame;
    syncAnnotationFrameInputs();
    updateFormState();
    const form = document.getElementById('anno-form');
    if (form && form.classList.contains('hidden')) showNewAnnotationForm(false);
}

window.captureStartFrame = captureStartFrame;
window.captureEndFrame = captureEndFrame;


// ── Save ─────────────────────────────────────────────

async function saveAnnotation() {
    if (!isAnnotationEditor()) return;
    const labelInput = document.getElementById('anno-label-input');
    const label = labelInput?.value?.trim();
    if (!label) {
        if (labelInput) {
            labelInput.classList.add('border-red-500');
            labelInput.focus();
            const warn = document.getElementById('anno-label-warn');
            if (warn) warn.classList.remove('hidden');
            setTimeout(() => {
                labelInput.classList.remove('border-red-500');
                if (warn) warn.classList.add('hidden');
            }, 2500);
        }
        return;
    }

    const body = {
        label: label,
        start_frame_index: currentStartFrame,
        end_frame_index: currentEndFrame,
    };

    try {
        let res;
        if (editingAnnotationId) {
            const oldSeg = annotations.find(a => a.id === editingAnnotationId);
            if (oldSeg) _pushUndo('update', oldSeg);
            body.updated_at = _editingBaselineUpdatedAt || undefined;  // optimistic concurrency
            // 兼容历史候选段：人工编辑后直接确认为正式段
            if (_editingWasCandidate) body.status = 'confirmed';
            res = await fetch(`/api/v1/annotation/${editingAnnotationId}`, {
                method: 'PUT',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(body),
            });
        } else {
            res = await fetch(`/api/v1/episode/${currentEpisodeId}/annotations`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(body),
            });
            if (res.ok) {
                const created = await res.json();
                if (created && created.id) _pushUndo('create', { id: created.id });
            }
        }

        if (res.status === 409) {
            // Another device saved this slice while we were editing — reload
            // the latest version and re-enter edit mode so the user can
            // re-apply their changes without creating a duplicate slice.
            const conflictId = editingAnnotationId;
            alert(t('slice_conflict'));
            await loadAnnotations(currentEpisodeId);
            editAnnotation(conflictId);
            return;
        }

        if (res.ok) {
            hideAnnotationForm();
            await loadAnnotations(currentEpisodeId);
        } else {
            const err = await res.json();
            alert(t('slice_save_failed') + (err.detail || 'unknown'));
        }
    } catch (err) {
        alert(t('slice_save_failed') + err.message);
    }
}


// ── Edit ─────────────────────────────────────────────

function editAnnotation(annoId) {
    if (!isAnnotationEditor()) return;
    const seg = annotations.find(a => a.id === annoId);
    if (!seg) return;

    editingAnnotationId = annoId;
    _editingWasCandidate = (seg.status || 'confirmed') === 'candidate';
    _editingBaselineUpdatedAt = seg.updated_at || null;
    currentStartFrame = seg.start_frame_index;
    currentEndFrame = seg.end_frame_index;
    syncAnnotationFrameInputs();

    const form = document.getElementById('anno-form');
    const titleEl = document.getElementById('anno-form-title');
    if (form) form.classList.remove('hidden');
    if (titleEl) titleEl.textContent = t('edit_annotation');

    const labelInput = document.getElementById('anno-label-input');
    if (labelInput) labelInput.value = seg.label;

    const saveBtn = document.getElementById('btn-save-annotation');
    if (saveBtn) {
        const labelSpan = saveBtn.querySelector('[data-i18n]');
        if (labelSpan) labelSpan.textContent = t('save_changes');
    }

    updateFormState();
    renderAnnotationList();
    labelInput?.focus();
}


// ── Delete ───────────────────────────────────────────

async function deleteAnnotation(annoId) {
    if (!isAnnotationEditor()) return;
    if (!confirm(t('confirm_delete_anno'))) return;
    const seg = annotations.find(a => a.id === annoId);
    if (seg) _pushUndo('delete', seg);
    try {
        const res = await fetch(`/api/v1/annotation/${annoId}`, { method: 'DELETE' });
        if (res.ok) {
            hideAnnotationForm();
            await loadAnnotations(currentEpisodeId);
        }
    } catch (err) {
        alert(t('slice_delete_failed') + err.message);
    }
}


// ── Undo (Ctrl+Z) ────────────────────────────────────
// 纯前端撤销栈:每次标注修改前压入旧状态(上限 10 步),
// Ctrl+Z 撤销一步(删除→重建 / 新建→删 / 编辑·确认·边界→PUT 恢复)。
let _undoStack = [];
const _UNDO_MAX = 10;

function _pushUndo(op, seg) {
    if (!seg) return;
    _undoStack.push({ op, seg: JSON.parse(JSON.stringify(seg)) });
    if (_undoStack.length > _UNDO_MAX) _undoStack.shift();
}

async function undoAnnotation() {
    if (!isAnnotationEditor()) return;
    const entry = _undoStack.pop();
    if (!entry) return;
    const { op, seg } = entry;
    try {
        if (op === 'delete') {
            // 重建(关键帧等附属数据 v1 不恢复;label/区间/来源恢复)
            await fetch(`/api/v1/episode/${currentEpisodeId}/annotations`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    label: seg.label || '—',
                    start_frame_index: seg.start_frame_index,
                    end_frame_index: seg.end_frame_index,
                }),
            });
        } else if (op === 'create') {
            await fetch(`/api/v1/annotation/${seg.id}`, { method: 'DELETE' });
        } else {  // update:恢复旧字段
            await fetch(`/api/v1/annotation/${seg.id}`, {
                method: 'PUT',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    label: seg.label,
                    start_frame_index: seg.start_frame_index,
                    end_frame_index: seg.end_frame_index,
                    status: seg.status || 'confirmed',
                    updated_at: seg.updated_at || undefined,
                }),
            });
        }
        await loadAnnotations(currentEpisodeId);
    } catch (err) {
        console.error('undo failed:', err);
    }
}

document.addEventListener('keydown', (e) => {
    if ((e.ctrlKey || e.metaKey) && (e.key === 'z' || e.key === 'Z')) {
        if (!isAnnotationEditor()) return;
        const t = e.target;
        if (t && (t.tagName === 'INPUT' || t.tagName === 'TEXTAREA' || t.isContentEditable)) return;
        e.preventDefault();
        undoAnnotation();
    }
});


// ── Current annotation highlight ────────────────────

function _singleCurrentAnnoId(frameIndex, preferId) {
    /* 单一高亮规则:一次只允许一个段处于"当前"状态。
       preferId:点击卡片时指定该段(即使其起点落在另一段的区间内,
       也只高亮被点的这张);否则取"开始帧 ≤ 当前帧"的段中开始帧
       最大的(最近进入的那段)。段重叠时不再出现双高亮。 */
    if (preferId && annotations.some(a => a.id === preferId)) return preferId;
    let best = null;
    for (const a of annotations) {
        if (frameIndex >= a.start_frame_index && frameIndex <= a.end_frame_index) {
            if (!best || a.start_frame_index > best.start_frame_index) best = a;
        }
    }
    return best ? best.id : null;
}

function updateCurrentAnnotation(frameIndex, preferId) {
    updateAnnotationTimelineCursor(frameIndex);
    const currentId = _singleCurrentAnnoId(frameIndex, preferId);
    if (currentId === lastHighlightedAnnoId) {
        lastHighlightedFrame = frameIndex;
        return;
    }
    lastHighlightedAnnoId = currentId;
    // 只切换高亮 class,不重建列表 —— 重建 innerHTML 会让用户点击
    // 瞬间的 DOM 被替换,导致"点中间的卡片命中上面/下面的卡片"
    const listEl = document.getElementById('episode-detail-annos');
    if (listEl) {
        listEl.querySelectorAll('.annotation-card').forEach(card => {
            const active = card.dataset.annoId === currentId;
            card.classList.toggle('bg-blue-900/30', active);
            card.classList.toggle('border-blue-600/60', active);
            card.classList.toggle('bg-gray-800/40', !active);
            card.classList.toggle('border-gray-700/30', !active);
            card.classList.toggle('hover:bg-gray-800', !active);
            const label = card.querySelector('.annotation-card-label');
            if (label) label.classList.toggle('text-blue-200', active);
            // 点击来源(切片预览/时间轴段)时,当前卡片滚动到可视区,
            // 与右侧列表保持同步;播放中不滚动(避免打断人工浏览)
            if (active && preferId) {
                try { card.scrollIntoView({ block: 'nearest', behavior: 'smooth' }); } catch (_) {}
            }
        });
    }
    lastHighlightedFrame = frameIndex;
}
let lastHighlightedFrame = -1;
let lastHighlightedAnnoId = null;

if (typeof setOnFrameChange === 'function') {
    setOnFrameChange((frameIndex) => {
        updateCurrentAnnotation(frameIndex);
    });
}


// ── Helpers ──────────────────────────────────────────

function escHtml(str) {
    if (!str) return '';
    str = String(str);
    return str.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;');
}


// ── Init ─────────────────────────────────────────────

document.addEventListener('DOMContentLoaded', () => {
    const btnSave = document.getElementById('btn-save-annotation');
    if (btnSave) btnSave.addEventListener('click', saveAnnotation);

    const btnAi = document.getElementById('btn-ai-annotate');
    if (btnAi) btnAi.addEventListener('click', aiAnnotate);
    const btnZoomIn = document.getElementById('btn-zoom-in');
    const btnZoomOut = document.getElementById('btn-zoom-out');
    if (btnZoomIn) btnZoomIn.addEventListener('click', () => {
        const idx = ZOOM_STEPS.indexOf(timelineZoom.scale);
        const next = ZOOM_STEPS[Math.min(ZOOM_STEPS.length - 1, (idx < 0 ? 0 : idx) + 1)];
        _setZoom(next, timelineZoom.centerFrame);
    });
    if (btnZoomOut) btnZoomOut.addEventListener('click', () => {
        const idx = ZOOM_STEPS.indexOf(timelineZoom.scale);
        const prev = ZOOM_STEPS[Math.max(0, (idx < 0 ? 0 : idx) - 1)];
        _setZoom(prev, timelineZoom.centerFrame);
    });
    _bindTimelineZoom();

    const btnAnnoStart = document.getElementById('btn-anno-set-start');
    if (btnAnnoStart) btnAnnoStart.addEventListener('click', captureStartFrame);
    const btnAnnoEnd = document.getElementById('btn-anno-set-end');
    if (btnAnnoEnd) btnAnnoEnd.addEventListener('click', captureEndFrame);

    const startInput = document.getElementById('anno-start-frame-input');
    const endInput = document.getElementById('anno-end-frame-input');
    if (startInput) startInput.addEventListener('change', readAnnotationFrameInputs);
    if (endInput) endInput.addEventListener('change', readAnnotationFrameInputs);

    const btnBack10 = document.getElementById('btn-skip-back-10');
    const btnBack1 = document.getElementById('btn-step-back');
    const btnFwd1 = document.getElementById('btn-step-forward');
    const btnFwd10 = document.getElementById('btn-skip-forward-10');

    if (btnBack10) btnBack10.addEventListener('click', () => { if (typeof skipFrames === 'function') skipFrames(-10); });
    if (btnBack1) btnBack1.addEventListener('click', () => { if (typeof stepFrame === 'function') stepFrame(-1); });
    if (btnFwd1) btnFwd1.addEventListener('click', () => { if (typeof stepFrame === 'function') stepFrame(1); });
    if (btnFwd10) btnFwd10.addEventListener('click', () => { if (typeof skipFrames === 'function') skipFrames(10); });
});
