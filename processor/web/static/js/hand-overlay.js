/* Hand Skeleton SVG 叠加层 —— 在原始视频画面上实时绘制手部骨骼(第一版)。
 *
 * 控制方式:独立的 ☷ Preview Options 菜单(素材栏右侧,点击展开):
 *  - Hand Skeleton 开关(总开关,默认 ON;所有视频 tile 的骨骼一起显隐)
 *  - 数据状态行:Loading… / Available / No tracking data(无数据时开关禁用,
 *    点击菜单不会报错 —— 对应 Project_Test10_000002 这类只有手套数据的批次)
 *
 * 数据:GET /api/v1/video/{episode}/{camera}/hand-keypoints?start_frame&end_frame
 * (stereo_left/right 各自的 2D 归一化关键点,分段拉取,窗口缓存)。
 * 同步:统一帧回调(多回调注册,与标注高亮互不覆盖)+ 播放期 rAF 循环。
 * 绘制:与渲染视频完全一致的 demo 风格(五指分色/掌心灰线/白色腕点,
 * 尺寸按显示比例缩放);SVG pointer-events:none,不影响视频拖动/播放。
 */

(function () {
    'use strict';

    /* 与渲染视频(_draw_demo_style)完全一致的 demo 风格:
       五指分色、掌心灰线、白色腕点;尺寸按视频显示比例缩放。 */
    const FINGERS = {
        Thumb: { ids: [1, 2, 3, 4], color: '#0080ff' },
        Index: { ids: [5, 6, 7, 8], color: '#00ff00' },
        Middle: { ids: [9, 10, 11, 12], color: '#00ffff' },
        Ring: { ids: [13, 14, 15, 16], color: '#ff00ff' },
        Pinky: { ids: [17, 18, 19, 20], color: '#ff8000' },
    };
    const PALM_EDGES = [[0, 1], [0, 5], [5, 9], [9, 13], [13, 17], [0, 17]];
    const WINDOW_HALF = 250;    // 每次拉取 ±250 帧
    const PREFETCH_MARGIN = 60; // 到窗口边界前提前拉取下一窗口
    const PREFETCH_WINDOWS = 3; // 播放头前至少保持三个关键点窗口
    const INITIAL_BUFFER_FRAMES = 450; // 15s at the default 30 FPS
    const FULL_PRELOAD_MAX_FRAMES = 3000;
    const CACHE_HALF = 1500;    // 缓存保留 ±1500 帧,超出剔除

    const overlays = new Map(); // key → state
    let masterOn = true;

    /* ── 控制菜单(☷ Preview Options,漂浮可拖动)── */
    function bindMenu() {
        const box = document.getElementById('preview-options');
        const btn = document.getElementById('btn-preview-options');
        const menu = document.getElementById('preview-options-menu');
        const chk = document.getElementById('toggle-hand-skeleton');
        // The menu button must remain usable even if a newer/older template
        // temporarily lacks one optional toggle. Previously this early return
        // made Preview Options completely inert when any child was missing.
        if (!box || !btn || !menu) return;
        if (box.dataset.previewMenuBound === '1') return;
        box.dataset.previewMenuBound = '1';

        // 拖动:按住移动超过 5px 视为拖动,否则视为点击(展开/收起菜单)。
        // 坐标用 offsetLeft/offsetTop(相对视频工作区,与 style.left/top
        // 同坐标系)—— 之前用 getBoundingClientRect(视口坐标)导致
        // 拖动瞬间面板跳位。
        let dragStart = null;
        btn.addEventListener('pointerdown', (e) => {
            dragStart = { x: e.clientX, y: e.clientY,
                          bx: box.offsetLeft, by: box.offsetTop, moved: false };
            btn.setPointerCapture(e.pointerId);
        });
        btn.addEventListener('pointermove', (e) => {
            if (!dragStart) return;
            const dx = e.clientX - dragStart.x, dy = e.clientY - dragStart.y;
            if (Math.abs(dx) + Math.abs(dy) > 5) dragStart.moved = true;
            if (dragStart.moved) {
                box.style.left = Math.max(0, dragStart.bx + dx) + 'px';
                box.style.top = Math.max(0, dragStart.by + dy) + 'px';
                box.style.right = 'auto';
            }
        });
        btn.addEventListener('pointerup', (e) => {
            const wasDrag = dragStart && dragStart.moved;
            dragStart = null;
            if (!wasDrag) {
                e.stopPropagation();
                menu.classList.toggle('hidden');
                btn.setAttribute('aria-expanded', String(!menu.classList.contains('hidden')));
            }
        });
        document.addEventListener('click', (e) => {
            if (!menu.classList.contains('hidden') && !menu.contains(e.target) && !btn.contains(e.target)) {
                menu.classList.add('hidden');
                btn.setAttribute('aria-expanded', 'false');
            }
        });
        if (chk) chk.addEventListener('change', () => {
            masterOn = chk.checked;
            overlays.forEach(st => _render(st));
        });
        const chkTrail = document.getElementById('toggle-hand-trail');
        if (chkTrail) chkTrail.addEventListener('change', () => {
            trailOn = chkTrail.checked;
            overlays.forEach(st => _render(st));
        });
        const chkGlove = document.getElementById('toggle-glove-sensor');
        const chkHeatmap = document.getElementById('toggle-heatmap');
        const chkHand3dWorld = document.getElementById('toggle-hand3d-world');
        if (chkGlove) chkGlove.addEventListener('change', () => {
            gloveOn = chkGlove.checked;
            _applyTileVisibility();
        });
        if (chkHeatmap) chkHeatmap.addEventListener('change', () => {
            // 深度显示的唯一开关:开 = 直接挂载采集端深度视频 tile;
            // 关 = 卸除 source(布局随 workspaceSources 回收,不留空位)
            depthOn = chkHeatmap.checked;
            _applyTileVisibility();
        });
        if (chkHand3dWorld) chkHand3dWorld.addEventListener('change', () => {
            hand3dWorldOn = chkHand3dWorld.checked;
            // 开 = 自动挂载 3D 世界窗口;关 = 卸除(主区空单元格随之回收)
            _applyTileVisibility();
        });
        updateMenuStatus();
        updatePreviewMenuData();
    }

    /* 手套/深度/3D 世界窗口 tile 显隐开关(按素材类型前缀,只影响画布显示) */
    let gloveOn = true, depthOn = true, hand3dWorldOn = true;
    function _tilesByPrefix(prefix) {
        const grid = document.getElementById('video-grid');
        if (!grid) return [];
        return [...grid.querySelectorAll('[data-source-key]')]
            .filter(t => (t.dataset.sourceKey || '').startsWith(prefix));
    }
    function _applyTileVisibility() {
        // 关闭某个预览 tile 不能改变 Preview Options 容器本身的页面状态。
        // removeGroupedSource/renderGroupedWorkspace 会重排工作区,这里在
        // 前后各做一次轻量兜底,但返回批次列表时仍由 __episodeOpen=false
        // 正常隐藏。
        _restorePreviewOptions();
        ['hand:', 'glove:'].forEach(p =>
            _tilesByPrefix(p).forEach(t => { t.style.display = gloveOn ? '' : 'none'; }));
        // 深度 / 3D 世界窗口:直接挂/卸 source(布局随 workspaceSources
        // 回收,不留 display:none 空位;挂载幂等,重复调用无害)
        if (typeof window.setDepthTileVisible === 'function') {
            window.setDepthTileVisible(depthOn);
        }
        if (typeof window.setHand3dWorldVisible === 'function') {
            window.setHand3dWorldVisible(hand3dWorldOn);
        }
        _restorePreviewOptions();
    }

    function _restorePreviewOptions() {
        if (window.__episodeOpen !== true) return;
        const box = document.getElementById('preview-options');
        if (box) box.classList.remove('hidden');
    }

    /* 数据驱动:当前批次有什么数据,菜单才显示对应的开关行
       (media-groups 素材清单为准;批次切换时由 player.js 调用)。
       同时把 checkbox 勾选同步到真实状态 —— 修复"显示开启实际
       关闭"的 UI 与状态脱节(批次切换后 checkbox 残留旧勾选)。 */
    function updatePreviewMenuData() {
        if (typeof renderPreviewVideoSources === 'function') renderPreviewVideoSources();
        const rowGlove = document.getElementById('po-glove-row');
        const rowHeatmap = document.getElementById('po-heatmap-row');
        const rowHand3dWorld = document.getElementById('po-hand3d-world-row');
        const kinds = new Set(((typeof currentMediaGroups !== 'undefined' && currentMediaGroups)
            ? (currentMediaGroups.sources || []) : [])
            .map(s => s.kind));
        if (rowGlove) {
            const has = kinds.has('glove') || kinds.has('hand');
            rowGlove.classList.toggle('hidden', !has);
            rowGlove.classList.toggle('flex', has);
        }
        if (rowHeatmap) {
            // 深度显示的唯一开关(无独立 Depth 行):有深度数据才显示
            const has = kinds.has('depth');
            rowHeatmap.classList.toggle('hidden', !has);
            rowHeatmap.classList.toggle('flex', has);
        }
        if (rowHand3dWorld) {
            // 任何可读取的 hand_3d 产物都提供 3D 预览；D435 深度抬升
            // 仍会在 player.js 中标记为 world mode。
            const has = (typeof hand3dData !== 'undefined' && hand3dData
                         && hand3dData.hasPreview) || false;
            const label = rowHand3dWorld.querySelector('[data-i18n="hand3d_world"]');
            if (label && typeof t === 'function') {
                label.textContent = hand3dData && hand3dData.worldMode
                    ? t('hand3d_world') : t('3d_skeleton');
            }
            rowHand3dWorld.classList.toggle('hidden', !has);
            rowHand3dWorld.classList.toggle('flex', has);
        }
        // UI 勾选 = 真实状态(单一事实源:gloveOn/depthOn/hand3dWorldOn)
        const chkGlove = document.getElementById('toggle-glove-sensor');
        const chkHeatmap = document.getElementById('toggle-heatmap');
        const chkH3d = document.getElementById('toggle-hand3d-world');
        if (chkGlove) chkGlove.checked = gloveOn;
        if (chkHeatmap) chkHeatmap.checked = depthOn;
        if (chkH3d) chkH3d.checked = hand3dWorldOn;
        _applyTileVisibility();
        _restorePreviewOptions();
    }

    function updateMenuStatus() {
        const chk = document.getElementById('toggle-hand-skeleton');
        const statusEl = document.getElementById('preview-options-status');
        if (!chk) return;
        const states = [...overlays.values()];
        let text = '';
        if (states.length === 0) {
            text = 'No video loaded';
            chk.disabled = true;
        } else if (states.some(s => s.status === 'loading')) {
            text = 'Loading…';
            chk.disabled = true;
        } else if (states.some(s => s.status === 'ready')) {
            text = 'Available';
            chk.disabled = false;
        } else {
            text = 'No tracking data — run the workflow first';
            chk.disabled = true;
            // Keep the control's batch default consistent even when this
            // episode has no overlay artifact. It is disabled because there
            // is nothing to draw, but it must not turn the global default off
            // and leak that state into the next episode.
            chk.checked = true;
            masterOn = true;
        }
        if (statusEl) statusEl.textContent = text;
        if (!chk.disabled && !chk.checked) chk.checked = masterOn;
    }

    /* ── 叠加层生命周期 ── */
    function initHandOverlay(source, tile, holder) {
        if (source.kind !== 'video') return Promise.resolve(false); // no overlay
        // 不再限制具体相机:双目(stereo_left/right)与单目(mediapipe_hand)
        // 都由 /hand-keypoints 接口按相机提供数据 —— 无数据 404 →
        // 该 tile 显示 No Data,不做叠加(探测驱动,兼容两种手部模块)
        const key = (typeof groupedSourceKey === 'function')
            ? groupedSourceKey(source) : String(source.source_key || '');
        if (overlays.has(key)) return Promise.resolve(true);

        // 透明 SVG 层(pointer-events:none —— 不影响视频拖动和播放)
        const svg = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
        svg.setAttribute('class', 'hand-overlay-svg');
        svg.style.cssText = 'position:absolute;inset:0;width:100%;height:100%;pointer-events:none;z-index:5;';
        svg.setAttribute('width', '100%');
        svg.setAttribute('height', '100%');
        holder.appendChild(svg);

        const state = {
            key, source, tile, holder, svg,
            frames: new Map(),       // frameIndex → {h0, h1}
            fetched: [],             // 已拉取的窗口 [[start,end],...]:窗口内
                                     // 缓存没有的帧 = 该帧真无手(不再重拉)
            frameCount: 0,           // 手部骨骼数据实际帧数(接口 count)
            inflight: false,
            prefetchPending: false,
            prefetchEnd: -1,
            fullPreloaded: false,
            fullCacheKey: '',
            status: 'loading',       // loading | ready | nodata
        };
        overlays.set(key, state);
        updateMenuStatus();

        // 首帧探测:数据存在性 + 第一窗口
        const initialReady = (async () => {
            const count = Number(source.frame_count) || 0;
            const full = count > 0 && count <= FULL_PRELOAD_MAX_FRAMES;
            let ok = false;
            if (full) {
                ok = await _fetchWindow(state, 0, 0, count - 1);
            } else {
                // Long clips get 10–15 seconds of keypoints before the
                // playback barrier opens. Three further windows are queued
                // asynchronously while RGB is playing.
                const initialEnd = Math.min(count - 1, INITIAL_BUFFER_FRAMES - 1);
                ok = await _fetchWindow(state, 0, 0, initialEnd);
                if (ok) _prefetchWindows(state, initialEnd + 1, PREFETCH_WINDOWS);
            }
            state.status = ok ? 'ready' : 'nodata';
            updateMenuStatus();
            _render(state);
            return ok;
        })().catch(() => {
            state.status = 'nodata';
            updateMenuStatus();
            return false;
        });
        // player.js awaits this promise before removing the batch loading
        // barrier, so RGB and spatial keypoints become visible together.
        return initialReady;
    }

    function destroyHandOverlay(key) {
        const st = overlays.get(key);
        if (!st) return;
        overlays.delete(key);
        st.svg.remove();
        updateMenuStatus();
    }

    function destroyAllHandOverlays() {
        overlays.forEach((st, key) => destroyHandOverlay(key));
        // Overlay visibility is a per-episode default, not a global session
        // preference. A batch without tracking data may set masterOn=false;
        // reset it before the next batch so valid 2D data is shown directly.
        masterOn = true;
        const chk = document.getElementById('toggle-hand-skeleton');
        if (chk) chk.checked = true;
        updateMenuStatus();
    }

    function _inFetched(state, frame) {
        return state.fetched.some(([s, e]) => frame >= s && frame <= e);
    }

    async function _fetchWindow(state, center, requestedStart = null,
                                requestedEnd = null) {
        if (state.inflight) return false;
        const start = requestedStart == null
            ? Math.max(0, center - WINDOW_HALF) : Math.max(0, requestedStart);
        const end = requestedEnd == null
            ? center + WINDOW_HALF : Math.max(start, requestedEnd);
        const isFull = start === 0 && requestedEnd != null &&
            (Number(state.source.frame_count) || 0) <= end + 1;
        if (state.fetched.some(([s, e]) => start >= s && end <= e)) return true;
        state.inflight = true;
        try {
            const epoch = currentEpisodeId;
            const sessionToken = typeof getPlaybackSessionToken === 'function'
                ? getPlaybackSessionToken() : null;
            const isCurrent = () => typeof isCurrentPlaybackSession !== 'function'
                ? currentEpisodeId === epoch
                : isCurrentPlaybackSession(epoch, sessionToken);
            const cacheKey = `${epoch}:hand2d:${state.source.source_key}:${start}:${end}`;
            if (window.EgoMediaCache) {
                const cached = await window.EgoMediaCache.get(cacheKey);
                if (!isCurrent()) return false;
                const value = cached && cached.value;
                if (value && Array.isArray(value.frames)) {
                    value.frames.forEach(fr => state.frames.set(fr.f, fr));
                    state.fetched.push([start, end]);
                    state.prefetchEnd = end;
                    state.frameCount = value.frameCount || end + 1;
                    if (isFull) {
                        state.fullPreloaded = true;
                        state.fullCacheKey = cacheKey;
                    }
                    _evict(state, center);
                    return true;
                }
            }
            const signal = typeof getMediaLoadSignal === 'function'
                ? getMediaLoadSignal() : null;
            const res = await fetch(
                `/api/v1/video/${epoch}/${state.source.source_key}/hand-keypoints?start_frame=${start}&end_frame=${end}`,
                signal ? { signal } : {});
            if (!isCurrent()) return false;  // 丢弃过期响应
            if (!res.ok) return false;
            const data = await res.json();
            if (!isCurrent()) return false;
            (data.frames || []).forEach(fr => state.frames.set(fr.f, fr));
            if (data.count) state.frameCount = data.count;
            // 记录已拉取窗口:窗口内缓存没有的帧 = 该帧真无手,
            // 之后直接清空骨架显示,不再反复请求(接口只返回有手的帧)。
            state.fetched.push([start, end]);
            state.prefetchEnd = Math.max(state.prefetchEnd, end);
            if (isFull) {
                state.fullPreloaded = true;
                state.fullCacheKey = cacheKey;
            }
            if (window.EgoMediaCache) {
                window.EgoMediaCache.put(cacheKey, {
                    frames: data.frames || [], frameCount: state.frameCount,
                });
            }
            _evict(state, center);
            return true;
        } catch (_) {
            return false;
        } finally {
            state.inflight = false;
        }
    }

    async function _prefetchWindows(state, frame, count = PREFETCH_WINDOWS) {
        if (state.prefetchPending) return;
        state.prefetchPending = true;
        try {
            let start = Math.max(frame, state.prefetchEnd + 1);
            for (let i = 0; i < count; i++) {
                const center = start + WINDOW_HALF;
                const ok = await _fetchWindow(state, center);
                if (!ok) break;
                start = state.prefetchEnd + 1;
            }
        } finally {
            state.prefetchPending = false;
        }
    }

    function _prefetchWindow(state, frame) {
        // 预取只负责填充缓存，不触碰当前 SVG。这样播放到窗口边界时，
        // 当前帧仍由已提交画面保持，下一窗口完整返回后才进行一次切换。
        if (state.inflight || state.prefetchPending
                || state.prefetchEnd < 0
                || frame < state.prefetchEnd - PREFETCH_MARGIN) return;
        _prefetchWindows(state, frame, PREFETCH_WINDOWS);
    }

    function _evict(state, center) {
        if (state.fullPreloaded) return;
        for (const f of [...state.frames.keys()]) {
            if (Math.abs(f - center) > CACHE_HALF) state.frames.delete(f);
        }
    }

    function _clearSvg(state) {
        while (state.svg.firstChild) state.svg.removeChild(state.svg.firstChild);
    }

    /* ── 手部轨迹:最近帧的骨骼残影,越旧越淡 ──
       版本备忘:
       第 1 版 = 连续式、每 1 帧一个残影(TRAIL_STEP=1,最密);
       当前版 = 连续式、每 4 帧一个残影、往前取 20 帧(5 个残影)
               + 完整骨骼(连线+关键点)。
       切换密度只需改 TRAIL_STEP/TRAIL_FRAMES。 */
    let trailOn = false;
    const TRAIL_FRAMES = 5;    // 残影个数(20 帧 / 4 帧间隔 = 5 个)
    const TRAIL_STEP = 4;      // 残影间隔帧数(第 1 版 = 1)

    function _svgLine(state, pts, a, b, color, w, opacity, target) {
        if (!pts[a] || !pts[b]) return;
        const line = document.createElementNS('http://www.w3.org/2000/svg', 'line');
        line.setAttribute('x1', pts[a][0]); line.setAttribute('y1', pts[a][1]);
        line.setAttribute('x2', pts[b][0]); line.setAttribute('y2', pts[b][1]);
        line.setAttribute('stroke', color);
        line.setAttribute('stroke-width', String(Math.max(0.5, w)));
        line.setAttribute('stroke-linecap', 'round');
        line.setAttribute('opacity', String(opacity));
        (target || state.svg).appendChild(line);
    }

    /* 与后端 hand_render.py 对齐:按手部在当前画布中的实际像素范围
       缩放骨骼样式。固定 5/7/9px 在 640x480 的远景手部上会显得过粗。 */
    function _handStyleScale(pts, displayScale) {
        const valid = (pts || []).filter(p => p && Number.isFinite(p[0])
            && Number.isFinite(p[1]));
        if (valid.length < 2) return 1;
        const xs = valid.map(p => p[0]), ys = valid.map(p => p[1]);
        const span = Math.max(Math.max(...xs) - Math.min(...xs),
                              Math.max(...ys) - Math.min(...ys));
        // span is in displayed pixels; 160 is the reference source-pixel
        // hand size used by the Python renderer.
        return Math.max(0.45, Math.min(1,
            span / Math.max(1, 160 * (displayScale || 1))));
    }

    function _drawTrail(state, frame, lb) {
        /* 连续式残影(无隐藏期):每隔 TRAIL_STEP 帧一个,共
           TRAIL_FRAMES 个,完整骨骼 = 连线 + 关键点圆点。
           透明度线性衰减:最近最明显,越旧越淡;当前帧骨架覆盖其上。 */
        const { s, vw, vh, dx, dy } = lb;
        // 批量构建:所有节点先挂到 Fragment,一次 append(减少 DOM 变更次数,
        // 降低浏览器样式/布局抖动)
        const frag = document.createDocumentFragment();
        for (let k = 1; k <= TRAIL_FRAMES; k++) {
            const age = k * TRAIL_STEP;
            const entry = state.frames.get(frame - age);
            if (!entry) continue;
            const opacity = (1 - k / (TRAIL_FRAMES + 1)) * 0.55;  // 线性消失
            // 颜色与实际手部骨骼完全一致:五指分色 + 掌心灰 + 白腕点
            const ptColor = new Array(21).fill('#9ca3af');
            ptColor[0] = '#ffffff';
            for (const [finger, { ids, color }] of Object.entries(FINGERS)) {
                for (const idx of ids) ptColor[idx] = color;
            }
            for (const h of [entry.h0, entry.h1]) {
                if (!h || !h.k) continue;
                const pts = h.k.map(([x, y]) => (x == null || y == null)
                    ? null : [dx + x * vw * s, dy + y * vh * s]);
                const hs = _handStyleScale(pts, s);
                for (const [a, b] of PALM_EDGES) _svgLine(state, pts, a, b, '#c8c8c8', 1.2 * s * hs, opacity, frag);
                for (const [finger, { ids, color }] of Object.entries(FINGERS)) {
                    const chain = finger === 'Thumb' ? ids : [0, ...ids];
                    for (let i = 0; i < chain.length - 1; i++) {
                        _svgLine(state, pts, chain[i], chain[i + 1], color, 1.2 * s * hs, opacity, frag);
                    }
                }
                // 残影关键点(按关节对应手指色,透明度与连线一致)
                pts.forEach((p, i) => {
                    if (!p) return;
                    const c = document.createElementNS('http://www.w3.org/2000/svg', 'circle');
                    c.setAttribute('cx', p[0]); c.setAttribute('cy', p[1]);
                    c.setAttribute('r', String(Math.max(1, 2.2 * s * hs)));
                    c.setAttribute('fill', ptColor[i] || '#9ca3af');
                    c.setAttribute('opacity', String(opacity));
                    frag.appendChild(c);
                });
            }
        }
        state.svg.appendChild(frag);
    }

    /* Plyr 以 object-contain 显示视频:归一化坐标 → letterbox 内的像素 */
    function _letterbox(state) {
        const video = state.holder.querySelector('video');
        if (!video || !video.videoWidth) return null;
        const cw = state.holder.clientWidth, ch = state.holder.clientHeight;
        const vw = video.videoWidth, vh = video.videoHeight;
        const s = Math.min(cw / vw, ch / vh);
        return { s, vw, vh, dx: (cw - vw * s) / 2, dy: (ch - vh * s) / 2 };
    }

    function _render(state) {
        _renderAt(state, (typeof getCurrentFrame === 'function') ? getCurrentFrame() : 0);
    }

    function _renderAt(state, frame) {
        if (!masterOn || state.status !== 'ready') { _clearSvg(state); return; }
        const entry = state.frames.get(frame);
        if (entry) _prefetchWindow(state, frame);
        if (!entry) {
            if (_inFetched(state, frame)) {
                // 已拉取过的窗口里没有这帧 = 该帧真无手:清空骨架,
                // 不要继续显示上一帧的旧骨架(看起来会"错位/卡住")
                _clearSvg(state);
                return;
            }
            // 本帧尚未缓存：保持当前已提交画面，异步补窗口。
            // 窗口成功返回后，按最新播放帧一次性提交。
            _fetchWindow(state, frame).then(ok => {
                if (!ok) return;
                const latest = (typeof getCurrentFrame === 'function')
                    ? getCurrentFrame() : frame;
                _renderAt(state, latest);
            });
            return;
        }
        _clearSvg(state);
        const lb = _letterbox(state);
        if (!lb) return;
        const { s, vw, vh, dx, dy } = lb;
        // 批量构建:当前帧骨架也先挂 Fragment,一次 append
        const frag = document.createDocumentFragment();
        // 手部轨迹(先画旧帧残影,当前帧骨架覆盖其上)
        if (trailOn) _drawTrail(state, frame, lb);
        for (const h of [entry.h0, entry.h1]) {
            if (!h || !h.k) continue;
            const pts = h.k.map(([x, y]) => (x == null || y == null)
                ? null : [dx + x * vw * s, dy + y * vh * s]);
            const hs = _handStyleScale(pts, s);
            const addLine = (a, b, color, w) => {
                if (!pts[a] || !pts[b]) return;
                const line = document.createElementNS('http://www.w3.org/2000/svg', 'line');
                line.setAttribute('x1', pts[a][0]); line.setAttribute('y1', pts[a][1]);
                line.setAttribute('x2', pts[b][0]); line.setAttribute('y2', pts[b][1]);
                line.setAttribute('stroke', color);
                line.setAttribute('stroke-width', String(Math.max(0.75, w * s * hs)));
                line.setAttribute('stroke-linecap', 'round');
                line.setAttribute('opacity', '0.95');
                frag.appendChild(line);
            };
            const addDot = (i, r, fill, outline, ow) => {
                if (!pts[i]) return;
                const c = document.createElementNS('http://www.w3.org/2000/svg', 'circle');
                c.setAttribute('cx', pts[i][0]); c.setAttribute('cy', pts[i][1]);
                c.setAttribute('r', String(Math.max(1, r * s * hs)));
                c.setAttribute('fill', fill);
                c.setAttribute('stroke', outline);
                c.setAttribute('stroke-width', String(Math.max(0.5, ow * s * hs)));
                frag.appendChild(c);
            };
            // 掌心灰线(与渲染视频一致)
            for (const [a, b] of PALM_EDGES) addLine(a, b, '#c8c8c8', 2);
            // 五指分色骨骼线 + 关节圆点(thumb 从 1 起,其余从腕 0 连到指根)
            for (const [finger, { ids, color }] of Object.entries(FINGERS)) {
                const chain = finger === 'Thumb' ? ids : [0, ...ids];
                for (let i = 0; i < chain.length - 1; i++) addLine(chain[i], chain[i + 1], color, 3);
                for (const idx of ids) {
                    const isTip = idx === ids[ids.length - 1];
                    addDot(idx, isTip ? 7 : 5, color, 'rgb(30,30,30)', 1);
                }
            }
            // 腕部白圆
            addDot(0, 9, '#ffffff', 'rgb(40,40,40)', 2);
        }
        state.svg.appendChild(frag);
    }

    /* 统一帧回调:与标注高亮共用 setOnFrameChange(多回调) */
    function onFrame(frameIndex) {
        overlays.forEach(state => {
            if (!masterOn || state.status !== 'ready') return;
            _renderAt(state, frameIndex);
        });
    }

    /* 播放期间:rAF 循环按显示刷新率重画骨骼(60Hz)。
       timeupdate 只有 ~4Hz,直接用它驱动会在 25fps 播放时落后 6-7 帧,
       表现为"正常播放时骨骼没对齐";逐帧步进走精确回调,不受影响。 */
    let _rafLastFrame = -1;
    function _rafLoop() {
        requestAnimationFrame(_rafLoop);
        // player.js owns the central frame clock. Keeping this legacy loop
        // active would redraw 2D independently of the 3D canvas and could
        // make one layer appear one refresh behind the other.
        if (window.__egodataCentralFrameClock) return;
        if (!masterOn || overlays.size === 0) return;
        let playing = false;
        for (const st of overlays.values()) {
            const v = st.holder.querySelector('video');
            if (v && !v.paused && !v.ended && v.readyState >= 2) { playing = true; break; }
        }
        if (!playing) return;
        const t = (typeof getCurrentTime === 'function') ? getCurrentTime() : 0;
        const fps = (typeof getEpisodeFps === 'function') ? getEpisodeFps() : 25;
        const frame = Math.floor(t * fps + 0.002);
        if (frame === _rafLastFrame) return;
        _rafLastFrame = frame;
        overlays.forEach(st => {
            if (st.status !== 'ready') return;
            _renderAt(st, frame);
        });
    }
    requestAnimationFrame(_rafLoop);

    window.addEventListener('resize', () => overlays.forEach(_render));
    if (typeof setOnFrameChange === 'function') setOnFrameChange(onFrame);
    if (document.readyState !== 'loading') bindMenu();
    else document.addEventListener('DOMContentLoaded', bindMenu);

    window.initHandOverlay = initHandOverlay;
    window.destroyHandOverlay = destroyHandOverlay;
    window.destroyAllHandOverlays = destroyAllHandOverlays;
    window.updateHandOverlayMenu = updateMenuStatus;
    window.updatePreviewMenuData = updatePreviewMenuData;
    // 3D 世界窗口开关的真实状态(单一事实源):player.js 的自动挂载
    // 判断必须读这里,不能 `typeof hand3dWorldOn`(IIFE 内 let 在外
    // 不可见,永远 undefined → 自动挂载无视开关,造成"显示开实际关")
    window.isHand3dWorldOn = () => hand3dWorldOn;
    // 深度开关同样导出(深度 tile 自动挂载判断用)
    window.isDepthOn = () => depthOn;
})();
