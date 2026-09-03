/* Plyr video player management */
console.log('[player.js] build 202609030900-sync-barrier-central-clock');

const players = {};  // {camera_name: Plyr instance}
let currentEpisodeId = null;
let currentCameras = [];
let _episodeMediaLoadController = null;
// Every episode selection gets a new generation.  Episode ids normally differ,
// but reopening the same episode must also invalidate old async callbacks.
let _playbackSessionToken = 0;
let workspaceSources = [];
let workspaceCameras = [];
let currentHasSkeleton = false;
let _defaultLayoutApplied = false;  // balanced layout (videos + hand heatmaps) applied once

// The review canvas always plays the raw camera video. Hand skeletons are
// rendered by hand-overlay.js as an SVG layer on top of that video.
async function pickVideoUrl(episodeId, camera, preferSkeleton = false) {
    return `/api/v1/video/${episodeId}/${camera}/preview-stream`;
}

function getMediaLoadSignal() {
    return _episodeMediaLoadController ? _episodeMediaLoadController.signal : null;
}

function getPlaybackSessionToken() {
    return _playbackSessionToken;
}

function isCurrentPlaybackSession(episodeId, token) {
    return currentEpisodeId === episodeId &&
        (token == null || token === _playbackSessionToken);
}

function _destroyEpisodePlayers() {
    Object.values(players).forEach(player => {
        try { player.pause(); } catch (_) {}
        try { player.destroy(); } catch (_) {}
    });
    Object.keys(players).forEach(key => delete players[key]);
}

function _cancelEpisodeMediaLoad(clearCanvas = true) {
    // Invalidate callbacks before aborting requests.  Abort is not guaranteed
    // to prevent a response already resolved from reaching its .then().
    _playbackSessionToken += 1;
    if (_episodeMediaLoadController) {
        try { _episodeMediaLoadController.abort(); } catch (_) {}
        _episodeMediaLoadController = null;
    }
    clearTimeout(_loadWatchdog);
    masterPlaying = false;
    try { pauseAll(); } catch (_) {}
    if (typeof stopHeatmapSync === 'function') stopHeatmapSync();
    if (typeof cancelSegmentPlayback === 'function') cancelSegmentPlayback();
    if (typeof disconnectAnnotationSocket === 'function') disconnectAnnotationSocket();
    _destroyEpisodePlayers();
    currentImageTiles = [];
    currentDepthPreviewTiles = [];
    currentHand3dTiles = [];
    workspaceGrid = null;
    workspaceMainRow = null;
    workspaceHandFooter = null;
    workspaceHandSlots = { left: null, right: null };
    workspaceBottomExtra = null;
    if (typeof clearHandTiles === 'function') clearHandTiles();
    if (typeof destroyAllHandOverlays === 'function') destroyAllHandOverlays();
    if (clearCanvas) {
        const grid = document.getElementById('video-grid');
        if (grid) {
            grid.innerHTML = '<div class="flex items-center justify-center h-64 text-gray-600">' +
                '<div><div class="text-4xl mb-3 text-center"><iconify-icon icon="ant-design:video-camera-outlined"></iconify-icon></div>' +
                '<div>Select an episode to review →</div></div></div>';
        }
    }
}

function clearEpisodeWorkspace() {
    _cancelEpisodeMediaLoad(true);
    currentEpisodeId = null;
    currentCameras = [];
    currentMediaGroups = null;
    workspaceSources = [];
    workspaceCameras = [];
    workspaceGrid = null;
    workspaceMainRow = null;
    workspaceHandFooter = null;
    workspaceHandSlots = { left: null, right: null };
    workspaceBottomExtra = null;
    if (typeof clearAnnotationsNow === 'function') clearAnnotationsNow();
    hand3dData = null;
    hand3dDataBySource = {};
    hand3dFrameCache = { frame: -1, data: null, inflight: -1 };
    hand3dFrameCacheBySource = {};
    hand3dWindow = { start: -1, end: -1, frames: {}, inflight: false };
    hand3dWindowBySource = {};
    episodeTotalFrames = 0;
    episodeFps = 0;
    frameDataReady = false;
    const frameLabel = document.getElementById('annotation-timeline-frame');
    if (frameLabel) frameLabel.textContent = '0 / 0';
    disableFrameControls();
    hideVideoLoading();
    resetSourceBar();
}

function initPlayer(containerId, videoUrl) {
    const container = document.getElementById(containerId);
    if (!container) return null;

    // Remove existing video element
    container.innerHTML = `
        <video id="${containerId}-video" controls crossorigin playsinline preload="auto"></video>
    `;

    const player = new Plyr(`#${containerId}-video`, {
        // Frame-aligned mode: individual video controls are hidden. A single
        // master play/pause button + seek bar above the canvas drives every
        // video at exactly the same frame (see bindMasterSync).
        controls: [],
        keyboard: { focused: true, global: true },
    });

    player.source = {
        type: 'video',
        sources: [{ src: videoUrl, type: 'video/mp4' }]
    };

    // Notify when player is ready (metadata loaded, seekable)
    player.on('ready', () => {
        if (typeof onPlayerReady === 'function') onPlayerReady();
    });

    // 禁用双击全屏 — Plyr 3.7.8 无 doubleClick 配置项，
    // 其 dblclick 绑在 .plyr 容器 bubble 阶段，无法移除。
    // 方案：capture 阶段设标记 → 覆写 fullscreen.toggle 拦截。
    let dblClicked = false;
    player.elements.container.addEventListener('dblclick', () => {
        dblClicked = true;
    }, true); // capture 阶段，先于 Plyr 的 bubble handler 触发
    const _origToggle = player.fullscreen.toggle.bind(player.fullscreen);
    player.fullscreen.toggle = function () {
        if (dblClicked) {
            dblClicked = false;
            return;
        }
        return _origToggle();
    };

    return player;
}

/* ── 媒体组(双目兼容)────────────────────────────
   素材清单由后端 /media-groups API 驱动:双目组(stereo_left+stereo_right)、
   辅助流、单目、骨骼视频、手套传感器。前端按 kind 渲染,新增素材类型
   无需改动前端代码。主要交互是"拖拽素材到画布"自由组合;点击素材项
   或顶部标签只是便捷方式。                                  */

let currentMediaGroups = null;   // /media-groups 返回的 {groups, singles, sources}
let currentImageTiles = [];      // 画布中的图像素材 tile {img, source}(手套热力图/深度图)
let workspaceGrid = null;
let workspaceMainRow = null;
let workspaceHandFooter = null;
let workspaceHandSlots = { left: null, right: null };
let workspaceBottomExtra = null;
const _gloveSyncBound = new WeakSet();
let _suppressMasterEvents = false;
let hand3dData = null;          // /hand-3d 接口返回 {frames:[{fi,h0,h1}], baseline_m, ...}
let hand3dDataBySource = {};     // source_key -> per-device world metadata
let hand3dFrameCache = { frame: -1, data: null, inflight: -1 };  // 按帧懒加载缓存
let hand3dFrameCacheBySource = {};
let _hand3dTileEntry = null;    // 3D tile 引用,懒加载回调重绘用
let currentHand3dTiles = [];    // 画布中的 3D 骨骼 tile {canvas, source, rotX, rotY, ...}
let currentDepthPreviewTiles = []; // raw uint16 depth codes + frontend Canvas
const _depthLittleEndian = new Uint8Array(new Uint16Array([1]).buffer)[0] === 1;
// Keep one recently-used complete raw-code buffer across episode clicks. The
// decoded D435 stream is large, so bound this cache to one typical episode;
// the cache contains only uint16 code values, never a JET/RGB image.
const _depthFullCache = new Map();
const DEPTH_FULL_CACHE_MAX_BYTES = 420 * 1024 * 1024;

function _depthFullCacheKey(source) {
    return `${currentEpisodeId || ''}:${source?.source_key || ''}`;
}

function _rememberDepthFull(entry) {
    if (!entry?.fullCodes || !entry?.source) return;
    const key = _depthFullCacheKey(entry.source);
    const bytes = entry.fullCodes.byteLength || 0;
    _depthFullCache.delete(key);
    _depthFullCache.set(key, {
        codes: entry.fullCodes,
        width: entry.width,
        height: entry.height,
        pixelCount: entry.pixelCount,
        frameBytes: entry.frameBytes,
        frameCount: entry.frameCount,
        bytes,
    });
    let total = 0;
    for (const [cacheKey, value] of [..._depthFullCache.entries()].reverse()) {
        total += value.bytes || 0;
        if (total > DEPTH_FULL_CACHE_MAX_BYTES) _depthFullCache.delete(cacheKey);
    }
}

function _restoreDepthFull(entry) {
    if (!entry?.source) return false;
    const value = _depthFullCache.get(_depthFullCacheKey(entry.source));
    if (!value?.codes) return false;
    _depthFullCache.delete(_depthFullCacheKey(entry.source));
    _depthFullCache.set(_depthFullCacheKey(entry.source), value);
    Object.assign(entry, value, { fullReady: true, initialReady: true });
    entry.frames.clear();
    return true;
}

function _decodeDepthCodeBuffer(buffer) {
    if (_depthLittleEndian) return new Uint16Array(buffer);
    const view = new DataView(buffer);
    const codes = new Uint16Array(buffer.byteLength / 2);
    for (let i = 0; i < codes.length; i++) {
        codes[i] = view.getUint16(i * 2, true);
    }
    return codes;
}

function _depthCodesUrl(source, frame) {
    const template = source && (source.depth_codes_url || source.depth_preview_url);
    return template ? _frameUrl(template, frame) : '';
}

// A depth request contains raw uint16 codes, not a colorized image.  Larger
// sequential windows remove the per-request decoder/HTTP overhead that made
// the depth canvas fall behind the RGB master.
// Four seconds at the canonical 30 FPS. This is the bounded fallback for
// long episodes; unlike the old 60-frame window it leaves enough time for a
// remote FFmpeg read to finish before the playhead reaches the boundary.
const DEPTH_WINDOW_SIZE = 120;
const DEPTH_PREFETCH_MARGIN = 20;
const DEPTH_INITIAL_BUFFER_FRAMES = 120;
// At 848x480 uint16, 600 frames are about 465 MiB. Below this boundary it is
// safer to keep the complete decoded window set resident and guarantee zero
// mid-play disk/cache reads; above it use the bounded three-window runway.
const DEPTH_VERY_LONG_FRAMES = 600;
const DEPTH_VERY_LONG_INITIAL_WINDOWS = 3;
// Depth is display-only raw code data. Normal clips wait for all windows;
// very long clips wait for three windows (~12s at 30 FPS), then continue in
// the background without allowing the playhead to cross an empty window.
const DEPTH_FULL_PRELOAD = true;
// A 848x480 stream costs about 0.78 MiB per raw frame. Keeping a 1180-frame
// episode as one Uint16Array would consume ~920 MiB in the browser and can
// trigger GC pauses or tab eviction. Short clips still get the one-shot cache;
// long clips use the sequential window path below.
const DEPTH_FULL_PRELOAD_MAX_FRAMES = 240;

let _depthPlaybackStall = null;

function _depthFrameReady(frame) {
    return currentDepthPreviewTiles.length > 0 &&
        currentDepthPreviewTiles.every(entry =>
            (entry.fullReady && frame < entry.frameCount) ||
            (!entry.fullReady && entry.fullCodes && frame < entry.frameCount) ||
            entry.frames.has(frame));
}

function _stallPlaybackForDepth(frame) {
    if (_depthPlaybackStall || !masterPlaying) return;
    _depthPlaybackStall = { episodeId: currentEpisodeId, frame };
    masterPlaying = false;
    _suppressMasterEvents = true;
    pauseAll();
    _suppressMasterEvents = false;
    refreshPlayButton(false);
}

function _maybeResumeAfterDepth() {
    const stall = _depthPlaybackStall;
    if (!stall || stall.episodeId !== currentEpisodeId ||
        !_depthFrameReady(stall.frame)) return;
    _depthPlaybackStall = null;
    currentFrameTarget = stall.frame;
    seekToFrame(stall.frame);
    masterPlaying = true;
    playAll();
}

function _depthCodesWindowUrl(source, start, end) {
    let template = source && source.depth_codes_window_url;
    // A media-groups response can come from the browser cache and predate
    // the window URL field. Derive the endpoint from the canonical source
    // identity so an old response cannot regress playback to one request per
    // frame. This still transfers raw display codes only; JET stays in the
    // browser.
    if (!template && source && source.source_key && currentEpisodeId) {
        template = `/api/v1/video/${encodeURIComponent(currentEpisodeId)}` +
            `/depth-codes-window/${encodeURIComponent(source.source_key)}` +
            '?start_frame={start}&end_frame={end}';
    }
    if (template) {
        const url = template.replace('{start}', Math.max(0, start))
            .replace('{end}', Math.max(start, end));
        const revision = source && source.depth_cache_key;
        return revision
            ? `${url}${url.includes('?') ? '&' : '?'}v=${encodeURIComponent(revision)}`
            : url;
    }
    return '';
}

function _depthCodesFullUrl(source) {
    let template = source && source.depth_codes_full_url;
    if (!template && source && source.source_key && currentEpisodeId) {
        template = `/api/v1/video/${encodeURIComponent(currentEpisodeId)}` +
            `/depth-codes-full/${encodeURIComponent(source.source_key)}`;
    }
    if (!template) return '';
    const revision = source && source.depth_cache_key;
    return revision
        ? `${template}${template.includes('?') ? '&' : '?'}v=${encodeURIComponent(revision)}`
        : template;
}

function _fullDepthCodesForFrame(entry, frame) {
    if (!entry || !entry.fullCodes ||
        frame < 0 || frame >= entry.frameCount) return null;
    const begin = frame * entry.pixelCount;
    return entry.fullCodes.subarray(begin, begin + entry.pixelCount);
}

async function _preloadDepthCodes(entry) {
    if (!entry || !entry.source || entry.fullReady) return true;
    if (entry.fullInflight) return entry.fullInflight;
    if (_restoreDepthFull(entry)) {
        const codes = _fullDepthCodesForFrame(entry, 0);
        if (codes) DepthRenderer.render(entry.canvas, codes, entry.width, entry.height);
        return true;
    }
    const episodeAtRequest = currentEpisodeId;
    entry.initialReady = false;
    entry.fullInflight = (async () => {
        try {
            const response = await fetch(_depthCodesFullUrl(entry.source), {
                // The URL includes the source file revision. force-cache
                // therefore gives repeat clicks a disk-cache hit, while a
                // reprocessed MP4 receives a new URL and cannot show stale
                // depth codes.
                cache: 'force-cache', signal: getMediaLoadSignal() || undefined,
            });
            if (!response.ok) {
                throw new Error(`full depth code request failed: ${response.status}`);
            }
            const width = Number(response.headers.get('X-Depth-Width')) || 0;
            const height = Number(response.headers.get('X-Depth-Height')) || 0;
            const frameBytes = Number(response.headers.get('X-Depth-Frame-Bytes')) ||
                width * height * 2;
            const declaredFrames = Number(response.headers.get('X-Depth-Frames')) ||
                Number(entry.source.frame_count) || 0;
            if (!width || !height || !frameBytes) {
                throw new Error('invalid depth stream metadata');
            }

            // Allocate one contiguous raw-code buffer.  It is intentionally
            // not a canvas/image buffer: no pseudo-colour data is created or
            // persisted.
            let capacity = Math.max(1, declaredFrames) * frameBytes;
            let raw = new Uint8Array(capacity);
            let offset = 0;
            let publishedBuffer = null;
            let publishedFrames = 0;

            const publishBufferedFrames = () => {
                const availableBytes = Math.floor(offset / frameBytes) * frameBytes;
                const availableFrames = Math.floor(availableBytes / frameBytes);
                if (availableFrames <= publishedFrames) return availableFrames;
                if (!publishedBuffer || publishedBuffer.buffer !== raw.buffer) {
                    publishedBuffer = _decodeDepthCodeBuffer(raw.buffer);
                }
                entry.width = width;
                entry.height = height;
                entry.pixelCount = width * height;
                entry.frameBytes = frameBytes;
                entry.frameCount = availableFrames;
                entry.fullCodes = publishedBuffer;
                publishedFrames = availableFrames;
                // Paint only the first available frame during the sequential
                // preload. Rendering once per network chunk can mean hundreds
                // of 848x480 texture uploads before playback even starts and
                // is a major source of the apparent "stuck" loading state.
                // The completed stream is painted at the current frame below.
                const active = getActivePlayer();
                const displayTarget = active && Number.isFinite(active.currentTime)
                    ? Math.floor(active.currentTime * getEpisodeFps() + 0.002)
                    : (typeof currentFrameTarget === 'number' ? currentFrameTarget : 0);
                const firstAvailable = Math.min(
                    Math.max(0, displayTarget), availableFrames - 1);
                const displayCodes = publishedBuffer.subarray(
                    firstAvailable * entry.pixelCount,
                    (firstAvailable + 1) * entry.pixelCount);
                if (!entry.initialPainted && displayCodes.length === entry.pixelCount) {
                    entry.lastRenderedFrame = firstAvailable;
                    DepthRenderer.render(entry.canvas, displayCodes, width, height);
                    entry.initialPainted = true;
                }
                if (!entry.initialReady && availableFrames >=
                    Math.min(DEPTH_INITIAL_BUFFER_FRAMES, declaredFrames || availableFrames)) {
                    entry.initialReady = true;
                }
                _maybeResumeAfterDepth();
                return availableFrames;
            };

            const consume = (value) => {
                if (!value || !value.byteLength) return;
                if (offset + value.byteLength > raw.byteLength) {
                    const next = Math.max(offset + value.byteLength,
                        Math.ceil(raw.byteLength * 1.25));
                    const grown = new Uint8Array(next);
                    grown.set(raw.subarray(0, offset));
                    raw = grown;
                    publishedBuffer = null;
                }
                raw.set(value, offset);
                offset += value.byteLength;
                publishBufferedFrames();
            };

            const reader = response.body && response.body.getReader
                ? response.body.getReader() : null;
            if (!reader) {
                consume(new Uint8Array(await response.arrayBuffer()));
            } else {
                let chunks = 0;
                while (true) {
                    const part = await reader.read();
                    if (part.done) break;
                    consume(part.value || new Uint8Array());
                    // Keep RGB controls/layout responsive during a large
                    // background fill without rAF throttling in hidden tabs.
                    if ((++chunks % 64) === 0) {
                        await new Promise(resolve => setTimeout(resolve, 0));
                    }
                }
            }
            if (currentEpisodeId !== episodeAtRequest) return false;
            const usableBytes = Math.floor(offset / frameBytes) * frameBytes;
            if (usableBytes < frameBytes) throw new Error('empty depth code stream');
            const exact = raw.byteLength === usableBytes
                ? raw.buffer : raw.slice(0, usableBytes).buffer;
            entry.fullCodes = _decodeDepthCodeBuffer(exact);
            entry.frameCount = Math.floor(usableBytes / frameBytes);
            entry.fullReady = true;
            entry.initialReady = true;
            entry.frames.clear();
            _rememberDepthFull(entry);
            const active = getActivePlayer();
            const displayTarget = active && Number.isFinite(active.currentTime)
                ? Math.floor(active.currentTime * getEpisodeFps() + 0.002)
                : (typeof currentFrameTarget === 'number' ? currentFrameTarget : 0);
            const frameCodes = _fullDepthCodesForFrame(entry, displayTarget);
            if (frameCodes) DepthRenderer.render(entry.canvas, frameCodes, width, height);
            _maybeResumeAfterDepth();
            return true;
        } catch (error) {
            entry.fullError = error;
            console.warn('[player] background depth preload failed:', error);
            return false;
        }
    })().finally(() => {
        entry.fullInflight = null;
    });
    // Playback must not start while the full buffer is still arriving: doing
    // so makes the central clock repeatedly pause/resume at the buffer edge.
    // A subsequent click can return immediately through _restoreDepthFull().
    return entry.fullInflight;
}

// Normal/medium clips are completely fetched before playback. They are
// fetched as 120-frame responses so the browser can persist them in its HTTP
// disk cache without creating one huge 900MB Uint16Array. Only a few decoded
// windows remain in RAM; later playback requests are local cache hits.
async function _preloadDepthWindows(entry) {
    if (!entry || !entry.source) return false;
    if (entry.allPreloaded) return true;
    if (entry.allPreloadInflight) return entry.allPreloadInflight;
    const sourceFrameCount = Number(entry.source.frame_count) || 0;
    if (!sourceFrameCount) return false;
    const episodeAtRequest = currentEpisodeId;
    entry.allPreloadInflight = (async () => {
        try {
            for (let start = 0; start < sourceFrameCount; start += DEPTH_WINDOW_SIZE) {
                if (currentEpisodeId !== episodeAtRequest) return false;
                entry.preloadProgress = Math.min(1, start / sourceFrameCount);
                await _fetchDepthCodes(entry, start);
                if (!entry.frames.has(start)) {
                    throw new Error(`depth preload window missing: ${start}`);
                }
            }
            if (currentEpisodeId !== episodeAtRequest) return false;
            // The sequential pass evicts old decoded windows from RAM. Put
            // frame zero back from the browser cache before releasing the
            // playback barrier.
            await _fetchDepthCodes(entry, 0);
            entry.allPreloaded = true;
            entry.initialReady = true;
            entry.preloadProgress = 1;
            return true;
        } catch (error) {
            entry.fullError = error;
            console.warn('[player] complete depth window preload failed:', error);
            return false;
        }
    })().finally(() => {
        entry.allPreloadInflight = null;
    });
    return entry.allPreloadInflight;
}

async function _preloadDepthInitialWindows(entry, windowCount = DEPTH_VERY_LONG_INITIAL_WINDOWS) {
    if (!entry || !entry.source) return false;
    if (entry.initialBufferReady) return true;
    if (entry.initialPreloadInflight) return entry.initialPreloadInflight;
    const sourceFrameCount = Number(entry.source.frame_count) || 0;
    if (!sourceFrameCount) return false;
    const episodeAtRequest = currentEpisodeId;
    entry.initialPreloadInflight = (async () => {
        try {
            const limit = Math.min(sourceFrameCount,
                Math.max(1, windowCount) * DEPTH_WINDOW_SIZE);
            for (let start = 0; start < limit; start += DEPTH_WINDOW_SIZE) {
                if (currentEpisodeId !== episodeAtRequest) return false;
                await _fetchDepthCodes(entry, start);
            }
            entry.initialBufferReady = entry.frames.has(0);
            return entry.initialBufferReady;
        } catch (error) {
            console.warn('[player] initial depth buffer failed:', error);
            return false;
        } finally {
            entry.initialPreloadInflight = null;
        }
    })();
    return entry.initialPreloadInflight;
}

async function _fetchDepthCodes(entry, frame) {
    if (!entry || !entry.source || !entry.canvas || frame < 0) return;
    if (entry.fullReady) {
        const codes = _fullDepthCodesForFrame(entry, frame);
        if (codes) {
            if (entry.lastRenderedFrame === frame) return;
            entry.lastRenderedFrame = frame;
            DepthRenderer.render(entry.canvas, codes, entry.width, entry.height);
        }
        return;
    }
    if (entry.fullInflight) return;
    if (entry.frames.has(frame)) {
        const cached = entry.frames.get(frame);
        if (entry.lastRenderedFrame !== frame) {
            entry.lastRenderedFrame = frame;
            DepthRenderer.render(entry.canvas, cached.codes, cached.width, cached.height);
        }
        return;
    }
    if (entry.windowInflight && frame >= entry.windowInflight.start
            && frame <= entry.windowInflight.end) return;
    entry.pendingFrame = frame;
    if (entry.windowInflight != null) return;
    const target = entry.pendingFrame;
    entry.pendingFrame = null;
    const start = Math.floor(target / DEPTH_WINDOW_SIZE) * DEPTH_WINDOW_SIZE;
    // Once the source metadata is known, never ask the backend for a window
    // beyond the actual clip. Besides avoiding noisy 404s, this prevents a
    // failed tail request from looking like a playback stall on short clips.
    const sourceFrameCount = Number(entry.source.frame_count) || 0;
    if (sourceFrameCount && start >= sourceFrameCount) return;
    const end = sourceFrameCount
        ? Math.min(start + DEPTH_WINDOW_SIZE - 1, sourceFrameCount - 1)
        : start + DEPTH_WINDOW_SIZE - 1;
    entry.windowInflight = { start, end };
    const episodeAtRequest = currentEpisodeId;
    try {
        const windowUrl = _depthCodesWindowUrl(entry.source, start, end);
        const response = await fetch(windowUrl || _depthCodesUrl(entry.source, target), {
            cache: 'force-cache', signal: getMediaLoadSignal() || undefined,
        });
        if (!response.ok) throw new Error(`depth code request failed: ${response.status}`);
        const width = Number(response.headers.get('X-Depth-Width')) || 0;
        const height = Number(response.headers.get('X-Depth-Height')) || 0;
        const buffer = await response.arrayBuffer();
        if (currentEpisodeId !== episodeAtRequest || buffer.byteLength % 2) return;
        // The API declares uint16 little-endian. Common browsers can use a
        // zero-copy view; the fallback keeps the wire contract strict on an
        // unusual big-endian host.
        const codes = _decodeDepthCodeBuffer(buffer);
        const expected = width * height;
        const count = Number(response.headers.get('X-Depth-Frames')) || 1;
        const first = Number(response.headers.get('X-Depth-Start')) || start;
        if (!width || !height || codes.length < expected) return;
        for (let offset = 0; offset < count; offset++) {
            const begin = offset * expected;
            const frameCodes = codes.subarray(begin, begin + expected);
            if (frameCodes.length !== expected) break;
            entry.frames.set(first + offset, { codes: frameCodes, width, height });
        }
        if (entry.keepAllFrames) {
            entry.initialReady = true;
        } else {
            // Keep the current window plus several future windows. A FIFO
            // eviction could remove frame 0 while the long-clip prefetcher
            // is still filling frame 360, causing an unnecessary pause.
            const activeFrame = Math.max(0, Number(currentFrameTarget) || 0);
            const minFrame = Math.max(0, activeFrame - DEPTH_WINDOW_SIZE);
            const maxFrame = activeFrame + DEPTH_WINDOW_SIZE * 5 - 1;
            for (const frameKey of entry.frames.keys()) {
                if (frameKey < minFrame || frameKey > maxFrame) {
                    entry.frames.delete(frameKey);
                }
            }
        }
        if (!entry.keepAllFrames && expected && entry.source.frame_count) {
            const residentBytes = Number(entry.source.frame_count) * expected * 2;
            entry.keepAllFrames = residentBytes <= 420 * 1024 * 1024;
        }
        // Playback may have advanced while the request was in flight. Render
        // the actual master frame if it is in this window; a prefetch must
        // never jump the depth canvas ahead to the first prefetched frame.
        const active = getActivePlayer();
        const displayTarget = active && !active.paused &&
                Number.isFinite(active.currentTime)
            ? Math.floor(active.currentTime * getEpisodeFps() + 0.002)
            : (typeof currentFrameTarget === 'number' ? currentFrameTarget : target);
        const cached = entry.frames.get(displayTarget) ||
            (displayTarget === target ? entry.frames.get(target) : null);
        if (cached) {
            if (entry.lastRenderedFrame === displayTarget) return;
            entry.lastRenderedFrame = displayTarget;
            DepthRenderer.render(entry.canvas, cached.codes, cached.width, cached.height);
        }
        _maybeResumeAfterDepth();
    } catch (_) {
        // A stale request or a temporary seek race must not break playback.
    } finally {
        entry.windowInflight = null;
        if (entry.pendingFrame != null && currentEpisodeId === episodeAtRequest) {
            _fetchDepthCodes(entry, entry.pendingFrame);
        }
    }
}

function _queueDepthWindow(entry, frame) {
    if (!entry || entry.fullReady || entry.keepAllFrames || frame < 0) return;
    const start = Math.floor(frame / DEPTH_WINDOW_SIZE) * DEPTH_WINDOW_SIZE;
    const count = Number(entry.source?.frame_count) || 0;
    if (count && start >= count) return;
    if (entry.frames.has(start) ||
        (entry.windowInflight && entry.windowInflight.start === start) ||
        entry.windowQueue.includes(start)) return;
    entry.windowQueue.push(start);
    _pumpDepthWindowQueue(entry);
}

async function _pumpDepthWindowQueue(entry) {
    if (!entry || entry.windowPump) return;
    entry.windowPump = true;
    try {
        while (entry.windowQueue.length && currentEpisodeId) {
            if (entry.windowInflight) {
                await new Promise(resolve => setTimeout(resolve, 20));
                continue;
            }
            const start = entry.windowQueue.shift();
            await _fetchDepthCodes(entry, start);
        }
    } finally {
        entry.windowPump = false;
    }
}

function _refreshDepthTilesAt(frame) {
    if (!Number.isFinite(frame) || frame < 0) return;
    currentDepthPreviewTiles.forEach(entry => {
        _fetchDepthCodes(entry, frame);
        if (entry.fullReady) return;
        // Fetch the next sequential window before the current one is empty.
        // The server-side reader stays sequential, so this is substantially
        // cheaper than waiting for a random frame request at the boundary.
        const windowStart = Math.floor(frame / DEPTH_WINDOW_SIZE) * DEPTH_WINDOW_SIZE;
        const prefetchFrame = windowStart + DEPTH_WINDOW_SIZE;
        const sourceFrameCount = Number(entry.source.frame_count) || 0;
        if (sourceFrameCount > DEPTH_VERY_LONG_FRAMES) {
            // Maintain at least three future windows in the queue. They are
            // fetched one at a time so the server decoder remains sequential.
            for (let i = 1; i <= 3; i++) {
                const queuedFrame = windowStart + i * DEPTH_WINDOW_SIZE;
                if (!sourceFrameCount || queuedFrame < sourceFrameCount) {
                    _queueDepthWindow(entry, queuedFrame);
                }
            }
        } else if (frame >= windowStart + DEPTH_PREFETCH_MARGIN * 2 &&
            (!sourceFrameCount || prefetchFrame < sourceFrameCount) &&
            !entry.frames.has(prefetchFrame) && !entry.windowInflight) {
            _fetchDepthCodes(entry, prefetchFrame);
        }
    });
}
// 3D 世界坐标窗口缓存:窗口拉取(±HAND3D_WINDOW 帧),播放期零逐帧请求
let hand3dWindow = { start: -1, end: -1, frames: {}, inflight: false };
let hand3dWindowBySource = {};
const HAND3D_WINDOW = 250;
const HAND3D_INITIAL_BUFFER_FRAMES = 450;
const HAND3D_MAX_CACHED_FRAMES = 2500;
// 3D 世界窗口在接近边界前提前请求下一段。严格同步模式下不使用
// 上一帧补位，因此必须在真正越过边界前完成预取。
const HAND3D_PREFETCH_MARGIN = 60;
// Review episodes are normally short (hundreds/thousands of frames).  Load
// those point sequences once before playback so a window boundary can never
// leave the spatial canvas one or more frames behind RGB.  Very long episodes
// keep the bounded-window fallback to avoid turning a browser tab into a
// multi-hundred-megabyte JSON allocation.
const HAND3D_FULL_PRELOAD = true;
const HAND3D_FULL_PRELOAD_MAX_FRAMES = 5000;
const HAND3D_FULL_CACHE_MAX_EPISODES = 2;
const _hand3dFullCache = new Map(); // episode/source -> {frames, count}

// A workflow rerun replaces the canonical parquet.  Drop only that episode's
// browser-side point/depth buffers so a later click cannot show the old
// derived result; unrelated episode caches remain warm.
window.invalidateEpisodePlaybackCache = function (episodeId) {
    const prefix = `${episodeId || ''}:`;
    for (const key of _depthFullCache.keys()) {
        if (key.startsWith(prefix)) _depthFullCache.delete(key);
    }
    for (const key of _hand3dFullCache.keys()) {
        if (key.startsWith(prefix)) _hand3dFullCache.delete(key);
    }
    if (window.EgoMediaCache && episodeId) {
        window.EgoMediaCache.removeEpisode(episodeId).catch(() => {});
    }
};
// 世界坐标使用稳定的空间锚点和安全视距。世界坐标中的点不能根据
// 当前帧的单手/双手数量重新居中或缩放,否则检测状态变化会被误显示
// 成相机跳动。2m 同时为固定网格保留足够的可视边界,用户仍可滚轮缩放。
const HAND3D_WORLD_DEFAULT_CENTER = [0, 0, 0.7];
const HAND3D_WORLD_DEFAULT_DISTANCE = 2.0;
// World-preview sizing is screen-space only. The target is intentionally
// shared by depth and RGB-estimated 3D so different coordinate sources have
// the same visual scale without changing parquet/API coordinates.
const HAND3D_DISPLAY_SPAN_RATIO = 0.18;
const HAND3D_DISPLAY_SPAN_MIN_PX = 70;
const HAND3D_DISPLAY_SPAN_MAX_PX = 130;
const HAND3D_DISPLAY_SCALE_MIN = 0.70;
const HAND3D_DISPLAY_SCALE_MAX = 3.0;
// RGB/深度 3D 坐标统一为 X 右、Y 上、Z 向前；网格放在手的下方。
const HAND3D_WORLD_DEFAULT_GRID_Y = -0.25;
// Root-anchored preview: MediaPipe landmark 0 (Wrist) is placed on the
// ground grid at two stable left/right display anchors. The stored camera
// coordinates remain untouched; this is only a presentation transform.
// Fixed preview positions. Keep enough horizontal clearance so the two
// skeletons cannot visually merge on the shared 3D canvas.
const HAND3D_ROOT_ANCHOR_X = 0.22;
const HAND3D_ROOT_ANCHOR_LIFT = 0.01;
// 默认正面视角与左侧 RGB 画面一致；拖拽仍可改变俯仰角查看空间深度。
const HAND3D_DEFAULT_ELEVATION = 0;
const HAND3D_MAX_CAMERA_DISTANCE = 6.0;
// RGB-only 3D is relative/estimated rather than metric. Its current image
// model is smaller than a physical hand, so normalize only the preview
// geometry to a stable hand span comparable with the depth view.
const RGB_ESTIMATED_DISPLAY_HAND_SPAN = 0.14;
const RGB_ESTIMATED_DISPLAY_SCALE_MIN = 1.0;
const RGB_ESTIMATED_DISPLAY_SCALE_MAX = 5.0;

function _hand3dCacheFor(sourceKey) {
    const source = sourceKey || 'default';
    if (!hand3dWindowBySource[source]) {
        hand3dWindowBySource[source] = {
            start: -1, end: -1, frames: {}, inflight: false,
            inflightPromise: null,
        };
    }
    return hand3dWindowBySource[source];
}

function _hand3dFrameCacheFor(sourceKey) {
    const source = sourceKey || 'default';
    if (!hand3dFrameCacheBySource[source]) {
        hand3dFrameCacheBySource[source] = { frame: -1, data: null, inflight: -1 };
    }
    return hand3dFrameCacheBySource[source];
}

function _hand3dCached(frame, sourceKey) {
    const cache = _hand3dCacheFor(sourceKey);
    const fr = cache.frames[frame];
    return fr ? { h0: fr.h0 || null, h1: fr.h1 || null } : null;
}

function _hand3dFullCacheKey(sourceKey) {
    return `${currentEpisodeId || ''}:${sourceKey || 'default'}`;
}

function _restoreHand3DFull(cache, sourceKey) {
    const saved = _hand3dFullCache.get(_hand3dFullCacheKey(sourceKey));
    if (!saved || !saved.frames) return false;
    cache.frames = saved.frames;
    cache.start = 0;
    cache.end = Math.max(0, Number(saved.count || 0) - 1);
    cache.fullReady = true;
    return true;
}

function _rememberHand3DFull(cache, sourceKey, count) {
    const key = _hand3dFullCacheKey(sourceKey);
    _hand3dFullCache.delete(key);
    _hand3dFullCache.set(key, { frames: cache.frames, count });
    while (_hand3dFullCache.size > HAND3D_FULL_CACHE_MAX_EPISODES) {
        _hand3dFullCache.delete(_hand3dFullCache.keys().next().value);
    }
}

async function fetchHand3DFull(sourceKey) {
    const source = sourceKey || 'default';
    const cache = _hand3dCacheFor(source);
    if (cache.fullReady) return true;
    if (_restoreHand3DFull(cache, source)) return true;
    if (cache.fullInflight) return cache.fullInflight;
    const data = hand3dDataBySource[source] || hand3dData;
    const count = Number(data && data.count || 0);
    if (!count || count > HAND3D_FULL_PRELOAD_MAX_FRAMES) return false;
    const episodeAtRequest = currentEpisodeId;
    const signal = getMediaLoadSignal();
    const persistentKey = `${episodeAtRequest}:hand3d:${source}:${count}`;
    cache.fullInflight = (async () => {
        try {
            if (window.EgoMediaCache) {
                const saved = await window.EgoMediaCache.get(persistentKey);
                const value = saved && saved.value;
                if (value && Array.isArray(value.frames)
                        && Number(value.count || 0) === count) {
                    const frames = {};
                    value.frames.forEach(fr => { frames[fr.f] = fr; });
                    cache.frames = frames;
                    cache.start = 0;
                    cache.end = Math.max(0, count - 1);
                    cache.fullReady = true;
                    _rememberHand3DFull(cache, source, count);
                    return true;
                }
            }
            const res = await fetch(
                `/api/v1/video/${episodeAtRequest}/hand-3d?source_key=`
                + `${encodeURIComponent(source)}&start_frame=0&end_frame=${count - 1}`,
                signal ? { signal } : {});
            if (!res.ok || currentEpisodeId !== episodeAtRequest) return false;
            const payload = await res.json();
            if (currentEpisodeId !== episodeAtRequest) return false;
            const frames = {};
            (payload.frames || []).forEach(fr => { frames[fr.f] = fr; });
            cache.frames = frames;
            cache.start = 0;
            cache.end = Math.max(0, count - 1);
            cache.fullReady = true;
            _rememberHand3DFull(cache, source, count);
            if (window.EgoMediaCache) {
                window.EgoMediaCache.put(persistentKey, {
                    frames: payload.frames || [], count,
                });
            }
            return true;
        } catch (_) {
            return false;
        } finally {
            cache.fullInflight = null;
        }
    })();
    return cache.fullInflight;
}

function _hand3dPlaybackFrame(fallback = 0) {
    const player = getActivePlayer();
    if (player && !player.paused && !player.ended
            && Number.isFinite(player.currentTime)) {
        return Math.max(0, Math.floor(
            player.currentTime * getEpisodeFps() + 0.002));
    }
    return typeof currentFrameTarget === 'number' ? currentFrameTarget : fallback;
}

function _mergeHand3DWindow(cache, frames, start, end) {
    const merged = cache.frames || {};
    (frames || []).forEach(fr => { merged[fr.f] = fr; });
    cache.frames = merged;
    cache.start = cache.start < 0 ? start : Math.min(cache.start, start);
    cache.end = Math.max(cache.end, end);
    const keys = Object.keys(merged).map(Number).sort((a, b) => a - b);
    while (keys.length > HAND3D_MAX_CACHED_FRAMES) {
        delete merged[keys.shift()];
    }
    const remaining = Object.keys(merged).map(Number);
    cache.start = remaining.length ? Math.min(...remaining) : -1;
    cache.end = remaining.length ? Math.max(...remaining) : -1;
}

async function fetchHand3DWindow(center, sourceKey) {
    const source = sourceKey || 'default';
    const cache = _hand3dCacheFor(source);
    if (cache.fullReady) return true;
    if (cache.inflight) return cache.inflightPromise || false;
    const insideWindow = center >= cache.start && center <= cache.end;
    const insidePrefetchRange = center >= cache.start + HAND3D_PREFETCH_MARGIN
        && center <= cache.end - HAND3D_PREFETCH_MARGIN;
    if (insideWindow && insidePrefetchRange) return;
    const start = Math.max(0, center - HAND3D_WINDOW);
    const end = center + HAND3D_WINDOW;
    const episodeAtRequest = currentEpisodeId;
    const sessionToken = _playbackSessionToken;
    const signal = getMediaLoadSignal();
    cache.inflight = true;
    cache.inflightPromise = (async () => {
      try {
        const persistentKey = `${episodeAtRequest}:hand3d:${source}:${start}:${end}`;
        if (window.EgoMediaCache) {
            const saved = await window.EgoMediaCache.get(persistentKey);
            if (!isCurrentPlaybackSession(episodeAtRequest, sessionToken)) return false;
            const value = saved && saved.value;
            if (value && Array.isArray(value.frames)) {
                _mergeHand3DWindow(cache, value.frames,
                    Number(value.start ?? start), Number(value.end ?? end));
                for (const entry of currentHand3dTiles) {
                    if ((entry.source.source_key || 'default') === source) {
                        renderHand3DTile(entry, _hand3dPlaybackFrame());
                    }
                }
                return true;
            }
        }
        const res = await fetch(
            `/api/v1/video/${episodeAtRequest}/hand-3d?source_key=${encodeURIComponent(source)}&start_frame=${start}&end_frame=${end}`,
            signal ? { signal } : {});
        if (!isCurrentPlaybackSession(episodeAtRequest, sessionToken)) return false;
        if (!res.ok) return false;
        const data = await res.json();
        if (!isCurrentPlaybackSession(episodeAtRequest, sessionToken)) return false;
        _mergeHand3DWindow(cache, data.frames, data.start, data.end);
        if (window.EgoMediaCache) {
            window.EgoMediaCache.put(persistentKey, {
                frames: data.frames || [], start: data.start, end: data.end,
            });
        }
        // 只在完整响应成功后替换窗口，避免半更新状态被渲染。
        for (const entry of currentHand3dTiles) {
            if ((entry.source.source_key || 'default') === source) {
                renderHand3DTile(entry, _hand3dPlaybackFrame());
            }
        }
        return true;
      } catch (e) { /* 拉取失败保持旧缓存 */
        return false;
      }
    })();
    try {
        return await cache.inflightPromise;
    } finally {
        cache.inflight = false;
        cache.inflightPromise = null;
    }
}

async function _ensureHand3DInitialWindow(sourceKey) {
    const source = sourceKey || 'default';
    const data = hand3dDataBySource[source] || hand3dData;
    if (!data || !data.hasPreview) return true;
    const cache = _hand3dCacheFor(source);
    if (HAND3D_FULL_PRELOAD) {
        const loaded = await fetchHand3DFull(source);
        if (loaded) return true;
    }
    if (cache.start <= 0 && cache.end >= 0) return true;
    const count = Number(data.count || 0);
    const initialCenter = count > HAND3D_INITIAL_BUFFER_FRAMES
        ? Math.min(Math.floor(HAND3D_INITIAL_BUFFER_FRAMES / 2), count - 1) : 0;
    const loaded = Boolean(await fetchHand3DWindow(initialCenter, source));
    if (loaded && count > HAND3D_INITIAL_BUFFER_FRAMES) {
        // Keep three future windows queued without blocking the first frame.
        // They merge into the bounded 3D cache and are usually served from
        // IndexedDB after the first visit.
        (async () => {
            let center = initialCenter + HAND3D_WINDOW * 2;
            for (let i = 0; i < 3 && center < count; i++, center += HAND3D_WINDOW * 2) {
                await fetchHand3DWindow(center, source);
            }
        })();
    }
    return loaded;
}
// The hand endpoint returns a display-only compact crop of its 800x220
// canvas. The raw sensor values are not changed by this presentation crop.
const HAND_PREVIEW_ASPECT = 800 / 200;
const HAND_FOOTER_HEIGHT = 144;
const HAND_FOOTER_WITH_EXTRA_HEIGHT = 300;

function groupedSourceKey(source) {
    if (source.kind === 'stereo_group') return `stereo:${source.id}`;
    return `${source.kind}:${source.source_key || source.camera || source.id}`;
}

function groupedSourceDomId(source) {
    return groupedSourceKey(source).replace(/[^a-zA-Z0-9_-]/g, '_');
}

function groupedSourceLabel(source) {
    if (source.kind === 'skeleton') return `${source.label || source.source_key} · ${t('skeleton_suffix')}`;
    if (source.kind === 'glove') return source.label || `${source.source_key} · ${t('glove_suffix')}`;
    if (source.kind === 'depth') return source.label || `${source.source_key} · ${t('depth_suffix')}`;
    if (source.kind === 'stereo_group') return `${source.label} · ${t('stereo_pair')}`;
    return source.label || source.source_key || t('video_suffix');
}

function _isBottomSource(source) {
    return ['hand', 'depth', 'glove'].includes(source.kind);
}

function groupedSource(kind, source_key, meta) {
    return { kind, source_key, ...(meta || {}) };
}

/* 素材 → 画布条目:双目组展开为左右目并排,其余一对一 */
function expandWorkspaceItems(source) {
    if (source.kind === 'stereo_group') {
        return (source.members || []).map(m => ({
            kind: 'video', source_key: m.source_key, label: m.label,
            stream_url: m.stream_url, frame_count: m.frame_count, fps: m.fps,
        }));
    }
    return [{ ...source }];
}

function _sourceIsPlaced(source) {
    const placed = new Set(workspaceSources.map(item => groupedSourceKey(item)));
    if (source.kind === 'stereo_group') {
        const members = source.members || [];
        return members.length > 0 && members.every(member =>
            placed.has(`video:${member.source_key}`));
    }
    return placed.has(groupedSourceKey(source));
}

function _syncWorkspacePlayers() {
    const active = getActivePlayer();
    if (!active) return;
    Object.values(players).forEach(player => bindMasterSync(player));
    bindGloveSync(active);
    bindFrameDrift(active);
    if (typeof startHeatmapSync === 'function') startHeatmapSync(active);
    if (masterPlaying) {
        Object.values(players).forEach(player => {
            try { player.play(); } catch (e) {}
        });
    }
}

function addGroupedSource(source) {
    const added = [];
    expandWorkspaceItems(source).forEach(item => {
        const key = groupedSourceKey(item);
        if (!workspaceSources.some(existing => groupedSourceKey(existing) === key)) {
            workspaceSources.push(item);
            added.push(item);
        }
    });
    renderSourceBar();
    renderPreviewVideoSources();
    if (!added.length) return;

    // When an episode is already mounted, append only the new source. This
    // keeps existing Plyr instances and playback position untouched.
    if (workspaceMainRow && workspaceGrid) {
        const addedDepth = added.some(item => item.kind === 'depth');
        const resumeAfterDepth = addedDepth && masterPlaying;
        if (addedDepth) {
            // Depth is a display-only stream and must be buffered before it
            // joins an already playing workspace; otherwise RGB advances
            // while the first depth window is still decoding.
            masterPlaying = false;
            _suppressMasterEvents = true;
            pauseAll();
            _suppressMasterEvents = false;
            disableFrameControls();
        }
        const pending = added.map(item => _appendWorkspaceSource(item));
        _refreshWorkspaceLayout();
        Promise.all(pending).then(() => {
            if (addedDepth && currentEpisodeId) {
                enableFrameControls();
                if (resumeAfterDepth) {
                    masterPlaying = true;
                    playAll();
                }
            } else {
                _syncWorkspacePlayers();
            }
        });
    } else {
        renderGroupedWorkspace();
    }
}

function removeGroupedSource(source) {
    const key = groupedSourceKey(source);
    if (typeof destroyHandOverlay === 'function') destroyHandOverlay(key);
    const tile = workspaceGrid
        ? workspaceGrid.querySelector(`[data-source-key="${CSS.escape(key)}"]`)
        : null;
    const image = tile ? tile.querySelector('img,canvas') : null;
    const player = players[key];
    const wasPlaying = masterPlaying;
    if (player) {
        // Plyr can emit pause while destroy() is tearing down the removed
        // player. Suppress that transient event so remaining players do not
        // get paused as a side effect of removing one tile.
        _suppressMasterEvents = true;
        try { player.destroy(); } catch (e) {} finally {
            _suppressMasterEvents = false;
        }
        delete players[key];
    }
    if (image && source.kind === 'hand' && typeof unregisterHandTile === 'function') {
        unregisterHandTile(image);
    }
    currentImageTiles = currentImageTiles.filter(entry =>
        groupedSourceKey(entry.source) !== key);
    currentDepthPreviewTiles = currentDepthPreviewTiles.filter(entry =>
        groupedSourceKey(entry.source) !== key);
    workspaceSources = workspaceSources.filter(item => groupedSourceKey(item) !== key);
    renderSourceBar();
    renderPreviewVideoSources();
    if (!workspaceGrid || !workspaceMainRow) {
        renderGroupedWorkspace();
        if (window.__episodeOpen === true) {
            const previewOptions = document.getElementById('preview-options');
            if (previewOptions) previewOptions.classList.remove('hidden');
        }
        return;
    }
    if (tile) tile.remove();
    _refreshWorkspaceLayout();
    const activeAfterRemove = getActivePlayer();
    if (!activeAfterRemove) {
        if (typeof stopHeatmapSync === 'function') stopHeatmapSync();
        masterPlaying = false;
        refreshPlayButton(false);
    } else {
        masterPlaying = wasPlaying;
        _syncWorkspacePlayers();
    }
    // 删除深度/3D/手套 tile 只影响工作区素材,不应把批次详情的
    // Preview Options 一并隐藏。
    if (window.__episodeOpen === true) {
        const previewOptions = document.getElementById('preview-options');
        if (previewOptions) previewOptions.classList.remove('hidden');
    }
}

/* ── Source button bar ───────────────────────────────────────
   Every asset (stereo group / single cameras / aux streams /
   glove heatmaps / depth / hand-pressure tiles) appears as a
   button above the video canvas. Click or drag to add it to the
   canvas; the button disappears once placed and returns when the
   tile is removed. No palette drawer, no tabs.                 */

function _sourceBarButton(label, icon, key, source, accent, parent) {
    const bar = parent || document.getElementById('source-bar');
    if (!bar || _sourceIsPlaced(source)) return;
    const btn = document.createElement('button');
    btn.type = 'button';
    btn.draggable = true;
    btn.dataset.sourceKey = key;
    btn.title = label;
    btn.className = 'text-xs px-2 py-1 rounded border cursor-grab flex-shrink-0 flex items-center gap-1 ' +
        (accent
            ? 'border-cyan-700/60 bg-gray-800 text-gray-300 hover:border-cyan-500 hover:text-white'
            : 'border-gray-700 bg-gray-800 text-gray-300 hover:border-blue-500 hover:text-white');
    btn.innerHTML = `<iconify-icon icon="${icon}" class="icon-sm"></iconify-icon> ${label}`;
    let dragging = false;
    btn.addEventListener('click', () => {
        if (!dragging) addGroupedSource(source);
    });
    btn.addEventListener('dragstart', event => {
        dragging = true;
        event.dataTransfer.setData('application/x-egodata-source', JSON.stringify(_serializeSource(source)));
        event.dataTransfer.effectAllowed = 'copy';
        btn.classList.add('opacity-50');
    });
    btn.addEventListener('dragend', () => {
        btn.classList.remove('opacity-50');
        setTimeout(() => { dragging = false; }, 0);
    });
    bar.appendChild(btn);
}

function _videoSource(item) {
    return {
        kind: 'video', source_key: item.source_key, label: item.label,
        stream_url: item.stream_url, frame_count: item.frame_count, fps: item.fps,
    };
}

function addSource(source, key) {
    addGroupedSource(source);
}

function addHandTile(hand) {
    addSource({
        kind: 'hand', hand, source_key: `hand_${hand}`,
        label: t(hand === 'left' ? 'left_hand' : 'right_hand'),
    }, `hand:hand_${hand}`);
}

/* Swap two video tiles in place — DOM-level reorder only, so the Plyr
   players are NOT rebuilt and playback continues uninterrupted. */
function reorderTiles(fromKey, toKey, container) {
    const fromIdx = workspaceSources.findIndex(s => groupedSourceKey(s) === fromKey);
    const toIdx = workspaceSources.findIndex(s => groupedSourceKey(s) === toKey);
    if (fromIdx < 0 || toIdx < 0 || fromIdx === toIdx) return;
    const [item] = workspaceSources.splice(fromIdx, 1);
    workspaceSources.splice(toIdx, 0, item);

    const fromTile = container.querySelector(`[data-source-key="${CSS.escape(fromKey)}"]`);
    const toTile = container.querySelector(`[data-source-key="${CSS.escape(toKey)}"]`);
    if (fromTile && toTile) {
        container.insertBefore(fromTile, fromIdx < toIdx ? toTile.nextSibling : toTile);
    }
}

function renderSourceBar() {
    const bar = document.getElementById('source-bar');
    if (!bar) return;
    bar.innerHTML = '';
    // All preview sources are now controlled by Preview Options. Keep this
    // function as a compatibility no-op because older code still calls it.
    bar.classList.add('hidden');
}

function _previewVideoEntries() {
    const entries = [];
    const add = item => {
        if (!item || !item.source_key || entries.some(e => e.source_key === item.source_key)) return;
        entries.push(_videoSource(item));
    };
    if (currentMediaGroups) {
        (currentMediaGroups.groups || []).forEach(group => {
            (group.members || []).forEach(add);
            (group.aux || []).forEach(add);
        });
        (currentMediaGroups.singles || []).forEach(add);
    }
    if (!entries.length) {
        currentCameras.forEach(camera => add({
            source_key: camera, label: camera,
            stream_url: `/api/v1/video/${currentEpisodeId}/${camera}/preview-stream`,
        }));
    }
    return entries;
}

function _previewDeviceLabel(source) {
    const raw = String(source.device_name || source.source_key || source.label || '').trim();
    // D405_depth_rgb / D435_depth_rgb are RGB stream slots; retain the
    // physical device token (D405_depth / D435_depth) shown to the reviewer.
    const device = raw.replace(/_rgb$/i, '').replace(/_color$/i, '');
    const role = source.label && source.label !== 'Primary' && source.label !== device
        ? ` · ${source.label}` : '';
    return `${device || raw}${role}`;
}

function renderPreviewVideoSources() {
    const row = document.getElementById('po-video-row');
    const list = document.getElementById('po-video-list');
    const all = document.getElementById('po-video-all');
    if (!row || !list) return;

    const entries = _previewVideoEntries();
    list.innerHTML = '';
    row.classList.toggle('hidden', entries.length === 0);
    row.classList.toggle('block', entries.length > 0);
    if (!entries.length) return;

    const isPlaced = source => workspaceSources.some(item =>
        groupedSourceKey(item) === groupedSourceKey(source));
    entries.forEach(source => {
        const label = document.createElement('label');
        label.className = 'flex items-center gap-2 px-1 py-1 rounded hover:bg-gray-700/70 cursor-pointer';
        const input = document.createElement('input');
        input.type = 'checkbox';
        input.className = 'w-3 h-3 accent-cyan-500';
        input.checked = isPlaced(source);
        input.dataset.sourceKey = source.source_key;
        const text = document.createElement('span');
        text.className = 'truncate text-[11px] text-gray-300';
        text.title = source.source_key;
        text.textContent = _previewDeviceLabel(source);
        input.addEventListener('change', () => {
            if (input.checked) addGroupedSource(source);
            else removeGroupedSource(source);
        });
        label.append(input, text);
        list.appendChild(label);
    });
    if (all) {
        const selected = entries.filter(isPlaced).length;
        all.textContent = selected === entries.length ? 'None' : 'All';
        all.onclick = () => {
            const shouldAdd = selected !== entries.length;
            entries.forEach(source => {
                if (shouldAdd && !isPlaced(source)) addGroupedSource(source);
                if (!shouldAdd && isPlaced(source)) removeGroupedSource(source);
            });
            renderPreviewVideoSources();
        };
    }
}

/* ── Source bar refresh ── */
function resetSourceBar() {
    renderSourceBar();
}

function _serializeSource(source) {
    if (source.kind === 'stereo_group') {
        return { kind: 'stereo_group', id: source.id, label: source.label, members: source.members };
    }
    return {
        kind: source.kind, source_key: source.source_key, label: source.label,
        hand: source.hand,
        stream_url: source.stream_url, heatmap_url: source.heatmap_url,
        depth_url: source.depth_url,
        depth_preview_url: source.depth_preview_url,
        depth_codes_url: source.depth_codes_url,
        raw_depth_url: source.raw_depth_url,
        depth_cache_key: source.depth_cache_key,
        // Optional grayscale clock stream; never used as depth pixels.
        depth_video_url: source.depth_video_url,
        missing_frames: source.missing_frames,
        depth_scale: source.depth_scale,
        frame_count: source.frame_count, fps: source.fps,
        // 并排拼接标记(3D 拼接等宽画面)→ 画布 tile 跨两列
        stereo_side_by_side: source.stereo_side_by_side,
    };
}

/* 画布渲染:video/skeleton → Plyr 播放器,glove/depth → 图像 tile */
function _frameUrl(urlTemplate, frameIndex) {
    return urlTemplate.replace('{frame}', Math.max(0, frameIndex));
}

async function mountGroupedSource(source, tile) {
    const episodeAtMount = currentEpisodeId;
    if (!tile || currentEpisodeId !== episodeAtMount) return;
    if (source.kind === 'hand3d_world') {
        // 3D 手部世界坐标交互窗口:canvas 渲染 /hand-3d 逐帧数据
        // (worker 只产 parquet,前端实时投影;拖拽旋转 + 滚轮缩放)
        const canvas = document.createElement('canvas');
        canvas.className = 'w-full h-full block bg-black';
        tile.appendChild(canvas);
        const entry = { canvas, source, rotY: 0, rotX: 0, zoom: 1,
                        pan: [0, 0, 0], dragging: false, dragMode: '',
                        lastX: 0, lastY: 0, _lastCam: null,
                        _hasCommittedFrame: false };
        currentHand3dTiles.push(entry);
        initHand3dDrag(entry);
        // Do not reveal the spatial canvas before its initial frame window is
        // available. RGB/2D and 3D then enter the workspace together.
        await _ensureHand3DInitialWindow(source.source_key || 'default');
        if (currentEpisodeId !== episodeAtMount || !tile.isConnected) return;
        renderHand3DTile(entry, 0);
        return;
    }
    if (source.kind === 'depth' && (source.depth_codes_url || source.depth_preview_url)) {
        // The source video remains raw gray12le. Only the received uint16
        // codes are colorized in this browser canvas; no JET image is stored.
        const holder = document.createElement('div');
        holder.id = `player-container-${groupedSourceDomId(source)}`;
        holder.className = 'bg-black rounded overflow-hidden w-full h-full min-h-0 relative';
        tile.appendChild(holder);
        const canvas = document.createElement('canvas');
        canvas.className = 'w-full h-full object-contain bg-black block';
        canvas.setAttribute('aria-label', source.label || 'Depth preview');
        holder.appendChild(canvas);
        const entry = { canvas, source, frames: new Map(),
                        windowInflight: null, pendingFrame: null,
                        fullCodes: null, fullInflight: null, fullReady: false,
                        allPreloadInflight: null, allPreloaded: false,
                        initialPreloadInflight: null, initialBufferReady: false,
                        keepAllFrames: false,
                        windowQueue: [], windowPump: false,
                        preloadProgress: 0,
                        initialReady: false,
                        fullError: null,
                        width: 0, height: 0, pixelCount: 0,
                        frameBytes: 0, frameCount: 0 };
        currentDepthPreviewTiles.push(entry);
        if (window.DepthRenderer) {
            return DepthRenderer.loadJetLut().then(async () => {
                const sourceFrameCount = Number(source.frame_count) || 0;
                const allowFullPreload = DEPTH_FULL_PRELOAD &&
                    sourceFrameCount > 0 &&
                    sourceFrameCount <= DEPTH_FULL_PRELOAD_MAX_FRAMES;
                if (allowFullPreload) {
                    try {
                        const loaded = await _preloadDepthCodes(entry);
                        if (!loaded) {
                            await _fetchDepthCodes(entry,
                                typeof currentFrameTarget === 'number'
                                    ? currentFrameTarget : 0);
                        }
                        return;
                    } catch (error) {
                        // Keep a bounded-window fallback for old servers or a
                        // transient full-stream failure; the normal path is
                        // still the complete raw-code preload above.
                        console.warn('[player] full depth preload failed:', error);
                    }
                }
                const veryLong = sourceFrameCount > DEPTH_VERY_LONG_FRAMES;
                if (veryLong) {
                    // Very long clips open after three depth windows (~12s at
                    // 30 FPS), then the playback queue fills future windows
                    // without evicting frames the playhead is still using.
                    const initialLoaded = await _preloadDepthInitialWindows(entry);
                    if (initialLoaded) {
                        return;
                    }
                } else {
                    // Normal/medium clips still use the strict all-loaded
                    // barrier, with only a few decoded windows in RAM.
                    const allLoaded = await _preloadDepthWindows(entry);
                    if (allLoaded) return;
                }
                await _fetchDepthCodes(entry,
                    typeof currentFrameTarget === 'number' ? currentFrameTarget : 0);
            }).catch(error => {
                console.warn('[player] depth mount failed:', error);
            });
        }
        return;
    }
    if (source.kind === 'glove' || source.kind === 'depth' || source.kind === 'hand') {
        const img = source.kind === 'hand'
            ? document.createElement('canvas') : document.createElement('img');
        // Hand-pressure panels are generated on a wide 800x220 canvas. Keep
        // that native ratio instead of stretching the heatmap vertically.
        img.className = source.kind === 'hand'
            ? 'w-full block bg-gray-900 hand-heatmap-image'
            : 'w-full h-full object-contain bg-black';
        if (source.kind === 'hand') {
            img.style.objectFit = 'contain';
            img.style.width = '100%';
            img.style.height = 'auto';
            img.style.maxHeight = '100%';
            img.style.aspectRatio = String(HAND_PREVIEW_ASPECT);
            tile.style.alignItems = 'center';
            tile.style.justifyContent = 'center';
        }
        if (source.kind !== 'hand') img.alt = source.label || 'Sensor / Depth';
        tile.appendChild(img);
        const entry = { img, source };
        currentImageTiles.push(entry);
        const frame = typeof currentFrameTarget === 'number' ? currentFrameTarget : 0;
        if (source.kind === 'hand') {
            // Hand-pressure tile: use the full frames-data payload and draw
            // locally; this avoids one PNG request per video frame.
            if (typeof registerHandTile === 'function') registerHandTile(img, source.hand);
        } else if (source.kind === 'depth') {
            // Depth streams are mounted by the Canvas/code path above.
        } else {
            img.src = _frameUrl(
                source.heatmap_url || source.depth_preview_url || source.depth_url,
                frame,
            );
        }
        return;
    }
    const holder = document.createElement('div');
    holder.id = `player-container-${groupedSourceDomId(source)}`;
    holder.className = 'bg-black rounded overflow-hidden w-full h-full min-h-0';
    tile.appendChild(holder);
    const url = source.stream_url ||
        await pickVideoUrl(episodeAtMount, source.source_key);
    if (currentEpisodeId !== episodeAtMount || !tile.isConnected) return;
    const player = initPlayer(holder.id, url);
    if (player) players[groupedSourceKey(source)] = player;
    // 手部骨骼 SVG 叠加层(只加在原始视频上;skeleton/渲染 tile 不加)
    // The first keypoint window is part of the shared initial-frame barrier.
    if (typeof initHandOverlay === 'function') {
        await initHandOverlay(source, tile, holder);
    }
}

/* 图像素材(手套热力图/深度图)跟随画布内任一 player 的播放进度 */
function bindGloveSync(player) {
    if (!player || _gloveSyncBound.has(player)) return;
    _gloveSyncBound.add(player);
    player.on('timeupdate', () => {
        if (isFrameScrubbing()) return;
        // 暂停态下的 seek(逐帧步进)也走这里:timeupdate 会在 seek 后触发一次
        _refreshImageTilesAt(Math.floor((player.currentTime || 0) * getEpisodeFps() + 0.002));
    });
}

let _lastImageTileFrame = -1;
function _refreshImageTilesAt(frame) {
    if (!currentImageTiles.length && !currentDepthPreviewTiles.length) return;
    _refreshDepthTilesAt(frame);
    if (!currentImageTiles.length) return;
    if (frame === _lastImageTileFrame) return;
    _lastImageTileFrame = frame;
    currentImageTiles.forEach((entry) => {
        const { img, source } = entry;
        // Hand-pressure tiles are driven by heatmap.js (frame URLs); only
        // template-based non-depth tiles are refreshed here.
        let url;
        if (source.kind === 'depth' || source.kind === 'hand') {
            return;
        } else {
            url = source.heatmap_url || source.depth_preview_url || source.depth_url;
        }
        if (url) img.src = _frameUrl(url, frame);
    });
}

/* 播放期逐帧刷新图像素材。

   不能用 rAF 作为视频时钟: rAF 跟显示器刷新(通常 60Hz),而视频可能是
   30FPS/变速/掉帧,于是深度画布会在同一个视频帧上重复上传,或先于视频
   画面切到下一帧。优先使用 HTMLVideoElement.requestVideoFrameCallback,
   让 RGB 视频真正送入 compositor 的那一帧驱动所有派生图层。回调内部
   只发布一个整数帧号,深度 renderer 采用 latest-only,不创建伪彩色文件。
   旧浏览器才走低频 rAF fallback。 */
let _imageRafStarted = false;
let _centralLastFrame = -1;
let _presentationClockToken = 0;
let _presentationClockVideo = null;
let _presentationClockHandle = null;

function _rawVideoElement(player) {
    if (!player) return null;
    return player.media || player.elements?.media ||
        player.elements?.container?.querySelector('video') || null;
}

function _publishPresentedFrame(video, metadata) {
    if (!video || video.paused || video.ended) return;
    const mediaTime = Number.isFinite(metadata?.mediaTime)
        ? metadata.mediaTime : (video.currentTime || 0);
    const frame = _clampFrameIndex(Math.floor(
        mediaTime * getEpisodeFps() + 0.002));
    // currentTime assignment is a seek and can make a healthy slave stutter.
    // Only correct a real multi-frame drift; initial play/seek alignment is
    // handled by playAll/seekToFrame.
    const maxDrift = Math.max(0.08, 2.5 / getEpisodeFps());
    Object.values(players).forEach(player => {
        const slaveVideo = _rawVideoElement(player);
        if (!slaveVideo || slaveVideo === video) return;
        if (Math.abs((slaveVideo.currentTime || 0) - mediaTime) > maxDrift) {
            try { slaveVideo.currentTime = mediaTime; } catch (_) {}
        }
    });
    // 深度帧未到时暂停整组播放器，避免 RGB 继续前进而深度画面停在旧帧。
    if (currentDepthPreviewTiles.length && !_depthFrameReady(frame)) {
        _stallPlaybackForDepth(frame);
        _refreshDepthTilesAt(frame);
        return;
    }
    if (frame !== _centralLastFrame) {
        _centralLastFrame = frame;
        updateFrameDisplay(frame);
    }
    _refreshImageTilesAt(frame);
}

function _schedulePresentationFrame(video, token) {
    if (token !== _presentationClockToken ||
        video !== _presentationClockVideo || video.paused || video.ended) return;
    if (typeof video.requestVideoFrameCallback === 'function') {
        _presentationClockHandle = video.requestVideoFrameCallback((_, metadata) => {
            _presentationClockHandle = null;
            if (token !== _presentationClockToken ||
                video !== _presentationClockVideo) return;
            _publishPresentedFrame(video, metadata);
            _schedulePresentationFrame(video, token);
        });
        return;
    }
    // Compatibility fallback: one rAF still publishes at most one integer
    // video frame, so a 30FPS source is never rendered twice per frame.
    _presentationClockHandle = requestAnimationFrame(() => {
        _presentationClockHandle = null;
        if (token !== _presentationClockToken ||
            video !== _presentationClockVideo) return;
        _publishPresentedFrame(video, null);
        _schedulePresentationFrame(video, token);
    });
}

function _ensurePresentationClock() {
    const video = _rawVideoElement(getActivePlayer());
    if (!video) return;
    if (video !== _presentationClockVideo) {
        _presentationClockToken += 1;
        _presentationClockVideo = video;
        _presentationClockHandle = null;
    }
    if (!video.paused && !video.ended && !_presentationClockHandle) {
        _schedulePresentationFrame(video, _presentationClockToken);
    }
}

function _startImageTileRaf() {
    if (_imageRafStarted) return;
    _imageRafStarted = true;
    window.__egodataCentralFrameClock = true;
    // This is only a cheap watchdog for play/pause and player replacement.
    // Actual frame work is scheduled by requestVideoFrameCallback above.
    const watchdog = () => {
        _ensurePresentationClock();
        setTimeout(watchdog, 100);
    };
    watchdog();
}

function _setPlayerToCurrentFrame(player) {
    if (!player) return;
    const apply = () => {
        const time = Math.max(0, currentFrameTarget / (episodeFps || 30));
        try { player.currentTime = time; } catch (e) {}
    };

    // Set immediately when metadata is already available, and set again on
    // ready for newly-added videos whose media element is not seekable yet.
    apply();
    if (typeof player.on === 'function' && typeof player.off === 'function') {
        const onReady = () => {
            player.off('ready', onReady);
            apply();
        };
        player.on('ready', onReady);
    }
}

function _restorePlayersToCurrentFrame() {
    Object.values(players).forEach(_setPlayerToCurrentFrame);
}

/* 并排拼接类源(3D 拼接等):内容是左右两路视频拼成的宽画面
   (2560×800),tile 应横跨两列显示,而不是挤在单格里。
   hand3d 是当前的 3D 拼接成品;未来任何带 stereo_side_by_side
   标记的双目骨骼拼接视频自动同样处理。 */
function _isWideSource(source) {
    return Boolean(source) && (
        source.kind === 'hand3d' ||
        source.stereo_side_by_side === true
    );
}

function _createWorkspaceTile(source, movable, reorderContainer) {
    const tile = document.createElement('div');
    tile.className = 'relative bg-black rounded overflow-hidden min-w-0 min-h-0 flex flex-col';
    tile.dataset.sourceKey = groupedSourceKey(source);
    tile.style.minWidth = '0';
    tile.style.width = '100%';
    tile.style.height = '100%';
    if (_isWideSource(source)) {
        // 横跨两列:与"两路原图各占一格"的总宽度一致,宽画面不再被压成
        // 单格(3D 拼接在 2 列布局下原本只占左侧格,右侧空置)
        tile.style.gridColumn = 'span 2';
    }
    if (source.kind === 'hand') {
        // Match the generated 800x220 hand canvas so the card itself does not
        // introduce letterbox space after the image returns to contain mode.
        tile.classList.remove('bg-black');
        tile.classList.add('bg-gray-900');
        tile.style.height = 'auto';
        tile.style.maxHeight = '100%';
        tile.style.aspectRatio = String(HAND_PREVIEW_ASPECT);
        tile.style.alignSelf = 'center';
    }

    const label = document.createElement('span');
    label.className = 'absolute z-10 top-1 left-1 bg-black/70 text-gray-200 text-[10px] px-1.5 py-0.5 rounded';
    label.textContent = groupedSourceLabel(source);
    tile.appendChild(label);

    const remove = document.createElement('button');
    remove.type = 'button';
    remove.className = 'absolute z-10 top-1 right-1 bg-black/70 text-gray-300 hover:text-white text-xs w-5 h-5 rounded';
    remove.textContent = '\u00d7';
    remove.title = t('remove_source');
    remove.onclick = event => {
        event.stopPropagation();
        removeGroupedSource(source);
    };
    tile.appendChild(remove);

    if (!movable) return tile;

    tile.draggable = true;
    tile.addEventListener('dragstart', event => {
        event.dataTransfer.setData('application/x-egodata-reorder', groupedSourceKey(source));
        event.dataTransfer.effectAllowed = 'move';
        tile.classList.add('opacity-50');
    });
    tile.addEventListener('dragend', () =>
        tile.classList.remove('opacity-50', 'ring-2', 'ring-blue-500'));
    tile.addEventListener('dragover', event => {
        const types = event.dataTransfer.types || [];
        if (types.includes('application/x-egodata-reorder') ||
            types.includes('application/x-egodata-source')) {
            event.preventDefault();
            event.stopPropagation();
            event.dataTransfer.dropEffect = types.includes('application/x-egodata-reorder')
                ? 'move' : 'copy';
            tile.classList.add('ring-2', 'ring-blue-500');
        }
    });
    tile.addEventListener('dragleave', () =>
        tile.classList.remove('ring-2', 'ring-blue-500'));
    tile.addEventListener('drop', event => {
        event.preventDefault();
        event.stopPropagation();
        tile.classList.remove('ring-2', 'ring-blue-500');
        const fromKey = event.dataTransfer.getData('application/x-egodata-reorder');
        if (fromKey && fromKey !== groupedSourceKey(source)) {
            reorderTiles(fromKey, groupedSourceKey(source), reorderContainer);
            return;
        }
        const raw = event.dataTransfer.getData('application/x-egodata-source');
        if (raw) {
            try { addGroupedSource(JSON.parse(raw)); } catch (e) {}
        }
    });
    return tile;
}

function _createHandFooter() {
    const footer = document.createElement('div');
    footer.className = 'min-w-0 min-h-0 overflow-hidden grid grid-cols-2 gap-3';
    footer.dataset.role = 'hand-footer';
    const extra = document.createElement('div');
    extra.className = 'min-w-0 min-h-0 overflow-hidden grid gap-3';
    extra.dataset.role = 'bottom-extra';
    extra.style.gridColumn = '1 / -1';
    workspaceBottomExtra = extra;
    footer.appendChild(extra);
    ['left', 'right'].forEach(hand => {
        const slot = document.createElement('div');
        slot.className = 'min-w-0 min-h-0 overflow-hidden bg-gray-900 rounded';
        slot.dataset.handSlot = hand;
        footer.appendChild(slot);
        workspaceHandSlots[hand] = slot;
    });
    return footer;
}

function _refreshWorkspaceLayout() {
    if (!workspaceGrid || !workspaceMainRow) return;
    const mainSources = workspaceSources.filter(source => !_isBottomSource(source));
    const bottomSources = workspaceSources.filter(_isBottomSource);
    const handSources = workspaceSources.filter(source => source.kind === 'hand');
    const columns = mainSources.length <= 1 ? 1 : 2;
    // 行数按"占用的格子数"算:并排拼接类(3D 拼接等)跨两列,占 2 格;
    // 单列布局下(只有一个源)跨列无意义,按 1 格计
    const slots = mainSources.reduce(
        (n, source) => n + (columns >= 2 && _isWideSource(source) ? 2 : 1), 0);
    const rows = Math.max(1, Math.ceil(slots / columns));

    const hasBottomExtras = bottomSources.some(source => source.kind !== 'hand');
    const bottomHeight = handSources.length
        ? (hasBottomExtras ? HAND_FOOTER_WITH_EXTRA_HEIGHT : HAND_FOOTER_HEIGHT)
        : 260;
    workspaceGrid.style.gridTemplateRows = bottomSources.length
        ? `minmax(0, 1fr) ${bottomHeight}px`
        : 'minmax(0, 1fr)';
    if (!bottomSources.length && workspaceHandFooter) {
        workspaceHandFooter.remove();
        workspaceHandFooter = null;
        workspaceHandSlots = { left: null, right: null };
        workspaceBottomExtra = null;
    }
    workspaceMainRow.style.gridTemplateColumns =
        `repeat(${columns}, minmax(0, 1fr))`;
    workspaceMainRow.style.gridTemplateRows =
        `repeat(${rows}, minmax(0, 1fr))`;

    if (!mainSources.length) {
        if (!workspaceMainRow.querySelector('[data-role="empty-workspace"]')) {
            workspaceMainRow.innerHTML =
                `<div data-role="empty-workspace" class="col-span-full row-span-full flex items-center justify-center text-gray-600">${t('drag_here')}</div>`;
        }
    } else {
        const empty = workspaceMainRow.querySelector('[data-role="empty-workspace"]');
        if (empty) empty.remove();
        mainSources.forEach(source => {
            const tile = workspaceMainRow.querySelector(
                `[data-source-key="${CSS.escape(groupedSourceKey(source))}"]`);
            if (tile) {
                // 跨列状态统一在这里兜底修正:只有多列布局下并排拼接类
                // 才跨两列(单列时 span 2 会创建隐式列导致布局错位)
                tile.style.gridColumn =
                    columns >= 2 && _isWideSource(source) ? 'span 2' : '';
                workspaceMainRow.appendChild(tile);
            }
        });
    }

    ['left', 'right'].forEach(hand => {
        const slot = workspaceHandSlots[hand];
        if (!slot) return;
        const source = handSources.find(item => item.hand === hand);
        const tile = source
            ? workspaceGrid.querySelector(`[data-source-key="${CSS.escape(groupedSourceKey(source))}"]`)
            : null;
        if (tile && tile.parentNode !== slot) slot.appendChild(tile);
        slot.style.display = source ? '' : 'none';
    });

    if (workspaceBottomExtra) {
        const extras = bottomSources.filter(source => source.kind !== 'hand');
        workspaceBottomExtra.style.display = extras.length ? 'grid' : 'none';
        workspaceBottomExtra.style.gridTemplateColumns =
            `repeat(${Math.min(2, Math.max(1, extras.length))}, minmax(0, 1fr))`;
        extras.forEach(source => {
            const tile = workspaceGrid.querySelector(
                `[data-source-key="${CSS.escape(groupedSourceKey(source))}"]`);
            if (tile && tile.parentNode !== workspaceBottomExtra) {
                workspaceBottomExtra.appendChild(tile);
            }
        });
    }
    if (workspaceHandFooter) {
        const hasExtras = bottomSources.some(source => source.kind !== 'hand');
        const hasHands = handSources.length > 0;
        workspaceHandFooter.style.gridTemplateRows = hasExtras && hasHands
            ? 'minmax(0, 1fr) minmax(0, 1fr)'
            : 'minmax(0, 1fr)';
    }
}

async function _appendWorkspaceSource(source) {
    if (!workspaceGrid || !workspaceMainRow) return;
    const movable = !_isBottomSource(source);
    const tile = _createWorkspaceTile(source, movable, workspaceMainRow);
    if (_isBottomSource(source)) {
        if (!workspaceHandFooter) {
            workspaceHandFooter = _createHandFooter();
            workspaceGrid.appendChild(workspaceHandFooter);
        }
        if (source.kind === 'hand') {
            const slot = workspaceHandSlots[source.hand];
            if (slot) slot.appendChild(tile);
        } else if (workspaceBottomExtra) {
            workspaceBottomExtra.appendChild(tile);
        }
    } else {
        const empty = workspaceMainRow.querySelector('[data-role="empty-workspace"]');
        if (empty) empty.remove();
        workspaceMainRow.appendChild(tile);
    }
    _refreshWorkspaceLayout();
    await mountGroupedSource(source, tile);
    const player = players[groupedSourceKey(source)];
    _setPlayerToCurrentFrame(player);
    _syncWorkspacePlayers();
}

function renderGroupedWorkspace() {
    const grid = document.getElementById('video-grid');
    if (!grid) return;
    const renderEpoch = currentEpisodeId;  // 防快速切换批次:过期渲染不得启用控件/揭遮罩
    const renderToken = _playbackSessionToken;
    masterPlaying = false;
    refreshPlayButton(false);
    if (typeof stopHeatmapSync === 'function') stopHeatmapSync();
    Object.values(players).forEach(player => { try { player.destroy(); } catch (e) {} });
    Object.keys(players).forEach(key => delete players[key]);
    currentImageTiles = [];
    currentDepthPreviewTiles = [];
    currentHand3dTiles = [];
    if (typeof clearHandTiles === 'function') clearHandTiles();

    workspaceGrid = grid;
    workspaceMainRow = null;
    workspaceHandFooter = null;
    workspaceHandSlots = { left: null, right: null };
    workspaceBottomExtra = null;
    grid.innerHTML = '';
    grid.style.position = 'relative';
    grid.className = 'flex-1 min-w-0 min-h-0 p-3 overflow-hidden grid gap-3';
    grid.ondragover = event => {
        event.preventDefault();
        grid.classList.add('ring-1', 'ring-blue-500/50');
    };
    grid.ondragleave = () => grid.classList.remove('ring-1', 'ring-blue-500/50');
    grid.ondrop = event => {
        event.preventDefault();
        grid.classList.remove('ring-1', 'ring-blue-500/50');
        const raw = event.dataTransfer.getData('application/x-egodata-source');
        if (raw) {
            try { addGroupedSource(JSON.parse(raw)); } catch (e) {}
        } else {
            const hand = event.dataTransfer.getData('application/x-egodata-hand');
            if (hand) addHandTile(hand);
        }
    };

    const mainSources = workspaceSources.filter(source => !_isBottomSource(source));
    const bottomSources = workspaceSources.filter(_isBottomSource);
    const handSources = workspaceSources.filter(source => source.kind === 'hand');
    const hasBottomExtras = bottomSources.some(source => source.kind !== 'hand');
    grid.style.gridTemplateRows = bottomSources.length
        ? `minmax(0, 1fr) ${handSources.length
            ? (hasBottomExtras ? HAND_FOOTER_WITH_EXTRA_HEIGHT : HAND_FOOTER_HEIGHT)
            : 260}px`
        : 'minmax(0, 1fr)';

    const mainRow = document.createElement('div');
    mainRow.className = 'min-w-0 min-h-0 overflow-hidden grid gap-3';
    mainRow.dataset.role = 'main-workspace';
    workspaceMainRow = mainRow;
    grid.appendChild(mainRow);

    if (!mainSources.length) {
        mainRow.innerHTML =
            `<div data-role="empty-workspace" class="col-span-full row-span-full flex items-center justify-center text-gray-600">${t('drag_here')}</div>`;
    }

    if (bottomSources.length) {
        const footer = _createHandFooter();
        workspaceHandFooter = footer;
        grid.appendChild(footer);
    }
    _refreshWorkspaceLayout();
    renderSourceBar();

    const mounted = [];
    mainSources.forEach(source => {
        const tile = _createWorkspaceTile(source, true, mainRow);
        mainRow.appendChild(tile);
        mounted.push(mountGroupedSource(source, tile));
    });
    bottomSources.forEach(source => {
        const tile = _createWorkspaceTile(source, false, null);
        if (source.kind === 'hand') {
            const slot = workspaceHandSlots[source.hand];
            if (slot) slot.appendChild(tile);
        } else if (workspaceBottomExtra) {
            workspaceBottomExtra.appendChild(tile);
        }
        mounted.push(mountGroupedSource(source, tile));
    });

    Promise.all(mounted).then(async () => {
        if (!isCurrentPlaybackSession(renderEpoch, renderToken)) return;  // 本次渲染作废
        _restorePlayersToCurrentFrame();
        const active = getActivePlayer();
        if (!active) { hideVideoLoading(); return; }
        _syncWorkspacePlayers();
        refreshPlayButton(false);
        try {
            await loadFrameData(renderEpoch, renderToken);
        } catch (e) {
            if (isCurrentPlaybackSession(renderEpoch, renderToken)) hideVideoLoading();
            return;
        }
        if (!isCurrentPlaybackSession(renderEpoch, renderToken)) return;
        if (!_defaultLayoutApplied && typeof hasHandSensor === 'function') {
            _defaultLayoutApplied = true;
            ['left', 'right'].forEach(hand => {
                if (hasHandSensor(hand) &&
                    !workspaceSources.some(source => source.kind === 'hand' && source.hand === hand)) {
                    addHandTile(hand);
                }
            });
        }
        // A newly opened episode always starts at its own frame zero. Do this
        // after the real FPS/frame count arrive so all overlays and media use
        // the same exact time origin, regardless of the previous episode.
        pauseAll();
        seekToFrame(0);
        updateFrameDisplay(0);
        _refreshDepthTilesAt(0);
        if (typeof startHeatmapSync === 'function') startHeatmapSync(getActivePlayer());
        if (typeof loadAnnotations === 'function') {
            await loadAnnotations(renderEpoch).catch(() => {});
        }
        if (!isCurrentPlaybackSession(renderEpoch, renderToken)) return;
        // 深度首帧必须先落到 Canvas，再放开播放控件；否则用户会先
        // 看到黑色深度窗，RGB 已经开始跑，随后每到深度边界就暂停。
        // 短批次在这里等待整段 raw code preload，长批次等待首个 120
        // 帧窗口，后续窗口由播放时钟提前预取。
        const depthReady = await _awaitDepthInitialReady(renderEpoch);
        if (!isCurrentPlaybackSession(renderEpoch, renderToken)) return;
        if (!depthReady) {
            console.warn('[player] depth initial frame not ready; keep playback gated');
            return;
        }
        // 全部就绪才放开控件/揭遮罩:消除「第二个视频晚 ready → 开头不同步」
        // 与「帧数据返回前用旧 fps 交互 → 帧不对齐」两个竞态窗口
        await _awaitPlayersReady();
        if (!isCurrentPlaybackSession(renderEpoch, renderToken)) return;
        enableFrameControls();
        hideVideoLoading();
    }).catch(err => {
        // 兜底:任何未预期异常都不能让遮罩卡死/控件锁死(仅当前批次生效)
        console.error('[player] workspace load failed:', err);
        if (isCurrentPlaybackSession(renderEpoch, renderToken)) {
            enableFrameControls();
            hideVideoLoading();
        }
    });
}

function renderGroupedWorkspaceLegacy() {
    const grid = document.getElementById('video-grid');
    if (!grid) return;
    if (typeof stopHeatmapSync === 'function') stopHeatmapSync();
    Object.values(players).forEach(player => { try { player.destroy(); } catch (e) {} });
    Object.keys(players).forEach(key => delete players[key]);
    currentImageTiles = [];
    currentDepthPreviewTiles = [];
    if (typeof clearHandTiles === 'function') clearHandTiles();
    grid.innerHTML = '';
    grid.style.position = 'relative';
    grid.ondragover = event => event.preventDefault();
    grid.ondrop = event => {
        event.preventDefault();
        try {
            const raw = event.dataTransfer.getData('application/x-egodata-source');
            if (raw) {
                const src = JSON.parse(raw);
                if (src.kind === 'hand') addHandTile(src.hand);  // heatmaps drop into the fixed bottom row
                else addGroupedSource(src);
            } else {
                const hand = event.dataTransfer.getData('application/x-egodata-hand');
                if (hand) addHandTile(hand);
            }
        } catch (e) {}
    };
    if (!workspaceSources.length) {
        grid.innerHTML = '<div class="flex items-center justify-center text-gray-600">' + t('drag_here') + '</div>';
        return;
    }

    // Two fixed zones: a video area on top (free drag-to-reorder) and a
    // heatmap area below (left-hand on the left column, right-hand on the
    // right column — always, per requirement).
    const isImageSource = s => ['glove', 'depth', 'hand'].includes(s.kind);
    const vids = workspaceSources.filter(s => !isImageSource(s));
    const imgs = workspaceSources.filter(s => isImageSource(s));

    // The canvas is a fixed viewport. Use nested CSS grids so adding a
    // source only repartitions the existing pixels; it must never create a
    // page-level scroll or increase the review area's height.
    grid.className = 'flex-1 min-w-0 min-h-0 p-3 overflow-hidden grid gap-3';
    grid.style.display = 'grid';
    grid.style.gridTemplateRows = imgs.length
        ? 'minmax(0, 1fr) 190px'
        : 'minmax(0, 1fr)';

    const topRow = document.createElement('div');
    topRow.className = 'min-w-0 min-h-0 overflow-hidden';
    const bottomRow = document.createElement('div');
    bottomRow.className = 'min-w-0 min-h-0 overflow-hidden';

    // Choose the layout automatically from the number of video sources.
    // Every row is explicitly constrained inside topRow, so even four or
    // more sources remain inside the fixed canvas.
    const videoCount = vids.length;
    const videoColumns = videoCount <= 1 ? 1 : 2;
    const videoRows = Math.max(1, Math.ceil(videoCount / videoColumns));
    topRow.style.display = 'grid';
    topRow.style.gridTemplateColumns = `repeat(${videoColumns}, minmax(0, 1fr))`;
    topRow.style.gridTemplateRows = `repeat(${videoRows}, minmax(0, 1fr))`;
    topRow.style.gap = '12px';
    if (imgs.length) {
        const imageColumns = Math.min(2, Math.max(1, imgs.length));
        const imageRows = Math.max(1, Math.ceil(imgs.length / imageColumns));
        bottomRow.style.display = 'grid';
        bottomRow.style.gridTemplateColumns = `repeat(${imageColumns}, minmax(0, 1fr))`;
        bottomRow.style.gridTemplateRows = `repeat(${imageRows}, minmax(0, 1fr))`;
        bottomRow.style.gap = '12px';
    }

    const makeTile = (source, inVideoRow) => {
        const tile = document.createElement('div');
        tile.className = 'relative bg-black rounded overflow-hidden min-w-0 min-h-0 flex flex-col';
        tile.dataset.sourceKey = groupedSourceKey(source);
        tile.style.minWidth = '0';
        tile.style.width = '100%';
        tile.style.height = '100%';
        const label = document.createElement('span');
        label.className = 'absolute z-10 top-1 left-1 bg-black/70 text-gray-200 text-[10px] px-1.5 py-0.5 rounded';
        label.textContent = groupedSourceLabel(source);
        tile.appendChild(label);
        const remove = document.createElement('button');
        remove.type = 'button';
        remove.className = 'absolute z-10 top-1 right-1 bg-black/70 text-gray-300 hover:text-white text-xs w-5 h-5 rounded';
        remove.textContent = '×';
        remove.title = t('remove_source');
        remove.onclick = () => removeGroupedSource(source);
        tile.appendChild(remove);
        if (inVideoRow) {
            // Drag to reorder videos (DOM-level swap — playback keeps running)
            tile.draggable = true;
            tile.addEventListener('dragstart', e => {
                e.dataTransfer.setData('application/x-egodata-reorder', groupedSourceKey(source));
                e.dataTransfer.effectAllowed = 'move';
                tile.classList.add('opacity-50');
            });
            tile.addEventListener('dragend', () => tile.classList.remove('opacity-50', 'ring-2', 'ring-blue-500'));
            tile.addEventListener('dragover', e => {
                if (e.dataTransfer.types && e.dataTransfer.types.includes('application/x-egodata-reorder')) {
                    e.preventDefault();
                    e.dataTransfer.dropEffect = 'move';
                    tile.classList.add('ring-2', 'ring-blue-500');
                }
            });
            tile.addEventListener('dragleave', () => tile.classList.remove('ring-2', 'ring-blue-500'));
            tile.addEventListener('drop', e => {
                e.preventDefault();
                e.stopPropagation();  // don't trigger the canvas asset drop
                tile.classList.remove('ring-2', 'ring-blue-500');
                const fromKey = e.dataTransfer.getData('application/x-egodata-reorder');
                if (fromKey && fromKey !== groupedSourceKey(source)) {
                    reorderTiles(fromKey, groupedSourceKey(source), topRow);
                }
            });
        }
        return tile;
    };

    const restoreFrame = typeof currentFrameTarget === 'number' ? currentFrameTarget : 0;
    const mounted = [];
    if (vids.length) {
        grid.appendChild(topRow);
        vids.forEach(source => {
            const tile = makeTile(source, true);
            topRow.appendChild(tile);
            mounted.push(mountGroupedSource(source, tile));
        });
    }
    if (imgs.length) {
        grid.appendChild(bottomRow);
        // Heatmaps keep their fixed slots: left hand → left column, right → right
        const handTiles = imgs.filter(s => s.kind === 'hand')
            .sort((a, b) => (a.hand === 'left' ? 0 : 1) - (b.hand === 'left' ? 0 : 1));
        const otherImgs = imgs.filter(s => s.kind !== 'hand');
        [...handTiles, ...otherImgs].forEach(source => {
            const tile = makeTile(source, false);
            bottomRow.appendChild(tile);
            mounted.push(mountGroupedSource(source, tile));
        });
    }
    Promise.all(mounted).then(() => {
        if (restoreFrame > 0) {
            const time = restoreFrame / (episodeFps || 30);
            Object.values(players).forEach(player => { player.currentTime = time; });
        }
        const active = getActivePlayer();
        if (active && currentEpisodeId) {
            bindGloveSync(active);
            Object.values(players).forEach(p => bindMasterSync(p));
            bindFrameDrift(active);
            refreshPlayButton(false);
            loadFrameData(currentEpisodeId, getPlaybackSessionToken()).then(() => {
                // Balanced default layout: videos on top, hand heatmaps below —
                // left-hand heatmap under the left view, right-hand under the
                // right view (2×2 grid). Runs once on first default load; the
                // heatmaps stay removable (buttons return to the source bar).
                if (!_defaultLayoutApplied && typeof hasHandSensor === 'function') {
                    _defaultLayoutApplied = true;
                    const added = [];
                    ['left', 'right'].forEach(hand => {
                        if (hasHandSensor(hand) &&
                            !workspaceSources.some(s => s.kind === 'hand' && s.hand === hand)) {
                            workspaceSources.push({
                                kind: 'hand', hand, source_key: `hand_${hand}`,
                                label: t(hand === 'left' ? 'left_hand' : 'right_hand'),
                            });
                            added.push(hand);
                        }
                    });
                    if (added.length) {
                        renderSourceBar();
                        renderGroupedWorkspace();
                        return;
                    }
                }
                if (typeof startHeatmapSync === 'function') startHeatmapSync(active);
                if (typeof loadAnnotations === 'function') loadAnnotations(currentEpisodeId);
            });
        }
    });
}

/* 默认画布:双目组 → 左右目并排;多物理设备单目流 → 全部显示 */
function _defaultWorkspaceFromGroups() {
    // 保留已自动挂载的 3D 世界窗口:loadHand3D 可能先于本函数完成
    // 并已把 hand3d_world 加入 workspaceSources,整表重建不能冲掉
    // (否则开关显示开启、窗口实际不存在 —— 竞态脱节 bug)
    const keep = workspaceSources.filter(s =>
        s.kind === 'hand3d_world' || s.kind === 'depth');
    // Depth is a first-class default preview source. It must be present in
    // workspaceSources before renderGroupedWorkspace takes its mount Promise
    // snapshot; adding it immediately after render creates a race where the
    // RGB tiles are ready but the depth canvas remains black with no request.
    const depthSources = (currentMediaGroups.sources || [])
        .filter(s => s.kind === 'depth')
        .filter(() => typeof window.isDepthOn !== 'function' || window.isDepthOn())
        .map(_serializeSource);
    const retained = [...keep, ...depthSources.filter(depth =>
        !keep.some(item => groupedSourceKey(item) === groupedSourceKey(depth)))];
    workspaceSources = [];
    const grp = (currentMediaGroups.groups || [])[0];
    if (grp && (grp.members || []).length) {
        workspaceSources = grp.members.map(m => ({
            kind: 'video', source_key: m.source_key, label: m.label,
            stream_url: m.stream_url, frame_count: m.frame_count, fps: m.fps,
        }));
        workspaceSources.push(...retained);
        return;
    }
    const singles = currentMediaGroups.singles || [];
    if (singles.length) {
        workspaceSources = singles.map(single => ({
            kind: 'video', source_key: single.source_key, label: single.label,
            stream_url: single.stream_url, frame_count: single.frame_count, fps: single.fps,
        }));
    }
    workspaceSources.push(...retained);
}

async function loadGroupedEpisodeVideo(episodeId, cameras, options = {}) {
    _cancelEpisodeMediaLoad(true);
    const loadToken = _playbackSessionToken;
    _episodeMediaLoadController = new AbortController();
    const mediaSignal = _episodeMediaLoadController.signal;
    currentEpisodeId = episodeId;
    currentCameras = cameras || [];
    currentHasSkeleton = Boolean(options.hasSkeleton);
    _suppressMasterEvents = true;
    pauseAll();
    _suppressMasterEvents = false;
    masterPlaying = false;
    refreshPlayButton(false);
    currentMediaGroups = null;
    workspaceSources = [];
    workspaceCameras = currentCameras.slice();
    currentImageTiles = [];
    currentDepthPreviewTiles = [];
    currentHand3dTiles = [];
    hand3dData = null;
    hand3dDataBySource = {};
    hand3dFrameCache = { frame: -1, data: null, inflight: -1 };
    hand3dFrameCacheBySource = {};
    hand3dWindow = { start: -1, end: -1, frames: {}, inflight: false };
    hand3dWindowBySource = {};
    _depthPlaybackStall = null;
    _hand3dViewAnchor = null;
    _h3dBaseDist = null;
    _h3dGridY = null;
    _h3dCentroid.filters = [null, null];
    _h3dCentroid.prevLabels = [null, null];
    _h3dCentroid.prevC = [null, null];
    _h3dCentroid.t0 = null;
    currentFrameTarget = 0;
    _centralLastFrame = -1;
    _lastDisplayedFrame = -1;
    _pendingSeekFrame = null;
    // Seed the UI with the selected episode's canonical metadata immediately.
    // frameDataReady remains false until frames-data has been validated.
    episodeTotalFrames = Math.max(0, Number(options.frameCount) || 0);
    episodeFps = Math.max(0, Number(options.fps) || 0);
    frameDataReady = false;
    _resetFrameUiForEpisode(0, episodeTotalFrames);
    _watchdogRetried = false;    // 每个批次允许一次自愈重渲染
    cancelSegmentPlayback();     // 切批次即终止段播放会话(旧会话的播放器即将被销毁)
    _defaultLayoutApplied = false;
    if (typeof resetHeatmapFrameState === 'function') resetHeatmapFrameState(episodeId);
    if (typeof clearAnnotationsNow === 'function') clearAnnotationsNow();  // 立即清掉上一批次残留的切片/overlay
    if (typeof stopHeatmapSync === 'function') stopHeatmapSync();
    if (typeof clearHandTiles === 'function') clearHandTiles();
    if (typeof destroyAllHandOverlays === 'function') destroyAllHandOverlays();
    resetSourceBar();
    disableFrameControls();
    showVideoLoading();
    // 自愈保险:加载链任何一环卡住,10 秒后强制启用控件并揭开遮罩 ——
    // 控件宁可可用,不可锁死("点了没反应"必须被兜底修掉)。
    // 若此时还没有任何活跃播放器(挂载失败/被冲掉),重渲染工作区一次。
    clearTimeout(_loadWatchdog);
    _loadWatchdog = setTimeout(() => {
        if (!isCurrentPlaybackSession(episodeId, loadToken)) return;
        const depthStillBuffering = currentDepthPreviewTiles.some(entry =>
            (entry.fullInflight && !entry.fullReady) ||
            (entry.allPreloadInflight && !entry.allPreloaded));
        if (!_controlsEnabled && !depthStillBuffering) {
            console.warn('[player] 加载自愈:强制启用控件');
            enableFrameControls();
            hideVideoLoading();
        }
        if (!getActivePlayer() && !_watchdogRetried) {
            _watchdogRetried = true;
            console.warn('[player] 加载自愈:无活跃播放器 → 重渲染工作区');
            try { renderGroupedWorkspace(); } catch (e) {
                console.error('[player] 自愈重渲染失败:', e);
            }
        }
    }, 10000);
    // Fetch 3D metadata in parallel with media-groups, but await both before
    // building the workspace. The 3D tile itself then awaits its first data
    // window together with each RGB overlay's first keypoint window.
    const hand3dReady = loadHand3D(episodeId, loadToken);

    const grid = document.getElementById('video-grid');
    if (!grid) { hideVideoLoading(); return; }

    // 媒体组 API:双目分组 + 素材清单(新/旧数据都兼容)
    try {
        // Media groups include transport URLs. They must not be served from
        // the browser's GET cache after a backend/frontend rollout.
        const res = await fetch(
            `/api/v1/episode/${episodeId}/media-groups?_v=202609022200`,
            { signal: mediaSignal, cache: 'no-store' });
        if (!isCurrentPlaybackSession(episodeId, loadToken)) return;  // 丢弃过期响应
        if (res.ok) {
            currentMediaGroups = await res.json();
            await hand3dReady;
            if (!isCurrentPlaybackSession(episodeId, loadToken)) return;
            renderSourceBar();
            _defaultWorkspaceFromGroups();
            // Add the selected spatial source before renderGroupedWorkspace
            // takes its Promise.all(mounted) snapshot.
            if (typeof ensureHand3dWorldTile === 'function'
                    && hand3dData && hand3dData.hasPreview
                    && (typeof window.isHand3dWorldOn !== 'function'
                        || window.isHand3dWorldOn())) {
                ensureHand3dWorldTile();
            }
            renderGroupedWorkspace();
            renderPreviewVideoSources();
            // 3D 世界窗口兜底:开关开且有产物 → 确保已挂载
            if (typeof ensureHand3dWorldTile === 'function'
                    && hand3dData && hand3dData.hasPreview
                    && (typeof window.isHand3dWorldOn !== 'function'
                        || window.isHand3dWorldOn())) {
                ensureHand3dWorldTile();
            }
            // 深度窗口兜底:☷ 深度开关开且有深度素材 → 自动挂载
            // (素材栏已无深度按钮,挂载入口统一走 ☷)
            if (typeof ensureDepthTile === 'function'
                    && (typeof window.isDepthOn !== 'function'
                        || window.isDepthOn())) {
                ensureDepthTile();
            }
            updateInfoBar();
            if (typeof updatePreviewMenuData === 'function') updatePreviewMenuData();
            return;
        }
    } catch (e) {
        if (e?.name === 'AbortError' || !isCurrentPlaybackSession(episodeId, loadToken)) return;
        /* fall through → 旧逻辑 */
    }

    if (!isCurrentPlaybackSession(episodeId, loadToken)) return;

    // 兜底:API 不可用时按 camera_names 平铺到素材按钮条
    renderSourceBar();
    if (!currentCameras.length) {
        grid.innerHTML = '<div class="flex items-center justify-center h-64 text-gray-600">' + t('no_video') + '</div>';
        hideVideoLoading();
        return;
    }
    workspaceSources = currentCameras.slice(0, 2).map(camera => groupedSource('video', camera, { label: camera }));
    renderGroupedWorkspace();
    renderPreviewVideoSources();
    updateInfoBar();
    if (typeof updatePreviewMenuData === 'function') updatePreviewMenuData();
}


function loadEpisodeVideo(episodeId, cameras, options = {}) {
    _cancelEpisodeMediaLoad(true);
    const loadToken = _playbackSessionToken;
    _episodeMediaLoadController = new AbortController();
    currentEpisodeId = episodeId;
    currentCameras = cameras || [];
    _suppressMasterEvents = true;
    pauseAll();
    _suppressMasterEvents = false;
    masterPlaying = false;
    refreshPlayButton(false);
    currentFrameTarget = 0;
    _centralLastFrame = -1;
    _lastDisplayedFrame = -1;
    _pendingSeekFrame = null;  // reset — prevent stale seek target from previous episode
    episodeTotalFrames = Math.max(0, Number(options.frameCount) || 0);
    episodeFps = Math.max(0, Number(options.fps) || 0);
    frameDataReady = false;
    _resetFrameUiForEpisode(0, episodeTotalFrames);
    cancelSegmentPlayback(); // 切批次即终止段播放会话(旧会话的播放器即将被销毁)
    if (typeof resetHeatmapFrameState === 'function') resetHeatmapFrameState(episodeId);
    if (typeof clearAnnotationsNow === 'function') clearAnnotationsNow();  // 立即清掉上一批次残留的切片/overlay
    _defaultLayoutApplied = false;
    showVideoLoading();
    stopHeatmapSync();
    if (typeof clearHandTiles === 'function') clearHandTiles();
    // The legacy single-video fallback uses the same hand-overlay state as
    // the grouped workspace. Clear it here too, otherwise a no-data episode
    // can leave the global skeleton switch disabled for the next episode.
    if (typeof destroyAllHandOverlays === 'function') destroyAllHandOverlays();
    currentHand3dTiles = [];
    hand3dData = null;
    hand3dDataBySource = {};
    hand3dFrameCache = { frame: -1, data: null, inflight: -1 };
    hand3dFrameCacheBySource = {};
    hand3dWindow = { start: -1, end: -1, frames: {}, inflight: false };
    hand3dWindowBySource = {};
    _depthPlaybackStall = null;
    _hand3dViewAnchor = null;
    _h3dBaseDist = null;
    _h3dGridY = null;
    _h3dCentroid.filters = [null, null];
    _h3dCentroid.prevLabels = [null, null];
    _h3dCentroid.prevC = [null, null];
    _h3dCentroid.t0 = null;
    loadHand3D(episodeId, loadToken);
    resetSourceBar();
    disableFrameControls();  // disabled until player fires 'ready'

    const grid = document.getElementById('video-grid');
    const tabs = document.getElementById('camera-tabs');

    if (!grid || !tabs) { hideVideoLoading(); return; }

    // Clear existing
    grid.innerHTML = '';

    tabs.innerHTML = '<span class="text-gray-500 text-sm py-2">Cameras:</span>';

    // Destroy old players
    Object.values(players).forEach(p => p.destroy());
    Object.keys(players).forEach(k => delete players[k]);

    if (!cameras || cameras.length === 0) {
        grid.innerHTML = '<div class="flex items-center justify-center h-64 text-gray-600">' + t('no_video') + '</div>';
        hideVideoLoading();
        return;
    }

    // Create tabs
    cameras.forEach((cam, i) => {
        const btn = document.createElement('button');
        btn.textContent = cam;
        btn.className = i === 0
            ? 'px-3 py-1 text-sm rounded-t bg-blue-600 text-white'
            : 'px-3 py-1 text-sm rounded-t bg-gray-800 text-gray-400 hover:bg-gray-700';
        btn.onclick = () => switchTab(cam);
        tabs.appendChild(btn);
    });

    // Create video container(s)
    const fallbackColumns = cameras.length <= 1 ? 1 : 2;
    const fallbackRows = Math.max(1, Math.ceil(cameras.length / fallbackColumns));
    grid.className = 'flex-1 min-w-0 min-h-0 p-3 overflow-hidden grid gap-3';
    grid.style.gridTemplateColumns = `repeat(${fallbackColumns}, minmax(0, 1fr))`;
    grid.style.gridTemplateRows = `repeat(${fallbackRows}, minmax(0, 1fr))`;
    grid.style.position = 'relative';  // anchor for skeleton canvas overlay

    cameras.forEach(async cam => {
        const div = document.createElement('div');
        div.id = `player-container-${cam}`;
        div.className = 'bg-black rounded overflow-hidden min-w-0 min-h-0 w-full h-full';
        grid.appendChild(div);

        const url = await pickVideoUrl(episodeId, cam);
        if (!isCurrentPlaybackSession(episodeId, loadToken)) return;
        players[cam] = initPlayer(`player-container-${cam}`, url);
        if (players[cam] && typeof initHandOverlay === 'function') {
            const source = { kind: 'video', source_key: cam, label: cam };
            initHandOverlay(source, div, div);
        }
    });

    // Start heatmap sync with first camera's player
    const firstCam = cameras[0];
    const loadEpoch = episodeId;  // capture for guard — discard if episode changed
    if (firstCam && players[firstCam]) {
        Object.values(players).forEach(p => bindMasterSync(p));
        bindFrameDrift(players[firstCam]);
        loadFrameData(episodeId, loadToken).then(async () => {
            if (!isCurrentPlaybackSession(loadEpoch, loadToken)) return;  // stale
            startHeatmapSync(players[firstCam]);
            // Now FPS + frameCount are correct — safe to render annotation overlay
            if (typeof loadAnnotations === 'function') {
                await loadAnnotations(episodeId).catch(() => {});
            }
            if (!isCurrentPlaybackSession(loadEpoch, loadToken)) return;
            // 全部就绪才放开控件/揭遮罩(与 grouped 路径同一门控)
            await _awaitPlayersReady();
            if (!isCurrentPlaybackSession(loadEpoch, loadToken)) return;
            enableFrameControls();
            hideVideoLoading();
        });
    }

    // Update info bar
    updateInfoBar();
}


function switchTab(camera) {
    // Highlight the active tab
    const tabs = document.getElementById('camera-tabs');
    if (tabs) {
        tabs.querySelectorAll('button').forEach(btn => {
            if (btn.textContent === camera) {
                btn.className = 'px-3 py-1 text-sm rounded-t bg-blue-600 text-white';
            } else {
                btn.className = 'px-3 py-1 text-sm rounded-t bg-gray-800 text-gray-400 hover:bg-gray-700';
            }
        });
    }

    // Show only selected camera
    const grid = document.getElementById('video-grid');
    if (grid) {
        grid.className = 'flex-1 min-w-0 min-h-0 p-3 overflow-hidden grid grid-cols-1 gap-3';
        if (typeof destroyAllHandOverlays === 'function') destroyAllHandOverlays();
        grid.innerHTML = '';
        const oldContainer = document.getElementById(`player-container-${camera}`);
        const div = document.createElement('div');
        div.id = `player-container-${camera}`;
        div.className = 'bg-black rounded overflow-hidden min-w-0 min-h-0 w-full h-full';
        grid.appendChild(div);

        // Re-init player if needed
        if (players[camera]) {
            players[camera].destroy();
        }
        pickVideoUrl(currentEpisodeId, camera).then(url => {
            players[camera] = initPlayer(`player-container-${camera}`, url);
            if (players[camera] && typeof initHandOverlay === 'function') {
                const source = { kind: 'video', source_key: camera, label: camera };
                initHandOverlay(source, div, div);
            }
            // Re-bind heatmap/frame sync to the new player instance
            Object.values(players).forEach(p => bindMasterSync(p));
            bindFrameDrift(players[camera]);
            startHeatmapSync(players[camera]);
            // Re-render annotation overlay on the rebuilt player
            if (typeof annotations !== 'undefined' && annotations.length > 0) {
                renderAnnotationOverlay(annotations, episodeTotalFrames);
            }
        });
    }
}


function playAll() {
    // 已播完(ended)后再次点击:HTML5 在 ended 状态 play() 是 no-op,
    // 必须先把所有播放器 seek 回第 0 帧才能从头重播。
    const master = getActivePlayer();
    if (master) {
        const dur = master.duration || 0;
        if (master.ended || (dur > 0 && (master.currentTime || 0) >= dur - 0.1)) {
            seekToFrame(0);
        }
    }
    // HTMLMediaElement.play() on several videos is not a barrier: each
    // element can start on a different refresh tick. Snap the complete set
    // to the master's integer frame before starting, so RGB, overlays and
    // the display-only 3D canvas all share one time origin. This does not
    // touch the stored parquet/video data.
    const syncMaster = getActivePlayer();
    if (syncMaster && !syncMaster.ended) {
        const frame = _clampFrameIndex(Math.floor(
            (syncMaster.currentTime || 0) * getEpisodeFps() + 0.002));
        const time = frame / getEpisodeFps();
        currentFrameTarget = frame;
        Object.values(players).forEach(p => {
            try {
                if (Math.abs((p.currentTime || 0) - time) > 0.001) {
                    p.currentTime = time;
                }
            } catch (e) {}
        });
        if (_centralLastFrame !== frame) {
            _centralLastFrame = frame;
            updateFrameDisplay(frame);
        }
    }
    Object.values(players).forEach(p => {
        try { p.play(); } catch(e) {}
    });
    // play() enters the native media state asynchronously. Arm the
    // presentation clock on the next task so the first displayed RGB frame
    // also becomes the first frame for depth/3D/sensor layers.
    setTimeout(_ensurePresentationClock, 0);
}


function pauseAll() {
    Object.values(players).forEach(p => {
        try { p.pause(); } catch(e) {}
    });
}

/* ── Frame-aligned master control ────────────────────────
   One play/pause button drives every video on the canvas.
   The master player's timeupdate forces slaves to the same
   time (±50ms drift correction); the master seek bar seeks
   all videos to the exact same frame.                     */

let masterPlaying = false;
const _syncedPlayers = new WeakSet();   // play/pause broadcast bound
const _driftBound = new WeakSet();      // frame-drift correction bound (master only)

function refreshPlayButton(playing) {
    const btn = document.getElementById('btn-play-all');
    if (!btn) return;
    const icon = btn.querySelector('[data-play-icon]');
    const label = btn.querySelector('.play-label');
    if (icon) {
        icon.setAttribute('icon', playing
            ? 'ant-design:pause-circle-filled'
            : 'ant-design:caret-right-filled');
    }
    if (label) label.textContent = playing ? t('pause_btn') : t('play_btn');
    btn.setAttribute('aria-pressed', String(Boolean(playing)));
}

function _lastValidFrame() {
    return Math.max(0, (episodeTotalFrames || 0) - 1);
}

function _clampFrameIndex(frameIndex) {
    const numeric = Number(frameIndex);
    const safeFrame = Number.isFinite(numeric) ? Math.floor(numeric) : 0;
    return episodeTotalFrames > 0
        ? Math.max(0, Math.min(_lastValidFrame(), safeFrame))
        : Math.max(0, safeFrame);
}

function stopPlaybackAtLastFrame() {
    cancelSegmentPlayback();
    const lastFrame = _lastValidFrame();
    masterPlaying = false;
    // Prevent a pause event from one view from starting another pause cascade
    // while all synchronized views are being snapped together.
    _suppressMasterEvents = true;
    pauseAll();
    _suppressMasterEvents = false;
    seekToFrame(lastFrame);
    updateFrameDisplay(lastFrame);
    refreshPlayButton(false);
}

function togglePlayAll() {
    const master = getActivePlayer();
    if (!master) {
        // 诊断:打印具体原因到控制台,不再"静默没反应"
        console.warn('[player] 播放不可用:无活跃播放器',
            { players: Object.keys(players), workspaceSources:
              workspaceSources.map(s => groupedSourceKey(s)) });
        return;
    }
    // 只挡"播放器未就绪"。帧数据缺失(如 frames-data 接口瞬时失败)不
    // 挡播放:按时间同步不依赖 fps,图像 tile 短暂用默认 30fps 可接受;
    // 若用 frameDataReady 一起挡,接口抖动会导致播放按钮"点了没反应"。
    if (!playerReady) {
        console.warn('[player] 播放不可用:播放器未就绪(ready 事件未触发)');
        return;
    }
    cancelSegmentPlayback();
    if (masterPlaying) {
        pauseAll();
    } else {
        // Set the broadcast lock before calling play(). Otherwise the first
        // native play event sees masterPlaying=false and recursively calls
        // playAll(), which can seek the video backwards by one frame/time tick.
        masterPlaying = true;
        playAll();
    }
    // button state follows the players' play/pause events
}

/* Bind play/pause broadcast to ANY player — clicking the play button on
   any view (left, right, aux) plays/pauses the whole frame-aligned set. */
function bindMasterSync(player) {
    if (!player || _syncedPlayers.has(player)) return;
    _syncedPlayers.add(player);
    player.on('play', () => {
        if (_suppressMasterEvents) return;
        if (!masterPlaying) {
            masterPlaying = true;    // broadcast lock (prevents re-entry)
            playAll();
        }
        refreshPlayButton(true);
    });
    player.on('pause', () => {
        if (_suppressMasterEvents) return;
        if (masterPlaying) {
            masterPlaying = false;   // broadcast lock
            pauseAll();
        }
        refreshPlayButton(false);
    });
    player.on('ended', () => {
        if (_suppressMasterEvents) return;
        stopPlaybackAtLastFrame();
    });
}

/* Frame-drift correction on the master player: while playing, every slave
   is snapped back within half a frame; on pause all slaves are snapped to
   the exact master time so every view shows the identical frame. */
function bindFrameDrift(master) {
    if (!master || _driftBound.has(master)) return;
    _driftBound.add(master);
    master.on('timeupdate', () => {
        if (!masterPlaying || _frameScrubbing) return;
        const t = master.currentTime || 0;
        const maxDrift = Math.max(0.08, 2.5 / getEpisodeFps());
        Object.values(players).forEach(p => {
            if (p === master) return;
            try {
                if (Math.abs(p.currentTime - t) > maxDrift) p.currentTime = t;
            } catch (e) {}
        });
    });
    master.on('pause', () => {
        const t = master.currentTime || 0;
        Object.values(players).forEach(p => {
            if (p === master) return;
            try {
                if (Math.abs(p.currentTime - t) > 0.005) p.currentTime = t;
            } catch (e) {}
        });
    });
}


function updateInfoBar() {
    const bar = document.getElementById('episode-info');
    if (!bar) return;
    // 顶部信息条:组名 / fps / 帧数 / 同步状态
    let groupText = '';
    if (currentMediaGroups) {
        groupText = (currentMediaGroups.groups || []).map(g => g.label).join(' / ');
        if (!groupText && (currentMediaGroups.singles || []).length) groupText = t('single_mono');
    }
    groupText = groupText || (currentCameras.length > 0 ? currentCameras.length + ' Cameras' : '');
    const depthMissing = currentMediaGroups
        ? (currentMediaGroups.sources || [])
            .filter(source => source.kind === 'depth')
            .reduce((total, source) => total + (source.missing_frames || []).length, 0)
        : 0;
    const syncMarkup = depthMissing > 0
        ? `<span class="text-yellow-400" title="${depthMissing} depth frames are missing"><iconify-icon icon="ant-design:warning-outlined" class="icon-sm"></iconify-icon> Depth missing ${depthMissing}</span>`
        : `<span class="text-green-400"><iconify-icon icon="ant-design:sync-outlined" class="icon-sm"></iconify-icon> ${t('synced_ok')}</span>`;
    bar.innerHTML = `
        <span class="font-medium text-gray-200">${groupText}</span>
        <span class="text-gray-600">|</span>
        <span class="font-mono">${getEpisodeFps()} FPS · ${episodeTotalFrames || 0} ${t('frame')}</span>
        <span class="text-gray-600">|</span>
        ${syncMarkup}
    `;
    bar.classList.remove('hidden');
    bar.classList.add('flex');
    updateEpisodeDetailSyncStatus();
}

function updateEpisodeDetailSyncStatus() {
    const el = document.getElementById('episode-detail-sync-status');
    if (!el) return;

    const depthSources = (currentMediaGroups?.sources || [])
        .filter(source => source.kind === 'depth');
    if (!depthSources.length) {
        el.classList.add('hidden');
        el.innerHTML = '';
        return;
    }

    const missingFrames = [...new Set(depthSources.flatMap(source => source.missing_frames || []))]
        .sort((a, b) => a - b);
    if (!missingFrames.length) {
        el.className = 'flex items-center gap-1 text-green-400';
        el.innerHTML = '<iconify-icon icon="ant-design:check-circle-outlined" class="icon-sm"></iconify-icon> Depth data complete';
        return;
    }

    const preview = missingFrames.slice(0, 4).join(', ');
    const suffix = missingFrames.length > 4 ? ', …' : '';
    el.className = 'flex items-start gap-1 text-yellow-400';
    el.innerHTML = `<iconify-icon icon="ant-design:warning-outlined" class="icon-sm mt-0.5"></iconify-icon>
        <span>Depth data missing ${missingFrames.length} frame${missingFrames.length === 1 ? '' : 's'}
        <span class="text-yellow-500/80">(${preview}${suffix})</span></span>`;
}


// ═══════════════════════════════════════════════════════════
// Frame-level controls
// ═══════════════════════════════════════════════════════════

const _frameChangeCallbacks = new Set();
let episodeFps = 30;
let episodeTotalFrames = 0;

function _syncFrameLabels(frameIndex) {
    const frame = Math.max(0, Math.floor(Number(frameIndex) || 0));
    const detailEl = document.getElementById('detail-current-frame');
    if (detailEl) detailEl.textContent = frame;
    const heatmapEl = document.getElementById('heatmap-frame');
    if (heatmapEl) heatmapEl.textContent = frame;
    const timelineEl = document.getElementById('annotation-timeline-frame');
    if (timelineEl) {
        timelineEl.textContent = episodeTotalFrames > 0
            ? frame + ' / ' + episodeTotalFrames : '0 / 0';
    }
}

function _resetFrameUiForEpisode(frameIndex = 0, totalFrames = 0) {
    currentFrameTarget = Math.max(0, Math.floor(Number(frameIndex) || 0));
    _lastDisplayedFrame = -1;
    const safeTotal = Math.max(0, Math.floor(Number(totalFrames) || 0));
    const detailTotal = document.getElementById('detail-total-frames');
    if (detailTotal && safeTotal > 0) detailTotal.textContent = safeTotal;
    if (typeof resetAnnotationTimelineForEpisode === 'function') {
        resetAnnotationTimelineForEpisode(safeTotal);
    }
    _syncFrameLabels(currentFrameTarget);
}

function setEpisodeFrameInfo(fps, totalFrames) {
    const prevFps = episodeFps;
    episodeFps = fps || 30;
    episodeTotalFrames = Math.max(0, Number(totalFrames) || 0);
    frameDataReady = true;
    if (episodeTotalFrames > 0) {
        currentFrameTarget = _clampFrameIndex(currentFrameTarget);
        const detailTotal = document.getElementById('detail-total-frames');
        if (detailTotal) detailTotal.textContent = episodeTotalFrames;
    }
    _syncFrameLabels(currentFrameTarget);
    updateInfoBar();
    // fps 变化且已有非零目标帧 → 重对齐:此前用旧 fps 换算的 seek 停在
    // 错误时间(loadFrameData 返回前点过 step/skip 的兜底修复)
    if (currentFrameTarget > 0 && episodeFps !== prevFps && !_frameScrubbing) {
        Object.values(players).forEach(_setPlayerToCurrentFrame);
    }
}

function setOnFrameChange(fn) {
    /* 多回调注册(标注高亮/手部骨骼叠加层各自挂一个,互不覆盖) */
    if (typeof fn === 'function') _frameChangeCallbacks.add(fn);
}

let currentFrameTarget = 0;  // integer frame index, avoids float error
let _pendingSeekFrame = null;  // target frame for in-flight seek, consumed by seeked handler
let _lastDisplayedFrame = -1;  // guard against redundant DOM updates on unchanged frame
let _frameScrubbing = false;
let _frameScrubWasPlaying = false;

function isFrameScrubbing() {
    return _frameScrubbing;
}

function beginFrameScrub() {
    if (_frameScrubbing) return _frameScrubWasPlaying;
    _frameScrubWasPlaying = masterPlaying;
    _frameScrubbing = true;

    // Pause the synchronized set while the pointer previews frames. This
    // prevents drift correction from competing with the user's drag.
    if (_frameScrubWasPlaying) {
        masterPlaying = false;
        _suppressMasterEvents = true;
        pauseAll();
        _suppressMasterEvents = false;
        refreshPlayButton(false);
    }
    return _frameScrubWasPlaying;
}

function previewFrame(frameIndex) {
    currentFrameTarget = _clampFrameIndex(frameIndex);
    updateFrameDisplay(currentFrameTarget);
    _refreshDepthTilesAt(currentFrameTarget);
}

function commitFrameScrub(frameIndex) {
    const resume = _frameScrubWasPlaying;
    _frameScrubbing = false;
    currentFrameTarget = _clampFrameIndex(frameIndex);
    const master = getActivePlayer();
    if (!master) {
        seekToFrame(currentFrameTarget);
        return;
    }
    if (!resume) {
        seekToFrame(currentFrameTarget);
        return;
    }
    const resumeAfterSeek = () => {
        master.off('seeked', resumeAfterSeek);
        masterPlaying = true;
        playAll();
    };
    master.on('seeked', resumeAfterSeek);
    seekToFrame(currentFrameTarget);
    // Some media backends do not emit `seeked` when the target time is
    // already buffered/current. Keep the resumed-play behavior deterministic.
    setTimeout(() => {
        if (!_frameScrubbing && _frameScrubWasPlaying && !masterPlaying) {
            master.off('seeked', resumeAfterSeek);
            masterPlaying = true;
            playAll();
        }
    }, 120);
}

function getActivePlayer() {
    for (const source of workspaceSources) {
        if (players[groupedSourceKey(source)]) return players[groupedSourceKey(source)];
    }
    for (const cam of currentCameras) {
        if (players[cam]) return players[cam];
    }
    return null;
}

function getCurrentFrame() {
    // During stepping, use the tracked integer target to avoid float precision loss.
    // During playback, timeupdate will sync currentFrameTarget from actual time.
    return currentFrameTarget;
}

function getCurrentTime() {
    const p = getActivePlayer();
    if (!p) return 0;
    return p.currentTime || 0;
}

function getEpisodeFps() {
    /* Single source of truth for FPS — use this everywhere instead of a local copy. */
    return episodeFps || 30;
}

function consumeSeekTarget() {
    /* Return the exact target frame from the most recent seekToFrame call, then clear it.
       This avoids the round-trip error where seeked recalculates a different frame
       from currentTime due to browser seek imprecision / millisecond truncation. */
    if (_pendingSeekFrame != null) {
        const f = _pendingSeekFrame;
        _pendingSeekFrame = null;
        return f;
    }
    return null;
}

function seekToFrame(frameIndex) {
    frameIndex = _clampFrameIndex(frameIndex);
    _pendingSeekFrame = frameIndex;  // remember exact target for seeked handler
    currentFrameTarget = frameIndex;
    // 不提前返回:播放器尚未挂载时目标帧已记录在 currentFrameTarget,
    // 新播放器挂载(_setPlayerToCurrentFrame)或就绪后会自动补 seek,
    // 避免"刚切换视频时点切片静默失败"。
    const time = frameIndex / (episodeFps || 30);
    Object.values(players).forEach(player => {
        try { player.currentTime = Math.max(0, time); } catch (e) {}
    });
}

// ── Annotation segment auto-play ──────────────────────

let _segmentPlayback = null;
let _segAutoStopHandler = null; // compatibility for stale callers

function cancelSegmentPlayback() {
    const session = _segmentPlayback;
    if (!session) return;
    _segmentPlayback = null;  // 先清引用,off 失败也不影响后续段播放
    if (session.player && session.handler) {
        // 会话的播放器可能已在切批次时被 destroy(Plyr 销毁后其内部
        // container 为 null,off() 会抛 TypeError)—— 兜底吞掉。
        try { session.player.off('timeupdate', session.handler); } catch (e) {}
    }
    if (session.player && session.startPlayback) {
        try { session.player.off('seeked', session.startPlayback); } catch (e) {}
    }
}

function legacyPlaySegment(startFrame, endFrame) {
    /* Legacy implementation kept only for compatibility with old callers. */
    const p = getActivePlayer();
    if (!p) return;

    // Remove any previous auto-stop handler(播放器可能已被销毁,兜底吞掉)
    if (_segAutoStopHandler) {
        try { p.off('timeupdate', _segAutoStopHandler); } catch (e) {}
        _segAutoStopHandler = null;
    }

    // Seek to start and play
    seekToFrame(startFrame);
    p.play().catch(() => {});

    // Register one-shot auto-pause after endFrame
    _segAutoStopHandler = () => {
        const cf = Math.floor(p.currentTime * getEpisodeFps() + 0.002);
        if (cf >= endFrame) {
            p.pause();
            seekToFrame(endFrame);  // snap to exact endFrame — timeupdate may overshoot
            p.off('timeupdate', _segAutoStopHandler);
            _segAutoStopHandler = null;
        }
    };
    p.on('timeupdate', _segAutoStopHandler);
}

// Final segment-playback implementation. It intentionally sits after the
// legacy handler above so older cached markup cannot bypass synchronized
// start/end handling.
function playSegment(startFrame, endFrame) {
    const p = getActivePlayer();
    if (!p) return;

    const start = _clampFrameIndex(startFrame);
    const end = Math.max(start, Math.min(_lastValidFrame(), Math.floor(Number(endFrame) || start)));
    cancelSegmentPlayback();

    const session = {
        player: p,
        endFrame: end,
        started: false,
        handler: null,
        startPlayback: null,
    };
    _segmentPlayback = session;

    const finish = () => {
        if (_segmentPlayback !== session) return;
        cancelSegmentPlayback();
        masterPlaying = false;
        _suppressMasterEvents = true;
        pauseAll();
        _suppressMasterEvents = false;
        seekToFrame(end);
        updateFrameDisplay(end);
        refreshPlayButton(false);
    };

    session.handler = () => {
        if (_segmentPlayback !== session || !session.started) return;
        const frame = Math.floor((p.currentTime || 0) * getEpisodeFps() + 0.002);
        if (frame >= end) finish();
    };
    session.startPlayback = () => {
        if (_segmentPlayback !== session || session.started) return;
        session.started = true;
        p.off('seeked', session.startPlayback);
        masterPlaying = true;
        playAll();
    };

    p.on('timeupdate', session.handler);
    p.on('seeked', session.startPlayback);
    masterPlaying = false;
    _suppressMasterEvents = true;
    pauseAll();
    _suppressMasterEvents = false;
    seekToFrame(start);
    // Buffered media can seek before emitting `seeked`.
    setTimeout(session.startPlayback, 120);
}

function stepFrame(delta) {
    const p = getActivePlayer();
    if (!p) return;
    p.pause();
    // Integer arithmetic — no float precision loss
    const newFrame = Math.max(0, Math.min(episodeTotalFrames - 1, currentFrameTarget + delta));
    seekToFrame(newFrame);
}

function skipFrames(delta) {
    stepFrame(delta);
}

// Update frame display in UI
function updateFrameDisplay(frameIndex) {
    // A media element may report its duration boundary as one frame past the
    // last actual image. Keep every UI surface on the valid frame range.
    frameIndex = _clampFrameIndex(frameIndex);
    // Skip if frame hasn't changed — prevents redundant DOM writes
    // that cause visual jitter on every timeupdate tick (~4 Hz)
    // but still refresh the timeline cursor. The timeline can be rebuilt after
    // the frame clock has already reached 0 (episode switch / annotation load),
    // so an early return must not leave the green cursor at the old position.
    if (frameIndex === _lastDisplayedFrame) {
        _syncFrameLabels(frameIndex);
        if (typeof updateAnnotationTimelineCursor === 'function') {
            updateAnnotationTimelineCursor(frameIndex);
        }
        return;
    }
    _lastDisplayedFrame = frameIndex;

    currentFrameTarget = frameIndex;
    _syncFrameLabels(frameIndex);
    const displayEl = document.getElementById('current-frame-display');
    if (displayEl) displayEl.textContent = 'Frame ' + frameIndex;

    _frameChangeCallbacks.forEach(fn => {
        try { fn(frameIndex); } catch (e) { /* 单个回调异常不影响其他 */ }
    });
}


// ═══════════════════════════════════════════════════════════
// Annotation overlay on Plyr progress bar
// ═══════════════════════════════════════════════════════════

function renderAnnotationOverlay(annotations, totalFrames) {
    // Remove existing overlays from ALL players (not just first)
    document.querySelectorAll('.anno-progress-overlay').forEach(el => el.remove());

    if (!annotations || annotations.length === 0) return;
    if (!totalFrames || totalFrames <= 0) return;  // guard: frame data not yet loaded

    const maxFrame = totalFrames || (annotations.length > 0
        ? Math.max(...annotations.map(a => a.end_frame_index))
        : 1);
    if (maxFrame <= 0) return;

    // Build segment elements once, clone into each player's progress bar
    const buildSegments = () => annotations.map(seg => {
        const leftPct = (seg.start_frame_index / maxFrame) * 100;
        const widthPct = ((seg.end_frame_index - seg.start_frame_index + 1) / maxFrame) * 100;
        if (widthPct <= 0) return null;

        const segEl = document.createElement('div');
        segEl.style.cssText = `
            position: absolute; left: ${leftPct}%; width: ${widthPct}%;
            top: 0; height: 100%; background: ${seg.color || '#3B82F6'};
            opacity: 0.55; pointer-events: auto; cursor: pointer;
            transition: opacity 0.15s;
        `;
        segEl.title = seg.label + ' (frame ' + seg.start_frame_index + '–' + seg.end_frame_index + ')';
        segEl.onmouseenter = () => { segEl.style.opacity = '0.85'; };
        segEl.onmouseleave = () => { segEl.style.opacity = '0.55'; };
        segEl.onclick = (e) => {
            e.stopPropagation();
            seekToFrame(seg.start_frame_index);
        };
        return segEl;
    }).filter(Boolean);

    // Apply overlay to every player's progress bar (multi-camera support)
    document.querySelectorAll('.plyr__progress').forEach(progressContainer => {
        const overlay = document.createElement('div');
        overlay.className = 'anno-progress-overlay';
        overlay.style.cssText = `
            position: absolute; top: 0; left: 0; width: 100%; height: 100%;
            pointer-events: none; z-index: 1; display: flex; border-radius: inherit;
            overflow: hidden;
        `;
        buildSegments().forEach(segEl => overlay.appendChild(segEl));

        const parentStyle = window.getComputedStyle(progressContainer);
        if (parentStyle.position === 'static') {
            progressContainer.style.position = 'relative';
        }
        progressContainer.appendChild(overlay);
    });
}

function clearAnnotationOverlay() {
    document.querySelectorAll('.anno-progress-overlay').forEach(el => el.remove());
}


// ═══════════════════════════════════════════════════════════
// Player ready state — controls enabled only after video loaded
// ═══════════════════════════════════════════════════════════

let playerReady = false;
let frameDataReady = false;  // fps/总帧数已就绪;未就绪时禁止 seek/播放(旧 fps 是脏值)
let _controlsEnabled = false;  // 控件当前是否已启用(自愈看门狗用)
let _loadWatchdog = null;
let _watchdogRetried = false;  // 自愈重渲染只试一次,防循环

function onPlayerReady() {
    playerReady = true;
    // 帧数据就绪前不放行控件:此时 fps 可能还是旧批次的脏值,
    // 提前放行会让 step/skip/播放用错误 fps 换算 → 帧不对齐。
    // 控件最终由渲染完成链(renderGroupedWorkspace/loadEpisodeVideo)启用。
    if (frameDataReady) enableFrameControls();
}

function disableFrameControls() {
    playerReady = false;
    _controlsEnabled = false;
    const controls = document.getElementById('frame-controls');
    if (controls) controls.classList.add('hidden');
    const btns = ['btn-skip-back-10', 'btn-step-back', 'btn-play-all', 'btn-step-forward', 'btn-skip-forward-10'];
    btns.forEach(id => {
        const el = document.getElementById(id);
        if (el) { el.disabled = true; el.classList.add('opacity-40', 'cursor-not-allowed'); }
    });
}

function enableFrameControls() {
    playerReady = true;
    _controlsEnabled = true;
    const controls = document.getElementById('frame-controls');
    if (controls) controls.classList.remove('hidden');
    const track = document.getElementById('annotation-timeline-track');
    if (track && typeof bindFrameTimelineSeek === 'function') {
        bindFrameTimelineSeek(track, episodeTotalFrames || 1);
    }
    const btns = ['btn-skip-back-10', 'btn-step-back', 'btn-play-all', 'btn-step-forward', 'btn-skip-forward-10'];
    btns.forEach(id => {
        const el = document.getElementById(id);
        if (el) { el.disabled = false; el.classList.remove('opacity-40', 'cursor-not-allowed'); }
    });
    // frame-controls 刚显示,其高度已参与工具条偏移 —— 重新定位播放工具条,
    // 否则工具条会按 controlsH=0 下沉,压到标注时间轴/切片块上
    if (typeof _positionPlaybackToolbar === 'function') {
        const markers = document.getElementById('annotation-timeline-markers');
        _positionPlaybackToolbar(markers ? markers.offsetHeight : 0);
    }
}

// Override stepFrame / seekToFrame to guard against unready state
const _origStepFrame = stepFrame;
stepFrame = function(delta) {
    if (!playerReady || !frameDataReady) return;
    _origStepFrame(delta);
};

const _origSeekToFrame = seekToFrame;
seekToFrame = function(frameIndex) {
    if (!playerReady || !frameDataReady) return;
    _origSeekToFrame(frameIndex);
};

/* ── 加载遮罩 + 就绪门控 ─────────────────────────────
   批次打开到「媒体组 + 全部播放器 + 帧数据 + 标注」就绪前,
   画布盖遮罩、控件禁用,防止用旧批次 fps 提前交互。        */
function showVideoLoading() {
    const ov = document.getElementById('video-loading-overlay');
    if (ov) ov.classList.remove('hidden');
}

function hideVideoLoading() {
    const ov = document.getElementById('video-loading-overlay');
    if (ov) ov.classList.add('hidden');
}

function _awaitPlayersReady(timeoutMs = 4000) {
    /* 等待当前全部播放器 ready(元数据可 seek)。已 ready 的立即放行,
       未 ready 等事件;单播放器与整体都有超时兜底,防止遮罩卡死。 */
    const list = Object.values(players);
    if (!list.length) return Promise.resolve();
    const checks = list.map(player => new Promise(resolve => {
        let done = false;
        const finish = () => { if (!done) { done = true; resolve(); } };
        if (player.ready) { finish(); return; }
        try {
            if (typeof player.once === 'function') player.once('ready', finish);
            else if (typeof player.on === 'function') player.on('ready', finish);
        } catch (e) { finish(); }
        setTimeout(finish, 2000);
    }));
    return Promise.race([
        Promise.all(checks),
        new Promise(resolve => setTimeout(resolve, timeoutMs)),
    ]);
}

async function _awaitDepthInitialReady(episodeId, timeoutMs = 20000) {
    const entries = currentDepthPreviewTiles.slice();
    if (!entries.length) return true;
    const deadline = performance.now() + timeoutMs;
    while (currentEpisodeId === episodeId && performance.now() < deadline) {
        const ready = entries.every(entry =>
            (Number(entry.source?.frame_count) > DEPTH_VERY_LONG_FRAMES
                ? (entry.initialBufferReady && entry.frameCount > 0)
                : (entry.allPreloaded && entry.frameCount > 0)) ||
            (entry.fullReady && entry.frameCount > 0) ||
            (Number(entry.source?.frame_count) <= 0 && entry.frames.has(0)) ||
            (entry.fullCodes && entry.frameCount > 0));
        if (ready) return true;
        await new Promise(resolve => setTimeout(resolve, 50));
    }
    return currentEpisodeId === episodeId && entries.every(entry =>
        (Number(entry.source?.frame_count) > DEPTH_VERY_LONG_FRAMES
            ? (entry.initialBufferReady && entry.frameCount > 0)
            : (entry.allPreloaded && entry.frameCount > 0)) ||
        (entry.fullReady && entry.frameCount > 0) ||
        (Number(entry.source?.frame_count) <= 0 && entry.frames.has(0)) ||
        (entry.fullCodes && entry.frameCount > 0));
}

/* ══ 手部骨骼 3D 视图 ══════════════════════════════════
   数据来自 /api/v1/video/{episode}/hand-3d(hand_3d 产物,
   可能是相对 3D 或相机米制坐标)。Canvas 透视投影,可拖拽旋转,
   随播放帧同步;手 0 = 青色,手 1 = 橙色,label 标记左右。        */

// MediaPipe 21 点连接拓扑(官方 HAND_CONNECTIONS)
const HAND3D_CONNECTIONS = [
    [0, 1], [1, 2], [2, 3], [3, 4],                 // 拇指
    [0, 5], [5, 6], [6, 7], [7, 8],                 // 食指
    [5, 9], [9, 10], [10, 11], [11, 12],            // 中指
    [9, 13], [13, 14], [14, 15], [15, 16],          // 无名指
    [13, 17], [17, 18], [18, 19], [19, 20],         // 小指
    [0, 17],                                        // 掌根
];

async function loadHand3D(episodeId, sessionToken = null) {
    try {
        const signal = getMediaLoadSignal();
        const res = await fetch(`/api/v1/video/${episodeId}/hand-3d`, signal ? { signal } : {});
        if (!isCurrentPlaybackSession(episodeId, sessionToken)) return;  // 丢弃过期响应
        if (!res.ok) return;
        const data = await res.json();
        if (!isCurrentPlaybackSession(episodeId, sessionToken)) return;
        if (!data) return;
        const availableOptions = data.available_sources && data.available_sources.length
            ? data.available_sources
            : [{ ...data, source_key: data.source_key || 'hand3d' }];
        // Stereo produces one 3D sequence per RGB view.  The server scores
        // valid finite landmarks/frames and exposes the winning source.  The
        // spatial preview intentionally mounts only that source; hand_0 and
        // hand_1 inside it remain available, so both physical hands are kept.
        const selectedKey = data.selected_source_key || data.source_key;
        const selectedOption = availableOptions.find(option =>
            option.source_key === selectedKey);
        const options = selectedOption ? [selectedOption] : [
            [...availableOptions].sort((a, b) =>
                (Number(b.valid_landmark_points || 0)
                 - Number(a.valid_landmark_points || 0))
                || (Number(b.valid_hand_frames || 0)
                    - Number(a.valid_hand_frames || 0))
                || (a.source_key === 'stereo_left' ? -1 : 1)
            )[0],
        ];
        hand3dDataBySource = {};
        options.forEach(option => {
            const item = { ...data, ...option };
            const relativeMode = item.unit === 'mediapipe_world_relative'
                || String(item.mode || '').toLowerCase().includes('relative');
            item.relativeMode = relativeMode;
            item.worldMode = item.world_preview === true
                && item.depth_available === true
                && item.source === 'depth_camera_meters'
                && item.unit === 'camera_meters';
            item.rgbEstimatedMode = item.rgb_estimated_preview === true
                || item.unit === 'rgb_estimated_meters'
                || String(item.mode || '').toLowerCase() === 'rgb_estimated_3d';
            item.spaceMode = item.worldMode || item.rgbEstimatedMode;
            // The depth lift stores OpenCV camera coordinates (Y down), while
            // RGBWorldTracker stores display coordinates (Y up). Normalize
            // only the depth path before projection; do not flip RGB points.
            const frameName = String(item.coordinate_frame || '').toLowerCase();
            item.cameraYDown = item.source === 'depth_camera_meters'
                && !frameName.includes('y_up');
            item.hasPreview = item.spaceMode;
            hand3dDataBySource[item.source_key] = item;
        });
        const firstSource = options[0].source_key || data.source_key || 'hand3d';
        hand3dData = hand3dDataBySource[firstSource] || data;
        // Mounting is intentionally done by loadGroupedEpisodeVideo after
        // media-groups is ready. That keeps this metadata request from
        // creating a late tile outside the initial loading barrier.
        if (typeof updatePreviewMenuData === 'function') updatePreviewMenuData();
        return hand3dData;
    } catch (e) { /* 无 3D 产物 → 不显示 3D 开关 */
        return null;
    }
}

/* ☷ "3D Hand World" 开关联动:开启且工作区未挂载时自动加 3D canvas 窗口。
   幂等性校验必须落到 DOM:源在 workspaceSources 但 tile 被重建冲掉时
   (加载竞态),列表存在 ≠ 窗口存在 —— 会造成"开关显示开实际关"。 */
window.ensureHand3dWorldTile = function () {
    const sources = Object.values(hand3dDataBySource || {})
        .filter(item => item && item.hasPreview);
    if (!sources.length && hand3dData && hand3dData.hasPreview) sources.push(hand3dData);
    sources.forEach(item => {
        const sourceKey = item.source_key || 'hand3d';
        const src = {
            kind: 'hand3d_world',
            source_key: sourceKey,
            label: `${item.device_name || sourceKey} · ${item.rgbEstimatedMode
                ? 'RGB Estimated 3D' : t('hand3d_world')}`,
        };
        const key = groupedSourceKey(src);
        const present = workspaceSources.some(s => groupedSourceKey(s) === key);
        if (present) {
            const tile = workspaceGrid
                ? workspaceGrid.querySelector(`[data-source-key="${CSS.escape(key)}"]`)
                : null;
            if (tile) return;
            workspaceSources = workspaceSources.filter(s => groupedSourceKey(s) !== key);
        }
        // During the first episode load the metadata can finish before the
        // workspace DOM exists. Queue the source in the source list; mounting
        // it here would recursively render an incomplete workspace.
        if (!workspaceGrid || !workspaceMainRow) {
            workspaceSources.push(src);
            return;
        }
        addGroupedSource(src);
    });
};

/* ☷ 开关关闭 → 从 workspaceSources 卸除 source(复用 removeGroupedSource
   的完整清理:销毁 Plyr、删 tile、布局回收)。display:none 只藏 tile,
   _refreshWorkspaceLayout 仍按源列表预留 260px 底部条 = "位置被占用",
   且再开时无可见变化 —— 必须卸源。再开由 ensure*Tile 重新挂载。 */
window.removeHand3dWorldTile = function () {
    workspaceSources.filter(s => s.kind === 'hand3d_world').forEach(src => {
        removeGroupedSource(src);
    });
    currentHand3dTiles = currentHand3dTiles.filter(
        e => e.source.kind !== 'hand3d_world');
};
window.removeDepthTile = function () {
    workspaceSources.filter(s => s.kind === 'depth').forEach(src => {
        removeGroupedSource(src);
    });
};

/* ☷ 开关统一入口(hand-overlay.js _applyTileVisibility 调用):
   on → 自动挂载(幂等);off → 卸除 source(布局回收)。 */
window.setHand3dWorldVisible = function (on) {
    if (on) window.ensureHand3dWorldTile();
    else window.removeHand3dWorldTile();
};
window.setDepthTileVisible = function (on) {
    if (on) window.ensureDepthTile();
    else window.removeDepthTile();
};

/* ☷ 深度开关联动:开启且批次有深度素材 → 自动挂载深度视频 tile
   (与 3D Hand World 同机制:素材栏不再提供深度按钮,唯一入口是 ☷)。
   同一幂等校验:源在列表但 DOM 无 tile 时重建。 */
window.ensureDepthTile = function () {
    const sources = (typeof currentMediaGroups !== 'undefined' && currentMediaGroups)
        ? (currentMediaGroups.sources || []) : [];
    const depths = sources.filter(s => s.kind === 'depth');
    depths.forEach(depth => {
        const src = _serializeSource(depth);
        const key = groupedSourceKey(src);
        const present = workspaceSources.some(s => groupedSourceKey(s) === key);
        if (present) {
            const tile = workspaceGrid
                ? workspaceGrid.querySelector(`[data-source-key="${CSS.escape(key)}"]`)
                : null;
            if (tile) return;
            workspaceSources = workspaceSources.filter(s => groupedSourceKey(s) !== key);
        }
        addGroupedSource(src);
    });
};

/* 按帧懒加载 3D 手部点:播放/拖动到哪一帧就取哪一帧(全量返回在
   60 分钟级批次上达 226MB/35s,会把页面拖死) */
async function fetchHand3DFrame(frame, sourceKey) {
    const source = sourceKey || 'default';
    const cache = _hand3dFrameCacheFor(source);
    if (!currentEpisodeId || cache.inflight === frame) return;
    const episodeAtRequest = currentEpisodeId;
    const sessionToken = _playbackSessionToken;
    const signal = getMediaLoadSignal();
    cache.inflight = frame;
    try {
        const res = await fetch(
            `/api/v1/video/${episodeAtRequest}/hand-3d?source_key=${encodeURIComponent(source)}&frame=${frame}`,
            signal ? { signal } : {});
        if (!isCurrentPlaybackSession(episodeAtRequest, sessionToken)) return;
        if (!res.ok) { cache.inflight = -1; return; }
        const data = await res.json();
        if (!isCurrentPlaybackSession(episodeAtRequest, sessionToken)) return;
        if (cache.inflight !== frame) return;  // 帧已跳走,丢弃过期响应
        hand3dFrameCacheBySource[source] = { frame, data, inflight: -1 };
        currentHand3dTiles
            .filter(entry => (entry.source.source_key || 'default') === source)
            .forEach(entry => renderHand3DTile(entry));
    } catch (e) {
        cache.inflight = -1;
    }
}

/* 米制 3D 点 → 绕 Y/X 轴旋转 → 屏幕坐标(等距投影) */
// ── 3D 世界坐标渲染(d435 demo RotatingSkeletonRenderer 的同款数学)──
// 五指分色 + 掌心灰线 + 腕部白点 + 地面网格 + 相机系坐标轴 + 腕部
// 深度标注;透视投影 f=(h/2)/tan(fov/2),静态视角 θ=π(正面)+ 仰角 25°,
// 世界坐标相机使用固定安全视距,不随当前可见手数量变化。用户拖拽 =
// 相对静态视角的偏转;滚轮缩放 = 相机距离。
const HAND3D_FINGERS = [
    // Keep the 3D tile identical to hand-overlay.js (the visible RGB video
    // overlay): thumb blue and pinky orange.
    { name: 'Thumb', ids: [1, 2, 3, 4], color: [0, 128, 255] },
    { name: 'Index', ids: [5, 6, 7, 8], color: [0, 255, 0] },
    { name: 'Middle', ids: [9, 10, 11, 12], color: [0, 255, 255] },
    { name: 'Ring', ids: [13, 14, 15, 16], color: [255, 0, 255] },
    { name: 'Pinky', ids: [17, 18, 19, 20], color: [255, 128, 0] },
];
const HAND3D_PALM = [[0, 1], [0, 5], [5, 9], [9, 13], [13, 17], [0, 17]];
let _hand3dViewAnchor = null;   // 世界坐标模式:统一的固定相机目标/世界原点
let _h3dBaseDist = null;        // 世界坐标模式:首帧锁定的相机距离(仅滚轮缩放改变)
let _h3dGridY = null;           // 世界坐标模式:首帧锁定的地面网格高度(视角固定)

// M1 质心锚定(demo _CentroidAnchor 展示路径的 JS 移植):每槽质心
// 强 One-Euro(freq 3.0/beta 0.3/dcutoff 0.3)+ 共模平移校正 ——
// 静态视角下手稳,"跟随乱动"的抖动主要来自逐帧质心漂移,这里吸收。
const _h3dCentroid = { filters: [null, null], prevLabels: [null, null],
                       prevC: [null, null], t0: null };

function _h3dMakeOneEuro(freqMin, beta, dcutoff) {
    let prev = null, prevDx = [0, 0, 0], prevTs = null;
    const alpha = (cutoff, dt) => {
        const tau = 1 / (2 * Math.PI * cutoff);
        return dt / (dt + tau);
    };
    return (x, ts) => {
        if (prev === null || prevTs === null) {
            prev = x.slice(); prevDx = [0, 0, 0]; prevTs = ts;
            return x.slice();
        }
        const dt = (ts - prevTs) / 1000;
        if (dt <= 1e-9) return prev.slice();
        const dx = x.map((v, i) => (v - prev[i]) / dt);
        const ad = alpha(dcutoff, dt);
        prevDx = dx.map((v, i) => ad * v + (1 - ad) * prevDx[i]);
        const fc = freqMin + beta * Math.hypot(prevDx[0], prevDx[1], prevDx[2]);
        const a = alpha(fc, dt);
        const out = x.map((v, i) => a * v + (1 - a) * prev[i]);
        prev = out.slice(); prevTs = ts;
        return out.slice();
    };
}

function _h3dAnchorApply(ptsBySlot, labels, state = _h3dCentroid) {
    if (state.t0 === null) state.t0 = performance.now();
    const ts = performance.now() - state.t0;
    const out = ptsBySlot.map(arr => arr.map(p => (p ? p.slice() : null)));
    for (let s = 0; s < 2; s++) {
        const finIdx = [];
        out[s].forEach((p, i) => { if (p) finIdx.push(i); });
        if (finIdx.length < 4) {
            state.filters[s] = null;
            state.prevC[s] = null;
            state.prevLabels[s] = labels[s];
            continue;
        }
        let c = [0, 0, 0];
        finIdx.forEach(i => { c[0] += out[s][i][0]; c[1] += out[s][i][1]; c[2] += out[s][i][2]; });
        c = c.map(v => v / finIdx.length);
        if (labels[s] !== state.prevLabels[s]) {
            // 重建帧且几何近(<0.1m)软衔接,否则硬重置(防跨手污染)
            const pc = state.prevC[s];
            if (state.filters[s] && pc && pc.every(Number.isFinite)
                    && Math.hypot(pc[0] - c[0], pc[1] - c[1], pc[2] - c[2]) < 0.10) {
                c = c.map((v, k) => 0.5 * pc[k] + 0.5 * v);
            }
            state.filters[s] = _h3dMakeOneEuro(3.0, 0.3, 0.3);
        }
        if (!state.filters[s]) state.filters[s] = _h3dMakeOneEuro(3.0, 0.3, 0.3);
        const ch = state.filters[s](c, ts);
        const delta = [ch[0] - c[0], ch[1] - c[1], ch[2] - c[2]];
        finIdx.forEach(i => {
            out[s][i][0] += delta[0];
            out[s][i][1] += delta[1];
            out[s][i][2] += delta[2];
        });
        state.prevC[s] = ch;
        state.prevLabels[s] = labels[s];
    }
    return out;
}

function _hand3dCam(valid, w, h, rotY, rotX, zoom,
                    baseDistOverride = null, targetOffset = null) {
    // demo RotatingSkeletonRenderer.render 同款相机布置:质心为目标、
    // fov 45°、静态正面视角 θ=π + 默认仰角 0°。世界坐标模式传入统一的
    // baseDist,播放期不随手部跨度自适应;默认视角静止,滚轮可调 zoom。
    const n = valid.length;
    const baseCentroid = valid.reduce(
        (a, p) => [a[0] + p[0], a[1] + p[1], a[2] + p[2]], [0, 0, 0])
        .map(v => v / n);
    const centroid = baseCentroid.map((value, index) => value
        + (targetOffset && Number.isFinite(targetOffset[index])
            ? targetOffset[index] : 0));
    const fov = 45 * Math.PI / 180;
    if (_h3dBaseDist === null && !Number.isFinite(baseDistOverride)) {
        const span = Math.max(
            Math.max(...valid.map(p => p[0])) - Math.min(...valid.map(p => p[0])),
            Math.max(...valid.map(p => p[1])) - Math.min(...valid.map(p => p[1])),
            Math.max(...valid.map(p => p[2])) - Math.min(...valid.map(p => p[2])));
        const half = span > 1e-6 ? span / 2 : 0.3;
        // 5.0 倍系数:手约占画面高度 1/5(默认视角按需求再拉远);
        // 距离下限 0.5m:手离相机远(跨度小)时默认视角也不会怼脸,
        // 需要更近/更远仍可用滚轮缩放
        _h3dBaseDist = Math.min(2.0, Math.max(0.5,
            6.0 * half / Math.tan(fov / 2)));
    }
    const baseDist = Number.isFinite(baseDistOverride)
        ? baseDistOverride : _h3dBaseDist;
    // Keep the default at 2m, but allow the user to zoom farther out when
    // inspecting a handless/world-only frame.
    const dist = Math.min(HAND3D_MAX_CAMERA_DISTANCE,
        Math.max(0.2, baseDist / (zoom || 1)));
    const theta = Math.PI + (rotY || 0);              // 静态正面 + 拖拽偏转
    // 仰角钳制 ±80°(标准轨道相机 pitch clamp):越过 ±90° 时
    // right = upW×fwd 与 up 基同时反向,画面绕屏幕中心 180° 翻转
    // —— 即"拖拽到达中心点后手部位置突变"的来源
    const elev = Math.max(-80 * Math.PI / 180, Math.min(80 * Math.PI / 180,
        HAND3D_DEFAULT_ELEVATION * Math.PI / 180 + (rotX || 0)));
    const ce = Math.cos(elev), se = Math.sin(elev);
    const eye = [
        centroid[0] + dist * Math.sin(theta) * ce,
        // 世界 Y 已统一为向上，因此正仰角时相机应位于目标上方。
        centroid[1] + dist * se,
        centroid[2] + dist * Math.cos(theta) * ce,
    ];
    // look-at 基:处理链路输出的世界坐标是 X 右、Y 上、Z 向前。
    // Canvas 的屏幕 Y 由投影公式负责翻转，因此这里不能再把世界上方向
    // 写成 -Y，否则 X/Y 轴都会被镜像，右侧空间视图会和原图相反。
    let fwd = [centroid[0] - eye[0], centroid[1] - eye[1], centroid[2] - eye[2]];
    const fn = Math.hypot(fwd[0], fwd[1], fwd[2]) + 1e-9;
    fwd = fwd.map(v => v / fn);
    const upW = [0, 1, 0];
    let right = [
        upW[1] * fwd[2] - upW[2] * fwd[1],
        upW[2] * fwd[0] - upW[0] * fwd[2],
        upW[0] * fwd[1] - upW[1] * fwd[0],
    ];
    const rn = Math.hypot(right[0], right[1], right[2]) + 1e-9;
    right = right.map(v => v / rn);
    const up = [
        fwd[1] * right[2] - fwd[2] * right[1],
        fwd[2] * right[0] - fwd[0] * right[2],
        fwd[0] * right[1] - fwd[1] * right[0],
    ];
    return { centroid, eye, fwd, right, up, f: (h / 2) / Math.tan(fov / 2), w, h };
}

function _projectHand3D(p, cam) {
    // 透视投影(demo _project 同款):u = cx + f·(d·right)/z,v = cy − f·(d·up)/z
    if (!p || !Number.isFinite(p[0]) || !Number.isFinite(p[1]) || !Number.isFinite(p[2])) {
        return [NaN, NaN];
    }
    const d = [p[0] - cam.eye[0], p[1] - cam.eye[1], p[2] - cam.eye[2]];
    const z = d[0] * cam.fwd[0] + d[1] * cam.fwd[1] + d[2] * cam.fwd[2];
    if (z <= 1e-4) return [NaN, NaN];
    const u = cam.w / 2 + cam.f * (d[0] * cam.right[0] + d[1] * cam.right[1] + d[2] * cam.right[2]) / z;
    const v = cam.h / 2 - cam.f * (d[0] * cam.up[0] + d[1] * cam.up[1] + d[2] * cam.up[2]) / z;
    return [u, v];
}

// 透视视图中的骨骼长度已经包含了距离缩放。使用它计算画笔尺寸，
// 避免手移动到远处后，固定像素半径的关节点互相盖住而看起来像圆球。
function _hand3dProjectedLength(a, b) {
    if (!a || !b || !Number.isFinite(a[0]) || !Number.isFinite(a[1])
            || !Number.isFinite(b[0]) || !Number.isFinite(b[1])) return 0;
    return Math.hypot(a[0] - b[0], a[1] - b[1]);
}

function _hand3dStrokeWidth(a, b, maxWidth) {
    const length = _hand3dProjectedLength(a, b);
    // 远处骨骼最小保留约 0.5px，近处不超过原来的视觉宽度。
    return Math.max(0.5, Math.min(maxWidth, length * 0.14));
}

function _hand3dStyleScale(proj) {
    const valid = (proj || []).filter(p => p && Number.isFinite(p[0])
        && Number.isFinite(p[1]));
    if (valid.length < 2) return 1;
    const xs = valid.map(p => p[0]), ys = valid.map(p => p[1]);
    const span = Math.max(Math.max(...xs) - Math.min(...xs),
                          Math.max(...ys) - Math.min(...ys));
    // Same reference size as the left SVG/hand_render style.
    return Math.max(0.45, Math.min(1, span / 160));
}

function _normalizeRgbEstimatedDisplayScale(slots, entry) {
    const handSpans = [];
    const all = [];
    for (const slot of slots) {
        const valid = slot.valid || [];
        if (valid.length < 4) continue;
        const xs = valid.map(p => p[0]), ys = valid.map(p => p[1]);
        handSpans.push(Math.max(Math.max(...xs) - Math.min(...xs),
                                Math.max(...ys) - Math.min(...ys)));
        all.push(...valid);
    }
    if (!handSpans.length || !all.length) return 1;
    const measured = handSpans.reduce((sum, value) => sum + value, 0)
        / handSpans.length;
    if (!(measured > 1e-6)) return 1;
    const targetScale = Math.max(RGB_ESTIMATED_DISPLAY_SCALE_MIN,
        Math.min(RGB_ESTIMATED_DISPLAY_SCALE_MAX,
            RGB_ESTIMATED_DISPLAY_HAND_SPAN / measured));
    // Do not resize the whole RGB skeleton independently on every frame.
    // Small 2D/PnP scale changes otherwise become visible as breathing and
    // jitter after the wrist root is anchored in the world tile.
    let scale = targetScale;
    if (entry) {
        const previous = Number(entry._rgbDisplayScale);
        scale = Number.isFinite(previous)
            ? previous * 0.92 + targetScale * 0.08
            : targetScale;
        entry._rgbDisplayScale = scale;
    }
    if (Math.abs(scale - 1) < 1e-3) return 1;
    const center = all.reduce((sum, p) => [
        sum[0] + p[0], sum[1] + p[1], sum[2] + p[2],
    ], [0, 0, 0]).map(value => value / all.length);
    for (const slot of slots) {
        slot.pts = slot.pts.map(p => p ? [
            center[0] + (p[0] - center[0]) * scale,
            center[1] + (p[1] - center[1]) * scale,
            center[2] + (p[2] - center[2]) * scale,
        ] : null);
        slot.fin = slot.pts.map(Boolean);
        slot.valid = slot.pts.filter(Boolean);
    }
    return scale;
}

function _normalizeDepthWorldDisplayScale(slots, entry, w, h) {
    // Calculate once from the first usable frame. Recomputing this per frame
    // would make the skeleton visibly breathe while playing.
    let scale = Number(entry?._worldDisplayScale);
    if (!Number.isFinite(scale)) {
        const spans = [];
        for (const slot of slots) {
            const valid = slot.valid || [];
            if (valid.length < 4) continue;
            const xs = valid.map(p => p[0]), ys = valid.map(p => p[1]);
            const zs = valid.map(p => p[2]);
            spans.push(Math.max(Math.max(...xs) - Math.min(...xs),
                                Math.max(...ys) - Math.min(...ys),
                                Math.max(...zs) - Math.min(...zs)));
        }
        const measured = spans.length
            ? spans.reduce((sum, value) => sum + value, 0) / spans.length : 0;
        const fov = 45 * Math.PI / 180;
        const focal = (Math.max(1, h) / 2) / Math.tan(fov / 2);
        const targetPx = Math.max(HAND3D_DISPLAY_SPAN_MIN_PX,
            Math.min(HAND3D_DISPLAY_SPAN_MAX_PX,
                Math.min(w, h) * HAND3D_DISPLAY_SPAN_RATIO));
        // At the fixed world camera distance, convert the target screen span
        // into a metric display span. This is only a preview transform.
        const targetWorldSpan = targetPx * HAND3D_WORLD_DEFAULT_DISTANCE / focal;
        scale = measured > 1e-6
            ? Math.max(HAND3D_DISPLAY_SCALE_MIN,
                Math.min(HAND3D_DISPLAY_SCALE_MAX, targetWorldSpan / measured)) : 1;
        if (entry) entry._worldDisplayScale = scale;
    }
    if (Math.abs(scale - 1) < 1e-3) return scale;
    slots.forEach(slot => {
        const wrist = slot.pts[0] || slot.valid[0];
        if (!wrist) return;
        slot.pts = slot.pts.map(point => point ? [
            wrist[0] + (point[0] - wrist[0]) * scale,
            wrist[1] + (point[1] - wrist[1]) * scale,
            wrist[2] + (point[2] - wrist[2]) * scale,
        ] : null);
        slot.fin = slot.pts.map(Boolean);
        slot.valid = slot.pts.filter(Boolean);
    });
    return scale;
}

function _hand3dJointRadius(proj, index, tip = false, styleScale = 1) {
    if (!proj[index] || !Number.isFinite(proj[index][0])
            || !Number.isFinite(proj[index][1])) return 0;
    const lengths = [];
    for (const [a, b] of HAND3D_CONNECTIONS) {
        if (a !== index && b !== index) continue;
        const length = _hand3dProjectedLength(proj[a], proj[b]);
        if (length > 0) lengths.push(length);
    }
    const preferred = (tip ? 7 : 5) * styleScale;
    if (!lengths.length) return Math.max(0.65, preferred);
    const mean = lengths.reduce((sum, value) => sum + value, 0) / lengths.length;
    // Prefer the canonical overlay size, but cap it by the projected local
    // bone size so distant joints never cover one another.
    return Math.max(0.65, Math.min(preferred, mean * (tip ? 0.18 : 0.16)));
}

function _drawHand3DWorldReference(ctx, w, h, state) {
    if (!state || !state.cam || !state.anchor) return;
    const { cam, anchor } = state;
    const px = p => _projectHand3D(p, cam);
    const line = (a, b, color, width) => {
        if (!Number.isFinite(a[0]) || !Number.isFinite(a[1])
                || !Number.isFinite(b[0]) || !Number.isFinite(b[1])) return;
        ctx.strokeStyle = color;
        ctx.lineWidth = _hand3dStrokeWidth(a, b, width);
        ctx.beginPath();
        ctx.moveTo(a[0], a[1]);
        ctx.lineTo(b[0], b[1]);
        ctx.stroke();
    };
    const yGrid = Number.isFinite(state.gridY) ? state.gridY : -0.25;
    // 网格是世界坐标的一部分，平移视角时不能跟着相机目标重新居中。
    const zc = anchor[2], xc = anchor[0];
    for (let x = xc - 0.4; x <= xc + 0.401; x += 0.05) {
        line(px([x, yGrid, zc - 0.45]), px([x, yGrid, zc + 0.45]),
             'rgba(58,58,58,0.9)', 1);
    }
    for (let z = zc - 0.45; z <= zc + 0.451; z += 0.05) {
        line(px([xc - 0.4, yGrid, z]), px([xc + 0.4, yGrid, z]),
             'rgba(58,58,58,0.9)', 1);
    }
    const axes = [[[0.1, 0, 0], 'rgb(60,60,255)', 'X'],
                  [[0, 0.1, 0], 'rgb(60,200,60)', 'Y'],
                  [[0, 0, 0.1], 'rgb(255,120,60)', 'Z']];
    for (const [v, color, name] of axes) {
        const o = px(anchor);
        const e = px([anchor[0] + v[0], anchor[1] + v[1], anchor[2] + v[2]]);
        line(o, e, color, 1);
        if (Number.isFinite(e[0])) {
            ctx.fillStyle = color;
            ctx.font = '9px sans-serif';
            ctx.fillText(name, e[0] + 3, e[1] - 3);
        }
    }
}

function _worldReferenceCam(state, w, h, entry) {
    if (!state) return null;
    if (state.center && Number.isFinite(state.baseDist)) {
        return _hand3dCam(
            [state.center], w, h,
            entry.rotY || 0, entry.rotX || 0, entry.zoom || 1,
            state.baseDist, entry.pan || [0, 0, 0]);
    }
    return state.cam || null;
}

function _defaultHand3DWorldReference(w, h, entry) {
    const anchor = HAND3D_WORLD_DEFAULT_CENTER.slice();
    const gridY = HAND3D_WORLD_DEFAULT_GRID_Y;
    return {
        center: anchor.slice(),
        baseDist: HAND3D_WORLD_DEFAULT_DISTANCE,
        // 视距固定在安全默认值。不能用首个有效帧拟合:首帧可能只有
        // 一只手或只有部分关键点,会让网格和后续双手画面突然缩放。
        fitted: true,
        anchor,
        gridY,
        rootAnchors: {
            left: [anchor[0] - HAND3D_ROOT_ANCHOR_X,
                   gridY + HAND3D_ROOT_ANCHOR_LIFT, anchor[2]],
            right: [anchor[0] + HAND3D_ROOT_ANCHOR_X,
                    gridY + HAND3D_ROOT_ANCHOR_LIFT, anchor[2]],
        },
    };
}

function _h3dRootAnchorKey(label, slotIndex) {
    const value = String(label || '').toLowerCase();
    if (value === 'left') return 'left';
    if (value === 'right') return 'right';
    // Older artifacts may not carry handedness. Keep a deterministic
    // fallback without changing the anchor when the other hand appears.
    return slotIndex === 0 ? 'left' : 'right';
}

function _h3dStableAnchorSide(entry, label, slotIndex) {
    if (!entry._anchorSides) entry._anchorSides = [null, null];
    const value = String(label || '').toLowerCase();
    const reported = value === 'left' || value === 'right' ? value : null;
    if (!entry._anchorSides[slotIndex] && reported) {
        const other = entry._anchorSides[slotIndex === 0 ? 1 : 0];
        // If two unstable labels claim the same side, preserve two distinct
        // display anchors rather than letting the hands overlap.
        entry._anchorSides[slotIndex] = (other && other === reported)
            ? (reported === 'left' ? 'right' : 'left') : reported;
    }
    return entry._anchorSides[slotIndex]
        || (slotIndex === 0 ? 'left' : 'right');
}

function _h3dRootAnchorApply(ptsBySlot, labels, state) {
    const rootAnchors = state && state.rootAnchors;
    if (!rootAnchors) return ptsBySlot;
    if (!state.lastRawRoots) state.lastRawRoots = [null, null];

    return ptsBySlot.map((points, slotIndex) => {
        const root = points && points[0]
            ? points[0]
            : state.lastRawRoots[slotIndex];
        if (!root || !root.every(Number.isFinite)) {
            return points.map(point => point ? point.slice() : null);
        }
        if (points && points[0]) state.lastRawRoots[slotIndex] = points[0].slice();
        const anchor = rootAnchors[
            _h3dRootAnchorKey(labels[slotIndex], slotIndex)
        ];
        return points.map(point => point ? [
            point[0] - root[0] + anchor[0],
            point[1] - root[1] + anchor[1],
            point[2] - root[2] + anchor[2],
        ] : null);
    });
}

function _h3dApplyFixedDisplayAnchors(slots, spaceMode, entry) {
    // The preview is intentionally a fixed layout. RGB-only coordinates are
    // camera-relative estimates, and MediaPipe-relative coordinates are
    // hand-local, so neither source can reliably define the two-hand layout.
    // Apply the same explicit wrist targets on every render path.
    const centerZ = spaceMode ? HAND3D_WORLD_DEFAULT_CENTER[2] : 0;
    const centerY = spaceMode
        ? HAND3D_WORLD_DEFAULT_GRID_Y + HAND3D_ROOT_ANCHOR_LIFT : 0;
    // Old RGB artifacts may contain duplicate handedness labels (commonly
    // both hands reported as Right).  The original 2D overlay is slot-based,
    // so it still looks correct, but label-only anchoring mirrors the two
    // hands in the 3D tile.  For that legacy case, initialize the semantic
    // sides from the raw camera-space wrist X order and then keep them fixed.
    const reported = slots.map(slot => String(slot.label || '').toLowerCase());
    const duplicateLabels = reported.length === 2
        && reported[0] && reported[0] === reported[1];
    if (duplicateLabels && !entry._duplicateAnchorSides) {
        const ordered = slots
            .map((slot, index) => ({ index, wrist: slot.pts[0] || slot.valid[0] }))
            .filter(item => item.wrist && Number.isFinite(item.wrist[0]))
            .sort((a, b) => a.wrist[0] - b.wrist[0]);
        if (ordered.length === 2) {
            entry._duplicateAnchorSides = [];
            entry._duplicateAnchorSides[ordered[0].index] = 'left';
            entry._duplicateAnchorSides[ordered[1].index] = 'right';
        }
    }
    slots.forEach((slot, slotIndex) => {
        const wrist = slot.pts[0] || slot.valid[0];
        if (!wrist) return;
        // Lock the semantic side once per tile. A transient per-frame
        // Left/Right classification must never move a whole hand to the
        // other side of the canvas, while the initial semantic orientation
        // remains compatible with both black-glove and bare-hand outputs.
        const side = duplicateLabels && entry._duplicateAnchorSides
            ? entry._duplicateAnchorSides[slotIndex]
            : _h3dStableAnchorSide(entry, slot.label, slotIndex);
        const target = [
            side === 'left' ? -HAND3D_ROOT_ANCHOR_X : HAND3D_ROOT_ANCHOR_X,
            centerY,
            centerZ,
        ];
        slot.pts = slot.pts.map(point => point ? [
            point[0] - wrist[0] + target[0],
            point[1] - wrist[1] + target[1],
            point[2] - wrist[2] + target[2],
        ] : null);
        slot.fin = slot.pts.map(Boolean);
        slot.valid = slot.pts.filter(Boolean);
    });
}

function _ensureHand3DWorldReference(w, h, entry) {
    const sourceData = hand3dDataBySource[entry.source.source_key] || hand3dData;
    if (!sourceData || !sourceData.spaceMode) return null;
    if (!entry._worldReference) entry._worldReference = _defaultHand3DWorldReference(w, h, entry);
    // Use the same world-space anchor for every episode. Hand data is shifted
    // relative to this stable reference, so its arrival cannot move the view.
    if (_hand3dViewAnchor === null) _hand3dViewAnchor = entry._worldReference.anchor.slice();
    if (_h3dGridY === null) _h3dGridY = HAND3D_WORLD_DEFAULT_GRID_Y;
    entry._worldReference.cam = _worldReferenceCam(entry._worldReference, w, h, entry);
    return entry._worldReference;
}

function renderHand3DTile(entry, frameOverride) {
    if (!entry || !entry.canvas) return;
    const canvas = entry.canvas;
    const rect = canvas.getBoundingClientRect();
    const dpr = window.devicePixelRatio || 1;
    const w = Math.max(1, rect.width), h = Math.max(1, rect.height);
    if (canvas.width !== Math.round(w * dpr) || canvas.height !== Math.round(h * dpr)) {
        canvas.width = Math.round(w * dpr);
        canvas.height = Math.round(h * dpr);
    }
    const ctx = canvas.getContext('2d');
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);

    const frame = typeof frameOverride === 'number'
        ? frameOverride
        : (typeof currentFrameTarget === 'number' ? currentFrameTarget : 0);
    _hand3dTileEntry = entry;
    const sourceKey = entry.source.source_key || 'default';
    const sourceData = hand3dDataBySource[sourceKey] || hand3dData;
    const worldMode = !!(sourceData && sourceData.worldMode);
    const spaceMode = !!(sourceData && sourceData.spaceMode);
    const cameraYDown = !!(sourceData && sourceData.cameraYDown);
    let fd = _hand3dCached(frame, sourceKey);
    const frameCache = _hand3dFrameCacheFor(sourceKey);
    if (!spaceMode && frameCache.frame === frame && frameCache.data) {
        fd = {
            h0: frameCache.data.h0 || null,
            h1: frameCache.data.h1 || null,
        };
    }
    if (sourceData && (sourceData.hasPreview || spaceMode)) {
        if (spaceMode) fetchHand3DWindow(frame, sourceKey);
        else fetchHand3DFrame(frame, sourceKey);
    }
    // 缓存窗口切换期间保留上一张已提交的 3D 画面，不清空画布。
    // 数据返回后本函数再次调用，并在下面一次性提交新帧，避免闪烁。
    // 若某帧确实没有手部数据，fd 是非 null 的空对象，仍会正常清空并
    // 绘制参考系；因此这里只处理“尚未拿到当前帧”的网络等待状态。
    if (fd === null && entry._hasCommittedFrame) return;
    ctx.clearRect(0, 0, w, h);
    ctx.fillStyle = 'rgba(3,7,18,0.95)';
    ctx.fillRect(0, 0, w, h);
    if (!fd || (!fd.h0 && !fd.h1)) {
        if (spaceMode) {
            _ensureHand3DWorldReference(w, h, entry);
            _drawHand3DWorldReference(ctx, w, h, entry._worldReference);
            return;
        }
        return;
    }

    entry._hasCommittedFrame = true;

    // 收集有效点(世界坐标模式:d435 相机系米制)
    const slots = [
        { key: 'h0', label: fd.h0 && fd.h0.label ? String(fd.h0.label) : '' },
        { key: 'h1', label: fd.h1 && fd.h1.label ? String(fd.h1.label) : '' },
    ];
    const all = [];
    for (const s of slots) {
        const hand = fd[s.key];
        s.pts = hand && hand.lm ? hand.lm.map(p => (
            p && Number.isFinite(p[0]) && Number.isFinite(p[1]) && Number.isFinite(p[2])
                ? [p[0], cameraYDown ? -p[1] : p[1], p[2]] : null)) : [];
        s.fin = s.pts.map(Boolean);
        s.valid = s.pts.filter(Boolean);
        all.push(...s.valid);
    }
    if (all.length < 4) {
        if (spaceMode) {
            _ensureHand3DWorldReference(w, h, entry);
        }
        if (spaceMode && entry._worldReference) {
            entry._worldReference.cam = _worldReferenceCam(entry._worldReference, w, h, entry);
            _drawHand3DWorldReference(ctx, w, h, entry._worldReference);
            return;
        }
        return;
    }

    // Keep the stored RGB estimate untouched. This is a presentation-only
    // normalization so RGB and depth tiles occupy a comparable visual scale.
    if (sourceData && sourceData.rgbEstimatedMode) {
        _normalizeRgbEstimatedDisplayScale(slots, entry);
        all.length = 0;
        slots.forEach(s => all.push(...s.valid));
    }
    if (sourceData && sourceData.worldMode) {
        _normalizeDepthWorldDisplayScale(slots, entry, w, h);
        all.length = 0;
        slots.forEach(s => all.push(...s.valid));
    }

    if (spaceMode) _ensureHand3DWorldReference(w, h, entry);

    if (spaceMode && entry._worldReference
            && entry._worldReference.fitted !== true) {
        // 兼容旧的 tile 状态,但不再依据当前帧的关键点尺度拟合。
        // 尤其不能让首个单手帧把相机拉近,否则第二只手出现时视角会跳。
        entry._worldReference.baseDist = HAND3D_WORLD_DEFAULT_DISTANCE;
        entry._worldReference.fitted = true;
    }

    // All 3D preview modes use the same fixed wrist layout. This is a
    // display-only transform; the original camera-relative coordinates in
    // parquet/API remain unchanged.
    _h3dApplyFixedDisplayAnchors(slots, spaceMode, entry);
    all.length = 0;
    slots.forEach(s => all.push(...s.valid));

    // 视图锚点:世界坐标模式使用真实的固定相机坐标,不按当前可见手的
    // 质心平移。否则双手变单手时 c0 会改变,剩余手会被人为挪动。
    // 相对 3D 仍按腕点独立归一化。
    if (!spaceMode && _hand3dViewAnchor === null) {
        const c0 = all.reduce((a, p) => [a[0] + p[0], a[1] + p[1], a[2] + p[2]], [0, 0, 0])
            .map(v => v / all.length);
        _hand3dViewAnchor = c0.slice();
    }
    // 世界坐标保留处理链路输出的绝对相机坐标。anchor 只用于
    // 相机/网格参考,不再把当前帧整体搬到 anchor。相对 3D 已在
    // 上面的腕点归一化阶段完成布局。
    const shifted = all.map(p => p.slice());

    let cam;
    if (spaceMode) {
        // The current hand points move and render, but never redefine the
        // shared initial world-coordinate viewpoint.
        entry._worldReference.cam = _worldReferenceCam(entry._worldReference, w, h, entry);
        cam = entry._worldReference.cam;
    } else {
        cam = _hand3dCam(shifted, w, h, entry.rotY || 0, entry.rotX || 0,
            entry.zoom || 1, null, entry.pan || [0, 0, 0]);
    }
    entry._lastCam = cam;
    const px = p => _projectHand3D(p, cam);
    const line = (a, b, color, width, styleScale = 1) => {
        if (!Number.isFinite(a[0]) || !Number.isFinite(a[1])
                || !Number.isFinite(b[0]) || !Number.isFinite(b[1])) return;
        ctx.strokeStyle = color;
        ctx.lineWidth = _hand3dStrokeWidth(a, b, width * styleScale);
        ctx.beginPath();
        ctx.moveTo(a[0], a[1]);
        ctx.lineTo(b[0], b[1]);
        ctx.stroke();
    };

    if (spaceMode) {
        // 世界坐标参考系独立于当前帧关键点,关键点缺失时仍保留。
        _drawHand3DWorldReference(ctx, w, h, entry._worldReference);
    }

    // painter 算法:远手先画(demo order.sort(key=-depth))
    const order = slots
        .filter(s => s.valid.length)
        .map(s => ({
            s,
            depth: s.valid.reduce((a, p) => a
                + ((p[0] - cam.eye[0]) * cam.fwd[0]
                   + (p[1] - cam.eye[1]) * cam.fwd[1]
                   + (p[2] - cam.eye[2]) * cam.fwd[2]), 0) / s.valid.length,
        }))
        .sort((a, b) => b.depth - a.depth);

    for (const { s } of order) {
        const proj = s.pts.map(p => px(p));
        const styleScale = _hand3dStyleScale(proj);
        // 掌心灰线
        for (const [a, b] of HAND3D_PALM) {
            if (s.fin[a] && s.fin[b]) {
                line(proj[a], proj[b], 'rgb(200,200,200)', 2, styleScale);
            }
        }
        // 五指分色链(拇指 1→2→3→4;其余 0→MCP→…→指尖)
        for (const f of HAND3D_FINGERS) {
            const chain = f.name === 'Thumb' ? f.ids : [0, ...f.ids];
            for (let i = 0; i < chain.length - 1; i++) {
                if (s.fin[chain[i]] && s.fin[chain[i + 1]]) {
                    line(proj[chain[i]], proj[chain[i + 1]],
                        `rgb(${f.color[0]},${f.color[1]},${f.color[2]})`, 3,
                        styleScale);
                }
            }
            for (const idx of f.ids) {
                if (!s.fin[idx]) continue;
                const r = _hand3dJointRadius(
                    proj, idx, idx === f.ids[f.ids.length - 1], styleScale);
                if (r <= 0) continue;
                ctx.fillStyle = `rgb(${f.color[0]},${f.color[1]},${f.color[2]})`;
                ctx.beginPath();
                ctx.arc(proj[idx][0], proj[idx][1], r, 0, Math.PI * 2);
                ctx.fill();
                ctx.strokeStyle = 'rgb(30,30,30)';
                ctx.lineWidth = Math.max(0.5, Math.min(1, r * 0.22));
                ctx.stroke();
            }
        }
        // 腕部白点
        if (s.fin[0]) {
            const wristRadius = Math.max(0.65, Math.min(
                9 * styleScale,
                _hand3dJointRadius(proj, 0, false, styleScale) * 1.45));
            ctx.fillStyle = '#fff';
            ctx.beginPath();
            ctx.arc(proj[0][0], proj[0][1], Math.max(1.2, Math.min(7, wristRadius)), 0, Math.PI * 2);
            ctx.fill();
            ctx.strokeStyle = 'rgb(40,40,40)';
            ctx.lineWidth = Math.max(0.5, Math.min(1.5, wristRadius * 0.22));
            ctx.stroke();
        }
        // 腕部深度标注只在相机米制世界坐标中显示。
        if (worldMode && s.fin[0] && Number.isFinite(proj[0][0])) {
            ctx.fillStyle = s.label === 'Right' ? 'rgb(0,255,0)' : 'rgb(255,200,0)';
            ctx.font = '11px sans-serif';
            ctx.fillText(`${s.label}  z=${s.pts[0][2].toFixed(2)}m`,
                proj[0][0] + 12, proj[0][1] - 12);
        }
    }

    // HUD
    ctx.fillStyle = 'rgba(235,235,235,0.9)';
    ctx.font = '11px sans-serif';
    ctx.textAlign = 'left';
    ctx.fillText(`frame ${frame}`, 12, 22);
    // The interaction/mode hint was only a debug HUD and is intentionally
    // omitted from the spatial preview.
}

function initHand3dDrag(entry) {
    if (!entry || !entry.canvas) return;
    const canvas = entry.canvas;
    canvas.style.cursor = 'grab';
    canvas.addEventListener('mousedown', e => {
        if (e.button !== 0 && e.button !== 1) return;
        entry.dragging = true;
        entry.dragMode = e.button === 1 ? 'pan' : 'rotate';
        entry.lastX = e.clientX;
        entry.lastY = e.clientY;
        canvas.style.cursor = entry.dragMode === 'pan' ? 'move' : 'grabbing';
        e.preventDefault();
    });
    window.addEventListener('mousemove', e => {
        if (!entry.dragging) return;
        // 左键是轨道旋转:水平向右看右侧,垂直向上看上方。
        const dx = e.clientX - entry.lastX;
        const dy = e.clientY - entry.lastY;
        if (entry.dragMode === 'pan') {
            // 让空间像被鼠标抓住一样移动:拖右→内容右移,拖下→内容下移。
            // 使用当前相机 right/up 基向量,旋转后平移仍符合屏幕方向。
            const cam = entry._lastCam;
            if (cam) {
                const distance = Math.max(0.2, Math.hypot(
                    cam.eye[0] - cam.centroid[0],
                    cam.eye[1] - cam.centroid[1],
                    cam.eye[2] - cam.centroid[2]));
                const worldPerPixel = distance / Math.max(1, cam.f);
                const pan = entry.pan || (entry.pan = [0, 0, 0]);
                for (let i = 0; i < 3; i++) {
                    pan[i] += (-dx * cam.right[i] + dy * cam.up[i])
                        * worldPerPixel;
                }
            }
        } else {
            // 水平向右拖动查看右侧；向上拖动查看上方。
            entry.rotY += dx * 0.01;
            entry.rotX += dy * 0.01;
        }
        entry.lastX = e.clientX;
        entry.lastY = e.clientY;
        renderHand3DTile(entry);
    });
    window.addEventListener('mouseup', () => {
        if (entry.dragging) {
            entry.dragging = false;
            entry.dragMode = '';
            canvas.style.cursor = 'grab';
        }
    });
    canvas.addEventListener('contextmenu', e => e.preventDefault());
    // 滚轮缩放(世界坐标模式:0.3x – 6x)
    canvas.addEventListener('wheel', e => {
        e.preventDefault();
        entry.zoom = Math.min(6, Math.max(0.3,
            (entry.zoom || 1) * Math.exp(-e.deltaY * 0.0015)));
        renderHand3DTile(entry);
    }, { passive: false });
}

/* 逐帧步进回调:3D 世界窗口随统一帧号重绘(多回调机制,与骨骼叠加层/
   标注高亮共用;闭包在调用时遍历,脚本加载时 tile 列表为空无影响) */
setOnFrameChange((f) => {
    for (const entry of currentHand3dTiles) renderHand3DTile(entry, f);
});

/* 图像素材(深度图/手套热力图)的逐帧 rAF 刷新循环(见 _startImageTileRaf) */
if (document.readyState !== 'loading') _startImageTileRaf();
else document.addEventListener('DOMContentLoaded', _startImageTileRaf);

/* 漂浮预览控件的显隐:点开批次详情后才显示(☷ 与 Slice Preview
   在标注/审核/审核通过页面可用);返回批次列表时隐藏。 */
function setPreviewPanelsVisible(open) {
    window.__episodeOpen = !!open;
    const po = document.getElementById('preview-options');
    if (po) po.classList.toggle('hidden', !open);
    const menu = document.getElementById('preview-options-menu');
    if (!open && menu) menu.classList.add('hidden');
    if (open) {
        renderPreviewVideoSources();
        if (typeof updatePreviewMenuData === 'function') updatePreviewMenuData();
    }
    const pt = document.getElementById('playback-toolbar');
    if (pt) pt.classList.toggle('hidden', !open);
    if (!open && typeof resetSlicePreview === 'function') resetSlicePreview('');
    if (typeof refreshSlicePreview === 'function') refreshSlicePreview();
}

window.ensurePreviewOptionsVisible = function () {
    if (window.__episodeOpen !== true) return;
    const box = document.getElementById('preview-options');
    if (box) box.classList.remove('hidden');
};

/* 诊断入口:播放按钮"点了没反应"时,在浏览器控制台(F12 → Console)输入
   __playerDebug() 回车,把输出发过来 —— 一次定位卡在哪个状态。 */
window.__playerDebug = function () {
    return {
        episode: currentEpisodeId,
        playerReady: playerReady,
        frameDataReady: frameDataReady,
        controlsEnabled: _controlsEnabled,
        players: Object.keys(players),
        activePlayer: Boolean(getActivePlayer()),
        workspaceSources: workspaceSources.map(s => groupedSourceKey(s)),
        overlayHidden: (document.getElementById('video-loading-overlay') || {})
            .classList.contains('hidden'),
    };
};
