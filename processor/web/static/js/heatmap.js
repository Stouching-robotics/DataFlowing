/* Bionic Hand Sensor — synced with video playback */

let heatmapFps = 30;
let heatmapSyncing = false;
const heatmapSyncBound = new WeakSet();
let currentEpisodeUuid = null;
let lastFrameIdx = -1;
let totalFrames = 0;

// Sensor → hand mapping (populated by loadFrameData)
let sensorRight = null;   // sensor name for right hand (e.g. "sensors_right")
let sensorLeft = null;    // sensor name for left hand  (e.g. "sensors_left")

// Hand-pressure tiles placed on the video canvas by drag & drop.
// Each entry: {canvas: <canvas element>, hand: 'left' | 'right'}.
// frames-data already contains the complete sensor sequence, so the browser
// renders it locally instead of requesting one PNG per frame.
let canvasHandTiles = [];
let sensorFramesByIndex = new Map();

function registerHandTile(canvas, hand) {
    canvasHandTiles.push({ canvas, hand });
    const frame = typeof getCurrentFrame === 'function' ? getCurrentFrame() : 0;
    const sensor = hand === 'left' ? sensorLeft : sensorRight;
    const values = _sensorValue(frame, sensor);
    if (canvas && values) renderHandSensorCanvas(canvas, values, hand === 'left');
}

function unregisterHandTile(canvas) {
    canvasHandTiles = canvasHandTiles.filter(entry => entry.canvas !== canvas);
}

function clearHandTiles() {
    canvasHandTiles = [];
}

function hasHandSensor(hand) {
    return hand === 'left' ? Boolean(sensorLeft) : Boolean(sensorRight);
}

function handTileUrl(frameIndex, hand) {
    const sensor = hand === 'left' ? sensorLeft : sensorRight;
    const params = new URLSearchParams({ frame_index: frameIndex });
    if (sensor) params.set('sensor', sensor);
    if (hand === 'left') params.set('mirror', 'true');
    params.set('compact', 'true');
    return `/api/v1/video/${currentEpisodeUuid}/hand?${params.toString()}`;
}


function _resolveSensorMapping(sensors) {
    /* Given available sensor names, decide which to use for left/right hand.

       - Only columns containing "sensor" are considered as hand pressure sensors
         (columns like hand_angles / hand_joints are not 16×16 pressure grids)
       - Contains "right" / "left" in name → map accordingly
       - Single sensor with direction hint → only that hand, other hand hidden
       - Single sensor without hint → both hands use it (old hardware compat)
       - Multiple sensors without hints → index 0→right, index 1→left
    */
    sensorRight = null;
    sensorLeft = null;

    if (!sensors || sensors.length === 0) return;

    // Filter to pressure-sensor columns only (ignore hand_angles, hand_joints, etc.).
    // Accepts "sensors_left/right", SenseGlove "left_glove"/"right_glove" and
    // tactile "left"/"right" naming.
    const pressureSensors = sensors.filter(s => {
        const n = s.toLowerCase();
        return n.includes('sensor') || n.includes('glove') || n.includes('tactile');
    });
    // If no pressure sensors found, fall back to all columns
    const candidates = pressureSensors.length > 0 ? pressureSensors : sensors;

    // Look for left/right hints in names
    for (const s of candidates) {
        const lower = s.toLowerCase();
        if (lower.includes('right') || lower.includes('_r')) sensorRight = s;
        if (lower.includes('left') || lower.includes('_l')) sensorLeft = s;
    }

    if (candidates.length === 1) {
        // Single sensor: only assign to the hand it names, if any
        if (!sensorRight && !sensorLeft) {
            // No direction hint → both hands same source (old hardware compat)
            sensorRight = candidates[0];
            sensorLeft = candidates[0];
        }
        // else: direction hint already set above, other hand stays null → hidden
        return;
    }

    // Multi-sensor: fill gaps with index-order fallback
    if (!sensorRight) sensorRight = candidates[0];
    if (!sensorLeft) sensorLeft = candidates.length >= 2 ? candidates[1] : candidates[0];
}

const HAND_SENSOR_REGIONS = [
    { rows: [0, 1, 2], cols: [14, 12, 13, 15], x: 250, y: 12, finger: true },
    { rows: [3, 4, 5], cols: [14, 12, 13, 15], x: 313, y: 12, finger: true },
    { rows: [6, 7, 8], cols: [14, 12, 13, 15], x: 376, y: 12, finger: true },
    { rows: [9, 10, 11], cols: [14, 12, 13, 15], x: 439, y: 12, finger: true },
    { rows: [12, 13, 14], cols: [14, 12, 13, 15], x: 502, y: 12, finger: true },
    { rows: Array.from({ length: 15 }, (_, i) => i), cols: [10, 9, 8, 6, 4], x: 280, y: 101, finger: false },
];

function _viridisSensorColor(value, vmax) {
    const stops = [
        [68, 1, 84], [59, 82, 139], [33, 145, 140],
        [94, 201, 98], [253, 231, 37],
    ];
    const ratio = Math.max(0, Math.min(1, Number(value || 0) / vmax));
    const pos = ratio * (stops.length - 1);
    const i = Math.min(stops.length - 2, Math.floor(pos));
    const t = pos - i;
    const a = stops[i], b = stops[i + 1];
    return `rgb(${Math.round(a[0] + (b[0] - a[0]) * t)},` +
        `${Math.round(a[1] + (b[1] - a[1]) * t)},` +
        `${Math.round(a[2] + (b[2] - a[2]) * t)})`;
}

function renderHandSensorCanvas(canvas, values, mirror) {
    if (!canvas) return;
    const width = 800, height = 200, cell = 16;
    if (canvas.width !== width || canvas.height !== height) {
        canvas.width = width;
        canvas.height = height;
    }
    canvas.style.aspectRatio = '4 / 1';
    const ctx = canvas.getContext('2d');
    const matrix = Array.isArray(values) && Array.isArray(values[0])
        ? values : [];
    const flat = matrix.flat ? matrix.flat() : [];
    const vmax = Math.max(1, ...flat.map(v => Number(v) || 0));
    ctx.save();
    if (mirror) {
        ctx.translate(width, 0);
        ctx.scale(-1, 1);
    }
    ctx.fillStyle = '#121212';
    ctx.fillRect(0, 0, width, height);
    ctx.font = '10px sans-serif';
    HAND_SENSOR_REGIONS.forEach(region => {
        const cs = cell;
        region.rows.forEach((row, i) => region.cols.forEach((col, j) => {
            const value = Number(matrix[row]?.[col]) || 0;
            const x = region.x + i * cs;
            const y = region.y + j * cs - 8;
            ctx.fillStyle = _viridisSensorColor(value, vmax);
            ctx.fillRect(x, y, cs, cs);
            ctx.strokeStyle = 'rgb(70,70,70)';
            ctx.strokeRect(x + 0.5, y + 0.5, cs - 1, cs - 1);
            ctx.fillStyle = vmax > 1 && value / vmax > 0.55 ? '#000' : '#fff';
            ctx.textAlign = 'center';
            ctx.textBaseline = 'middle';
            ctx.fillText(String(Math.round(value)), x + cs / 2, y + cs / 2);
        }));
    });
    ctx.restore();
}

function _sensorValue(frameIndex, sensor) {
    const row = sensorFramesByIndex.get(frameIndex);
    return row ? row[`observation_${sensor}`] : null;
}


async function loadFrameData(episodeId, sessionToken = null) {
    if (typeof isCurrentPlaybackSession === 'function' &&
        !isCurrentPlaybackSession(episodeId, sessionToken)) return;
    if (typeof currentEpisodeId !== 'undefined' && currentEpisodeId !== episodeId) return;  // 已切换批次
    currentEpisodeUuid = episodeId;
    lastFrameIdx = -1;
    _lastHandFrameIdx = -1;  // reset hand throttle on episode switch

    try {
        const signal = typeof getMediaLoadSignal === 'function'
            ? getMediaLoadSignal() : null;
        const cacheKey = `${episodeId}:glove:frames-data:v1`;
        let data = null;
        if (window.EgoMediaCache) {
            const cached = await window.EgoMediaCache.get(cacheKey);
            if (cached && cached.value) data = cached.value;
        }
        if (!data) {
            const res = await fetch(`/api/v1/episode/${episodeId}/frames-data`,
                signal ? { signal } : {});
            if ((typeof isCurrentPlaybackSession === 'function' &&
                 !isCurrentPlaybackSession(episodeId, sessionToken)) ||
                (typeof currentEpisodeId !== 'undefined' && currentEpisodeId !== episodeId)) return;  // stale — discard
            if (!res.ok) throw new Error(`frames-data request failed: ${res.status}`);
            data = await res.json();
            if (window.EgoMediaCache) {
                window.EgoMediaCache.put(cacheKey, data);
            }
        }
        if ((typeof isCurrentPlaybackSession === 'function' &&
             !isCurrentPlaybackSession(episodeId, sessionToken)) ||
            (typeof currentEpisodeId !== 'undefined' && currentEpisodeId !== episodeId)) return;  // stale — discard
        heatmapFps = data.fps || 30;

        // Resolve sensor → hand mapping from available sensors
        _resolveSensorMapping(data.sensors || []);

        // Keep the complete sensor payload indexed by the real frame number.
        // The source parquet may contain repeated/resampled rows, while the
        // API has already de-duplicated them to the video frame timeline.
        sensorFramesByIndex = new Map();
        (data.frames || []).forEach(frame => {
            if (frame && Number.isFinite(Number(frame.frame_index))) {
                sensorFramesByIndex.set(Number(frame.frame_index), frame);
            }
        });

        totalFrames = data.frame_count || 0;
        // Hand heatmaps are now draggable canvas tiles (top buttons), not a
        // fixed section under the video. Notify player.js to render the
        // drag-source buttons when sensor data exists.
        _updateHandLabels();
        updateHandImages(0);
        if (typeof renderSourceBar === 'function') renderSourceBar();

        // Notify player.js of frame info
        if (typeof setEpisodeFrameInfo === 'function') {
            setEpisodeFrameInfo(heatmapFps, totalFrames);
        }
    } catch (err) {
        const heatmapSection = document.getElementById('heatmap-section');
        if (heatmapSection) heatmapSection.classList.add('hidden');
    }
}

// 切换批次时清掉旧批次的帧状态:fps/总帧数在 loadFrameData 返回前保持
// "未知",避免同步换算(深度/手套 tile、热力图、seek)用到脏值。
// 同时把 uuid 立刻指向新批次,手部 tile 挂载时的首帧 URL 不会请求旧批次。
function resetHeatmapFrameState(episodeId) {
    heatmapFps = 0;
    totalFrames = 0;
    lastFrameIdx = -1;
    _lastHandFrameIdx = -1;
    sensorFramesByIndex = new Map();
    currentEpisodeUuid = episodeId || null;
}


function _updateHandLabels() {
    /* Show sensor name hints in hand labels. Hide hand when no sensor data available. */
    const leftContainer = document.querySelector('#heatmap-section .flex-1:first-child');
    const rightContainer = document.querySelector('#heatmap-section .flex-1:last-child');
    const leftLabel = leftContainer ? leftContainer.querySelector('.text-xs') : null;
    const rightLabel = rightContainer ? rightContainer.querySelector('.text-xs') : null;
    const imgL = document.getElementById('hand-left');
    const imgR = document.getElementById('hand-right');

    // 没有手部传感器数据 → 对应手部热力图容器保持隐藏,绝不显示空图
    if (leftContainer && imgL) {
        const baseLeft = (typeof t === 'function') ? t('left_hand') : 'Left Hand';
        if (sensorLeft) {
            leftLabel.textContent = baseLeft + ' (' + sensorLeft + ')';
            leftContainer.style.display = '';
            imgL.style.display = '';
        } else {
            leftLabel.textContent = baseLeft;
            leftContainer.style.display = 'none';
            imgL.style.display = 'none';
        }
    }
    if (rightContainer && imgR) {
        const baseRight = (typeof t === 'function') ? t('right_hand') : 'Right Hand';
        if (sensorRight) {
            rightLabel.textContent = baseRight + ' (' + sensorRight + ')';
            rightContainer.style.display = '';
            imgR.style.display = '';
        } else {
            rightLabel.textContent = baseRight;
            rightContainer.style.display = 'none';
            imgR.style.display = 'none';
        }
    }
}


let _lastHandFrameIdx = -1;

function updateHandImages(frameIndex) {
    if (!currentEpisodeUuid || frameIndex === lastFrameIdx) return;
    // Clamp to last frame to avoid 404 on out-of-range requests
    if (totalFrames > 0) frameIndex = Math.min(frameIndex, totalFrames - 1);
    lastFrameIdx = frameIndex;

    _lastHandFrameIdx = frameIndex;

    // Draw every frame from the already-loaded sensor sequence. No per-frame
    // HTTP request and no intentional frame skipping during playback.
    canvasHandTiles.forEach(({ canvas, hand }) => {
        const sensor = hand === 'left' ? sensorLeft : sensorRight;
        const values = _sensorValue(frameIndex, sensor);
        if (canvas && values) renderHandSensorCanvas(canvas, values, hand === 'left');
    });

    const el = document.getElementById('heatmap-frame');
    if (el) el.textContent = frameIndex;
}


function _readFps() {
    /* Single source of truth — always read from player.js to avoid divergence. */
    return typeof getEpisodeFps === 'function' ? getEpisodeFps() : (heatmapFps || 30);
}

function _timeToFrame(currentTime) {
    /* Standard video frame→time inverse: frame N occupies [N/fps, (N+1)/fps).
       The +0.002 epsilon guards against float truncation when the browser seek
       lands a fraction of a millisecond short of the exact frame boundary. */
    return Math.floor(currentTime * _readFps() + 0.002);
}

function _tryConsumeSeekTarget() {
    /* Return the exact target frame from seekToFrame, or null if none pending.
       MUST only be called from seeked/pause — timeupdate must never consume it,
       otherwise timeupdate can steal the target before seeked fires, causing
       the display to jump target→recalculated→back. */
    return typeof consumeSeekTarget === 'function' ? consumeSeekTarget() : null;
}

function startHeatmapSync(plyrInstance) {
    if (!plyrInstance || !currentEpisodeUuid) return;
    heatmapSyncing = true;
    if (heatmapSyncBound.has(plyrInstance)) return;
    heatmapSyncBound.add(plyrInstance);

    plyrInstance.on('timeupdate', () => {
        if (!heatmapSyncing) return;
        // Ignore intermediate seek states while the shared timeline is being
        // scrubbed; the final pointer release performs one committed refresh.
        if (typeof isFrameScrubbing === 'function' && isFrameScrubbing()) return;
        // NEVER consume seek target — timeupdate fires asynchronously and
        // can race ahead of seeked, stealing the target and causing flicker.
        const frameIndex = _timeToFrame(plyrInstance.currentTime);
        updateHandImages(frameIndex);
        if (typeof updateFrameDisplay === 'function') updateFrameDisplay(frameIndex);
    });

    plyrInstance.on('seeked', () => {
        if (!heatmapSyncing) return;
        // Prefer exact target from seekToFrame (zero round-trip error).
        // Fall back to time→frame for user-dragged seeks.
        let frameIndex = _tryConsumeSeekTarget();
        if (frameIndex == null) {
            frameIndex = _timeToFrame(plyrInstance.currentTime);
        }
        updateHandImages(frameIndex);
        if (typeof updateFrameDisplay === 'function') updateFrameDisplay(frameIndex);
    });

    plyrInstance.on('pause', () => {
        if (!heatmapSyncing) return;
        // On pause: prefer pending seek target (stepFrame pauses-then-seeks),
        // otherwise compute exact paused frame so captureStart/End see it.
        let frameIndex = _tryConsumeSeekTarget();
        if (frameIndex == null) {
            frameIndex = _timeToFrame(plyrInstance.currentTime);
        }
        if (typeof updateFrameDisplay === 'function') updateFrameDisplay(frameIndex);
    });
}


function stopHeatmapSync() {
    heatmapSyncing = false;
}

// player.js owns the single rAF frame clock. Registering here keeps glove
// sensor canvases on exactly the same integer frame as RGB, depth and 3D.
if (typeof setOnFrameChange === 'function') {
    setOnFrameChange(frameIndex => {
        if (heatmapSyncing) updateHandImages(frameIndex);
    });
}
