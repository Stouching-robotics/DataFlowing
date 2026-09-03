/* Dashboard — overview statistics, pipeline flow, task progress */
var _dashTimer = null;
var _dashVisible = true;
var _dashController = null;
var _dashLoadEpoch = 0;

// ── Formatting ────────────────────────────────────────

function fmtNum(n) {
    if (n == null) return '0';
    return n.toLocaleString('en-US');
}

function fmtToday(n) {
    if (!n || n <= 0) return '';
    return '+' + fmtNum(n) + ' today';
}

function timeAgo(isoStr) {
    if (!isoStr) return '';
    var diff = Date.now() - new Date(isoStr).getTime();
    var sec = Math.floor(diff / 1000);
    if (sec < 60) return 'just now';
    if (sec < 3600) return Math.floor(sec / 60) + 'm ago';
    if (sec < 86400) return Math.floor(sec / 3600) + 'h ago';
    return Math.floor(sec / 86400) + 'd ago';
}

function statusBadge(status) {
    var map = {
        'completed':  { cls: 'bg-yellow-900/50 text-yellow-400', label: 'Reviewing' },
        'to_review':  { cls: 'bg-yellow-900/50 text-yellow-400', label: 'Reviewing' },
        'reviewed':   { cls: 'bg-green-900/50 text-green-400',  label: 'Approved' },
        'approved':   { cls: 'bg-green-900/50 text-green-400',  label: 'Approved' },
        'processing': { cls: 'bg-blue-900/50 text-blue-400',    label: 'Processing' },
        'received':   { cls: 'bg-indigo-900/50 text-indigo-400', label: 'Received' },
        'failed':     { cls: 'bg-red-900/50 text-red-400',      label: 'Failed' },
    };
    var m = map[status] || { cls: 'bg-gray-800 text-gray-400', label: status || 'unknown' };
    return '<span class="text-xs px-2 py-0.5 rounded ' + m.cls + '">' + m.label + '</span>';
}


// ── Load all data ─────────────────────────────────────

async function refreshDashboard() {
    if (_dashController) _dashController.abort();
    var controller = new AbortController();
    _dashController = controller;
    var loadEpoch = ++_dashLoadEpoch;
    var btn = document.getElementById('btn-refresh');
    if (btn) {
        btn.disabled = true;
        var icon = btn.querySelector('iconify-icon');
        if (icon) icon.setAttribute('icon', 'ant-design:loading-outlined');
    }

    // These four panels are independent.  Request them together so a slow
    // remote-storage read cannot delay all of the other dashboard sections.
    const results = await Promise.allSettled([
        fetch('/api/v1/dashboard/overview', { signal: controller.signal }),
        fetch('/api/v1/dashboard/recent-episodes?limit=6', { signal: controller.signal }),
        fetch('/api/v1/projects/summary?limit=5', { signal: controller.signal }),
        fetch('/api/v1/dashboard/trend?days=30', { signal: controller.signal }),
    ]);

    if (loadEpoch !== _dashLoadEpoch) return;

    const [overview, recent, projectSummary, trend] = results;
    try {
        if (overview.status !== 'fulfilled' || !overview.value.ok) {
            throw new Error('Overview request failed');
        }
        renderStats(await overview.value.json());
        hideError();
    } catch (e) { showError(); }
    try {
        if (recent.status === 'fulfilled' && recent.value.ok) {
            renderRecent((await recent.value.json()).episodes || []);
        }
    } catch (e) { /* silent */ }
    try {
        if (projectSummary.status === 'fulfilled' && projectSummary.value.ok) {
            renderProjectSummary((await projectSummary.value.json()).tasks || []);
        }
    } catch (e) { /* silent */ }
    try {
        if (trend.status === 'fulfilled' && trend.value.ok) {
            const trendData = await trend.value.json();
            renderTrend(trendData.labels || [], trendData.counts || []);
        }
    } catch (e) { /* silent */ }

    if (btn && _dashController === controller) {
        btn.disabled = false;
        var icon = btn.querySelector('iconify-icon');
        if (icon) icon.setAttribute('icon', 'ant-design:reload-outlined');
    }
    if (_dashController === controller) _dashController = null;
}


// ── 4 Stat cards ──────────────────────────────────────

function setCardValue(id, value) {
    var el = document.getElementById(id);
    if (el) el.textContent = fmtNum(value);
}

function setCardToday(id, today) {
    var el = document.getElementById(id);
    if (!el) return;
    var text = fmtToday(today);
    if (text) {
        el.textContent = text;
        el.classList.remove('hidden');
    } else {
        el.classList.add('hidden');
    }
}

function renderStats(data) {
    document.querySelectorAll('.stat-card-skeleton').forEach(function(el) { el.remove(); });
    var real = document.getElementById('stat-cards-real');
    if (real) real.classList.remove('hidden');

    setCardValue('stat-total-total', data.total ? data.total.total : 0);
    setCardToday('stat-total-today', data.total ? data.total.today : 0);

    var reviewing = data.reviewing ? data.reviewing.total : 0;
    var approved  = data.approved  ? data.approved.total  : 0;
    var failed    = data.failed    ? data.failed.total    : 0;

    setCardValue('stat-reviewing-total', reviewing);
    setCardToday('stat-reviewing-today', data.reviewing ? data.reviewing.today : 0);
    setCardValue('stat-approved-total', approved);
    setCardToday('stat-approved-today', data.approved ? data.approved.today : 0);
    setCardValue('stat-failed-total', failed);

    // Donut chart
    renderDonut(reviewing, approved, failed);

}


// ── Projects (原 Task progress,任务概念已移除) ───────

function renderProjectSummary(projects) {
    var el = document.getElementById('task-progress-list');
    if (!el) return;
    if (!projects || projects.length === 0) {
        el.innerHTML = `<div class="text-center text-gray-600 text-sm py-4">${t('no_projects')}</div>`;
        return;
    }

    el.innerHTML = projects.map(function(p) {
        return '<a href="/tasks" class="block mb-3 last:mb-0 hover:bg-gray-800/30 rounded p-2 -mx-2 transition-colors">' +
            '<div class="flex items-center justify-between text-xs">' +
                '<span class="flex items-center gap-1.5 text-gray-300 truncate">' +
                    '<iconify-icon icon="ant-design:folder-outlined" class="icon-sm text-blue-500 flex-shrink-0"></iconify-icon>' +
                    escHtml(p.task_name || '') +
                '</span>' +
                '<span class="text-gray-500 ml-2 flex-shrink-0">' + fmtNum(p.total_episodes) + ' episodes</span>' +
            '</div>' +
        '</a>';
    }).join('');
}


// ── Recent episodes ──────────────────────────────────

function renderRecent(episodes) {
    var el = document.getElementById('recent-list');
    if (!el) return;
    if (!episodes || episodes.length === 0) {
        el.innerHTML = '<div class="text-center text-gray-600 text-sm py-4">No episodes yet</div>';
        return;
    }

    el.innerHTML = episodes.map(function(ep) {
        var cleanIcon = '';
        if (ep.cleaning_passed === false) {
            cleanIcon = ' <iconify-icon icon="ant-design:warning-filled" class="text-red-400 icon-sm" title="Cleaning failed"></iconify-icon>';
        } else if (ep.cleaning_passed === true) {
            cleanIcon = ' <iconify-icon icon="ant-design:check-circle-filled" class="text-green-400 icon-sm" title="Cleaning passed"></iconify-icon>';
        }
        return '<a href="/review" class="flex items-center gap-3 px-2 py-2 rounded hover:bg-gray-800/50 transition-colors text-xs border-b border-gray-800/50 last:border-0">' +
            '<span class="text-gray-300 w-24 truncate flex-shrink-0">' + (ep.task_name || '').slice(0, 18) + cleanIcon + '</span>' +
            '<span class="text-gray-500 w-12 text-right flex-shrink-0">' + (ep.frame_count || 0) + 'f</span>' +
            '<span class="w-20 flex-shrink-0">' + statusBadge(ep.status) + '</span>' +
            '<span class="text-gray-600 ml-auto flex-shrink-0">' + timeAgo(ep.received_at) + '</span>' +
        '</a>';
    }).join('');
}


// ── Donut chart (SVG) ────────────────────────────────

function renderDonut(reviewing, approved, failed) {
    var el = document.getElementById('donut-chart');
    if (!el) return;

    var total = reviewing + approved + failed || 1;

    var segments = [
        { value: reviewing, color: '#eab308', label: 'To Review' },  // yellow
        { value: approved,  color: '#22c55e', label: 'Approved'  },  // green
        { value: failed,    color: '#ef4444', label: 'Failed'    },  // red
    ];

    var radius = 60;
    var stroke = 16;
    var circumference = 2 * Math.PI * radius;
    var cx = 100, cy = 100, size = 200;

    // Build SVG
    var html = '<svg viewBox="0 0 ' + size + ' ' + size + '" class="w-48 h-48">';

    // Background ring
    html += '<circle cx="' + cx + '" cy="' + cy + '" r="' + radius + '" ' +
        'fill="none" stroke="#1f2937" stroke-width="' + stroke + '"/>';

    // Segments
    var offset = -Math.PI / 2; // start from top
    segments.forEach(function(seg) {
        if (seg.value <= 0) return;
        var dashLen = (seg.value / total) * circumference;
        var dashGap = circumference - dashLen;
        var rotation = (offset / (2 * Math.PI)) * 360;
        html += '<circle cx="' + cx + '" cy="' + cy + '" r="' + radius + '" ' +
            'fill="none" stroke="' + seg.color + '" stroke-width="' + stroke + '" ' +
            'stroke-linecap="butt" ' +
            'stroke-dasharray="' + dashLen.toFixed(1) + ' ' + dashGap.toFixed(1) + '" ' +
            'transform="rotate(' + rotation.toFixed(1) + ' ' + cx + ' ' + cy + ')" ' +
            'style="transition: stroke-dasharray 0.6s ease"/>';
        offset += (seg.value / total) * 2 * Math.PI;
    });

    // Center text
    html += '<text x="' + cx + '" y="' + (cy - 8) + '" text-anchor="middle" ' +
        'class="fill-white font-bold" style="font-size:22px">' + fmtNum(total) + '</text>';
    html += '<text x="' + cx + '" y="' + (cy + 12) + '" text-anchor="middle" ' +
        'class="fill-gray-500" style="font-size:11px">Total</text>';

    html += '</svg>';

    // Legend
    html += '<div class="flex items-center gap-4 ml-4 text-xs">';
    segments.forEach(function(seg) {
        html += '<div class="flex items-center gap-1.5">' +
            '<span class="w-2.5 h-2.5 rounded-sm flex-shrink-0" style="background:' + seg.color + '"></span>' +
            '<span class="text-gray-500">' + seg.label + '</span>' +
            '<span class="text-gray-300 font-mono">' + fmtNum(seg.value) + '</span>' +
        '</div>';
    });
    html += '</div>';

    el.innerHTML = html;
}


// ── Trend bar chart (div) ────────────────────────────

function renderTrend(labels, counts) {
    var el = document.getElementById('trend-chart');
    if (!el || !labels.length) return;

    var max = 0;
    for (var i = 0; i < counts.length; i++) {
        if (counts[i] > max) max = counts[i];
    }
    max = max || 1;

    var html = '<div class="flex items-end gap-px h-40">';
    for (var i = 0; i < labels.length; i++) {
        var h = Math.max(4, Math.round((counts[i] / max) * 100));
        var isMax = counts[i] === max && max > 0;
        var barCls = isMax
            ? 'bg-blue-500 hover:bg-blue-400'
            : 'bg-gray-700 hover:bg-gray-600';
        html += '<div class="flex-1 flex flex-col items-center justify-end group" style="height:100%">' +
            '<span class="text-xs text-gray-600 mb-1 group-hover:text-gray-300" style="font-size:9px">' + (counts[i] || '') + '</span>' +
            '<div class="w-full ' + barCls + ' rounded-t transition-colors" style="height:' + (counts[i] ? h + '%' : '4px') + '"></div>' +
        '</div>';
    }
    html += '</div>';

    // X-axis labels (show ~7 evenly spaced)
    var step = Math.max(1, Math.floor(labels.length / 7));
    html += '<div class="flex gap-px mt-1.5">';
    for (var i = 0; i < labels.length; i++) {
        html += '<div class="flex-1 text-center">';
        if (i % step === 0) {
            html += '<span class="text-xs text-gray-600" style="font-size:9px">' + labels[i] + '</span>';
        }
        html += '</div>';
    }
    html += '</div>';

    el.innerHTML = html;
}


// ── Error handling ───────────────────────────────────

function showError() {
    document.querySelectorAll('.stat-card-skeleton').forEach(function(el) { el.remove(); });
    var real = document.getElementById('stat-cards-real');
    if (real) real.classList.remove('hidden');
    var err = document.getElementById('stat-error');
    if (err) err.classList.remove('hidden');
}

function hideError() {
    var err = document.getElementById('stat-error');
    if (err) err.classList.add('hidden');
}

function escHtml(s) {
    if (!s) return '';
    return s.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;');
}


// ── Auto-refresh (30s) ───────────────────────────────

function startAutoRefresh() {
    stopAutoRefresh();
    _dashTimer = setInterval(function() {
        if (_dashVisible) refreshDashboard();
    }, 30000);
}

function stopAutoRefresh() {
    if (_dashTimer) { clearInterval(_dashTimer); _dashTimer = null; }
}

document.addEventListener('visibilitychange', function() {
    _dashVisible = !document.hidden;
    if (_dashVisible) refreshDashboard();
});


// ── Init ─────────────────────────────────────────────

document.addEventListener('DOMContentLoaded', function() {
    refreshDashboard();
    startAutoRefresh();
});
