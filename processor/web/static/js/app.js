/* Episode list, info panel, review/delete/export */

function escHtml(s) {
    if (!s) return '';
    return String(s).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;');
}

let selectedEpisodes = new Set();  // batch selection
let allEpisodes = [];
let hierarchyData = [];            // Project → Task → Episode tree for the list view
const _expandedProjects = new Set();  // expanded project ids (default: collapsed)
const _collapsedTasks = new Set();     // collapsed task ids
let _loadEpoch = 0;  // stale-response guard — discard out-of-order poll results
let _formatEpoch = 0;  // 导出格式徽标防竞态:快速切换批次时丢弃过期响应
let _silentReloadTimer = null;  // coalesce background list refreshes after actions
let _listLoadController = null; // abort obsolete hierarchy requests on fast navigation
let _episodesLoadedAt = 0;      // snapshot freshness for background reconciliation
let selectMode = false;  // selection mode toggle — hides checkboxes by default

// A short-lived cross-navigation snapshot makes the Review page paint
// immediately after leaving Projects/Workflow. The server remains
// authoritative: loadEpisodes still refreshes in the background.
const REVIEW_SNAPSHOT_CACHE_KEY = 'egodata.review.hierarchy.v1';
const REVIEW_SNAPSHOT_CACHE_TTL = 60 * 1000;

function readReviewSnapshotCache() {
    try {
        const cached = JSON.parse(sessionStorage.getItem(REVIEW_SNAPSHOT_CACHE_KEY) || 'null');
        if (!cached || !Array.isArray(cached.projects)
                || Date.now() - Number(cached.savedAt || 0) > REVIEW_SNAPSHOT_CACHE_TTL) return null;
        return cached;
    } catch (_) { return null; }
}

function saveReviewSnapshotCache(projects) {
    try {
        sessionStorage.setItem(REVIEW_SNAPSHOT_CACHE_KEY, JSON.stringify({
            savedAt: Date.now(), projects,
        }));
    } catch (_) { /* storage quota/private mode — network refresh still works */ }
}

function applyHierarchySnapshot(projects, savedAt = Date.now()) {
    hierarchyData = projects || [];
    _episodesLoadedAt = savedAt;
    allEpisodes = hierarchyData.flatMap(node =>
        (node.episodes || []).map(e => ({ ...e, task_description: node.project.name })));
    const visible = visibleEpisodesForCurrentView();
    updateTaskFilter(visible);
    updateEpisodeListCount(allEpisodes, '', document.getElementById('filter-task')?.value || '');
    renderEpisodeCards(hierarchyForCurrentView(), '', document.getElementById('filter-task')?.value || '');
}

// ── Nav tree → filter bridge (called from base.html) ──

function applyNavFilter(statusParam, taskName) {
    // Set hidden filter controls
    const statusSel = document.getElementById('filter-status');
    const taskSel = document.getElementById('filter-task');
    const searchInput = document.getElementById('search-input');
    if (statusSel) statusSel.value = statusParam;
    const requestedTask = taskName || '';
    if (taskSel) taskSel.value = requestedTask;
    if (searchInput) searchInput.value = requestedTask;
    // Update URL
    const url = new URL(document.location);
    url.searchParams.set('status', statusParam);
    url.searchParams.set('search', requestedTask);
    history.replaceState({}, '', url);
    // Close detail panel (if open), then filter the in-memory snapshot.
    backToList();
    if (hierarchyData.length) {
        renderEpisodeListFromMemory();
        scheduleSilentEpisodeRefresh(250);
    } else {
        loadEpisodes();
    }
    // Highlight nav tree
    if (typeof window.highlightNavActive === 'function') {
        window.highlightNavActive(statusParam, taskName);
    }
}
window.applyNavFilter = applyNavFilter;

function syncReviewShortcutActive(status) {
    const key = status === 'reviewed' ? 'approved'
        : (status === 'failed' ? 'failed' : 'reviewing');
    document.querySelectorAll('.sidebar-sub[data-review-status]').forEach(link => {
        link.classList.toggle('active', link.dataset.reviewStatus === key);
    });
}

function switchReviewStatus(status, options = {}) {
    const allowed = new Set(['completed', 'to_review', 'reviewed', 'failed']);
    const nextStatus = allowed.has(status) ? status : 'completed';
    const statusSel = document.getElementById('filter-status');
    const taskSel = document.getElementById('filter-task');
    const searchInput = document.getElementById('search-input');
    if (statusSel) statusSel.value = nextStatus;
    if (taskSel) taskSel.value = '';
    if (searchInput) searchInput.value = '';

    const url = new URL(window.location.href);
    url.searchParams.set('status', nextStatus);
    url.searchParams.delete('search');
    if (options.replace) history.replaceState({}, '', url);
    else history.pushState({}, '', url);

    syncReviewShortcutActive(nextStatus);
    backToList();
    if (hierarchyData.length) {
        renderEpisodeListFromMemory();
        // Keep the switch instant; only reconcile in the background when the
        // current snapshot is older than a moment.
        if (Date.now() - _episodesLoadedAt > 1000) scheduleSilentEpisodeRefresh(200);
    } else {
        loadEpisodes();
    }
}
window.switchReviewStatus = switchReviewStatus;

function installReviewStatusNavigation() {
    if (window.location.pathname !== '/review') return;
    const statusByShortcut = { reviewing: 'completed', approved: 'reviewed', failed: 'failed' };
    document.querySelectorAll('.sidebar-sub[data-review-status]').forEach(link => {
        link.addEventListener('click', event => {
            event.preventDefault();
            switchReviewStatus(statusByShortcut[link.dataset.reviewStatus] || 'completed');
        });
    });
    window.addEventListener('popstate', () => {
        const params = new URLSearchParams(window.location.search);
        const status = params.get('status') || 'completed';
        const statusSel = document.getElementById('filter-status');
        const searchInput = document.getElementById('search-input');
        if (statusSel) statusSel.value = status;
        if (searchInput) searchInput.value = params.get('search') || '';
        syncReviewShortcutActive(status);
        if (hierarchyData.length) renderEpisodeListFromMemory();
        else loadEpisodes();
    });
}

// ── Init from URL param ─────────────────────────────

function initStatusFromURL() {
    const params = new URLSearchParams(document.location.search);
    const statusParam = params.get('status') || 'completed';
    const sel = document.getElementById('filter-status');
    if (sel) sel.value = statusParam;

    const searchParam = params.get('search') || '';
    const searchInput = document.getElementById('search-input');
    if (searchInput && searchParam) searchInput.value = searchParam;
}

// ── Sort order ──────────────────────────────────────

const SORT_MODES = [
    { dir: 'asc',  label: 'Name 1→N', icon: 'ant-design:ordered-list-outlined' },
    { dir: 'desc', label: 'Name N→1', icon: 'ant-design:ordered-list-outlined' },
];
let sortModeIdx = 0;  // cycles 0→1→2→3→0

function currentSortMode() {
    return SORT_MODES[sortModeIdx];
}

function toggleSortOrder() {
    sortModeIdx = (sortModeIdx + 1) % SORT_MODES.length;
    const mode = currentSortMode();
    const icon = document.getElementById('sort-order-icon');
    const label = document.getElementById('sort-order-label');
    const btn = document.getElementById('btn-sort-order');
    if (icon) icon.setAttribute('icon', mode.icon);
    if (label) label.textContent = mode.label;
    if (btn) btn.title = 'Sort: ' + mode.label;
    // Re-sort in memory (no network round trip) — only refetch if not yet loaded
    if (hierarchyData.length === 0) {
        loadEpisodes();
    } else {
        renderEpisodeListFromMemory();
    }
}

// Extract trailing numeric suffix: "Chew_gum_0005" → 5, "ep_00020" → 20, "no_number" → 0
function extractEpisodeNumber(name) {
    const m = String(name || '').match(/(\d+)\s*$/);
    return m ? parseInt(m[1], 10) : 0;
}

// ── Selection mode ──────────────────────────────────

function toggleSelectMode() {
    selectMode = !selectMode;
    if (!selectMode) {
        selectedEpisodes.clear();  // exit selection → clear all
    } else {
        // 批量选择模式:自动展开所有项目,否则折叠状态下无法勾选视频
        (hierarchyData || []).forEach(node => {
            if (node.project && node.project.id) _expandedProjects.add(node.project.id);
        });
    }
    const btn = document.getElementById('btn-toggle-select');
    if (btn) {
        btn.textContent = selectMode ? 'Cancel' : 'Select';
        btn.className = selectMode
            ? 'bg-blue-600 hover:bg-blue-500 text-white text-xs px-2 py-0.5 rounded flex-shrink-0 transition-colors'
            : 'bg-gray-800 hover:bg-gray-700 text-gray-400 hover:text-gray-200 text-xs px-2 py-0.5 rounded flex-shrink-0 transition-colors';
    }
    renderEpisodeListFromMemory();
}

// ── Load ────────────────────────────────────────────

function currentReviewStatus() {
    return document.getElementById('filter-status')?.value || 'completed';
}

function isEpisodeTemporarilyHidden(ep) {
    // Reprocessing invalidates the previous review artifact immediately. Keep
    // the episode out of every list until the server publishes the new state;
    // otherwise the old card can remain visible as "Processing…".
    return Boolean(ep && (ep.status === 'processing'
        || ep._uiWorkflowState === 'queued'));
}

function matchesReviewStatus(status, filter = currentReviewStatus()) {
    if (!filter) return true;
    if (filter === 'completed' || filter === 'to_review') {
        if (status === 'completed' || status === 'to_review') return true;
        // processing 由 hierarchyForCurrentView 统一隐藏，避免用户打开
        // 旧产物；处理完成后由下一次静默刷新重新出现。
        return false;
    }
    if (filter === 'reviewed') {
        return status === 'reviewed' || status === 'approved';
    }
    return status === filter;
}

function matchesEpisodeSearch(ep, projectName, rawSearch) {
    const search = String(rawSearch || '').trim().toLowerCase();
    if (!search) return true;
    return (ep.name || '').toLowerCase().includes(search)
        || String(ep.id || '').toLowerCase().includes(search)
        || String(projectName || '').toLowerCase().includes(search);
}

// The server snapshot contains all live episode metadata. Status/search
// filtering happens here, so Reviewing ↔ Approved never starts another scan.
function hierarchyForCurrentView() {
    const status = currentReviewStatus();
    const search = document.getElementById('search-input')?.value || '';
    return hierarchyData.map(node => ({
        ...node,
        episodes: (node.episodes || []).filter(ep =>
            !isEpisodeTemporarilyHidden(ep)
            && matchesReviewStatus(ep.status, status)
            && matchesEpisodeSearch(ep, node.project?.name, search)),
    }));
}

function visibleEpisodesForCurrentView() {
    return hierarchyForCurrentView().flatMap(node =>
        (node.episodes || []).map(ep => ({ ...ep, task_description: node.project?.name })));
}

async function loadEpisodes(options = {}) {
    const silent = Boolean(options.silent);
    if (!silent && _silentReloadTimer) {
        clearTimeout(_silentReloadTimer);
        _silentReloadTimer = null;
    }
    const taskFilter = document.getElementById('filter-task')?.value || '';
    const listEl = document.getElementById('episode-list-inline');
    if (!listEl) return;

    // Paint the last successful hierarchy immediately on a new page load.
    // A normal request below still reconciles status/uploads in the background.
    if (!silent && hierarchyData.length === 0) {
        const cached = readReviewSnapshotCache();
        if (cached) applyHierarchySnapshot(cached.projects, Number(cached.savedAt) || Date.now());
    }

    // Save scroll position before innerHTML rebuild (prevent 15s poll reset)
    const scrollContainer = document.getElementById('episode-list-section');
    const savedScrollTop = scrollContainer ? scrollContainer.scrollTop : 0;

    if (!silent && hierarchyData.length === 0) {
        listEl.innerHTML = `<div class="p-4 text-center text-gray-500 text-sm">${t('loading')}</div>`;
    }

    if (_listLoadController) _listLoadController.abort();
    const controller = new AbortController();
    _listLoadController = controller;
    const fetchEpoch = ++_loadEpoch;
    try {
        // Hierarchy view: Project → Task (upload batch) → Episodes.
        // Status and search are applied after the full snapshot arrives.

        // ``fetchEpoch`` was captured before dispatch so stale responses are ignored.
        const res = await fetch('/api/v1/projects/hierarchy', { signal: controller.signal });
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        const data = await res.json();
        if (fetchEpoch !== _loadEpoch) return;  // newer request in flight — discard
        saveReviewSnapshotCache(data.projects || []);
        applyHierarchySnapshot(data.projects || []);
        // Restore scroll position after render
        if (scrollContainer) {
            requestAnimationFrame(() => { scrollContainer.scrollTop = savedScrollTop; });
        }
    } catch (err) {
        if (err?.name === 'AbortError') return;
        if (!silent && hierarchyData.length === 0) {
            listEl.innerHTML = `<div class="p-4 text-center text-red-400 text-sm">${t('load_failed')}: ${err.message}</div>`;
        }
    } finally {
        if (_listLoadController === controller) _listLoadController = null;
    }
}


function updateEpisodeListCount(episodes, search, taskFilter) {
    const countEl = document.getElementById('episode-list-count');
    if (!countEl) return;
    let filtered = episodes.filter(ep => !isEpisodeTemporarilyHidden(ep)
        && matchesReviewStatus(ep.status));
    if (taskFilter) filtered = filtered.filter(ep => (ep.task_description || '') === taskFilter);
    if (search) {
        const s = search.toLowerCase();
        filtered = filtered.filter(ep =>
            (ep.name || '').toLowerCase().includes(s) ||
            (ep.task_description || '').toLowerCase().includes(s) ||
            String(ep.id).toLowerCase().includes(s));
    }
    countEl.textContent = filtered.length + ' episodes';
}

function updateTaskFilter(episodes) {
    // filter-task is now a hidden input (not a select). Values are set by applyNavFilter().
    // This function remains as a hook: clear the value if the task no longer exists in results.
    const sel = document.getElementById('filter-task');
    if (!sel) return;
    const current = sel.value;
    if (current) {
        const taskNames = new Set(episodes.map(e => e.task_description || 'unknown'));
        if (!taskNames.has(current)) {
            sel.value = '';
        }
    }
}


function renderEpisodeCards(hierarchy, search, taskFilter) {
    const listEl = document.getElementById('episode-list-inline');
    if (!listEl) return;

    // Auto-exit selection mode when switching away from Reviewed
    const statusFilter = document.getElementById('filter-status')?.value || '';
    if (selectMode && statusFilter !== 'reviewed') {
        selectMode = false;
        selectedEpisodes.clear();
    }

    // 项目文件夹**始终渲染**(含空项目/刚创建的项目):几个项目就显示几个
    // 文件夹,哪怕里面没有批次;只有没有任何项目时才显示 No data。
    if (hierarchy.length === 0) {
        listEl.innerHTML = `<div class="p-4 text-center text-gray-500 text-sm">${t('no_data')}</div>`;
        updateBatchUI(visibleEpisodesForCurrentView());
        return;
    }

    // 任务概念已移除:项目 → Episodes 两层;组内按 Name 1→N / N→1 按钮排序
    const sortMode = currentSortMode();
    const dir = sortMode.dir === 'asc' ? 1 : -1;
    let html = '';
    hierarchy.forEach((node) => {
        const episodes = (node.episodes || []).filter(ep => {
            if (taskFilter && (node.project.name || '') !== taskFilter) return false;
            return true;
        });
        if (taskFilter && (node.project.name || '') !== taskFilter) return;
        // 组内按尾部序号排序(Test1_000012 → 12),只排副本不动原始数据
        const sorted = [...episodes].sort((a, b) =>
            (extractEpisodeNumber(a.name) - extractEpisodeNumber(b.name)) * dir);
        // 默认折叠:只在用户点击展开过的项目才展开视频卡片
        const pCollapsed = !_expandedProjects.has(node.project.id);
        html += `
        <div class="project-group border-b border-gray-800">
            <div class="project-header px-3 py-2 flex items-center gap-2 cursor-pointer hover:bg-gray-800/50 select-none"
                 onclick="toggleProjectCollapse('${node.project.id}')">
                <iconify-icon icon="ant-design:folder-outlined" class="text-blue-500"></iconify-icon>
                <span class="text-sm font-medium text-gray-200 truncate">${node.project.name}</span>
                <span class="text-xs text-gray-500 flex-shrink-0">${sorted.length}</span>
                <iconify-icon icon="ant-design:${pCollapsed ? 'right' : 'down'}-outlined" class="text-gray-600 ml-auto"></iconify-icon>
            </div>
            <div class="project-body ${pCollapsed ? 'hidden' : ''}">`;
        if (sorted.length === 0) {
            html += `<div class="text-center text-gray-600 text-xs py-2">${t('no_tasks_in_project')}</div>`;
        }
        sorted.forEach(ep => { html += episodeCardHtml(ep, node.project.name); });
        html += `</div></div>`;
    });
    listEl.innerHTML = html;
    updateBatchUI(visibleEpisodesForCurrentView());
}

function toggleProjectCollapse(projectId) {
    if (_expandedProjects.has(projectId)) _expandedProjects.delete(projectId);
    else _expandedProjects.add(projectId);
    renderEpisodeListFromMemory();
}

// Apply small action results locally so the list responds immediately. The
// server remains authoritative; a coalesced silent refresh reconciles state
// after the filesystem-backed hierarchy has caught up.
function renderEpisodeListFromMemory() {
    const taskFilter = document.getElementById('filter-task')?.value || '';
    const visible = visibleEpisodesForCurrentView();
    updateTaskFilter(visible);
    updateEpisodeListCount(allEpisodes, '', taskFilter);
    renderEpisodeCards(hierarchyForCurrentView(), '', taskFilter);
}

function removeEpisodeFromLocalList(episodeId) {
    const id = String(episodeId);
    selectedEpisodes.delete(id);
    hierarchyData.forEach(node => {
        node.episodes = (node.episodes || []).filter(ep => String(ep.id) !== id);
    });
    allEpisodes = allEpisodes.filter(ep => String(ep.id) !== id);
    renderEpisodeListFromMemory();
}

function updateEpisodeInLocalList(episodeId, patch) {
    const id = String(episodeId);
    hierarchyData.forEach(node => {
        const ep = (node.episodes || []).find(item => String(item.id) === id);
        if (ep) Object.assign(ep, patch);
    });
    const flat = allEpisodes.find(ep => String(ep.id) === id);
    if (flat) Object.assign(flat, patch);
    renderEpisodeListFromMemory();
}

function scheduleSilentEpisodeRefresh(delay = 1200) {
    clearTimeout(_silentReloadTimer);
    _silentReloadTimer = setTimeout(() => {
        _silentReloadTimer = null;
        loadEpisodes({ silent: true });
    }, delay);
}

function toggleTaskCollapse(taskId) {
    if (_collapsedTasks.has(taskId)) _collapsedTasks.delete(taskId);
    else _collapsedTasks.add(taskId);
    renderEpisodeListFromMemory();
}

function episodeCardHtml(ep, taskName) {
    const activeClass = currentEpisodeId === ep.id ? 'active' : '';
    // 前端入队标记(点击后立即显示)或后端真实状态(静默刷新后保持显示)
    const isWorkflowQueued = ep._uiWorkflowState === 'queued' || ep.status === 'processing';
    const cardClass = isWorkflowQueued
        ? 'cursor-default opacity-80'
        : 'cursor-pointer';
    const cardClick = isWorkflowQueued ? '' : `onclick="selectEpisode('${ep.id}')"`;
    const isReviewed = ep.status === 'reviewed' || ep.status === 'approved';
    const isFailed = ep.status === 'failed';
    const statusDot = isWorkflowQueued
        ? '<span class="inline-block w-2 h-2 rounded-full bg-blue-400 mr-1"></span>'
        : (isFailed
        ? '<span class="inline-block w-2 h-2 rounded-full bg-red-400 mr-1"></span>'
        : (isReviewed
            ? '<span class="inline-block w-2 h-2 rounded-full bg-green-400 mr-1"></span>'
            : '<span class="inline-block w-2 h-2 rounded-full bg-yellow-400 mr-1"></span>'));
    const statusText = isWorkflowQueued ? t('processing')
        : (isFailed ? t('stat_failed') : (isReviewed ? t('reviewed') : t('reviewing')));
    const statusColor = isWorkflowQueued ? 'text-blue-400'
        : (isFailed ? 'text-red-400' : (isReviewed ? 'text-green-400' : 'text-yellow-400'));
    const cameraCount = (ep.camera_names || []).length;
    const timestamp = ep.timestamp || '';
    const time = timestamp ? timestamp : new Date(ep.created_at).toLocaleDateString('zh-CN');

    const checked = selectedEpisodes.has(ep.id) ? 'checked' : '';
    const checkbox = (isReviewed && selectMode)
        ? `<input type="checkbox" class="batch-checkbox w-3.5 h-3.5 rounded accent-blue-600 flex-shrink-0"
                  data-episode-id="${ep.id}" ${checked}
                  onclick="event.stopPropagation();toggleBatchSelect('${ep.id}', this.checked)">`
        : '';

    const isAnnotationPage = window.EGODATA_PAGE_MODE === 'annotation';
    const statusFilter = document.getElementById('filter-status')?.value
        || new URLSearchParams(document.location.search).get('status')
        || 'completed';
    const isReviewingMode = statusFilter === 'completed' || statusFilter === 'to_review';
    const userRole = String(window.__EGO_USER__?.role || '').toLowerCase();
    const canReprocess = userRole === 'admin' || userRole === 'engineer';
    // processing(含前端入队标记)显示禁用的"处理中"状态按钮,不要求角色;
    // 主动重跑按钮仍只对 admin/engineer 开放且要求 completed/to_review。
    const workflowButton = !isAnnotationPage && isReviewingMode
        ? (isWorkflowQueued
            ? `<button disabled
                       class="mt-1.5 w-full bg-blue-950/60 text-blue-300/80 text-xs px-3 py-1 rounded cursor-wait">
                       <iconify-icon icon="ant-design:loading-outlined" class="icon-sm"></iconify-icon> ${t('processing')}</button>`
            : (canReprocess && (ep.status === 'completed' || ep.status === 'to_review')
                ? `<button onclick="event.stopPropagation();reprocessEpisode('${ep.id}')"
                     title="Re-run the bound workflow for this episode"
                     class="mt-1.5 w-full bg-purple-900/70 hover:bg-purple-800 text-purple-200 text-xs px-3 py-1 rounded">
                     <iconify-icon icon="ant-design:tool-outlined" class="icon-sm"></iconify-icon> ${t('reprocess')}</button>`
                : ''))
        : '';
    const buttons = isAnnotationPage ? '' : (isFailed
        ? `<div class="flex gap-2">
             <button onclick="event.stopPropagation();retryEpisode('${ep.id}')"
                     class="bg-blue-800 hover:bg-blue-700 text-blue-200 text-xs px-3 py-1 rounded flex-1"><iconify-icon icon="ant-design:reload-outlined" class="icon-sm"></iconify-icon> ${t('retry_review')}</button>
             <button onclick="event.stopPropagation();deleteEpisode('${ep.id}')"
                     class="bg-gray-800 hover:bg-red-900 text-gray-400 hover:text-red-300 text-xs px-3 py-1 rounded flex-1"><iconify-icon icon="ant-design:delete-outlined" class="icon-sm"></iconify-icon> ${t('delete')}</button>
           </div>`
        : (isReviewed
            ? `<div class="flex gap-2">
                 <button onclick="event.stopPropagation();downloadEpisode('${ep.id}')"
                         title="${t('export_hint')}"
                         class="bg-blue-800 hover:bg-blue-700 text-blue-200 text-xs px-3 py-1 rounded flex-1"><iconify-icon icon="ant-design:export-outlined" class="icon-sm"></iconify-icon> ${t('export')}</button>
                 <button onclick="event.stopPropagation();unreviewEpisode('${ep.id}')"
                         class="bg-yellow-800 hover:bg-yellow-700 text-yellow-200 text-xs px-3 py-1 rounded flex-1"><iconify-icon icon="ant-design:rollback-outlined" class="icon-sm"></iconify-icon> ${t('unreview')}</button>
               </div>`
            : `<div class="flex gap-2">
                 <button onclick="event.stopPropagation();markReviewed('${ep.id}')"
                         class="bg-green-800 hover:bg-green-700 text-green-200 text-xs px-3 py-1 rounded flex-1"><iconify-icon icon="ant-design:check-outlined" class="icon-sm"></iconify-icon> ${t('approve')}</button>
                 <button onclick="event.stopPropagation();deleteEpisode('${ep.id}')"
                         class="bg-gray-800 hover:bg-red-900 text-gray-400 hover:text-red-300 text-xs px-3 py-1 rounded"><iconify-icon icon="ant-design:delete-outlined" class="icon-sm"></iconify-icon></button>
               </div>${workflowButton}`));

    return `
    <div class="episode-card ${activeClass} p-3 border-b border-gray-800 hover:bg-gray-800/50 ${cardClass} flex gap-2"
         data-episode-id="${ep.id}"
         ${cardClick}
         ${isWorkflowQueued ? `title="${t('processing')}"` : ''}>
        ${checkbox}
        <div class="flex-1 min-w-0">
        <div class="flex items-center gap-1.5 mb-2">
            ${statusDot}
            <span class="text-xs ${statusColor}">${statusText}</span>
        </div>
        <div class="flex items-center gap-1.5 mb-1">
            ${typeof ep.episode_index === 'number'
                ? `<div class="text-sm font-semibold text-blue-300 truncate" title="${ep.name || ''}">#${ep.episode_index}</div>`
                : `<div class="text-sm truncate text-gray-200">${ep.name || taskName || ep.id.slice(0, 8)}</div>`}
        </div>
        <div class="text-xs text-gray-500 space-y-0.5 mb-2">
            <div><iconify-icon icon="ant-design:field-time-outlined" class="icon-sm"></iconify-icon> ${ep.fps || 30}FPS · ${ep.frame_count || 0}${t('frame')}</div>
            ${cameraCount > 0 ? `<div><iconify-icon icon="ant-design:video-camera-outlined" class="icon-sm"></iconify-icon> ${cameraCount} ${t('cameras')}</div>` : ''}
            <div><iconify-icon icon="ant-design:calendar-outlined" class="icon-sm"></iconify-icon> ${time}</div>
        </div>
        ${buttons}
        </div>
    </div>`;
}


// ── Back to list ─────────────────────────────────────

function backToList() {
    // Closing detail must release decoders, 3D canvases, timers, sockets and
    // in-flight media requests. Hiding the panel alone leaves the old video
    // workload alive behind Reviewing/Approved and makes the next click lag.
    if (typeof clearEpisodeWorkspace === 'function') clearEpisodeWorkspace();
    const detail = document.getElementById('episode-detail');
    const listSection = document.getElementById('episode-list-section');
    if (detail) detail.classList.add('hidden');
    if (listSection) listSection.classList.remove('hidden');
    // 返回批次列表 → 隐藏漂浮预览控件(点开视频后才有)
    if (typeof setPreviewPanelsVisible === 'function') setPreviewPanelsVisible(false);
    const warning = document.getElementById('input-warning-banner');
    if (warning) {
        warning.classList.add('hidden');
        warning.innerHTML = '';
    }
}

function renderInputWarning(ep) {
    const warning = document.getElementById('input-warning-banner');
    if (!warning) return;
    const missing = [];
    const matched = [];
    (ep.exceptions || []).filter(ex => ex.kind === 'input_missing').forEach(ex => {
        (ex.missing || []).forEach(value => {
            if (!missing.includes(String(value))) missing.push(String(value));
        });
        (ex.matched || []).forEach(value => {
            if (!matched.includes(String(value))) matched.push(String(value));
        });
    });
    if (!missing.length) {
        warning.classList.add('hidden');
        warning.innerHTML = '';
        return;
    }
    warning.innerHTML =
        '<div class="flex items-start gap-2">' +
            '<iconify-icon icon="ant-design:warning-filled" class="text-amber-300 text-base flex-shrink-0 mt-0.5"></iconify-icon>' +
            '<div class="min-w-0">' +
                '<div class="font-medium text-amber-100">' + escHtml(t('input_missing_title')) + '</div>' +
                '<div class="mt-0.5 text-amber-300/90">' + escHtml(t('input_missing_detail')) + '</div>' +
                '<div class="mt-1 font-mono break-words">' + escHtml(missing.join(', ')) + '</div>' +
                (matched.length ? '<div class="mt-1 text-emerald-300/90">✓ ' + escHtml(matched.join(', ')) + '</div>' : '') +
            '</div>' +
        '</div>';
    warning.classList.remove('hidden');
}

// ── Batch selection ──────────────────────────────────

function toggleBatchSelect(episodeId, checked) {
    if (checked) {
        selectedEpisodes.add(episodeId);
    } else {
        selectedEpisodes.delete(episodeId);
    }
    // Filter consistently with renderEpisodeCards: taskFilter + search
    const taskFilter = document.getElementById('filter-task')?.value || '';
    const search = (document.getElementById('search-input')?.value || '').toLowerCase();
    let visible = visibleEpisodesForCurrentView();
    if (taskFilter) visible = visible.filter(ep => (ep.task_description || '') === taskFilter);
    if (search) {
        const s = search.toLowerCase();
        visible = visible.filter(ep =>
            (ep.name || '').toLowerCase().includes(s) ||
            (ep.task_description || '').toLowerCase().includes(s) ||
            String(ep.id).toLowerCase().includes(s));
    }
    updateBatchUI(visible);
}


function toggleSelectAll(checked) {
    if (!selectMode) return;
    const statusFilter = document.getElementById('filter-status')?.value || '';
    if (statusFilter !== 'reviewed') return;

    const search = (document.getElementById('search-input')?.value || '').toLowerCase();
    const taskFilter = document.getElementById('filter-task')?.value || '';
    let filtered = visibleEpisodesForCurrentView();
    if (taskFilter) filtered = filtered.filter(ep => (ep.task_description || '') === taskFilter);
    if (search) {
        filtered = filtered.filter(ep =>
            (ep.name || '').toLowerCase().includes(search) ||
            (ep.task_description || '').toLowerCase().includes(search) ||
            String(ep.id).toLowerCase().includes(search)
        );
    }

    if (checked) {
        filtered.forEach(ep => selectedEpisodes.add(ep.id));
    } else {
        filtered.forEach(ep => selectedEpisodes.delete(ep.id));
    }
    renderEpisodeListFromMemory();
}


function updateBatchUI(filteredEpisodes) {
    const selectAllBar = document.getElementById('select-all-bar');
    const batchBar = document.getElementById('batch-bar');
    const selectBtn = document.getElementById('btn-toggle-select');
    const statusFilter = document.getElementById('filter-status')?.value || '';
    const isReviewedMode = statusFilter === 'reviewed';
    const showSelection = selectMode && isReviewedMode;

    // Select button: only visible in Reviewed/Approved category
    if (selectBtn) {
        selectBtn.classList.toggle('hidden', !isReviewedMode);
    }
    // Force-exit selection mode when switching away from Reviewed
    if (!isReviewedMode && selectMode) {
        selectMode = false;
        selectedEpisodes.clear();
        if (selectBtn) {
            selectBtn.textContent = 'Select';
            selectBtn.className = 'bg-gray-800 hover:bg-gray-700 text-gray-400 hover:text-gray-200 text-xs px-2 py-0.5 rounded flex-shrink-0 transition-colors';
        }
    }

    // Show/hide select-all bar only in selection mode
    if (selectAllBar) {
        selectAllBar.classList.toggle('hidden', !showSelection);
    }

    if (!showSelection) {
        if (!selectMode) selectedEpisodes.clear();
        if (batchBar) batchBar.classList.add('hidden');
        return;
    }

    // Update counts
    const reviewedList = filteredEpisodes.filter
        ? filteredEpisodes.filter(ep => ep.status === 'reviewed' || ep.status === 'approved')
        : [];
    const reviewedCount = reviewedList.length;
    const selCount = reviewedList.filter(ep => selectedEpisodes.has(ep.id)).length;

    const selectCount = document.getElementById('select-count');
    if (selectCount) selectCount.textContent = `(${selCount}/${reviewedCount})`;

    const selectAllCb = document.getElementById('select-all-checkbox');
    if (selectAllCb) {
        selectAllCb.checked = reviewedCount > 0 && selCount === reviewedCount;
        selectAllCb.indeterminate = selCount > 0 && selCount < reviewedCount;
    }

    // Show/hide batch bar
    if (batchBar) {
        batchBar.classList.toggle('hidden', selCount === 0);
    }
    const batchCount = document.getElementById('batch-count');
    if (batchCount) batchCount.textContent = selCount + ' selected';
}


async function batchDownload() {
    if (selectedEpisodes.size === 0) return;
    const ids = Array.from(selectedEpisodes);
    const done = await startReviewExport(ids, null, null);
    if (done && selectMode) toggleSelectMode();
}


function waitForExportJob(jobId, button) {
    return new Promise(resolve => {
        const poll = async () => {
            try {
                const res = await fetch(`/api/v1/export/${jobId}`);
                const job = await res.json();
                if (job.status === 'completed') {
                    resolve(true);
                    return;
                }
                if (job.status === 'failed') {
                    alert('Export failed: ' + (job.error || 'unknown'));
                    resolve(false);
                    return;
                }
            } catch (err) {
                alert('Export status failed: ' + err.message);
                resolve(false);
                return;
            }
            setTimeout(poll, 800);
        };
        poll();
    }).finally(() => {
        if (button) {
            button.disabled = false;
            button.dataset.exporting = '';
            button.classList.remove('opacity-60', 'cursor-wait');
        }
    });
}


async function startReviewExport(ids, format, datasetName, button = null) {
    if (!ids || ids.length === 0) return false;
    if (button) {
        button.disabled = true;
        button.dataset.exporting = '1';
        button.classList.add('opacity-60', 'cursor-wait');
    }
    try {
        const payload = {
            episode_ids: ids,
            split_ratio: 0.9,
        };
        if (datasetName) payload.dataset_name = datasetName;
        // Normal review-page exports omit the format deliberately: the
        // backend resolves it from the selected project's workflow node.
        if (format) payload.export_format = format;
        const res = await fetch('/api/v1/export/start', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload),
        });
        if (!res.ok) {
            const err = await res.json().catch(() => ({}));
            alert('Export failed: ' + (err.detail || 'unknown'));
            if (button) {
                button.disabled = false;
                button.dataset.exporting = '';
                button.classList.remove('opacity-60', 'cursor-wait');
            }
            return false;
        }
        const job = await res.json();
        const done = await waitForExportJob(job.id, button);
        if (done) {
            window.location.href = `/api/v1/export/download/${job.id}`;
            return true;
        }
        return false;
    } catch (err) {
        alert('Export failed: ' + err.message);
        if (button) {
            button.disabled = false;
            button.dataset.exporting = '';
            button.classList.remove('opacity-60', 'cursor-wait');
        }
        return false;
    }
}


// ── Select / Info Panel ──────────────────────────────

function selectEpisode(episodeId) {
    const ep = allEpisodes.find(e => e.id === episodeId);
    if (!ep) return;
    // Reprocessing keeps the list card visible, but its previous artifacts are
    // no longer a valid review target. Do not reopen stale media while the
    // worker is replacing the processed output.
    if (ep.status === 'processing' || ep._uiWorkflowState === 'queued') {
        backToList();
        return;
    }

    // Annotation is a separate workspace: start every selected file with a
    // clean editor while leaving the review page's approval flow untouched.
    if (window.EGODATA_PAGE_MODE === 'annotation' && typeof hideAnnotationForm === 'function') {
        hideAnnotationForm();
    }

    // Highlight card
    document.querySelectorAll('.episode-card').forEach(el => el.classList.remove('active'));
    const card = document.querySelector('.episode-card[data-episode-id="' + episodeId + '"]');
    if (card) card.classList.add('active');

    // Load video
    const cameras = ep.camera_names || [];
    const playbackMeta = { frameCount: ep.frame_count, fps: ep.fps };
    if (typeof loadGroupedEpisodeVideo === 'function') {
        loadGroupedEpisodeVideo(episodeId, cameras, {
            hasSkeleton: Boolean(ep.has_skeleton),
            ...playbackMeta,
        });
    } else {
        loadEpisodeVideo(episodeId, cameras, playbackMeta);
    }

    // Switch to detail view: hide list, show detail panel
    const listSection = document.getElementById('episode-list-section');
    const detail = document.getElementById('episode-detail');
    if (listSection) listSection.classList.add('hidden');
    if (detail) detail.classList.remove('hidden');
    // 进入批次详情(审核/通过/标注页)→ 显示漂浮预览控件
    if (typeof setPreviewPanelsVisible === 'function') setPreviewPanelsVisible(true);
    renderInputWarning(ep);

    // Fill detail header
    const nameEl = document.getElementById('detail-episode-name');
    if (nameEl) {
        const episodeNumber = typeof ep.episode_index === 'number'
            ? `#${ep.episode_index}` : '';
        const title = ep.task_description || ep.name || ep.id.slice(0, 8);
        nameEl.textContent = episodeNumber ? `${episodeNumber} · ${title}` : title;
        nameEl.title = ep.id || title;
    }
    const statusBadge = document.getElementById('detail-status-badge');
    if (statusBadge) {
        const isReviewed = ep.status === 'reviewed' || ep.status === 'approved';
        const isFailed = ep.status === 'failed';
        const isProcessing = ep.status === 'processing';
        statusBadge.className = 'text-xs flex-shrink-0 ' + (isFailed ? 'text-red-400' : (isReviewed ? 'text-green-400' : (isProcessing ? 'text-blue-400' : 'text-yellow-400')));
        statusBadge.textContent = isFailed ? t('stat_failed') : (isReviewed ? t('reviewed') : (isProcessing ? t('processing') : t('reviewing')));
    }

    // Fill detail info
    const infoDiv = document.getElementById('episode-detail-info');
    const meta = ep.meta || {};
    const timestamp = meta.timestamp || '';
    const time = timestamp || new Date(ep.created_at).toLocaleString('zh-CN');
    const isReviewed = ep.status === 'reviewed' || ep.status === 'approved';
    const isFailed = ep.status === 'failed';
    const duration = ep.fps > 0 ? (ep.frame_count / ep.fps).toFixed(1) : '0';
    const cameraList = (ep.camera_names || []).join(', ');

    let cleaningHTML = '';
    if (ep.cleaning_report) {
        const cr = ep.cleaning_report;
        if (cr.passed) {
            cleaningHTML = '<div class="flex items-center gap-1 text-green-400 text-xs"><iconify-icon icon="ant-design:check-circle-filled" class="icon-sm"></iconify-icon> Cleaning passed</div>';
        } else {
            const failList = (cr.checks || []).filter(c => !c.passed).map(c => c.name).join(', ');
            cleaningHTML = '<div class="flex items-center gap-1 text-red-400 text-xs" title="' + failList + '"><iconify-icon icon="ant-design:warning-filled" class="icon-sm"></iconify-icon> Cleaning failed</div>';
        }
    }

    // 上传不匹配/运行失败异常:点击文件后在此详情显示(不在项目列表显示)
    let exceptionHTML = '';
    if (ep.exceptions && ep.exceptions.length) {
        exceptionHTML = ep.exceptions.map(function (ex) {
            const kindLabel = ex.kind === 'run_failed'
                ? t('exception_kind_failed')
                : (ex.kind === 'input_missing' ? t('exception_kind_missing') : t('exception_kind_mismatch'));
            const safeMsg = escHtml(ex.message || '');
            const msg = safeMsg ? ': ' + safeMsg : '';
            const color = ex.kind === 'input_missing' ? 'text-amber-300' : 'text-red-400';
            return '<div class="flex items-center gap-1 ' + color + ' text-xs" title="' + safeMsg + '">' +
                '<iconify-icon icon="ant-design:warning-filled" class="icon-sm"></iconify-icon> ' + kindLabel + msg + '</div>';
        }).join('');
    }

    const isAnnotationPage = window.EGODATA_PAGE_MODE === 'annotation';
    const buttons = isAnnotationPage ? '' : (isFailed
        ? '<button onclick="retryEpisode(\'' + ep.id + '\')" class="flex-1 bg-blue-800 hover:bg-blue-700 text-blue-200 py-1.5 rounded text-xs"><iconify-icon icon="ant-design:reload-outlined" class="icon-sm"></iconify-icon> ' + t('retry_review') + '</button>' +
          '<button onclick="deleteEpisode(\'' + ep.id + '\')" class="bg-gray-800 hover:bg-red-900 text-gray-400 hover:text-red-300 py-1.5 px-3 rounded text-xs"><iconify-icon icon="ant-design:delete-outlined" class="icon-sm"></iconify-icon></button>'
        : (ep.status === 'processing'
        ? '<button disabled class="flex-1 bg-blue-950/60 text-blue-300/80 py-1.5 rounded text-xs cursor-wait"><iconify-icon icon="ant-design:loading-outlined" class="icon-sm"></iconify-icon> ' + t('processing') + '</button>' +
          '<button onclick="deleteEpisode(\'' + ep.id + '\')" class="bg-gray-800 hover:bg-red-900 text-gray-400 hover:text-red-300 py-1.5 px-3 rounded text-xs"><iconify-icon icon="ant-design:delete-outlined" class="icon-sm"></iconify-icon></button>'
        : (isReviewed
        ? '<button onclick="downloadEpisode(\'' + ep.id + '\')" title="' + t('export_hint') + '" class="flex-1 bg-blue-800 hover:bg-blue-700 text-blue-200 py-1.5 rounded text-xs"><iconify-icon icon="ant-design:export-outlined" class="icon-sm"></iconify-icon> ' + t('export') + '</button>' +
          '<button onclick="unreviewEpisode(\'' + ep.id + '\')" class="flex-1 bg-yellow-800 hover:bg-yellow-700 text-yellow-200 py-1.5 rounded text-xs"><iconify-icon icon="ant-design:rollback-outlined" class="icon-sm"></iconify-icon> ' + t('unreview') + '</button>'
        : '<button onclick="markReviewed(\'' + ep.id + '\')" class="flex-1 bg-green-700 hover:bg-green-600 text-white py-1.5 rounded text-xs font-medium"><iconify-icon icon="ant-design:check-outlined" class="icon-sm"></iconify-icon> ' + t('approve') + '</button>' +
          '<button onclick="deleteEpisode(\'' + ep.id + '\')" class="bg-gray-800 hover:bg-red-900 text-gray-400 hover:text-red-300 py-1.5 px-3 rounded text-xs"><iconify-icon icon="ant-design:delete-outlined" class="icon-sm"></iconify-icon></button>')));

    if (infoDiv) {
        infoDiv.innerHTML =
            '<div class="space-y-1.5 text-xs">' +
                (cleaningHTML || '') +
                (exceptionHTML || '') +
                '<div class="text-gray-500 space-y-0.5">' +
                    '<div><iconify-icon icon="ant-design:field-time-outlined" class="icon-sm"></iconify-icon> ' + ep.frame_count + ' frames · ' + ep.fps + ' FPS · ' + duration + 's</div>' +
                    (cameraList ? '<div><iconify-icon icon="ant-design:video-camera-outlined" class="icon-sm"></iconify-icon> ' + cameraList + '</div>' : '') +
                    '<div><iconify-icon icon="ant-design:calendar-outlined" class="icon-sm"></iconify-icon> ' + time + '</div>' +
                    '<div id="episode-detail-sync-status" class="hidden"></div>' +
                    '<div id="episode-detail-ai-status" class="hidden"></div>' +
                '</div>' +
                '<div class="flex items-center gap-2 bg-gray-800 rounded px-3 py-1.5">' +
                    '<span class="text-gray-500">Frame</span>' +
                    '<span id="heatmap-frame" class="text-blue-400 font-mono font-bold text-lg">-</span>' +
                    '<span class="text-gray-600">/ ' + ep.frame_count + '</span>' +
                '</div>' +
                '<div class="flex gap-2">' + buttons + '</div>' +
                '<div id="episode-detail-format"></div>' +
            '</div>';
        if (typeof updateEpisodeDetailSyncStatus === 'function') updateEpisodeDetailSyncStatus();

        // 导出格式徽标(工作流连接驱动:HDF5 / LeRobot v2.1 / v3.0 / Raw 兜底)
        const fmtEl = document.getElementById('episode-detail-format');
        if (fmtEl) {
            fmtEl.innerHTML = '';
            if (!isAnnotationPage && isReviewed) {
                const fmtEpoch = ++_formatEpoch;
                fetch(`/api/v1/export/episode-format/${episodeId}`)
                    .then(r => r.json())
                    .then(d => {
                        if (fmtEpoch !== _formatEpoch) return;  // 已切换批次,丢弃过期响应
                        const el = document.getElementById('episode-detail-format');
                        if (!el) return;
                        const label = escHtml(d.label || 'Raw');
                        const badge = d.available
                            ? '<span class="text-blue-300 font-medium">' + label + '</span>'
                            : '<span class="text-gray-500">' + label + '</span>';
                        el.innerHTML = '<div class="flex items-center gap-1 text-gray-500"><iconify-icon icon="ant-design:export-outlined" class="icon-sm"></iconify-icon> ' +
                            t('export') + ': ' + badge +
                            (d.available ? ' · <button id="btn-re-export" onclick="reExportEpisode()" class="text-blue-400 hover:underline text-xs">' + t('re_export') + '</button>' : '') +
                            '</div>';
                    })
                    .catch(() => {});
            }
        }
    }

// Re-export:只重建导出,不重跑检测(复用最新 run 的检测产物 +
// 当前工作流的导出配置/连线)。Review 详情 Export 徽标旁按钮。
// (挂 window:inline onclick 需要全局可见)
window.reExportEpisode = async function () {
    if (!currentEpisodeId) return;
    const btn = document.getElementById('btn-re-export');
    if (!btn || btn.dataset.busy === '1') return;
    btn.dataset.busy = '1';
    btn.textContent = t('re_exporting') + '…';
    let finalStatus = 'done';
    try {
        const res = await fetch(`/api/v1/export/re-export/${currentEpisodeId}`, { method: 'POST' });
        if (!res.ok) throw new Error((await res.json().catch(() => ({}))).detail || res.status);
        for (let i = 0; i < 300; i++) {
            await new Promise(r => setTimeout(r, 1000));
            let st = { status: 'running' };
            try {
                st = await (await fetch(`/api/v1/export/re-export/${currentEpisodeId}/status`)).json();
            } catch (_) { /* keep polling */ }
            if (st.status === 'done' || st.status === 'failed' || st.status === 'interrupted') {
                finalStatus = st.status;
                if (st.status !== 'done') {
                    alert(t('re_export_failed') + ': ' + (st.detail || st.status));
                } else {
                    alert(t('re_export_done') + ': ' + (st.detail || ''));
                }
                break;
            }
        }
    } catch (err) {
        finalStatus = 'failed';
        alert(t('re_export_failed') + ': ' + (err.message || err));
    }
    btn.dataset.busy = '';
    btn.textContent = t('re_export');
    if (finalStatus === 'done') {
        // 刷新徽标(格式可能已变)
        const fmtEpoch = ++_formatEpoch;
        try {
            const d = await (await fetch(`/api/v1/export/episode-format/${currentEpisodeId}`)).json();
            if (fmtEpoch !== _formatEpoch || !currentEpisodeId) return;
            const el = document.getElementById('episode-detail-format');
            if (!el) return;
            const label = escHtml(d.label || 'Raw');
            const badge = d.available
                ? '<span class="text-blue-300 font-medium">' + label + '</span>'
                : '<span class="text-gray-500">' + label + '</span>';
            el.innerHTML = '<div class="flex items-center gap-1 text-gray-500"><iconify-icon icon="ant-design:export-outlined" class="icon-sm"></iconify-icon> ' +
                t('export') + ': ' + badge +
                (d.available ? ' · <button id="btn-re-export" onclick="reExportEpisode()" class="text-blue-400 hover:underline text-xs">' + t('re_export') + '</button>' : '') +
                '</div>';
        } catch (_) { /* 刷新失败下次进入详情自动重取 */ }
    }
}

    // Update frame info header
    const frameHeader = document.getElementById('frame-info-header');
    const totalFramesEl = document.getElementById('detail-total-frames');
    if (frameHeader) frameHeader.classList.remove('hidden');
    if (totalFramesEl) totalFramesEl.textContent = ep.frame_count || 0;

    // Show frame controls
    if (typeof showAnnotationUI === 'function') showAnnotationUI();

    // Annotations are loaded by loadEpisodeVideo after frame data is ready
    // (avoid race: loadAnnotations needs episodeTotalFrames from loadFrameData)
}


// ── Review ──────────────────────────────────────────

async function markReviewed(episodeId) {
    try {
        const res = await fetch(`/api/v1/episode/${episodeId}/review`, { method: 'POST' });
        if (res.ok) {
            backToList();
            removeEpisodeFromLocalList(episodeId);
            scheduleSilentEpisodeRefresh();
        }
    } catch (err) {
        alert('Operation failed: ' + err.message);
    }
}


async function unreviewEpisode(episodeId) {
    try {
        const res = await fetch(`/api/v1/episode/${episodeId}/unreview`, { method: 'POST' });
        if (res.ok) {
            backToList();
            removeEpisodeFromLocalList(episodeId);
            scheduleSilentEpisodeRefresh();
        }
    } catch (err) {
        alert('Operation failed: ' + err.message);
    }
}


async function retryEpisode(episodeId) {
    try {
        const res = await fetch(`/api/v1/episode/${episodeId}/retry`, { method: 'POST' });
        if (res.ok) {
            backToList();
            removeEpisodeFromLocalList(episodeId);
            scheduleSilentEpisodeRefresh();
        }
    } catch (err) {
        alert('Operation failed: ' + err.message);
    }
}


// ── Reprocess(主动重新处理)──────────────────────────

async function reprocessEpisode(episodeId) {
    if (!confirm(t('confirm_reprocess'))) return;
    // The worker will replace the canonical parquet. Do not reuse the old
    // full depth/3D browser buffers when this episode is opened again.
    if (typeof window.invalidateEpisodePlaybackCache === 'function') {
        window.invalidateEpisodePlaybackCache(episodeId);
    }
    // Clear the old detail workspace before waiting for the queue response.
    // Reprocessing is asynchronous; the old media must not remain active
    // while the user navigates to another review section.
    backToList();
    // Mark the local snapshot before the POST resolves. This closes the small
    // window in which a fast click could reopen the old completed artifact.
    updateEpisodeInLocalList(episodeId, { _uiWorkflowState: 'queued' });
    try {
        const res = await fetch(`/api/v1/episode/${episodeId}/reprocess`, { method: 'POST' });
        if (res.ok) {
            scheduleSilentEpisodeRefresh();
        } else {
            updateEpisodeInLocalList(episodeId, { _uiWorkflowState: null });
            const err = await res.json();
            alert(t('reprocess_failed') + (err.detail || res.status));
        }
    } catch (err) {
        updateEpisodeInLocalList(episodeId, { _uiWorkflowState: null });
        alert(t('reprocess_failed') + err.message);
    }
}


// ── Delete ──────────────────────────────────────────

async function deleteEpisode(episodeId) {
    if (!confirm(t('confirm_trash'))) return;

    try {
        const res = await fetch(`/api/v1/episode/${episodeId}/delete`, { method: 'POST' });
        if (res.ok) {
            updateTrashBadge();  // 垃圾桶徽标立即 +1,无需刷新页面
            backToList();       // instant feedback — close detail, show list
            if (currentEpisodeId === episodeId) {
                document.getElementById('video-grid').innerHTML =
                    '<div class="flex items-center justify-center h-64 text-gray-600">' + t('deleted_msg') + '</div>';
                // Clear annotations
                annotations = [];
                if (typeof clearAnnotationOverlay === 'function') clearAnnotationOverlay();
            }
            removeEpisodeFromLocalList(episodeId);
            scheduleSilentEpisodeRefresh();
        }
    } catch (err) {
        alert(t('op_failed') + ': ' + err.message);
    }
}


async function restoreEpisode(episodeId) {
    try {
        const res = await fetch(`/api/v1/episode/${episodeId}/restore`, { method: 'POST' });
        if (res.ok) {
            updateTrashBadge();  // 恢复后徽标立即 -1
            loadTrashList();
            if (typeof refreshReviewTree === 'function') refreshReviewTree();
        }
    } catch (err) {
        alert(t('restore_failed') + ': ' + err.message);
    }
}


async function permanentDeleteEpisode(episodeId) {
    if (!confirm(t('confirm_permanent'))) return;
    try {
        const res = await fetch(`/api/v1/episode/${episodeId}/permanent`, { method: 'DELETE' });
        if (res.ok) {
            updateTrashBadge();
            loadTrashList();
            if (typeof refreshReviewTree === 'function') refreshReviewTree();
        }
    } catch (err) {
        alert(t('delete_failed') + ': ' + err.message);
    }
}


async function purgeTrash() {
    if (!confirm(t('confirm_purge'))) return;
    try {
        const res = await fetch('/api/v1/episodes/purge-trash', { method: 'POST' });
        if (res.ok) {
            updateTrashBadge();
            loadTrashList();
            if (typeof refreshReviewTree === 'function') refreshReviewTree();
        }
    } catch (err) {
        alert(t('purge_failed') + ': ' + err.message);
    }
}


// ── Export ────────────────────────────────────────────

async function downloadEpisode(episodeId) {
    // A single episode already has its workflow-published export product.
    // That endpoint returns the version selected in the workflow export node.
    window.location.href = `/api/v1/export/download-episode/${episodeId}`;
}


async function exportSelected() {
    const reviewed = allEpisodes.filter(e => e.status === 'reviewed');
    if (reviewed.length === 0) return alert(t('no_reviewed'));
    window.location.href = '/api/v1/export/download-reviewed';
}


// ── Export Page ───────────────────────────────────────

async function loadExportJobs() {
    const listEl = document.getElementById('export-list');
    if (!listEl) return;

    try {
        const res = await fetch('/api/v1/export/list?limit=50');
        const jobs = await res.json();

        if (jobs.length === 0) {
            listEl.innerHTML = '<tr><td colspan="5" class="text-center text-gray-500 py-8">No export jobs</td></tr>';
            return;
        }

        listEl.innerHTML = jobs.map(job => {
            const time = new Date(job.created_at).toLocaleString('zh-CN');
            const statusColors = { pending: 'text-yellow-400', running: 'text-blue-400', completed: 'text-green-400', failed: 'text-red-400' };
            const statusText = { pending: 'Pending', running: 'Running', completed: 'Completed', failed: 'Failed' };
            const progressBar = job.status === 'running'
                ? `<div class="w-full bg-gray-700 rounded h-2"><div class="bg-blue-500 h-2 rounded" style="width:${job.progress}%"></div></div>`
                : job.status === 'completed' ? '100%' : '-';
            const actions = job.status === 'completed'
                ? `<a href="/api/v1/export/download/${job.id}" class="text-blue-400 hover:underline text-xs"><iconify-icon icon="ant-design:download-outlined" class="icon-sm"></iconify-icon> Download</a>`
                : job.status === 'running'
                    ? `<button onclick="loadExportJobs()" class="text-gray-400 hover:underline text-xs">Refresh</button>`
                    : '';

            return `<tr class="border-b border-gray-800">
                <td class="px-4 py-2">${job.dataset_name}</td>
                <td class="px-4 py-2 ${statusColors[job.status]||'text-gray-400'}">${statusText[job.status]||job.status}</td>
                <td class="px-4 py-2">${progressBar}</td>
                <td class="px-4 py-2 text-xs text-gray-400">${time}</td>
                <td class="px-4 py-2">${actions}</td>
            </tr>`;
        }).join('');
    } catch (err) {
        listEl.innerHTML = `<tr><td colspan="5" class="text-center text-red-400 py-8">Load failed</td></tr>`;
    }
}


async function startExport() {
    const name = document.getElementById('new-dataset-name')?.value;
    const ratio = parseFloat(document.getElementById('new-split-ratio')?.value) || 0.9;
    if (!name) return alert('Please enter a dataset name');

    try {
        const res = await fetch('/api/v1/export/start', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ dataset_name: name, episode_ids: null, split_ratio: ratio }),
        });
        if (res.ok) { document.getElementById('new-dataset-name').value = ''; loadExportJobs(); }
        else { const err = await res.json(); alert('Export creation failed: ' + (err.detail || 'unknown')); }
    } catch (err) {
        alert('Export creation failed: ' + err.message);
    }
}


// ── Init ────────────────────────────────────────────

document.addEventListener('DOMContentLoaded', () => {
    // Apply data-i18n translations
    document.querySelectorAll('[data-i18n]').forEach(el => {
        el.textContent = t(el.dataset.i18n);
    });
    document.querySelectorAll('[data-i18n-placeholder]').forEach(el => {
        el.placeholder = t(el.dataset.i18nPlaceholder);
    });
    // Init nav labels
    const navReview = document.getElementById('nav-review');
    const navTrash = document.getElementById('nav-trash');
    if (navReview) navReview.textContent = t('video_review');
    if (navTrash) navTrash.textContent = t('trash');

    if (document.getElementById('episode-list-inline')) {
        initStatusFromURL();
        installReviewStatusNavigation();
        loadEpisodes();
        setInterval(() => { loadEpisodes({ silent: true }); updateTrashBadge(); }, 15000);
        updateTrashBadge();
    }
    if (document.getElementById('export-list')) {
        loadExportJobs();
        setInterval(loadExportJobs, 10000);
    }
    if (document.getElementById('trash-list')) {
        loadTrashList();
    }
});


async function loadTrashList() {
    const listEl = document.getElementById('trash-list');
    if (!listEl) return;

    listEl.innerHTML = `<div class="p-4 text-center text-gray-500 text-sm">${t('loading')}</div>`;

    try {
        const res = await fetch('/api/v1/episodes?status=deleted&limit=200');
        const data = await res.json();
        const episodes = data.episodes || [];

        if (episodes.length === 0) {
            listEl.innerHTML = `<div class="p-8 text-center text-gray-500"><iconify-icon icon="ant-design:delete-outlined" class="icon-md"></iconify-icon> ${t('trash_empty')}</div>`;
        } else {
            const now = new Date();
            listEl.innerHTML = episodes.map(ep => {
                const delTime = new Date(ep.deleted_at);
                const daysLeft = Math.max(0, Math.ceil(7 - (now - delTime) / (1000 * 60 * 60 * 24)));
                const time = new Date(ep.created_at).toLocaleString('zh-CN');
                return `
                <div class="bg-gray-800 rounded p-4 flex items-center justify-between">
                    <div class="flex-1">
                        <div class="text-sm font-medium text-gray-200">${ep.task_description || ep.name || 'unknown'}</div>
                        <div class="text-xs text-gray-500 mt-1">
                            <iconify-icon icon="ant-design:field-time-outlined" class="icon-sm"></iconify-icon> ${ep.fps}FPS · ${ep.frame_count}${t('frame')} · <iconify-icon icon="ant-design:video-camera-outlined" class="icon-sm"></iconify-icon> ${(ep.camera_names||[]).join(', ') || t('not_found')}
                        </div>
                        <div class="text-xs text-gray-500"><iconify-icon icon="ant-design:calendar-outlined" class="icon-sm"></iconify-icon> ${time}</div>
                    </div>
                    <div class="text-xs text-gray-400 mr-4"><iconify-icon icon="ant-design:hourglass-outlined" class="icon-sm"></iconify-icon> ${t('days_left')} ${daysLeft} ${t('day_unit')}</div>
                    <div class="flex gap-2">
                        <button onclick="restoreEpisode('${ep.id}')"
                                class="bg-green-700 hover:bg-green-600 text-white text-xs px-3 py-1.5 rounded"><iconify-icon icon="ant-design:rollback-outlined" class="icon-sm"></iconify-icon> ${t('restore')}</button>
                        <button onclick="permanentDeleteEpisode('${ep.id}')"
                                class="bg-red-800 hover:bg-red-700 text-red-200 text-xs px-3 py-1.5 rounded"><iconify-icon icon="ant-design:delete-outlined" class="icon-sm"></iconify-icon> ${t('permanent_delete')}</button>
                    </div>
                </div>`;
            }).join('');
        }
        updateTrashBadge();
    } catch (err) {
        listEl.innerHTML = `<div class="p-4 text-center text-red-400 text-sm">${t('load_failed')}</div>`;
    }
}


// updateTrashBadge 由 base.html 全局提供(所有页面生效),删除/恢复/清空后
// 各操作函数调用它立即刷新徽标,无需刷新页面。
