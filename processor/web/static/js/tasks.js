/* Project management — hierarchy: Project → Task (upload batch) → Episode (video).
   Replaces the old task-definition page. */

let editingProjectId = null;
const _expanded = new Set();  // project ids with open task list
let _projectsSnapshot = [];
let _projectsLoaded = false;
let _projectsLoadController = null;
const _projectTreeSnapshot = new Map();
const PROJECTS_SNAPSHOT_CACHE_KEY = 'egodata.projects.snapshot.v1';
const PROJECTS_SNAPSHOT_CACHE_TTL = 60 * 1000;

function readProjectsSnapshotCache() {
    try {
        const cached = JSON.parse(sessionStorage.getItem(PROJECTS_SNAPSHOT_CACHE_KEY) || 'null');
        if (!cached || !Array.isArray(cached.projects)
                || Date.now() - Number(cached.savedAt || 0) > PROJECTS_SNAPSHOT_CACHE_TTL) return null;
        return cached;
    } catch (_) { return null; }
}

function saveProjectsSnapshotCache(projects) {
    try {
        sessionStorage.setItem(PROJECTS_SNAPSHOT_CACHE_KEY, JSON.stringify({
            savedAt: Date.now(), projects,
        }));
    } catch (_) { /* storage quota/private mode — network refresh still works */ }
}

function escHtml(s) {
    if (!s) return '';
    return String(s).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;');
}

function escAttr(s) {
    if (!s) return '';
    return escHtml(s).replace(/'/g, '&#39;');
}

async function apiFetch(url, options = {}) {
    const res = await fetch(url, {
        headers: { 'Content-Type': 'application/json', ...(options.headers || {}) },
        ...options,
    });
    if (!res.ok) {
        const detail = await res.json().catch(() => ({}));
        throw new Error(detail.detail || `HTTP ${res.status}`);
    }
    return res.status === 204 ? null : res.json();
}

// 一个项目绑定一个当前工作流。旧项目文件仍兼容 workflow_ids 数组,
// 但前端选择和保存都只保留一个,与后端派发规则保持一致。
let _selectedWorkflowIds = [];
let _workflowNameById = {};   // id → name(下拉框内显示已选名称)

function updateWorkflowDropdownLabel() {
    const label = document.getElementById('wf-dropdown-label');
    if (!label) return;
    if (_selectedWorkflowIds.length > 0) {
        // 框内直接显示已选工作流名(最多 2 个,超出 +N)
        const names = _selectedWorkflowIds.slice(0, 2).map(id => _workflowNameById[id]).filter(Boolean);
        label.textContent = names.length > 0
            ? (_selectedWorkflowIds.length > 2 ? `${names.join(', ')} +${_selectedWorkflowIds.length - 2}` : names.join(', '))
            : t('workflow_selected').replace('%s', _selectedWorkflowIds.length);
        label.classList.remove('text-gray-400');
        label.classList.add('text-gray-200');
    } else {
        label.textContent = t('workflow_placeholder');
        label.classList.remove('text-gray-200');
        label.classList.add('text-gray-400');
    }
}

// 工作流下拉:点击按钮展开,点选项行切换当前唯一工作流,点外部关闭
// 顶部固定 "+ New Workflow" 行:在项目中直接新建工作流(命名即建、自动选中)
// 命名用内联输入(替代浏览器 prompt):点 + 行 → 就地变输入框,回车创建、Esc/失焦取消
let _wfNameEditing = false;

function startWorkflowNameInput() {
    _wfNameEditing = true;
    loadWorkflowsForSelect().then(() => {
        const input = document.getElementById('wf-new-name');
        if (input) { input.focus(); input.select(); }
    });
}

function cancelWorkflowNameInput() {
    if (!_wfNameEditing) return;
    _wfNameEditing = false;
    loadWorkflowsForSelect();
}

async function commitWorkflowName(input) {
    if (!_wfNameEditing) return;
    const name = (input.value || '').trim();
    _wfNameEditing = false;   // 先退出编辑态,再创建
    if (!name) { loadWorkflowsForSelect(); return; }
    try {
        const w = await apiFetch('/api/v1/workflows', {
            method: 'POST',
            body: JSON.stringify({ name, graph: { nodes: [], edges: [] }, status: 'draft' }),
        });
        _selectedWorkflowIds = [w.id];
        await loadWorkflowsForSelect();
        updateWorkflowDropdownLabel();
    } catch (e) {
        alert(t('workflow_create_failed') + ': ' + e.message);
        loadWorkflowsForSelect();   // 失败也退出编辑态
    }
}

async function loadWorkflowsForSelect() {
    try {
        const d = await apiFetch('/api/v1/workflows?limit=100');
        const panel = document.getElementById('wf-dropdown-panel');
        if (!panel) return;
        _workflowNameById = {};
        d.workflows.forEach(w => { _workflowNameById[w.id] = w.name; });
        const newRow = _wfNameEditing
            ? `<input id="wf-new-name" type="text" maxlength="128" placeholder="${t('new_workflow_prompt')}"
                      class="w-full bg-gray-800 border border-blue-600 rounded px-2 py-1.5 text-xs text-gray-200 placeholder-gray-500 focus:outline-none"
                      onkeydown="if(event.key==='Enter'){commitWorkflowName(this)}else if(event.key==='Escape'){cancelWorkflowNameInput()}"
                      onblur="if(document.getElementById('wf-new-name')){setTimeout(()=>{cancelWorkflowNameInput()},120)}">`
            : `<div class="flex items-center gap-2 px-2 py-1.5 cursor-pointer hover:bg-gray-800 rounded text-xs text-blue-400 font-medium"
                  onclick="startWorkflowNameInput()">
                <iconify-icon icon="ant-design:plus-outlined" class="icon-sm shrink-0"></iconify-icon>
                <span>${t('new_workflow_panel')}</span>
            </div>`;
        const listHtml = d.workflows.length === 0
            ? `<div class="text-xs text-gray-600 px-2 py-1.5">${t('workflow_placeholder')}</div>`
            : d.workflows.map(w => {
                const checked = _selectedWorkflowIds.includes(w.id);
                return `
                <div class="wf-row flex items-center gap-2 px-2 py-1.5 cursor-pointer hover:bg-gray-800 rounded text-xs text-gray-300"
                     data-id="${w.id}" onclick="toggleWorkflowRow(this)">
                    <span class="wf-check w-3.5 h-3.5 rounded border flex items-center justify-center text-[9px] leading-none shrink-0
                                 ${checked ? 'bg-blue-600 border-blue-500 text-white' : 'border-gray-600 text-transparent'}">✓</span>
                    <span class="truncate flex-1">${escHtml(w.name)}${w.is_preset ? `<span class="text-[9px] text-yellow-500 border border-yellow-800 rounded px-1 ml-1 align-middle">${t('template_badge')}</span>` : ''}${w.node_count === 0 ? `<span class="text-[9px] text-yellow-500 border border-yellow-800 rounded px-1 ml-1 align-middle">⚠ ${t('workflow_empty')}</span>` : ''}</span>
                    <a href="/workflow-studio?workflow=${w.id}${w.project_id ? `&project=${encodeURIComponent(w.project_id)}` : ''}" onclick="event.stopPropagation()" title="Edit workflow"
                       class="text-blue-400 hover:text-blue-300 shrink-0 text-[12px]">
                        <iconify-icon icon="ant-design:edit-outlined" class="icon-sm"></iconify-icon>
                    </a>
                </div>`;
            }).join('');
        panel.innerHTML = newRow + listHtml;
        updateWorkflowDropdownLabel();
    } catch (e) { console.error('load workflows failed', e); }
}

function toggleWorkflowRow(row) {
    const id = row.dataset.id;
    const idx = _selectedWorkflowIds.indexOf(id);
    if (idx >= 0) {
        _selectedWorkflowIds = [];
    } else {
        _selectedWorkflowIds = [id];
    }
    // Re-render all rows so the previous selection is visibly cleared.
    loadWorkflowsForSelect();
    updateWorkflowDropdownLabel();
}

// 面板开合:按钮切换(箭头旋转),点外部关闭
function closeWorkflowDropdown() {
    const panel = document.getElementById('wf-dropdown-panel');
    const arrow = document.getElementById('wf-dropdown-arrow');
    if (panel) panel.classList.add('hidden');
    if (arrow) arrow.classList.remove('rotate-180');
}

(function initWorkflowDropdown() {
    document.addEventListener('click', (e) => {
        const panel = document.getElementById('wf-dropdown-panel');
        const toggle = document.getElementById('wf-dropdown-toggle');
        const arrow = document.getElementById('wf-dropdown-arrow');
        if (!panel || !toggle) return;
        if (toggle.contains(e.target)) {
            const open = panel.classList.toggle('hidden');
            if (arrow) arrow.classList.toggle('rotate-180', !open);
            return;
        }
        if (!panel.contains(e.target)) {
            closeWorkflowDropdown();
        }
    });
})();

function getSelectedWorkflowIds() {
    return _selectedWorkflowIds.slice();
}

function renderProjectsFromMemory() {
    const listEl = document.getElementById('projects-list');
    if (!listEl) return;
    const search = (document.getElementById('projects-search')?.value || '').toLowerCase();
    const projects = _projectsSnapshot.filter(p => !search || p.name.toLowerCase().includes(search));

    if (projects.length === 0) {
        listEl.innerHTML = `<div class="text-center text-gray-600 text-sm py-8">${t('no_projects')}</div>`;
        return;
    }
    listEl.innerHTML = projects.map(p => projectCard(p)).join('');
    projects.filter(p => _expanded.has(p.id)).forEach(p => renderTree(p.id));
}

async function loadProjects(options = {}) {
    const force = Boolean(options && options.force);
    const listEl = document.getElementById('projects-list');
    if (!listEl) return;
    if (_projectsLoaded && !force) {
        renderProjectsFromMemory();
        return;
    }
    if (!force && !_projectsLoaded) {
        const cached = readProjectsSnapshotCache();
        if (cached) {
            _projectsSnapshot = cached.projects;
            _projectsLoaded = true;
            renderProjectsFromMemory();
        }
    }

    if (_projectsLoadController) _projectsLoadController.abort();
    const controller = new AbortController();
    _projectsLoadController = controller;
    try {
        const d = await apiFetch('/api/v1/projects?limit=200', { signal: controller.signal });
        if (_projectsLoadController !== controller) return;
        _projectsSnapshot = d.projects || [];
        saveProjectsSnapshotCache(_projectsSnapshot);
        _projectsLoaded = true;
        if (force) _projectTreeSnapshot.clear();
        renderProjectsFromMemory();
    } catch (err) {
        if (err?.name === 'AbortError') return;
        listEl.innerHTML = `<div class="text-center text-red-400 text-sm py-8">${t('project_load_failed')}: ${err.message}</div>`;
    } finally {
        if (_projectsLoadController === controller) _projectsLoadController = null;
    }
}

function projectCard(p) {
    const badge = p.status === 'active'
        ? '<span class="text-green-400 text-xs border border-green-800 rounded px-1.5 py-0.5">active</span>'
        : '<span class="text-gray-500 text-xs border border-gray-700 rounded px-1.5 py-0.5">paused</span>';
    // 工作流名 → 可点击链接,跳转到 Studio 对应工作流(项目为主导,工作流跟随项目)
    // 后端已把 workflow_ids 与 workflow_names 对齐(悬空引用过滤);
    // 这里 zip 后按名字过滤兜底,绝不显示假名(如 'Workflow')
    const wfIds = (p.workflow_ids && p.workflow_ids.length ? p.workflow_ids : []);
    const wfNames = (p.workflow_names && p.workflow_names.length ? p.workflow_names
                     : (p.workflow_name ? [p.workflow_name] : []));
    const wfLinks = wfIds
        .map((id, i) => ({ id, name: wfNames[i] }))
        .filter(w => w.name);
    const wf = wfLinks.length
        ? `<span class="text-xs flex items-center gap-1 flex-wrap">
            <iconify-icon icon="ant-design:deployment-unit-outlined" class="icon-sm text-blue-400"></iconify-icon>
            ${wfLinks.map(w => `
                <a href="/workflow-studio?workflow=${w.id}&project=${p.id}"
                   onclick="event.stopPropagation()" title="Edit workflow"
                   class="text-blue-400 hover:underline">${w.name}</a>`)
                .join('<span class="text-gray-600">·</span>')}
          </span>`
        : `<span class="text-xs text-gray-600">${t('unbound_workflow')}</span>`;
    // 项目条数: 已上传(累计,含删除)/ 目标 M 条 —— 统计上传过的所有数据
    const target = parseInt(p.target_episodes, 10) || 0;
    const uploaded = parseInt(p.uploaded_total, 10) || 0;
    const countLabel = target > 0
        ? `${uploaded} / ${target} ${t('ep_count')}`
        : `${uploaded} ${t('ep_count')}`;
    const newBatches = parseInt(p.new_batch_count, 10) || 0;
    const reuploads = parseInt(p.reupload_count, 10) || 0;
    const unclassified = parseInt(p.unclassified_upload_count, 10) || 0;
    const uploadDetail = `
        <span class="text-gray-600 text-[11px]" title="${t('upload_total')}: ${uploaded}">
            ${t('upload_new')}: ${newBatches} · ${t('upload_reupload')}: ${reuploads}${unclassified ? ` · ${t('upload_unclassified')}: ${unclassified}` : ''}
        </span>`;
    const countBtn = `
        <button onclick="event.stopPropagation(); editProject('${p.id}')"
                title="${t('edit_project')} (${t('target_episodes')}: ${target > 0 ? target : t('target_unlimited')})"
                class="hover:text-blue-400">
            <iconify-icon icon="ant-design:inbox-outlined" class="icon-sm"></iconify-icon> ${countLabel}
        </button>`;
    return `
    <div class="bg-gray-900 rounded-lg border border-gray-800 overflow-hidden">
        <div class="p-4 cursor-pointer hover:bg-gray-800 transition-colors" onclick="toggleTree('${p.id}')">
            <div class="flex items-center justify-between">
                <div class="flex items-center gap-3">
                    <iconify-icon icon="ant-design:folder-outlined" class="text-xl text-blue-500"></iconify-icon>
                    <div>
                        <div class="flex items-center gap-2">
                            <span class="font-medium text-gray-200">${escHtml(p.name)}</span>${badge}
                        </div>
                        <div class="flex items-center gap-3 mt-1 text-gray-500 text-xs">
                            ${wf}
                            ${p.device_type ? `<span><iconify-icon icon="ant-design:video-camera-outlined" class="icon-sm"></iconify-icon> ${escHtml(p.device_type)}</span>` : ''}
                            ${(p.observed_inputs && p.observed_inputs.length)
                                ? `<span class="text-gray-600 text-[11px]"><iconify-icon icon="ant-design:api-outlined" class="icon-sm"></iconify-icon> ${t('observed_inputs')}: ${escHtml(p.observed_inputs.join(', '))}</span>` : ''}
                            <span class="text-gray-400">${countBtn}</span>
                            ${uploadDetail}
                        </div>
                        ${p.description ? `<div class="text-gray-600 text-xs mt-1">${p.description}</div>` : ''}
                    </div>
                </div>
                <div class="flex items-center gap-2">
                    <button onclick="event.stopPropagation(); editProject('${p.id}')"
                            class="text-gray-500 hover:text-blue-400 text-xs px-2 py-1 rounded hover:bg-gray-800"><iconify-icon icon="ant-design:edit-outlined" class="icon-sm"></iconify-icon> ${t('edit_project')}</button>
                    <button onclick="event.stopPropagation(); deleteProject('${p.id}', '${p.name}')"
                            class="text-gray-500 hover:text-red-400 text-xs px-2 py-1 rounded hover:bg-gray-800"><iconify-icon icon="ant-design:delete-outlined" class="icon-sm"></iconify-icon> ${t('delete_project')}</button>
                    <iconify-icon icon="ant-design:down-outlined" class="text-gray-500 ${_expanded.has(p.id) ? 'rotate-180' : ''}" style="transition: transform .15s"></iconify-icon>
                </div>
            </div>
        </div>
        <div id="tree-${p.id}" class="hidden border-t border-gray-800"></div>
    </div>`;
}

function toggleTree(projectId) {
    if (_expanded.has(projectId)) {
        _expanded.delete(projectId);
        loadProjects();
        return;
    }
    _expanded.add(projectId);
    loadProjects();
}

async function renderTree(projectId) {
    const el = document.getElementById(`tree-${projectId}`);
    if (!el) return;
    el.classList.remove('hidden');
    el.innerHTML = `<div class="text-center text-gray-600 text-sm py-4">${t('loading')}</div>`;
    try {
        // 批次列表:报错不在此显示(点击文件后在 Review 详情里看)
        let d = _projectTreeSnapshot.get(projectId);
        if (!d) {
            d = await apiFetch(`/api/v1/projects/${projectId}/tree`);
            _projectTreeSnapshot.set(projectId, d);
        }
        if (!d.tasks || d.tasks.length === 0) {
            el.innerHTML = `<div class="text-center text-gray-600 text-sm py-4">${t('no_tasks_in_project')}</div>`;
            return;
        }
        el.innerHTML = d.tasks.map(taskRow).join('');
    } catch (err) {
        el.innerHTML = `<div class="text-center text-red-400 text-sm py-4">${t('project_load_failed')}: ${err.message}</div>`;
    }
}

function taskRow(task) {
    // 注意:参数名不能用 t —— 会遮蔽全局 i18n 翻译函数 t()
    // 一个上传压缩包 = 一个批次 = 一行显示(不再多级嵌套)
    // 报错不在此显示(点击文件后在 Review 详情里看)
    const e = (task.episodes || [])[0] || {};
    const statusColor = { completed: 'text-green-400', to_review: 'text-yellow-400', reviewed: 'text-blue-400', failed: 'text-red-400' }[e.status] || 'text-gray-400';
    const name = task.name || e.name || e.id;
    const camera = e.camera || '';
    const frames = e.frame_count || 0;
    // 已彻底删除的历史条目(目录已删,记录保留,统计不丢)
    const isPurged = task.purged || e.status === 'purged';
    const statusHtml = isPurged
        ? `<span class="text-xs text-gray-600 shrink-0" title="${t('status_purged')}">${t('status_purged')}</span>`
        : `<span class="text-xs ${statusColor} shrink-0">${e.status}</span>`;
    const nameHtml = isPurged
        ? `<span class="text-sm text-gray-600 truncate" title="${t('status_purged')}">${name}</span>`
        : `<a href="/review?search=${encodeURIComponent(name)}" class="text-sm text-gray-300 hover:text-blue-400 truncate">${name}</a>`;
    return `
    <div class="px-4 py-2.5 border-b border-gray-800 last:border-0 flex items-center justify-between gap-3 ${isPurged ? 'opacity-70' : ''}">
        <div class="flex items-center gap-2 min-w-0">
            <iconify-icon icon="ant-design:delete-outlined" class="text-gray-700 shrink-0"></iconify-icon>
            ${nameHtml}
            ${camera && !isPurged ? `<span class="text-xs text-gray-600 shrink-0">${camera}</span>` : ''}
            ${!isPurged ? `<span class="text-xs text-gray-600 shrink-0">${frames} ${t('frame')}</span>` : ''}
            ${statusHtml}
        </div>
        <span class="text-xs text-gray-600 shrink-0">${task.created_at ? new Date(task.created_at).toLocaleString() : ''}</span>
    </div>`;
}

// ── Project form ─────────────────────────────────────────

function showProjectForm() {
    editingProjectId = null;
    document.getElementById('project-form-title').textContent = t('new_project');
    document.getElementById('pf-name').value = '';
    document.getElementById('pf-description').value = '';
    document.getElementById('pf-status').value = 'active';
    _selectedWorkflowIds = [];
    document.getElementById('pf-target').value = '';
    loadWorkflowsForSelect();
    closeWorkflowDropdown();  // 打开表单时下拉收起
    document.getElementById('project-form').classList.remove('hidden');
}

function hideProjectForm() {
    document.getElementById('project-form').classList.add('hidden');
}

async function editProject(id) {
    try {
        const p = await apiFetch(`/api/v1/projects/${id}`);
        editingProjectId = id;
        document.getElementById('project-form-title').textContent = t('edit_project') + ': ' + p.name;
        document.getElementById('pf-name').value = p.name;
        document.getElementById('pf-description').value = p.description || '';
        document.getElementById('pf-status').value = p.status;
        document.getElementById('pf-target').value = p.target_episodes || '';
        _selectedWorkflowIds = (p.workflow_ids || []).slice();
        await loadWorkflowsForSelect();
        closeWorkflowDropdown();
        document.getElementById('project-form').classList.remove('hidden');
    } catch (e) { alert(t('project_load_failed') + ': ' + e.message); }
}

// 表单校验 + 组装请求体;不合法 → alert 并返回 null
function buildProjectBody() {
    const name = document.getElementById('pf-name').value.trim();
    if (!name) { alert(t('project_name')); return null; }
    const targetVal = parseInt(document.getElementById('pf-target').value, 10);
    const target = Number.isFinite(targetVal) && targetVal > 0 ? targetVal : 0;
    const status = document.getElementById('pf-status').value;
    const wfIds = getSelectedWorkflowIds();
    // 项目可以先 active 接收数据；没有绑定工作流时只入库不处理，
    // 绑定并保存工作流后由后端自动回填历史批次。
    // device_type 已移除(手工标记被 observed_inputs 自动记录替代);
    // 编辑时后端只更新 body 里的键,老项目的 device_type 值保留
    return {
        body: {
            name,
            workflow_id: wfIds[0] || null,
            workflow_ids: wfIds,
            description: document.getElementById('pf-description').value.trim() || null,
            status: status,
            params: target > 0 ? { target_episodes: target } : {},
        },
        wfIds,
    };
}

async function saveProject() {
    const built = buildProjectBody();
    if (!built) return;
    const isNew = !editingProjectId;
    try {
        const saved = isNew
            ? await apiFetch('/api/v1/projects', { method: 'POST', body: JSON.stringify(built.body) })
            : await apiFetch(`/api/v1/projects/${editingProjectId}`, { method: 'PUT', body: JSON.stringify(built.body) });
        hideProjectForm();
        loadProjects({ force: true });
        // 只有新建项目才弹窗提醒并跳转工作流画布(可取消,项目已保存不会丢);
        // 编辑已有项目只保存,不打扰。画布上可直接用模板或新建工作流。
        if (isNew && confirm(t('project_saved_go_build'))) {
            const q = new URLSearchParams();
            if (built.wfIds.length) q.set('workflow', built.wfIds[0]);
            q.set('project', saved.id);
            window.location = `/workflow-studio?${q.toString()}`;
        }
    } catch (e) { alert(t('project_save_failed') + ': ' + e.message); }
}

async function deleteProject(id, name) {
    if (!confirm(t('confirm_delete_project') + name + t('confirm_delete_project_suffix'))) return;
    try {
        await apiFetch(`/api/v1/projects/${id}`, { method: 'DELETE' });
        _expanded.delete(id);
        loadProjects({ force: true });
    } catch (e) { alert(t('project_delete_failed') + ': ' + e.message); }
}

// ── Init ─────────────────────────────────────────────────

document.addEventListener('DOMContentLoaded', () => {
    if (document.getElementById('projects-list')) {
        loadProjects();
    }
});
