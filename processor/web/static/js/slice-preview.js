/* Slice Preview 漂浮面板 —— 视频标注页右侧的切片预览(只读跳转)。
 *
 * 行为:
     *  - 第 1 条 = 当前帧所在切片(无则显示最近的"下一段"待看);
     *    下方 2 条 = 接下来要看的切片,随播放自动滚动
 *  - 点击条目 → seek 到该切片起点;条目上的 ▶ → playSegment 播放该切片
 *  - 标题栏按住拖动;▾ 折叠/展开;标注、审核和审核通过页面显示
 *  - 数据与右侧标注卡片同源(annotations / sortedAnnotations),不做编辑
 */

(function () {
    'use strict';

    let box = null, listEl = null, countEl = null, collapsed = false;
    let _lastFirstIdx = -1;   // 列表窗口第一位的段索引(变化才重渲染)
    let _episodeKey = null;    // 防止切换批次后复用旧 DOM 窗口

    const FALLBACK_COLOR = '#3b82f6';

    function esc(s) {
        return String(s == null ? '' : s)
            .replace(/&/g, '&amp;').replace(/</g, '&lt;')
            .replace(/>/g, '&gt;').replace(/"/g, '&quot;');
    }

    function colorOf(seg, i) {
        if (typeof annotationColor === 'function') {
            try { return annotationColor(seg, i); } catch (_) { /* fallthrough */ }
        }
        return FALLBACK_COLOR;
    }

    function canDisplay() {
        const mode = window.EGODATA_PAGE_MODE;
        return (mode === 'annotation' || mode === 'review')
            && window.__episodeOpen === true;
    }

    function activeEpisodeKey() {
        // currentEpisodeId is declared by player.js as a top-level lexical
        // binding, so guard the lookup for older pages that do not load it.
        try {
            return typeof currentEpisodeId !== 'undefined'
                ? String(currentEpisodeId || '') : '';
        } catch (_) {
            return '';
        }
    }

    function resetPanelContent() {
        _lastFirstIdx = -1;
        if (listEl) listEl.innerHTML = '';
        if (countEl) countEl.textContent = '';
    }

    function resetSlicePreview(episodeKey) {
        _episodeKey = episodeKey == null ? activeEpisodeKey() : String(episodeKey || '');
        resetPanelContent();
    }

    function ensurePanel() {
        if (box) {
            // 点开批次详情才显示;标注/审核/审核通过页面都显示
            box.style.display = canDisplay()
                ? '' : 'none';
            return canDisplay();
        }
        if (!canDisplay()) return false;

        box = document.createElement('div');
        box.id = 'slice-preview';
        // Align the panel header with the Preview Options button, which also
        // starts at workspace-relative top:48px.
        box.style.cssText = 'position:absolute;right:12px;top:48px;z-index:45;width:264px;' +
            'background:rgba(15,23,42,.55);backdrop-filter:blur(8px);border:1px solid rgba(55,65,81,.8);' +
            'border-radius:12px;box-shadow:0 8px 24px rgba(0,0,0,.35);overflow:hidden;';
        (document.getElementById('review-workspace') || document.body).appendChild(box);

        // 标题栏(拖动手柄 + 折叠)
        const head = document.createElement('div');
        head.style.cssText = 'display:flex;align-items:center;gap:6px;padding:8px 10px;' +
            'cursor:move;user-select:none;border-bottom:1px solid rgba(31,41,55,.7);font-size:12px;color:#e5e7eb;';
        const title = document.createElement('span');
        title.textContent = 'Slices Preview';
        head.appendChild(title);
        countEl = document.createElement('span');
        countEl.style.cssText = 'color:#9ca3af;font-family:monospace;font-size:10px;';
        head.appendChild(countEl);
        const fold = document.createElement('button');
        fold.type = 'button';
        fold.innerHTML = '<iconify-icon icon="ant-design:down-outlined" class="icon-sm"></iconify-icon>';
        fold.style.cssText = 'margin-left:auto;background:none;border:none;color:#9ca3af;' +
            'cursor:pointer;padding:0 2px;';
        fold.title = 'Collapse / Expand';
        fold.setAttribute('aria-label', 'Collapse / Expand slices preview');
        fold.setAttribute('aria-expanded', 'true');
        // The header is also a drag handle and captures the pointer. Stop
        // that gesture at the button so the browser can dispatch a real
        // click to the fold control instead of the parent header.
        fold.addEventListener('pointerdown', (e) => e.stopPropagation());
        fold.addEventListener('mousedown', (e) => e.stopPropagation());
        fold.addEventListener('click', (e) => {
            e.preventDefault();
            e.stopPropagation();
            collapsed = !collapsed;
            listEl.style.display = collapsed ? 'none' : '';
            fold.setAttribute('aria-expanded', String(!collapsed));
            fold.querySelector('iconify-icon').setAttribute('icon',
                collapsed ? 'ant-design:right-outlined' : 'ant-design:down-outlined');
        });
        head.appendChild(fold);
        box.appendChild(head);

        listEl = document.createElement('div');
        listEl.style.cssText = 'padding:6px;display:flex;flex-direction:column;gap:4px;';
        box.appendChild(listEl);

        // 拖动(与 ☷ Preview Options 同一套指针逻辑)。
        // 坐标用 offsetLeft/offsetTop(相对视频工作区,与 style.left/top
        // 同坐标系)—— getBoundingClientRect 是视口坐标,混用会跳位。
        let dragStart = null;
        head.addEventListener('pointerdown', (e) => {
            dragStart = { x: e.clientX, y: e.clientY,
                          bx: box.offsetLeft, by: box.offsetTop, moved: false };
            head.setPointerCapture(e.pointerId);
        });
        head.addEventListener('pointermove', (e) => {
            if (!dragStart) return;
            const dx = e.clientX - dragStart.x, dy = e.clientY - dragStart.y;
            if (Math.abs(dx) + Math.abs(dy) > 5) dragStart.moved = true;
            if (dragStart.moved) {
                box.style.left = Math.max(0, dragStart.bx + dx) + 'px';
                box.style.top = Math.max(0, dragStart.by + dy) + 'px';
                box.style.right = 'auto';
            }
        });
        head.addEventListener('pointerup', () => { dragStart = null; });
        return true;
    }

    function currentWindow(segs) {
        const frame = (typeof getCurrentFrame === 'function') ? getCurrentFrame() : -1;
        if (!segs.length) return { first: -1, items: [] };
        let curIdx = -1;
        // 与右侧卡片同一套"单一当前段"规则(段重叠时取开始帧最近的段),
        // 避免切片预览和卡片高亮不一致导致的"定位跳回/错位"
        if (typeof _singleCurrentAnnoId === 'function') {
            const id = _singleCurrentAnnoId(frame, null);
            curIdx = segs.findIndex(s => s.id === id);
        }
        if (curIdx === -1) {
            curIdx = segs.findIndex(s => s.start_frame_index > frame);
            if (curIdx === -1) {
                return { first: -2, items: [] };  // 已过全部切片
            }
        }
        return { first: curIdx, items: segs.slice(curIdx, curIdx + 3) };
    }

    function renderItem(seg, globalIdx, isCurrent) {
        const item = document.createElement('div');
        const color = colorOf(seg, globalIdx);
        item.dataset.annoId = seg.id;
        item.style.cssText = 'display:flex;align-items:center;gap:7px;padding:7px 8px;' +
            'border-radius:8px;cursor:pointer;border:1px solid;font-size:12px;' +
            (isCurrent
                ? 'border:#0891b2;background:rgba(8,145,178,.18);'
                : 'border:#1f2937;background:rgba(31,41,55,.45);opacity:.88;') +
            'transition:background .12s;';
        item.onmouseenter = () => { if (!isCurrent) item.style.background = 'rgba(55,65,81,.6)'; };
        item.onmouseleave = () => { if (!isCurrent) item.style.background = 'rgba(31,41,55,.45)'; };

        const strip = document.createElement('div');
        strip.style.cssText = `width:3px;align-self:stretch;border-radius:2px;background:${color};flex-shrink:0;`;
        item.appendChild(strip);

        const mid = document.createElement('div');
        mid.style.cssText = 'flex:1;min-width:0;';
        const fps = (typeof getEpisodeFps === 'function') ? getEpisodeFps() : 30;
        const isCandidate = (seg.status || 'confirmed') === 'candidate';
        const boundaryBad = seg.boundary_ok === false || seg.label_matched === false;
        mid.innerHTML = `
            <div style="color:${isCurrent ? '#e0f2fe' : '#d1d5db'};white-space:normal;overflow-wrap:anywhere;line-height:1.35;">
                ${isCandidate ? '<span style="color:#22d3ee;" title="AI candidate">✨</span> ' : ''}${boundaryBad ? '<span style="color:#f59e0b;" title="边界可疑或标签未命中词表">⚠</span> ' : ''}${globalIdx + 1} · ${esc(seg.label || '—')}
            </div>
            <div style="color:#9ca3af;font-family:monospace;font-size:10px;">
                ${seg.start_frame_index}-${seg.end_frame_index} · ${(seg.start_frame_index / fps).toFixed(1)}s
            </div>`;
        item.appendChild(mid);

        const play = document.createElement('button');
        play.type = 'button';
        play.innerHTML = '<iconify-icon icon="ant-design:caret-right-outlined" class="icon-sm"></iconify-icon>';
        play.title = 'Play this slice';
        play.style.cssText = 'background:none;border:none;color:#9ca3af;cursor:pointer;' +
            'padding:2px 4px;flex-shrink:0;';
        play.addEventListener('click', (e) => {
            e.stopPropagation();
            if (typeof playSegment === 'function') {
                playSegment(seg.start_frame_index, seg.end_frame_index);
            }
        });
        item.appendChild(play);

        // Review and Approved are read-only. Only Annotation Studio exposes
        // the editor action; click/seek/play remains available in all views.
        if (window.EGODATA_PAGE_MODE === 'annotation') {
            const edit = document.createElement('button');
            edit.type = 'button';
            edit.innerHTML = '<iconify-icon icon="ant-design:edit-outlined" class="icon-sm"></iconify-icon>';
            edit.title = 'Edit this slice';
            edit.style.cssText = 'background:none;border:none;color:#9ca3af;cursor:pointer;' +
                'padding:2px 4px;flex-shrink:0;';
            edit.addEventListener('click', (e) => {
                e.stopPropagation();
                if (typeof editAnnotation === 'function') editAnnotation(seg.id);
            });
            item.appendChild(edit);
        }

        item.addEventListener('click', () => {
            if (typeof seekToFrame === 'function') seekToFrame(seg.start_frame_index);
            // 同步右侧卡片:优先高亮被点的段(段重叠时不错亮)并滚动到可视区
            if (typeof updateCurrentAnnotation === 'function') {
                updateCurrentAnnotation(seg.start_frame_index, seg.id);
            }
        });
        return item;
    }

    function refreshSlicePreview() {
        const episodeKey = activeEpisodeKey();
        if (_episodeKey !== episodeKey) resetSlicePreview(episodeKey);
        if (!ensurePanel()) {
            // Do not retain visible content when returning to the episode
            // list; this also prevents stale slices on the next open.
            if (!canDisplay()) resetPanelContent();
            return;
        }
        const segs = (typeof sortedAnnotations === 'function')
            ? sortedAnnotations() : (window.annotations || []).slice();
        const win = currentWindow(segs);

        if (win.first === -1) {
            _lastFirstIdx = -3;
            listEl.innerHTML = '<div style="color:#6b7280;text-align:center;padding:10px 4px;font-size:11px;">No slices</div>';
            countEl.textContent = '';
            return;
        }
        if (win.first === -2) {
            _lastFirstIdx = -4;
            listEl.innerHTML = '<div style="color:#6b7280;text-align:center;padding:10px 4px;font-size:11px;">All slices reviewed</div>';
            countEl.textContent = '';
            return;
        }
        // 窗口首段没变 → 不重建(播放中每秒 ~25-60 次帧回调,只在跨段时刷新)
        if (win.first === _lastFirstIdx) return;
        _lastFirstIdx = win.first;
        countEl.textContent = `[${win.first + 1} / ${segs.length}]`;
        listEl.innerHTML = '';
        win.items.forEach((seg, i) => {
            listEl.appendChild(renderItem(seg, win.first + i, i === 0));
        });
    }

    /* 帧变化驱动(多回调注册,与标注高亮/骨骼叠加互不覆盖) */
    if (typeof setOnFrameChange === 'function') {
        setOnFrameChange(() => refreshSlicePreview());
    }
    window.refreshSlicePreview = refreshSlicePreview;
    window.resetSlicePreview = resetSlicePreview;
})();
