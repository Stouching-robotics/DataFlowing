import { useEffect, useRef, useState } from 'react';
import { ReactFlowProvider } from '@xyflow/react';
import { WorkflowCanvas } from './components/WorkflowCanvas/WorkflowCanvas';
import { NodePalette } from './components/WorkflowCanvas/NodePalette';
import { IconifyIcon } from './components/WorkflowCanvas/IconifyIcon';
import { clearDraft, normalizeWorkflowGraphForEditor, useWorkflowStore } from './store/workflowStore';
import { useUiStore } from './store/uiStore';
import { ToastContainer, GlobalDialog } from './components/ui/AppDialogs';
import { deleteWorkflow, getWorkflow, getWorkflowUsage, listModules, updateWorkflow } from './api/workflows';
import { getCurrentUser } from './api/auth';
import { hydrateNodeTypes } from './components/WorkflowCanvas/nodes/registry';
import type { WorkflowListItem } from './types/workflow';

/**
 * Compare only persisted workflow semantics. React Flow adds presentation
 * metadata (labels, handles, measured dimensions, selection state, etc.)
 * while the registry also hydrates old nodes on load. Those fields must not
 * turn an identical saved graph into a dirty draft.
 */
function comparableGraph(graph: any) {
  const stableValue = (value: any): any => {
    if (!value || typeof value !== 'object') return value;
    if (Array.isArray(value)) return value.map(stableValue);
    return Object.keys(value).sort().reduce((result, key) => {
      result[key] = stableValue(value[key]);
      return result;
    }, {} as Record<string, any>);
  };
  const nodes = (graph?.nodes || []).map((node: any) => ({
    id: node.id,
    type: node.type || 'workflowNode',
    position: {
      x: Number(node.position?.x || 0),
      y: Number(node.position?.y || 0),
    },
    parentId: node.parentId || null,
    data: {
      nodeType: node.data?.nodeType || null,
      config: stableValue(node.data?.config || {}),
    },
  })).sort((a: any, b: any) => String(a.id).localeCompare(String(b.id)));
  const edges = (graph?.edges || []).map((edge: any) => ({
    id: edge.id || null,
    source: edge.source,
    target: edge.target,
    sourceHandle: edge.sourceHandle || null,
    targetHandle: edge.targetHandle || null,
    type: edge.type || null,
  })).sort((a: any, b: any) => String(a.id).localeCompare(String(b.id)));
  return JSON.stringify({ nodes, edges });
}

function graphsMatch(left: any, right: any) {
  return comparableGraph(left) === comparableGraph(right);
}

export default function App() {
  useEffect(() => {
    listModules().then((items) => hydrateNodeTypes(items as any)).catch(() => {
      // The built-in registry remains available when the API is offline.
    });
  }, []);

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      const ctrl = e.ctrlKey || e.metaKey;
      // Ctrl+S 由 WorkflowToolbar 统一处理(与 Save 按钮同权限/确认流程)
      if (ctrl && e.key === 'z' && !e.shiftKey) {
        e.preventDefault();
        useWorkflowStore.getState().undo();
      }
      if (ctrl && (e.key === 'y' || (e.key === 'z' && e.shiftKey))) {
        e.preventDefault();
        useWorkflowStore.getState().redo();
      }
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, []);

  // Auto-load most recent workflow or restore unsaved draft on mount.
  // 优先级:URL ?workflow= > localStorage draft > 最后创建的工作流。
  // 显式导航(?workflow=)必须跳过 draft 恢复,否则项目页跳转会被旧草稿顶掉。
  useEffect(() => {
    const store = useWorkflowStore.getState();
    const params = new URLSearchParams(window.location.search);
    const wfParam = params.get('workflow');
    const projectParam = params.get('project');

    // 当前用户角色(决定预设编辑权限);失败 → userRole 保持 null(按非 admin)
    getCurrentUser()
      .then((u) => useWorkflowStore.setState({ userRole: u.role }))
      .catch(() => {});
    // ?project= 是显式项目上下文。没有它时，等工作流加载完成后再用
    // workflow.project_id 解析输入源，避免直接从工作流模块进入时退回全局设备。
    if (projectParam) useWorkflowStore.setState({ workflowProjectId: projectParam });

    const restoreDraftOrFirst = async () => {
      // Check for unsaved draft first
      try {
        const raw = localStorage.getItem('workflow_draft');
        if (raw) {
          let draft = JSON.parse(raw);
          const normalizedDraft = draft?.nodes && Array.isArray(draft.nodes)
            ? normalizeWorkflowGraphForEditor({ nodes: draft.nodes, edges: draft.edges || [] })
            : null;
          // Older builds left the draft in localStorage after loading a saved
          // workflow. An identical draft is not an unsaved edit.
          if (draft.workflowId) {
            try {
              const saved = await getWorkflow(String(draft.workflowId));
              if (normalizedDraft && graphsMatch(normalizedDraft, saved.graph)) {
                clearDraft();
                draft = null;
              }
            } catch {
              // Keep a real draft when the API is temporarily unavailable.
            }
          }
          if (draft && draft.nodes && Array.isArray(draft.nodes) && draft.nodes.length > 0) {
            const normalized = normalizedDraft || normalizeWorkflowGraphForEditor(draft);
            useWorkflowStore.setState({
              nodes: normalized.nodes,
              edges: normalized.edges,
              workflowId: draft.workflowId || null,
              workflowProjectId: projectParam || draft.workflowProjectId || null,
              workflowName: draft.workflowName || 'Untitled',
              isDirty: true,
            });
            store.loadWorkflowList();
            const draftProjectId = projectParam || draft.workflowProjectId;
            store.loadInputsForWorkflow(draftProjectId);
            return;
          }
        }
      } catch {}

      // No draft — load the last created workflow by default
      // (workflows.json 追加顺序,末尾 = 最新创建)
      store.loadWorkflowList().then(() => {
        const { workflows } = useWorkflowStore.getState();
        if (workflows.length > 0) {
          store.loadWorkflow(workflows[workflows.length - 1].id).then(() => {
            const current = useWorkflowStore.getState();
            // A workflow's persisted project wins over a stale URL project.
            return current.loadInputsForWorkflow(current.workflowProjectId || projectParam);
          });
        } else if (projectParam) {
          store.loadInputsForWorkflow(projectParam);
        } else {
          // A new/direct workflow has no project input context. Keep the
          // palette's fixed device categories, but do not show global online
          // devices from another project in the node cards.
          store.loadInputsForWorkflow(null);
        }
      });
    };

    if (wfParam) {
      store.loadWorkflowList();
      store.loadWorkflow(wfParam).then(() => {
        const current = useWorkflowStore.getState();
        return current.loadInputsForWorkflow(current.workflowProjectId || projectParam);
      }).catch(() => {
        // URL 工作流不存在(被删)→ 回退 draft/首个工作流
        void restoreDraftOrFirst();
      });
      return;
    }
    void restoreDraftOrFirst();
  }, []);

  return (
    <ReactFlowProvider>
      <div className="h-full w-full flex flex-col">
        <WorkflowToolbar />
        <div className="flex-1 flex min-h-0">
          <WorkflowCanvas />
          <NodePalette />
        </div>
      </div>
      <ToastContainer />
      <GlobalDialog />
    </ReactFlowProvider>
  );
}

/** Inline toolbar — workflow switcher + rename + Save.
 *  运行是自动的:数据上传后按项目绑定的工作流自动入队执行,无手动 Run。
 *  预设权限:非 admin 对预设只读(保存 → 另存为);admin 可改/可新增/可取消预设。 */
interface ConfirmDialogState {
  title: string;
  message: string;
  confirmLabel: string;
  danger?: boolean;
  /** 次按钮(如"另存为副本");存在则渲染在 Cancel 与主按钮之间。 */
  secondaryLabel?: string;
  onConfirm: () => void | Promise<void>;
  onSecondary?: () => void | Promise<void>;
}

function WorkflowToolbar() {
  const { workflowName, workflowId, isDirty, isSaving, workflows, userRole, isPresetWorkflow, saveWorkflowSafe, saveWorkflowAs, loadWorkflow, loadWorkflowList } =
    useWorkflowStore();
  const [editingName, setEditingName] = useState(false);
  const [nameDraft, setNameDraft] = useState(workflowName);
  const [msg, setMsg] = useState('');
  const [confirmDialog, setConfirmDialog] = useState<ConfirmDialogState | null>(null);
  const [dialogBusy, setDialogBusy] = useState(false);
  // 新建工作流:选择 Blank / 模板(模板样式 = 只含后处理链,设备按实际采集端添加)
  const [templateDialog, setTemplateDialog] = useState(false);
  const [templateBusy, setTemplateBusy] = useState(false);

  const isAdmin = userRole === 'admin';
  const presetReadOnly = isPresetWorkflow && !isAdmin;

  const flash = (s: string) => { setMsg(s); setTimeout(() => setMsg(''), 2000); };

  const syncUrl = () => {
    const s = useWorkflowStore.getState();
    if (!s.workflowId) return;
    const params = new URLSearchParams(window.location.search);
    params.set('workflow', s.workflowId);
    if (s.workflowProjectId) params.set('project', s.workflowProjectId);
    else params.delete('project');
    const q = params.toString();
    window.history.replaceState(null, '', `/workflow-studio${q ? `?${q}` : ''}`);
  };

  /** 预设且非 admin:prompt 另存为(权限现状保持)。 */
  const runPresetSaveAs = async (s = useWorkflowStore.getState()) => {
    const name = await useUiStore.getState().prompt(
      'This is a template workflow and cannot be overwritten. Save a copy as:',
      { title: 'Save a copy', defaultValue: (s.workflowName || 'Untitled') + ' (copy)', confirmLabel: 'Save' },
    );
    if (!name || !name.trim()) return;
    try {
      await s.saveWorkflowAs(name.trim());
      await s.loadWorkflowList();
      syncUrl();
      flash('Saved as copy');
    } catch (err: any) { useUiStore.getState().pushToast(err.message || 'Save failed', 'error'); }
  };

  /** 直接保存(store 兜底 403 → 转另存为)。 */
  const runDirectSaveInner = async () => {
    const s = useWorkflowStore.getState();
    const r = await s.saveWorkflowSafe();
    if (r === 'needs-save-as') { await runPresetSaveAs(s); return; }
    if (r === 'saved') flash('Saved');
  };

  /** 共享工作流(被 ≥2 项目使用):改全局前提示影响范围。 */
  const runDirectSave = async () => {
    const s = useWorkflowStore.getState();
    if (s.workflowId) {
      try {
        const usage = await getWorkflowUsage(s.workflowId);
        if (usage.project_count >= 2) {
          const names = usage.projects.map((p) => p.name).join(', ');
          setConfirmDialog({
            title: 'Shared workflow',
            message: `This workflow is used by ${usage.project_count} projects (${names}). Saving directly will affect all of them.`,
            confirmLabel: 'Save directly',
            secondaryLabel: 'Save as copy',
            onConfirm: () => runDirectSaveInner(),
            onSecondary: () => runSaveAsCopy(),
          });
          return;
        }
      } catch (err) {
        // usage 查询失败不阻断保存(后端 403 仍由 saveWorkflowSafe 兜底)
        console.warn('usage check failed, saving directly', err);
      }
    }
    await runDirectSaveInner();
  };

  const runSaveAsCopy = async () => {
    const s = useWorkflowStore.getState();
    const name = await useUiStore.getState().prompt('Save a copy as:', {
      title: 'Save a copy', defaultValue: (s.workflowName || 'Untitled') + ' (copy)', confirmLabel: 'Save',
    });
    if (!name || !name.trim()) return;
    try {
      await s.saveWorkflowAs(name.trim());
      await s.loadWorkflowList();
      syncUrl();
      flash('Saved as copy');
    } catch (err: any) { useUiStore.getState().pushToast(err.message || 'Save failed', 'error'); }
  };

  const saveSafe = async () => {
    const s = useWorkflowStore.getState();
    // 新工作流(未保存):先命名再保存,不直接创建 "Untitled"
    if (!s.workflowId) {
      setNameDraft(s.workflowName);
      setEditingName(true);
      return;
    }
    // 预设 + 非 admin:只读,强制另存为(现状)
    if (presetReadOnly) { await runPresetSaveAs(s); return; }
    // 空工作流提醒:0 节点不会处理任何上传数据
    if (s.workflowId && s.nodes.length === 0) {
      setConfirmDialog({
        title: 'Save empty workflow?',
        message: 'This workflow has no nodes, so it will not process any uploaded data. Save anyway?',
        confirmLabel: 'Save anyway',
        onConfirm: () => runDirectSave(),
      });
      return;
    }
    await runDirectSave();
  };

  const commitName = async () => {
    const name = nameDraft.trim();
    setEditingName(false);
    if (!name) { setNameDraft(workflowName); return; }
    if (name === workflowName) return;
    if (presetReadOnly) {
      flash('Template workflows can only be renamed by admin. Save a copy instead.');
      return;
    }
    if (workflowId) {
      // 已有工作流:改 store 里的名称后保存(更新后端 name)
      useWorkflowStore.setState({ workflowName: name });
      try {
        await saveWorkflowSafe();
        await loadWorkflowList();   // 刷新下拉列表,让新名称立即显示
        flash('Renamed');
      } catch { /* saveWorkflowSafe rethrows non-403 errors */ }
    } else {
      // 未保存的新工作流:直接以新名称创建
      await saveWorkflowAs(name);
      await loadWorkflowList();
      flash('Saved');
    }
  };

  const applyWorkflowSwitch = async (id: string) => {
    const store = useWorkflowStore.getState();
    if (id === '__new__') {
      // 直接新建空白工作流(熟练用户手动拖节点);
      // 模板不在这里:工具栏 Template 按钮 = 应用模板到当前工作流
      store.newWorkflow();
      const params = new URLSearchParams(window.location.search);
      const projectId = params.get('project');
      if (projectId) {
        useWorkflowStore.setState({ workflowProjectId: projectId });
        await useWorkflowStore.getState().loadInputsForWorkflow(projectId);
      } else {
        await useWorkflowStore.getState().loadInputsForWorkflow(null);
      }
      params.delete('workflow');
      const q = params.toString();
      window.history.replaceState(null, '', `/workflow-studio${q ? `?${q}` : ''}`);
    } else if (id) {
      await loadWorkflow(id);
      const current = useWorkflowStore.getState();
      // If another switch completed later, this invocation is stale and must
      // not start a second request for the wrong project.
      if (current.workflowId !== id) return;
      const projectFromUrl = new URLSearchParams(window.location.search).get('project');
      const projectId = current.workflowProjectId || projectFromUrl;
      await current.loadInputsForWorkflow(projectId);
      if (useWorkflowStore.getState().workflowId !== id) return;
      syncUrl();
    }
  };

  /** 应用模板到当前工作流(替换画布,保留工作流 ID/名称)。
   *  清掉 URL 里的 workflow 参数,未保存刷新时走草稿恢复而非旧后端内容。 */
  const pickTemplate = async (t: WorkflowListItem | null) => {
    setTemplateBusy(true);
    try {
      const store = useWorkflowStore.getState();
      if (t) await store.newWorkflowFromTemplate(t);
      const params = new URLSearchParams(window.location.search);
      params.delete('workflow');
      const q = params.toString();
      window.history.replaceState(null, '', `/workflow-studio${q ? `?${q}` : ''}`);
    } catch (err: any) {
      flash(err?.message || 'Failed to load template');
    } finally {
      setTemplateBusy(false);
      setTemplateDialog(false);
    }
  };

  const handleSwitch = async (e: React.ChangeEvent<HTMLSelectElement>) => {
    const id = e.target.value;
    const store = useWorkflowStore.getState();
    if (store.isDirty) {
      setConfirmDialog({
        title: 'Discard unsaved changes?',
        message: 'Your current workflow has unsaved changes. Switching workflows will discard them.',
        confirmLabel: 'Discard and switch',
        danger: true,
        onConfirm: () => applyWorkflowSwitch(id),
      });
      return;
    }
    await applyWorkflowSwitch(id);
  };

  /** 打开模板选择弹窗:与 handleSwitch 一致的 isDirty 保护,
   *  防止未保存改动被"从模板新建"无声丢弃。 */
  const openTemplateDialog = () => {
    const store = useWorkflowStore.getState();
    if (store.isDirty) {
      setConfirmDialog({
        title: 'Discard unsaved changes?',
        message: 'Your current workflow has unsaved changes. Starting from a template or a blank workflow will discard them.',
        confirmLabel: 'Discard and continue',
        danger: true,
        onConfirm: () => setTemplateDialog(true),
      });
      return;
    }
    setTemplateDialog(true);
  };

  const togglePreset = () => {
    const s = useWorkflowStore.getState();
    if (!s.workflowId) return;
    const nextPreset = !isPresetWorkflow;
    setConfirmDialog({
      title: nextPreset ? 'Make this a template workflow?' : 'Remove template status?',
      message: nextPreset
        ? 'This workflow will become a shared template for all users.'
        : 'This workflow will become a regular workflow. Its nodes and connections will not change.',
      confirmLabel: nextPreset ? 'Make Template' : 'Remove template status',
      onConfirm: async () => {
        try {
          await updateWorkflow(s.workflowId as string, { is_preset: nextPreset });
          await loadWorkflowList();
          useWorkflowStore.setState({ isPresetWorkflow: nextPreset });
          flash(nextPreset ? 'Template created' : 'Template removed');
        } catch (err: any) { flash(err.message || 'Failed'); }
      },
    });
  };

  // 执行后若执行期间弹了新对话框(如"共享提醒"→"空工作流"链),保留新对话框
  const closeDialogIfCurrent = (dialog: ConfirmDialogState) => {
    setDialogBusy(false);
    setConfirmDialog((cur) => (cur === dialog ? null : cur));
  };

  const confirmDialogAction = async () => {
    const dialog = confirmDialog;
    if (!dialog || dialogBusy) return;
    setDialogBusy(true);
    try {
      await dialog.onConfirm();
    } catch (err: any) {
      flash(err?.message || 'Operation failed');
    } finally {
      closeDialogIfCurrent(dialog);
    }
  };

  const confirmDialogSecondary = async () => {
    const dialog = confirmDialog;
    if (!dialog || dialogBusy) return;
    setDialogBusy(true);
    try {
      await dialog.onSecondary?.();
    } catch (err: any) {
      flash(err?.message || 'Operation failed');
    } finally {
      closeDialogIfCurrent(dialog);
    }
  };

  // Ctrl+S 与 Save 按钮同一套确认流程(空工作流/共享提醒/预设另存为)
  const saveSafeRef = useRef(saveSafe);
  useEffect(() => { saveSafeRef.current = saveSafe; });
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if ((e.ctrlKey || e.metaKey) && e.key === 's') {
        e.preventDefault();
        saveSafeRef.current();
      }
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, []);

  const dot = isDirty ? 'bg-yellow-500' : msg ? 'bg-green-500' : 'bg-gray-600';

  return (
    <div className="relative flex items-center gap-2 px-4 py-1.5 bg-gray-900 border-b border-gray-800 shrink-0">
      <span className={`w-2 h-2 rounded-full shrink-0 ${dot}`} />
      <select
        value={workflowId || ''}
        onChange={handleSwitch}
        onClick={() => loadWorkflowList()}
        className="bg-transparent border border-gray-700 rounded px-1.5 py-0.5 text-xs font-semibold text-gray-200 focus:outline-none focus:border-blue-500 cursor-pointer max-w-[200px]"
      >
        {!workflowId && <option value="" disabled hidden>New Workflow</option>}
        {workflows.map((w) => (
          <option key={w.id} value={w.id} className="bg-gray-800 text-gray-200">{w.name}{w.is_preset ? ' (Template)' : ''}</option>
        ))}
        {workflowId && !workflows.find((w) => w.id === workflowId) && (
          <option value={workflowId} className="bg-gray-800 text-gray-200">{workflowName}</option>
        )}
        <option value="__new__" className="bg-gray-800 text-gray-400">+ New Workflow</option>
      </select>
      {editingName ? (
        <>
          <input autoFocus value={nameDraft} onChange={(e) => setNameDraft(e.target.value)}
            onKeyDown={(e) => { if (e.key === 'Enter') commitName(); if (e.key === 'Escape') { setNameDraft(workflowName); setEditingName(false); } }}
            onBlur={commitName}
            className="bg-gray-800 border border-gray-600 rounded px-1.5 py-0.5 text-xs text-white w-48 focus:outline-none focus:border-blue-500"
            title="Press Enter to save" />
        </>
      ) : (
        <span
          onClick={() => {
            if (presetReadOnly) { flash('Template workflows can only be renamed by admin. Save a copy instead.'); return; }
            setNameDraft(workflowName); setEditingName(true);
          }}
          title="Click to rename workflow"
          className="text-xs font-semibold text-gray-200 hover:text-blue-400 hover:underline cursor-pointer whitespace-nowrap">
          {workflowName} {isPresetWorkflow && <span className="text-[9px] text-yellow-400 border border-yellow-700 rounded px-1 ml-1 align-middle">Template</span>}
          {!presetReadOnly && <span className="text-[10px] text-gray-500 ml-1">✎</span>}
        </span>
      )}
      {msg && <span className="text-[11px] text-green-400">{msg}</span>}
      <div className="flex-1" />
      <button
        onClick={() => { loadWorkflowList(); openTemplateDialog(); }}
        title="Start a new workflow from a template (processing chain) or a blank canvas"
        className="text-[11px] text-blue-400 hover:text-blue-300 hover:bg-blue-900/30 border border-blue-900/60 px-1.5 py-0.5 rounded">
        <IconifyIcon icon="ant-design:project-outlined" className="text-[12px] align-[-2px] mr-0.5" /> Template
      </button>
      {isAdmin && workflowId && (
        <button onClick={togglePreset} className="text-[11px] text-yellow-400 hover:text-yellow-300 hover:bg-yellow-900/30 px-1.5 py-0.5 rounded">
          {isPresetWorkflow ? 'Unset Template' : 'Make Template'}
        </button>
      )}
      <button onClick={saveSafe} disabled={isSaving} className="text-[11px] bg-blue-600 hover:bg-blue-500 text-white px-2 py-0.5 rounded disabled:opacity-50">{isSaving ? '...' : 'Save'}</button>
      {workflowId && !presetReadOnly && (
        <button onClick={() => setConfirmDialog({
          title: 'Delete this workflow?',
          message: 'This action cannot be undone. The workflow definition will be removed.',
          confirmLabel: 'Delete workflow',
          danger: true,
          onConfirm: async () => {
            await deleteWorkflow(workflowId);
            await loadWorkflowList();
            const { workflows: updated } = useWorkflowStore.getState();
            if (updated.length > 0) await loadWorkflow(updated[0].id);
            else useWorkflowStore.getState().newWorkflow();
            syncUrl();
          },
        })} className="text-[11px] text-red-400 hover:text-red-300 hover:bg-red-900/30 px-1.5 py-0.5 rounded">Del</button>
      )}
      {confirmDialog && (
        <div className="fixed inset-0 z-[100] flex items-center justify-center bg-black/60 p-4" role="presentation">
          <div className="w-full max-w-sm rounded-lg border border-gray-700 bg-gray-900 shadow-2xl" role="dialog" aria-modal="true" aria-labelledby="workflow-dialog-title">
            <div className="flex items-start gap-3 border-b border-gray-800 px-4 py-3">
              <span className={`mt-0.5 flex h-7 w-7 shrink-0 items-center justify-center rounded-full ${confirmDialog.danger ? 'bg-red-900/50 text-red-300' : 'bg-blue-900/50 text-blue-300'}`}>
                <span className="text-sm font-bold">{confirmDialog.danger ? '!' : '?'}</span>
              </span>
              <div className="min-w-0">
                <h2 id="workflow-dialog-title" className="text-sm font-semibold text-gray-100">{confirmDialog.title}</h2>
                <p className="mt-1 text-xs leading-5 text-gray-400">{confirmDialog.message}</p>
              </div>
            </div>
            <div className="flex justify-end gap-2 px-4 py-3">
              <button type="button" disabled={dialogBusy} onClick={() => setConfirmDialog(null)} className="rounded border border-gray-700 px-3 py-1.5 text-xs text-gray-300 hover:bg-gray-800 disabled:opacity-50">Cancel</button>
              {confirmDialog.secondaryLabel && (
                <button type="button" disabled={dialogBusy} onClick={confirmDialogSecondary}
                        className="rounded border border-gray-700 px-3 py-1.5 text-xs text-gray-300 hover:bg-gray-800 disabled:opacity-50">
                  {dialogBusy ? 'Working...' : confirmDialog.secondaryLabel}
                </button>
              )}
              <button type="button" disabled={dialogBusy} onClick={confirmDialogAction} className={`rounded px-3 py-1.5 text-xs text-white disabled:opacity-50 ${confirmDialog.danger ? 'bg-red-700 hover:bg-red-600' : 'bg-blue-600 hover:bg-blue-500'}`}>
                {dialogBusy ? 'Working...' : confirmDialog.confirmLabel}
              </button>
            </div>
          </div>
        </div>
      )}
      {templateDialog && (
        <div className="fixed inset-0 z-[100] flex items-center justify-center bg-black/60 p-4" role="presentation">
          <div className="w-full max-w-sm rounded-lg border border-gray-700 bg-gray-900 shadow-2xl" role="dialog" aria-modal="true" aria-labelledby="template-dialog-title">
            <div className="border-b border-gray-800 px-4 py-3">
              <h2 id="template-dialog-title" className="text-sm font-semibold text-gray-100">Start from Template</h2>
              <p className="mt-1 text-xs leading-5 text-gray-400">
                Templates contain the processing chain only — input device cards appear as your
                collectors report data. Applying a template replaces the current canvas.
              </p>
            </div>
            <div className="px-4 py-3 space-y-1.5">
              {workflows.filter((w) => w.is_preset).map((t) => (
                <button type="button" key={t.id} disabled={templateBusy} onClick={() => pickTemplate(t)}
                  className="w-full text-left rounded border border-gray-700 px-3 py-2 hover:border-blue-600 hover:bg-gray-800 disabled:opacity-50">
                  <div className="flex items-center gap-2">
                    <IconifyIcon icon="ant-design:database-outlined" className="text-[14px] text-gray-500 shrink-0" />
                    <div className="min-w-0">
                      <div className="text-xs text-gray-200 truncate">{t.name}</div>
                      <div className="text-[10px] text-gray-500 mt-0.5">Template · {t.node_count ?? 0} nodes · processing chain only</div>
                    </div>
                  </div>
                </button>
              ))}
              {!workflows.some((w) => w.is_preset) && (
                <div className="text-[11px] text-gray-600 px-1 pt-1">No templates available — an admin can mark workflows as templates.</div>
              )}
              <div className="text-[10px] text-gray-600 pt-1">
                Applying a template replaces the current canvas — your workflow name and ID stay the same.
              </div>
            </div>
            <div className="flex justify-end gap-2 px-4 py-3">
              <button type="button" disabled={templateBusy} onClick={() => setTemplateDialog(false)}
                className="rounded border border-gray-700 px-3 py-1.5 text-xs text-gray-300 hover:bg-gray-800 disabled:opacity-50">Cancel</button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
