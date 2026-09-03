import { useState, useCallback } from 'react';
import { useWorkflowStore } from '../../store/workflowStore';
import { useUiStore } from '../../store/uiStore';
import { runWorkflow } from '../../api/workflows';

interface Props { onOpenDrawer: () => void }

export function PipelineToolbar({ onOpenDrawer }: Props) {
  const { workflowName, workflowId, isDirty, isSaving, saveWorkflow, saveWorkflowAs, newWorkflow } = useWorkflowStore();
  const [showNameInput, setShowNameInput] = useState(false);
  const [nameDraft, setNameDraft] = useState(workflowName);
  const [feedback, setFeedback] = useState('');

  const flash = (msg: string) => { setFeedback(msg); setTimeout(() => setFeedback(''), 2000); };

  const handleSave = async () => {
    if (workflowId) { await saveWorkflow(); flash('✓ Saved'); }
    else setShowNameInput(true);
  };

  const handleSaveAs = () => { setShowNameInput(true); };

  const confirmSaveAs = async () => {
    const name = nameDraft.trim(); if (!name) return;
    await saveWorkflowAs(name); setShowNameInput(false); flash('✓ Saved');
  };

  const handleRun = async () => {
    if (!workflowId) { await saveWorkflow(); }
    const id = useWorkflowStore.getState().workflowId;
    if (!id) return;
    try { await runWorkflow(id); useUiStore.getState().pushToast('Workflow run queued (placeholder — Phase 3)'); }
    catch (e: any) { useUiStore.getState().pushToast(`Run failed: ${e.message}`, 'error'); }
  };

  const handleExport = () => {
    const s = useWorkflowStore.getState();
    const json = JSON.stringify({ name: s.workflowName, nodes: s.nodes, edges: s.edges }, null, 2);
    const blob = new Blob([json], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a'); a.href = url; a.download = `${s.workflowName.replace(/\s+/g,'_')}.json`; a.click();
    URL.revokeObjectURL(url);
  };

  const statusDot = isDirty ? 'bg-yellow-500' : feedback ? 'bg-green-500' : 'bg-gray-600';

  return (
    <div className="flex items-center gap-3 px-4 py-2 bg-gray-900 border-b border-gray-800 shrink-0">
      <button onClick={onOpenDrawer} className="text-gray-400 hover:text-white p-1.5 rounded hover:bg-gray-800" title="Workflow list">☰</button>
      <span className={`w-2 h-2 rounded-full shrink-0 ${statusDot}`} />
      {showNameInput ? (
        <>
          <input autoFocus value={nameDraft} onChange={(e) => setNameDraft(e.target.value)}
            onKeyDown={(e) => { if (e.key==='Enter') confirmSaveAs(); if (e.key==='Escape') setShowNameInput(false); }}
            onBlur={() => { if (!nameDraft.trim()) setShowNameInput(false); }}
            className="bg-gray-800 border border-gray-600 rounded px-2 py-1 text-sm text-white w-52 focus:outline-none focus:border-blue-500" />
          <button onClick={confirmSaveAs} className="text-xs bg-blue-600 hover:bg-blue-500 text-white px-2 py-1 rounded">Save</button>
        </>
      ) : (
        <span className="text-sm font-semibold text-gray-200">{workflowName}</span>
      )}
      {feedback && <span className="text-xs text-green-400">{feedback}</span>}
      <div className="flex-1" />
      <button onClick={newWorkflow} className="text-xs text-gray-400 hover:text-white px-2 py-1.5 rounded hover:bg-gray-800">New</button>
      <button onClick={handleSave} disabled={isSaving} className="text-xs bg-blue-600 hover:bg-blue-500 text-white px-3 py-1.5 rounded disabled:opacity-50">{isSaving ? '...' : 'Save'}</button>
      <button onClick={handleSaveAs} className="text-xs text-gray-400 hover:text-white px-2 py-1.5 rounded hover:bg-gray-800">Save As</button>
      <button onClick={handleExport} className="text-xs text-gray-400 hover:text-white px-2 py-1.5 rounded hover:bg-gray-800">Export</button>
      <button onClick={handleRun} className="text-xs bg-green-700 hover:bg-green-600 text-white px-3 py-1.5 rounded">▶ Run</button>
    </div>
  );
}
