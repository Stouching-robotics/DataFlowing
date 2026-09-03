import { useEffect } from 'react';
import { useWorkflowStore } from '../../store/workflowStore';
import type { WorkflowListItem } from '../../types/workflow';

interface Props { onClose: () => void }

export function WorkflowDrawer({ onClose }: Props) {
  const { workflows, listLoading, loadWorkflowList, loadWorkflow, loadInputsForWorkflow, newWorkflow } = useWorkflowStore();

  useEffect(() => { loadWorkflowList(); }, [loadWorkflowList]);

  const handleSelect = async (wf: WorkflowListItem) => {
    await loadWorkflow(wf.id);
    const projectFromUrl = new URLSearchParams(window.location.search).get('project');
    await loadInputsForWorkflow(projectFromUrl);
    onClose();
  };
  const handleNew = () => { newWorkflow(); onClose(); };

  return (
    <div className="fixed inset-0 z-50 flex">
      <div className="absolute inset-0 bg-black/60" onClick={onClose} />
      <div className="relative w-80 bg-gray-900 border-r border-gray-800 h-full flex flex-col shadow-2xl">
        <div className="flex items-center justify-between px-4 py-3 border-b border-gray-800">
          <h3 className="text-sm font-semibold text-gray-200">Workflows</h3>
          <button onClick={handleNew} className="text-xs bg-blue-600 hover:bg-blue-500 text-white px-2 py-1 rounded">+ New</button>
        </div>
        <div className="flex-1 overflow-y-auto">
          {listLoading ? (
            <div className="py-8 text-center text-sm text-gray-500">Loading...</div>
          ) : workflows.length === 0 ? (
            <div className="flex flex-col items-center py-12 text-gray-600 gap-2">
              <span className="text-3xl">📋</span>
              <span className="text-xs">No workflows yet</span>
            </div>
          ) : workflows.map((wf) => (
            <button key={wf.id} onClick={() => handleSelect(wf)}
              className="w-full text-left px-4 py-3 border-b border-gray-800/50 hover:bg-gray-800/50 transition-colors">
              <div className="flex items-center gap-2">
                <span className="text-sm text-gray-200 truncate">{wf.name}</span>
                <span className={`ml-auto w-1.5 h-1.5 rounded-full ${wf.status === 'active' ? 'bg-green-500' : wf.status === 'archived' ? 'bg-gray-600' : 'bg-yellow-500'}`} />
              </div>
              <div className="text-[11px] text-gray-600 mt-1">{wf.status} · {new Date(wf.updated_at).toLocaleDateString()}</div>
            </button>
          ))}
        </div>
      </div>
    </div>
  );
}
