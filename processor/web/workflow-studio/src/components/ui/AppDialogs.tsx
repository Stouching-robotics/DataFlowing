import { useEffect, useState } from 'react';
import { useUiStore } from '../../store/uiStore';

/** 全局 toast 消息容器(右上角,自动消失)。 */
export function ToastContainer() {
  const toasts = useUiStore((s) => s.toasts);
  const removeToast = useUiStore((s) => s.removeToast);
  if (toasts.length === 0) return null;
  return (
    <div className="fixed top-3 right-3 z-[300] flex flex-col gap-2 max-w-xs pointer-events-none">
      {toasts.map((t) => (
        <div key={t.id}
          className={`pointer-events-auto rounded border px-3 py-2 text-xs shadow-lg ${t.type === 'error' ? 'border-red-800 bg-red-950/95 text-red-200' : 'border-gray-700 bg-gray-900/95 text-gray-200'}`}>
          <div className="flex items-start gap-2">
            <span className="flex-1 leading-snug break-words">{t.msg}</span>
            <button type="button" onClick={() => removeToast(t.id)} className="text-gray-500 hover:text-gray-300 shrink-0">×</button>
          </div>
        </div>
      ))}
    </div>
  );
}

/** 全局确认/输入弹窗(替代 window.confirm / window.prompt)。 */
export function GlobalDialog() {
  const dialog = useUiStore((s) => s.dialog);
  const resolveDialog = useUiStore((s) => s.resolveDialog);
  const [value, setValue] = useState('');
  useEffect(() => {
    if (dialog) setValue(dialog.defaultValue || '');
  }, [dialog]);
  if (!dialog) return null;
  const isPrompt = dialog.kind === 'prompt';
  return (
    <div className="fixed inset-0 z-[200] flex items-center justify-center bg-black/60 p-4" role="presentation">
      <div className="w-full max-w-sm rounded-lg border border-gray-700 bg-gray-900 shadow-2xl" role="dialog" aria-modal="true">
        <div className="flex items-start gap-3 border-b border-gray-800 px-4 py-3">
          <span className={`mt-0.5 flex h-7 w-7 shrink-0 items-center justify-center rounded-full ${dialog.danger ? 'bg-red-900/50 text-red-300' : 'bg-blue-900/50 text-blue-300'}`}>
            <span className="text-sm font-bold">{dialog.danger ? '!' : '?'}</span>
          </span>
          <div className="min-w-0 flex-1">
            <h2 className="text-sm font-semibold text-gray-100">{dialog.title}</h2>
            <p className="mt-1 text-xs leading-5 text-gray-400">{dialog.message}</p>
            {isPrompt && (
              <input autoFocus value={value} onChange={(e) => setValue(e.target.value)}
                onKeyDown={(e) => { if (e.key === 'Enter') resolveDialog(value.trim()); if (e.key === 'Escape') resolveDialog(null); }}
                className="mt-2 w-full bg-gray-800 border border-gray-600 rounded px-2 py-1.5 text-xs text-white focus:outline-none focus:border-blue-500" />
            )}
          </div>
        </div>
        <div className="flex justify-end gap-2 px-4 py-3">
          <button type="button" onClick={() => resolveDialog(null)}
            className="rounded border border-gray-700 px-3 py-1.5 text-xs text-gray-300 hover:bg-gray-800">Cancel</button>
          <button type="button" onClick={() => resolveDialog(isPrompt ? value.trim() : true)}
            className={`rounded px-3 py-1.5 text-xs text-white ${dialog.danger ? 'bg-red-700 hover:bg-red-600' : 'bg-blue-600 hover:bg-blue-500'}`}>
            {dialog.confirmLabel || (isPrompt ? 'OK' : 'Confirm')}
          </button>
        </div>
      </div>
    </div>
  );
}
