import { create } from 'zustand';

/** 应用内全局 UI 状态:toast 消息 + 统一确认/输入弹窗。
 *  替代 window.alert / window.confirm / window.prompt —— 不依赖浏览器原生弹窗。 */

export interface ToastItem {
  id: number;
  msg: string;
  type: 'info' | 'error';
}

export interface DialogRequest {
  kind: 'confirm' | 'prompt';
  title: string;
  message: string;
  defaultValue?: string;
  confirmLabel?: string;
  danger?: boolean;
  resolve: (value: boolean | string | null) => void;
}

let toastId = 0;

interface UiState {
  toasts: ToastItem[];
  pushToast: (msg: string, type?: 'info' | 'error') => void;
  removeToast: (id: number) => void;
  dialog: DialogRequest | null;
  /** 应用内确认(替代 window.confirm)。 */
  confirm: (message: string, opts?: { title?: string; confirmLabel?: string; danger?: boolean }) => Promise<boolean>;
  /** 应用内输入(替代 window.prompt);取消返回 null。 */
  prompt: (message: string, opts?: { title?: string; defaultValue?: string; confirmLabel?: string }) => Promise<string | null>;
  resolveDialog: (value: boolean | string | null) => void;
}

export const useUiStore = create<UiState>((set, get) => ({
  toasts: [],
  pushToast: (msg, type = 'info') => {
    const id = ++toastId;
    set((s) => ({ toasts: [...s.toasts, { id, msg, type }] }));
    setTimeout(() => get().removeToast(id), 4000);
  },
  removeToast: (id) => set((s) => ({ toasts: s.toasts.filter((t) => t.id !== id) })),
  dialog: null,
  confirm: (message, opts) => new Promise<boolean>((resolve) => {
    set({
      dialog: {
        kind: 'confirm', title: opts?.title || 'Confirm', message,
        confirmLabel: opts?.confirmLabel, danger: opts?.danger,
        resolve: (v) => resolve(v as boolean),
      },
    });
  }),
  prompt: (message, opts) => new Promise<string | null>((resolve) => {
    set({
      dialog: {
        kind: 'prompt', title: opts?.title || 'Input', message,
        defaultValue: opts?.defaultValue, confirmLabel: opts?.confirmLabel,
        resolve: (v) => resolve(v as string | null),
      },
    });
  }),
  resolveDialog: (value) => {
    const d = get().dialog;
    if (!d) return;
    d.resolve(value);
    set({ dialog: null });
  },
}));
