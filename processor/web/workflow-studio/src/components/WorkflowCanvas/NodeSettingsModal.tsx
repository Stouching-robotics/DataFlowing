import { useState } from 'react';
import { useWorkflowStore } from '../../store/workflowStore';
import type { WorkflowNodeData } from '../../types/workflow';

interface Props {
  nodeId: string;
  onClose: () => void;
}

const API_MODELS: Record<string, string[]> = {
  kimi: ['kimi-k3'],
  qwen: [
    'qwen3.8-max', 'qwen3.7-plus', 'qwen3.7-flash',
    'qwen3-vl-plus', 'qwen3-vl-32b-thinking',
    'qwen3.5-flash', 'qwen3.5-omni-flash',
    'kimi-k2.6',
  ],
  // Qwen3-VL 开源 30B-A3B 系列(SiliconFlow/自部署 vLLM;
  // DashScope 端点请用 qwen3-vl-* 官方模型名)
  siliconflow: [
    'Qwen/Qwen3-VL-30B-A3B-Instruct',
    'Qwen/Qwen3-VL-30B-A3B-Thinking',
    // Qwen3-VL-235B 已从硅基流动下线(2026-04),由 Qwen3.5 系列替代
    'Qwen/Qwen3.5-397B-A17B',
    'Qwen/Qwen3.5-122B-A10B',
    'Qwen/Qwen3.6-35B-A3B',
    'Qwen/Qwen3.6-27B',
    'Qwen/Qwen3-Omni-30B-A3B-Instruct',
    'Qwen/Qwen3-Omni-30B-A3B-Thinking',
    'zai-org/GLM-4.5V',
    'Pro/moonshotai/Kimi-K2.6',
    'MiniMaxAI/MiniMax-M2.5',
  ],
};

const API_DEFAULT_BASE_URLS: Record<string, string> = {
  kimi: 'https://api.moonshot.ai/v1',
  qwen: 'https://dashscope.aliyuncs.com/compatible-mode/v1',
  siliconflow: 'https://api.siliconflow.cn/v1',
};

interface ApiEntry {
  id: number;
  vendor: string;
  model: string;
  key: string;
  baseUrl: string;
}

let _entrySeq = 0;
const nextEntryId = () => ++_entrySeq;

function entriesFromConfig(config: Record<string, unknown> | undefined): ApiEntry[] {
  const cfg = config || {};
  const providers = cfg.api_providers;
  if (Array.isArray(providers) && providers.length) {
    return providers.map((p) => {
      const item = (p || {}) as Record<string, unknown>;
      return {
        id: nextEntryId(),
        vendor: String(item.vendor || 'kimi'),
        model: String(item.model || ''),
        key: String(item.key || ''),
        baseUrl: String(item.base_url || ''),
      };
    });
  }
  // 兼容旧配置:单套 api_vendor/api_model/api_key/api_base_url 折叠成第一条
  return [{
    id: nextEntryId(),
    vendor: String(cfg.api_vendor || 'kimi'),
    model: String(cfg.api_model || ''),
    key: String(cfg.api_key || ''),
    baseUrl: String(cfg.api_base_url || ''),
  }];
}

/**
 * AI Annotation 卡片 ⚙ 设置弹窗:配置输出语言并保存多套 API 供应商配置(每套 =
 * 厂商/模型/Key/base URL)。顶部切换器选择当前编辑的 API,字段区与
 * 原单套表单完全一致。运行时(标注页点 AI 标注)从列表里选一套执行,
 * 单套直接执行 —— 不并发、不合并。Save → 写入节点 config:
 * `api_providers` 数组 + 首套同步写回旧字段(兼容旧运行链路)。
 */
export function NodeSettingsModal({ nodeId, onClose }: Props) {
  const node = useWorkflowStore((s) => s.nodes.find((n) => n.id === nodeId));
  const data = node?.data as unknown as WorkflowNodeData | undefined;
  const schema = data?.configSchema || [];
  const vendorField = schema.find((f) => f.name === 'api_vendor');
  const modelField = schema.find((f) => f.name === 'api_model');
  const baseUrlField = schema.find((f) => f.name === 'api_base_url');
  const languageField = schema.find((f) => f.name === 'prompt_language');

  const [entries, setEntries] = useState<ApiEntry[]>(() =>
    entriesFromConfig(data?.config as Record<string, unknown> | undefined));
  const [activeId, setActiveId] = useState<number>(() => entries[0]?.id ?? 0);
  const [testing, setTesting] = useState(false);
  const [testResult, setTestResult] = useState<{ ok: boolean; message: string } | null>(null);
  const [language, setLanguage] = useState(() => {
    const value = String(data?.config?.prompt_language || languageField?.default || 'zh').toLowerCase();
    return value === 'en' ? 'en' : 'zh';
  });

  const active = entries.find((e) => e.id === activeId) || entries[0];
  const vendor = active?.vendor ?? 'kimi';
  const model = active?.model ?? '';
  const key = active?.key ?? '';
  const baseUrl = active?.baseUrl ?? '';

  const vendorOptions = Array.from(new Set([
    ...((vendorField?.options as string[]) || []),
    'kimi', 'qwen', 'siliconflow',
  ]));

  const updateActive = (patch: Partial<ApiEntry>) => {
    setEntries((prev) => prev.map((e) => (e.id === activeId ? { ...e, ...patch } : e)));
    setTestResult(null);
  };

  const handleVendorChange = (nextVendor: string) => {
    const nextModels = API_MODELS[nextVendor] || [];
    const knownDefaults = new Set(Object.values(API_DEFAULT_BASE_URLS));
    const patch: Partial<ApiEntry> = { vendor: nextVendor };
    if (nextModels.length && !nextModels.includes(model)) patch.model = nextModels[0];
    // 切换厂商时,地址只要是"任一厂商的默认地址"(或为空)就替换成新厂商
    // 默认 —— 防止带着上一家的默认地址显示成错地址;用户自定义地址保留。
    if (!baseUrl || knownDefaults.has(baseUrl)) {
      patch.baseUrl = API_DEFAULT_BASE_URLS[nextVendor] || '';
    }
    updateActive(patch);
  };

  const addEntry = () => {
    const fresh: ApiEntry = {
      id: nextEntryId(),
      vendor: 'kimi',
      model: API_MODELS.kimi[0] || '',
      key: '',
      baseUrl: '',
    };
    setEntries((prev) => [...prev, fresh]);
    setActiveId(fresh.id);
    setTestResult(null);
  };

  const removeActive = () => {
    if (entries.length <= 1) return;
    const rest = entries.filter((e) => e.id !== activeId);
    setEntries(rest);
    setActiveId(rest[0].id);
    setTestResult(null);
  };

  const clearTestResult = () => setTestResult(null);

  const handleTestConnection = async () => {
    if (testing) return;
    if (!model.trim()) {
      setTestResult({ ok: false, message: 'Model required' });
      return;
    }
    if (!key.trim()) {
      setTestResult({ ok: false, message: 'API key required' });
      return;
    }
    setTesting(true);
    setTestResult(null);
    try {
      const response = await fetch('/api/v1/ai-annotation/test-connection', {
        method: 'POST',
        credentials: 'include',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          api_vendor: vendor.trim() || 'kimi',
          api_model: model.trim(),
          api_key: key.trim(),
          api_base_url: baseUrl.trim(),
        }),
      });
      const payload = await response.json().catch(() => ({}));
      setTestResult({
        ok: response.ok && payload.ok === true,
        message: String(payload.message || (response.ok ? 'Test failed' : `Request failed (${response.status})`)),
      });
    } catch (error) {
      setTestResult({
        ok: false,
        message: `Request failed: ${error instanceof Error ? error.message : 'network error'}`,
      });
    } finally {
      setTesting(false);
    }
  };

  const handleSave = () => {
    const providers = entries.map((e) => ({
      vendor: e.vendor.trim() || 'kimi',
      model: e.model.trim(),
      key: e.key.trim(),
      base_url: e.baseUrl.trim(),
    }));
    const first = providers[0] || { vendor: 'kimi', model: '', key: '', base_url: '' };
    useWorkflowStore.setState((state) => ({
      nodes: state.nodes.map((n) =>
        n.id === nodeId ? {
          ...n,
          data: {
            ...n.data,
            config: {
              ...(n.data.config || {}),
              api_providers: providers,
              // 首套同步写回旧字段:旧运行链路(未升级的页面/后端)仍可用
              api_vendor: first.vendor,
              api_model: first.model,
              api_key: first.key,
              api_base_url: first.base_url,
              prompt_language: language === 'en' ? 'en' : 'zh',
            },
          },
        } : n,
      ) as any,
      isDirty: true,
    }));
    onClose();
  };

  const modelOptions = API_MODELS[vendor] || (modelField?.options as string[]) || [];
  // Keep a previously saved/custom model visible when opening the modal.
  // Otherwise a controlled <select> can show a blank value and silently
  // replace an older provider-specific model on the next vendor switch.
  const visibleModelOptions = model && !modelOptions.includes(model)
    ? [model, ...modelOptions]
    : modelOptions;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center">
      <div className="absolute inset-0 bg-black/60" onClick={onClose} />
      <div className="relative w-96 bg-gray-900 border border-gray-700 rounded-lg shadow-2xl">
        <div className="flex items-center justify-between px-4 py-3 border-b border-gray-800">
          <h3 className="text-sm font-semibold text-gray-200">AI Annotation · Settings</h3>
          <button onClick={onClose} aria-label="Close"
            className="text-gray-500 hover:text-gray-300 text-base leading-none">✕</button>
        </div>
        <div className="p-4 space-y-3">
          <label className="block">
            <span className="text-xs text-gray-400">Label language</span>
            <select
              aria-label="Label language"
              value={language}
              onChange={(e) => setLanguage(e.target.value === 'en' ? 'en' : 'zh')}
              className="mt-1 w-full bg-gray-800 border border-gray-700 rounded px-2 py-1.5 text-sm text-gray-200">
              <option value="zh">中文</option>
              <option value="en">English</option>
            </select>
            <span className="mt-1 block text-[10px] text-gray-500">
              Applies to the next AI annotation run.
            </span>
          </label>
          {String(data?.config?.vlm_provider || 'local') === 'local' && (
            <div className="rounded border border-gray-800 bg-gray-950/50 px-2.5 py-2 text-[10px] text-gray-500">
              Local VLM selected. Language setting is shared with API mode.
            </div>
          )}
          {/* 多 API 切换器:选择当前编辑的 API,新增/删除整套配置 */}
          <div className="flex items-center gap-1.5">
            <select value={activeId} onChange={(e) => {
              setActiveId(Number(e.target.value));
              clearTestResult();
            }}
              className="flex-1 min-w-0 bg-gray-800 border border-gray-700 rounded px-2 py-1.5 text-sm text-gray-200">
              {entries.map((e, idx) => (
                <option key={e.id} value={e.id}>
                  {`#${idx + 1} · ${e.vendor}${e.model ? ` · ${e.model}` : ''}`}
                </option>
              ))}
            </select>
            <button onClick={addEntry} title="Add API"
              className="shrink-0 text-xs px-2 py-1.5 rounded border border-dashed border-gray-600 text-gray-400 hover:text-gray-200 hover:border-gray-400">+</button>
            <button onClick={removeActive} disabled={entries.length <= 1} title="Remove this API"
              className="shrink-0 text-xs px-2 py-1.5 rounded border border-gray-700 text-gray-500 hover:text-red-400 hover:border-red-800 disabled:opacity-30">✕</button>
          </div>
          <label className="block">
            <span className="text-xs text-gray-400">API vendor</span>
            <select value={vendor} onChange={(e) => handleVendorChange(e.target.value)}
              className="mt-1 w-full bg-gray-800 border border-gray-700 rounded px-2 py-1.5 text-sm text-gray-200">
              {vendorOptions.map((o) => (
                <option key={o} value={o}>{o}</option>
              ))}
            </select>
          </label>
          <label className="block">
            <span className="text-xs text-gray-400">API model</span>
            <select value={model} onChange={(e) => { updateActive({ model: e.target.value }); }}
              className="mt-1 w-full bg-gray-800 border border-gray-700 rounded px-2 py-1.5 text-sm text-gray-200">
              {visibleModelOptions.length ? visibleModelOptions.map((option) => (
                <option key={option} value={option}>{option}</option>
              )) : <option value={model}>{model}</option>}
            </select>
          </label>
          <label className="block">
            <span className="text-xs text-gray-400">API key (stored with this workflow)</span>
            <input type="password" value={key} onChange={(e) => updateActive({ key: e.target.value })}
              autoComplete="new-password"
              placeholder="sk-..."
              className="mt-1 w-full bg-gray-800 border border-gray-700 rounded px-2 py-1.5 text-sm text-gray-200" />
          </label>
          <label className="block">
            <span className="text-xs text-gray-400">API base URL (blank = official default)</span>
            <input type="text" value={baseUrl} onChange={(e) => updateActive({ baseUrl: e.target.value })}
              placeholder={String(baseUrlField?.default || API_DEFAULT_BASE_URLS[vendor] || '')}
              className="mt-1 w-full bg-gray-800 border border-gray-700 rounded px-2 py-1.5 text-sm text-gray-200" />
          </label>
          {testResult && (
            <div className={`rounded px-2.5 py-2 text-xs ${testResult.ok
              ? 'border border-green-800 bg-green-950/40 text-green-300'
              : 'border border-red-800 bg-red-950/40 text-red-300'}`}>
              <span className="mr-1">{testResult.ok ? '✓' : '✕'}</span>
              {testResult.message}
            </div>
          )}
        </div>
        <div className="flex justify-end gap-2 px-4 py-3 border-t border-gray-800">
          <button onClick={onClose}
            className="text-xs px-3 py-1.5 rounded bg-gray-800 hover:bg-gray-700 text-gray-300">Cancel</button>
          <button onClick={handleTestConnection} disabled={testing}
            className="text-xs px-3 py-1.5 rounded border border-cyan-700 bg-cyan-950/40 hover:bg-cyan-900/60 text-cyan-300 disabled:opacity-50">
            {testing ? 'Testing…' : 'Test Connection'}
          </button>
          <button onClick={handleSave}
            className="text-xs px-3 py-1.5 rounded bg-blue-600 hover:bg-blue-500 text-white">Save</button>
        </div>
      </div>
    </div>
  );
}
