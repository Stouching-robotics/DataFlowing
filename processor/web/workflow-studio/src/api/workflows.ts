import type { WorkflowRecord, WorkflowListResponse, WorkflowRunRecord, WorkflowUsage } from '../types/workflow';

const BASE = '/api/v1/workflows';

export async function req<T>(url: string, opts?: RequestInit): Promise<T> {
  const res = await fetch(url, {
    credentials: 'include',
    headers: { 'Content-Type': 'application/json', ...opts?.headers },
    ...opts,
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail || `HTTP ${res.status}`);
  }
  if (res.status === 204) return undefined as T;
  return res.json();
}

export function listWorkflows(limit = 50, offset = 0, status?: string) {
  const p = new URLSearchParams({ limit: String(limit), offset: String(offset) });
  if (status) p.set('status', status);
  return req<WorkflowListResponse>(`${BASE}?${p}`);
}

export function getWorkflow(id: string) {
  return req<WorkflowRecord>(`${BASE}/${id}`);
}

export function createWorkflow(data: {
  name: string;
  project_id?: string | null;
  description?: string;
  graph: { nodes: unknown[]; edges: unknown[] };
  node_configs?: Record<string, unknown>;
  status?: string;
}) {
  return req<WorkflowRecord>(BASE, { method: 'POST', body: JSON.stringify(data) });
}

export function updateWorkflow(
  id: string,
  data: Partial<{ name: string; project_id: string | null; description: string; graph: unknown; node_configs: unknown; status: string; is_preset: boolean }>,
) {
  return req<WorkflowRecord>(`${BASE}/${id}`, { method: 'PUT', body: JSON.stringify(data) });
}

export function deleteWorkflow(id: string) {
  return req<void>(`${BASE}/${id}`, { method: 'DELETE' });
}

export function runWorkflow(id: string, episodeId?: string) {
  const query = episodeId ? `?episode_id=${encodeURIComponent(episodeId)}` : '';
  return req<WorkflowRunRecord>(`${BASE}/${id}/run${query}`, { method: 'POST' });
}

export function listModules() {
  return req<Array<Record<string, unknown>>>(`${BASE}/modules`);
}

export function listRuns(id: string, limit = 10) {
  return req<WorkflowRunRecord[]>(`${BASE}/${id}/runs?limit=${limit}`);
}

export function getWorkflowUsage(id: string) {
  return req<WorkflowUsage>(`${BASE}/${id}/usage`);
}

export function getRun(id: string) {
  return req<WorkflowRunRecord>(`/api/v1/workflows/runs/${id}`);
}

export function retryRun(id: string) {
  return req<WorkflowRunRecord>(`/api/v1/workflows/runs/${id}/retry`, { method: 'POST' });
}
