/** Project APIs — input-source derivation for the Studio palette. */
import { req } from './workflows';
import type { InputSourcesResponse, WorkflowBindings } from '../types/workflow';

export function getProjectInputSources(projectId: string) {
  return req<InputSourcesResponse>(`/api/v1/projects/${projectId}/input-sources`);
}

/** 项目设备命名映射(工作流共享,本项目按节点覆盖 source_key)。 */
export function getProjectBindings(projectId: string) {
  return req<{ project_id: string; workflow_bindings: WorkflowBindings }>(
    `/api/v1/projects/${projectId}/bindings`,
  );
}

export function putProjectBinding(
  projectId: string,
  data: { workflow_id: string; node_id: string; source_key: string | null },
) {
  return req<{ project_id: string; workflow_bindings: WorkflowBindings }>(
    `/api/v1/projects/${projectId}/bindings`,
    { method: 'PUT', body: JSON.stringify(data) },
  );
}

/** 持久化"项目 × 工作流"的可用输入源(首次构建记录,后续新增并入)。 */
export function putProjectWorkflowInputs(
  projectId: string,
  data: {
    workflow_id: string;
    camera_names?: string[];
    sensors?: string[];
    device_inputs?: Record<string, unknown>;
  },
) {
  return req<{ project_id: string; workflow_id: string }>(
    `/api/v1/projects/${projectId}/workflow-inputs`,
    { method: 'PUT', body: JSON.stringify(data) },
  );
}
