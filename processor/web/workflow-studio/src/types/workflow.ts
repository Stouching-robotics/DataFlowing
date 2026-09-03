/** Node/edge type definitions for Workflow Studio. */
import type { Node, Edge } from '@xyflow/react';

export type NodeCategory = 'input' | 'process' | 'review' | 'export';

export interface NodeTypeDescriptor {
  type: string;
  slug?: string;
  version?: string;
  category: NodeCategory;
  label: string;
  icon: string;
  color: string;
  /** 卡片悬停说明:模块用途 + 可连接性(如"只可连视频输入")。 */
  description?: string;
  inputs: { key: string; label: string }[];
  outputs: { key: string; label: string }[];
  defaultConfig: Record<string, unknown>;
  configSchema?: ConfigField[];
  executionTarget?: string;
  capabilities?: string[];
}

export interface ConfigField {
  name: string;
  type: string;
  label: string;
  default?: unknown;
  options?: string[];
  min?: number;
  max?: number;
  step?: number;
}

export interface WorkflowNodeData {
  label: string;
  category: NodeCategory;
  icon: string;
  color: string;
  config: Record<string, unknown>;
  nodeType: string;
  inputs: { key: string; label: string }[];
  outputs: { key: string; label: string }[];
  configSchema?: ConfigField[];
  executionTarget?: string;
  [key: string]: unknown;
}

export type WorkflowNode = Node<WorkflowNodeData, 'workflowNode'>;
export type WorkflowEdge = Edge;

export interface WorkflowRecord {
  id: string;
  name: string;
  /** Project context used to resolve the workflow's physical input devices. */
  project_id?: string | null;
  project_name?: string | null;
  description: string | null;
  graph: { nodes: WorkflowNode[]; edges: WorkflowEdge[] };
  node_configs: Record<string, Record<string, unknown>>;
  status: 'draft' | 'active' | 'archived';
  is_preset?: boolean;
  created_at: string;
  updated_at: string;
}

export interface WorkflowListItem {
  id: string;
  name: string;
  status: string;
  is_preset?: boolean;
  node_count?: number;
  updated_at: string;
}

/** 工作流被哪些项目绑定(Studio 改全局时提示影响范围)。 */
export interface WorkflowUsage {
  workflow_id: string;
  project_count: number;
  projects: { id: string; name: string }[];
}

/** 项目级设备命名映射:工作流共享,项目按节点覆盖 source_key。 */
export type NodeBinding = { source_key?: string; source_keys?: string };
export type WorkflowBindings = Record<string, Record<string, NodeBinding>>;

export interface WorkflowListResponse {
  workflows: WorkflowListItem[];
  total: number;
  limit: number;
  offset: number;
}

export interface InputSourceModuleInfo {
  type: string;
  label: string;
  matched: boolean;
  matched_keys: string[];
}

export interface DeviceInputSource {
  id: string;
  name: string;
  /** Standardized UI name; name remains the raw compatibility identifier. */
  display_name?: string;
  kind?: string;
  device_type?: 'rgbd_camera' | 'stereo_rgbd_camera' | 'mono_rgb' | 'stereo_rgb' | 'glove_sensor' | string;
  lens?: string;
  label?: string;
  input_type: string;
  source_key?: string;
  source_keys: string[];
  /** Depth-only streams belonging to this physical RGB-D camera. */
  depth_keys?: string[];
  slots?: string[];
}

export interface InputSourcesResponse {
  project_id?: string;
  project_name?: string;
  has_episodes: boolean;
  has_online_devices?: boolean;
  camera_names?: string[];
  /** Map actual video/source slot to physical device name from meta/info.json. */
  device_names?: Record<string, string>;
  device_display_names?: Record<string, string>;
  devices?: Array<{ key?: string; kind?: string; name?: string; display_name?: string; slots?: string[] }>;
  device_sources?: DeviceInputSource[];
  sensors?: string[];
  device_inputs?: { cameras: string[]; sensors: string[] };
  modules: InputSourceModuleInfo[];
}

export interface WorkflowRunRecord {
  id: string;
  workflow_id: string;
  episode_id?: string | null;
  status: string;
  started_at?: string | null;
  finished_at?: string | null;
  node_states: Record<string, { type?: string; status?: string; progress?: number; error?: string }>;
  error_log?: string | null;
  worker_id?: string | null;
  progress: number;
  attempt: number;
  outputs: Record<string, unknown>;
  created_at: string;
}
