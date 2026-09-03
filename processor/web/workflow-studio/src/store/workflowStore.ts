
import { create } from 'zustand';
import { addEdge, applyNodeChanges, applyEdgeChanges, type Node, type Edge, type OnNodesChange, type OnEdgesChange, type OnConnect } from '@xyflow/react';
import type { ConfigField, WorkflowListItem, WorkflowNodeData, InputSourcesResponse, DeviceInputSource, NodeBinding, WorkflowBindings } from '../types/workflow';
import * as api from '../api/workflows';
import { getProjectInputSources, getProjectBindings, putProjectBinding } from '../api/projects';
import { getDeviceInputSources } from '../api/devices';
import { canonicalNodeType, getNodeType } from '../components/WorkflowCanvas/nodes/registry';
import { useUiStore } from './uiStore';

interface HistoryEntry {
  nodes: Node<WorkflowNodeData>[];
  edges: Edge[];
}

const MAX_HISTORY = 30;

interface WorkflowState {
  nodes: Node<WorkflowNodeData>[];
  edges: Edge[];
  onNodesChange: OnNodesChange;
  onEdgesChange: OnEdgesChange;
  onConnect: OnConnect;
  setNodes: (n: Node<WorkflowNodeData>[]) => void;
  setEdges: (e: Edge[]) => void;

  workflowId: string | null;
  /** Project owning this workflow; drives the input-device selector. */
  workflowProjectId: string | null;
  workflowName: string;
  workflowStatus: string;
  isDirty: boolean;
  isSaving: boolean;

  // 权限 + 预设:userRole null 按非 admin 处理(安全默认)
  userRole: string | null;
  isPresetWorkflow: boolean;
  setUserRole: (role: string) => void;
  // 参考项目(URL ?project=)的输入源,驱动输入卡片
  referenceProjectId: string | null;
  referenceInputs: InputSourcesResponse | null;
  loadReferenceInputs: (projectId: string) => Promise<void>;
  /** Load the device palette for the active workflow/project context. */
  loadInputsForWorkflow: (projectId?: string | null) => Promise<void>;
  // 项目级设备命名映射(workflow_id → node_id → 本项目命名)
  projectBindings: WorkflowBindings | null;
  /** 当前工作流的节点绑定(按节点索引,给画布渲染用)。 */
  currentWorkflowBindings: Record<string, NodeBinding>;
  setProjectBinding: (nodeId: string, sourceKey: string | null) => Promise<void>;
  // 全局在线设备输入源(无 ?project= 时):采集端上报能力驱动输入卡片
  globalInputs: InputSourcesResponse | null;
  /** Current project/global input-source request status. */
  inputsLoading: boolean;
  inputsError: string | null;
  loadGlobalInputs: () => Promise<void>;

  workflows: WorkflowListItem[];
  listLoading: boolean;

  // Undo history (session-only, max 30)
  history: HistoryEntry[];
  historyIndex: number;
  pushHistory: () => void;
  undo: () => void;
  redo: () => void;

  loadWorkflow: (id: string) => Promise<void>;
  saveWorkflow: () => Promise<void>;
  /** 权限感知保存:预设且非 admin → 'needs-save-as';否则保存,失败 rethrow。 */
  saveWorkflowSafe: () => Promise<'saved' | 'needs-save-as'>;
  saveWorkflowAs: (name: string) => Promise<void>;
  newWorkflow: () => void;
  /** 应用模板到当前工作流:剔除输入设备节点(模板样式 = 只含后处理链),
   *  设备节点由用户按实际采集端手动添加;保留当前工作流的 ID/名称。 */
  newWorkflowFromTemplate: (template: WorkflowListItem) => Promise<void>;
  loadWorkflowList: () => Promise<void>;
}

let idCounter = 1;

// A user can switch workflows before a previous graph/input request finishes.
// Only the newest request may update the canvas context.
let workflowLoadToken = 0;
let inputsLoadToken = 0;

const DRAFT_KEY = 'workflow_draft';
const WORKFLOW_LIST_CACHE_KEY = 'egodata.workflow-list.v1';
const WORKFLOW_LIST_CACHE_TTL = 60 * 1000;

function readWorkflowListCache(): WorkflowListItem[] | null {
  try {
    const cached = JSON.parse(sessionStorage.getItem(WORKFLOW_LIST_CACHE_KEY) || 'null');
    if (!cached || !Array.isArray(cached.workflows)
        || Date.now() - Number(cached.savedAt || 0) > WORKFLOW_LIST_CACHE_TTL) return null;
    return cached.workflows as WorkflowListItem[];
  } catch { return null; }
}

function saveWorkflowListCache(workflows: WorkflowListItem[]) {
  try {
    sessionStorage.setItem(WORKFLOW_LIST_CACHE_KEY, JSON.stringify({
      savedAt: Date.now(), workflows,
    }));
  } catch { /* storage quota/private mode — network refresh still works */ }
}

const DEVICE_CATEGORY_LABELS: Record<string, string> = {
  rgbd_camera: 'RGB-D Camera',
  stereo_rgbd_camera: 'Stereo RGB-D Camera',
  mono_rgb: 'RGB Camera',
  stereo_rgb: 'Stereo RGB Camera',
  glove_sensor: 'Glove Sensor',
};

const CAMERA_INPUT_TYPES = new Set([
  'mono_camera', 'rgbd_camera', 'rgb_camera', 'fisheye_camera',
  'stereo_camera', 'stereo_rgbd_camera',
]);

function cameraSourceKeys(node: Node<WorkflowNodeData>): string[] {
  const config = node.data?.config || {};
  return [config.source_key, config.source_keys, config.position]
    .flatMap((value) => String(value || '').split(','))
    .map((value) => value.trim().toLowerCase())
    .filter(Boolean);
}

function sourceMatchesNode(node: Node<WorkflowNodeData>, source: DeviceInputSource): boolean {
  const wanted = cameraSourceKeys(node);
  if (!wanted.length) return false;
  const available = [source.name, source.display_name, source.source_key,
    ...(source.source_keys || []), ...(source.depth_keys || [])]
    .map((value) => String(value || '').trim().toLowerCase())
    .filter(Boolean);
  return wanted.some((key) => available.includes(key));
}

/** Add display-only physical-device metadata to old input nodes.
 * This is deliberately not marked dirty and never changes config/edges: the
 * worker continues to use the historical source keys exactly as before. */
function enrichInputNodeDeviceMetadata(
  nodes: Node<WorkflowNodeData>[],
  inputs: InputSourcesResponse,
): Node<WorkflowNodeData>[] {
  const sources = inputs.device_sources || [];
  if (!sources.length) return nodes;
  return nodes.map((node) => {
    if (!CAMERA_INPUT_TYPES.has(node.data?.nodeType)) return node;
    const data = node.data;
    // Metadata written when a real source is dragged is authoritative. This
    // keeps later edits of source_key independent from the card category.
    if (data.device_type && DEVICE_CATEGORY_LABELS[String(data.device_type)]) return node;
    const matched = sources.find((source) => sourceMatchesNode(node, source));
    const deviceType = matched?.device_type
      || (data.nodeType === 'stereo_rgbd_camera' ? 'stereo_rgbd_camera'
        : data.nodeType === 'stereo_camera' ? 'stereo_rgb'
        : data.nodeType === 'rgbd_camera' ? 'rgbd_camera' : 'mono_rgb');
    const displayName = matched?.display_name
      || DEVICE_CATEGORY_LABELS[String(deviceType)]
      || data.device_display_name;
    if (!displayName && !deviceType) return node;
    return {
      ...node,
      data: { ...data, device_type: deviceType, device_display_name: displayName },
    } as Node<WorkflowNodeData>;
  });
}

function saveDraft(state: Partial<WorkflowState>) {
  try {
    const draft = {
      nodes: state.nodes,
      edges: state.edges,
      workflowId: state.workflowId,
      workflowProjectId: state.workflowProjectId,
      workflowName: state.workflowName,
      savedAt: Date.now(),
    };
    localStorage.setItem(DRAFT_KEY, JSON.stringify(draft));
  } catch { /* quota exceeded — ignore */ }
}

function loadDraft(): {
  nodes: Node<WorkflowNodeData>[];
  edges: Edge[];
  workflowId: string | null;
  workflowProjectId: string | null;
  workflowName: string;
} | null {
  try {
    const raw = localStorage.getItem(DRAFT_KEY);
    if (!raw) return null;
    const d = JSON.parse(raw);
    if (!d.nodes || !Array.isArray(d.nodes)) return null;
    return d;
  } catch { return null; }
}

export function clearDraft() {
  try { localStorage.removeItem(DRAFT_KEY); } catch {}
}

function _clone<T>(obj: T): T {
  return JSON.parse(JSON.stringify(obj));
}

/** 用注册表描述符补齐旧快照缺失的 schema 字段:模块升级新增参数后,
 *  已保存工作流的旧 configSchema 快照也能在设置弹窗里看到新字段。 */
function _mergeSchema(
  snapshot?: ConfigField[],
  fresh?: ConfigField[],
): ConfigField[] {
  if (!snapshot?.length) return fresh || [];
  if (!fresh?.length) return snapshot;
  const byName = new Map(snapshot.map((f) => [f.name, f]));
  const missing = fresh.filter((f) => !byName.has(f.name));
  return missing.length ? [...snapshot, ...missing] : snapshot;
}

/** Normalize old persisted graphs before they enter the editor. */
function _migrateWorkflowGraph(graph: { nodes?: Node<WorkflowNodeData>[]; edges?: Edge[] }) {
  const sourceTypes = new Map<string, string>();
  const nodes = (graph.nodes || []).map((node) => {
    const raw = node.data?.nodeType || '';
    const nodeType = canonicalNodeType(raw);
    sourceTypes.set(node.id, nodeType);
    const data = { ...node.data, nodeType };
    if (nodeType === 'rgbd_camera') {
      data.outputs = [
        { key: 'video', label: 'RGB Video' },
        { key: 'depth', label: 'Depth' },
      ];
    } else if (nodeType === 'stereo_rgbd_camera') {
      data.outputs = [
        { key: 'video_left', label: 'Left RGB Video' },
        { key: 'video_right', label: 'Right RGB Video' },
        { key: 'depth', label: 'Depth' },
      ];
    } else if (nodeType === 'rgb_to_2d_bare_hand'
        || nodeType === 'rgb_to_2d_black_glove') {
      data.inputs = [{ key: 'video', label: 'RGB Video' }];
      data.outputs = [{ key: 'hand_keypoints', label: 'Hand 2D' }];
    } else if (nodeType === 'rgbd_to_3d_bare_hand'
        || nodeType === 'rgbd_to_3d_black_glove') {
      data.inputs = [
        { key: 'video', label: 'RGB Video' },
        { key: 'depth', label: 'Depth' },
      ];
      data.outputs = [{ key: 'hand_3d', label: 'Hand 3D' }];
    }
    return {
      ...node,
      data,
    } as Node<WorkflowNodeData>;
  });
  const edges = (graph.edges || []).map((edge) => {
    const sourceType = sourceTypes.get(edge.source) || '';
    const sourceHandle = String(edge.sourceHandle || '');
    let nextHandle = sourceHandle;
    const resultOutput = sourceType === 'annotation' || sourceType === 'ai_annotation'
      ? 'annotation'
      : sourceType === 'human_review' || sourceType === 'ai_quality_review'
        ? 'reviewed'
        : '';
    if (resultOutput && sourceHandle === 'result') {
      nextHandle = resultOutput;
    } else if (['rgb_to_2d_bare_hand', 'rgb_to_2d_black_glove'].includes(sourceType)
        && sourceHandle.startsWith('hand_3d')) {
      nextHandle = sourceHandle.replace(/^hand_3d/, 'hand_keypoints');
    } else if (['rgbd_to_3d_bare_hand', 'rgbd_to_3d_black_glove'].includes(sourceType)
        && sourceHandle.startsWith('hand_keypoints')) {
      nextHandle = sourceHandle.replace(/^hand_keypoints/, 'hand_3d');
    }
    if (nextHandle === sourceHandle) return edge;
    return {
      ...edge,
      sourceHandle: nextHandle,
      id: edge.id
        ? String(edge.id).replace(sourceHandle, nextHandle)
        : edge.id,
    };
  });

  // Existing local drafts may be loaded without a round trip through the API.
  // Keep their RGB-D graph equivalent to the server migration: an RGB-D input
  // card feeds both typed inputs when the old graph already had the video edge.
  const edgeKeys = new Set(edges.map((edge) =>
    `${edge.source}|${edge.target}|${edge.targetHandle || ''}`));
  for (const edge of [...edges]) {
    if (edge.targetHandle !== 'video'
        || !['rgbd_camera', 'stereo_rgbd_camera']
          .includes(sourceTypes.get(edge.source) || '')
        || !['rgbd_to_3d_bare_hand', 'rgbd_to_3d_black_glove']
          .includes(sourceTypes.get(edge.target) || '')) {
      continue;
    }
    const key = `${edge.source}|${edge.target}|depth`;
    if (edgeKeys.has(key)) continue;
    edges.push({
      ...edge,
      id: `xy-edge__${edge.source}depth-${edge.target}depth`,
      sourceHandle: 'depth',
      targetHandle: 'depth',
    });
    edgeKeys.add(key);
  }
  return { nodes, edges };
}

/** 节点数据补齐:用注册表描述符填充缺失的 label/icon/color/端口/configSchema。
 *  后端 graph 节点可能只存了部分字段(精简存储/旧数据),画布需要完整描述符。
 *  loadWorkflow 与 newWorkflowFromTemplate 共用,保证两条加载路径行为一致。 */
function _hydrateNode(node: Node<WorkflowNodeData>): Node<WorkflowNodeData> {
  const nodeType = canonicalNodeType(node.data.nodeType);
  const descriptor = getNodeType(nodeType);
  if (!descriptor) return node;
  const canonicalLabelTypes = new Set([
    'rgb_camera', 'mono_camera', 'rgbd_camera', 'fisheye_camera', 'stereo_camera',
    'stereo_rgbd_camera',
    'rgbd_to_3d_bare_hand', 'rgb_to_2d_bare_hand',
    'rgbd_to_3d_black_glove', 'rgb_to_2d_black_glove', 'annotation', 'ai_annotation',
    'human_review', 'ai_quality_review', 'lerobot_export', 'hdf5_export',
  ]);
  const controlledPorts = new Set([
    'rgb_camera', 'mono_camera', 'fisheye_camera', 'rgbd_camera',
    'stereo_camera', 'stereo_rgbd_camera', 'glove_sensor', 'mediapipe_hand', 'annotation',
    'ai_annotation', 'human_review', 'ai_quality_review', 'lerobot_export',
    'hdf5_export',
    'rgbd_to_3d_bare_hand', 'rgb_to_2d_bare_hand',
    'rgbd_to_3d_black_glove', 'rgb_to_2d_black_glove',
  ]);
  return {
    ...node,
    data: {
      ...node.data,
      nodeType,
      label: canonicalLabelTypes.has(nodeType)
        ? descriptor.label : (node.data.label || descriptor.label),
      icon: node.data.icon || descriptor.icon,
      // Semantic module colors replace stale values saved by older workflow
      // snapshots; custom/unknown nodes retain their stored color.
      color: canonicalLabelTypes.has(nodeType)
        ? descriptor.color : (node.data.color || descriptor.color),
      inputs: controlledPorts.has(nodeType)
        ? descriptor.inputs : (node.data.inputs?.length ? node.data.inputs : descriptor.inputs),
      outputs: controlledPorts.has(nodeType)
        ? descriptor.outputs
        : (node.data.outputs?.length ? node.data.outputs : descriptor.outputs),
      configSchema: _mergeSchema(node.data.configSchema,
                                 descriptor.configSchema),
      executionTarget: node.data.executionTarget || descriptor.executionTarget,
    },
  } as Node<WorkflowNodeData>;
}

/** Normalize a graph from either the API or an old localStorage draft. */
export function normalizeWorkflowGraphForEditor(
  graph: { nodes?: Node<WorkflowNodeData>[]; edges?: Edge[] },
) {
  const migrated = _migrateWorkflowGraph(graph);
  return {
    nodes: (migrated.nodes || []).map(_hydrateNode),
    edges: migrated.edges || [],
  };
}

/** Validate a proposed edge: source output kind must match the target
 *  input kind. Most legacy ``data`` inputs are generic, but annotation nodes
 *  use that persisted key for an RGB-video-only semantic input. */
export function isConnectionValid(
  conn: { source: string; target: string; sourceHandle?: string | null; targetHandle?: string | null },
  nodes: Node<WorkflowNodeData>[],
): boolean {
  const sourceNode = nodes.find((n) => n.id === conn.source);
  const targetNode = nodes.find((n) => n.id === conn.target);
  if (!sourceNode || !targetNode) return false;

  const outputs = sourceNode.data.outputs || [];
  const inputs = targetNode.data.inputs || [];
  if (!outputs.length || !inputs.length) return true; // unknown ports — allow

  const out = outputs.find((o) => o.key === (conn.sourceHandle ?? outputs[0].key))
    || (outputs.length === 1 ? outputs[0] : null);
  const inp = inputs.find((i) => i.key === (conn.targetHandle ?? inputs[0].key))
    || (inputs.length === 1 ? inputs[0] : null);
  if (!out || !inp) return true;

  const targetType = canonicalNodeType(targetNode.data.nodeType);
  const sourceType = canonicalNodeType(sourceNode.data.nodeType);

  // Human/AI annotation consumes video. Keep the persisted target handle
  // ``data`` for compatibility, but reject glove sensor and hand-keypoint
  // outputs at the editor boundary. AI Quality Review intentionally remains
  // generic, so a Glove Sensor can connect directly to it.
  if ((targetType === 'annotation' || targetType === 'ai_annotation')
      && inp.key === 'data') {
    return ['video', 'video_left', 'video_right'].includes(out.key)
      && sourceType !== 'glove_sensor';
  }

  if (inp.key === 'data') return true;          // generic hand-off port

  // Metric RGB-D hand reconstruction is deliberately a typed connection:
  // its Depth port can only be fed by the Depth output of an RGB-D
  // camera. This prevents a stale/forged depth edge from silently producing
  // an unrelated 3D result. Missing ports are still allowed at save time;
  // the worker will skip the incomplete node when the workflow runs.
  const isMetric3d = targetType === 'rgbd_to_3d_bare_hand'
    || targetType === 'rgbd_to_3d_black_glove';
  if (isMetric3d && inp.key === 'depth') {
    return (sourceType === 'rgbd_camera' || sourceType === 'stereo_rgbd_camera')
      && out.key === 'depth';
  }

  // 视频类端口兼容:stereo_camera 输出 video_left/video_right,
  // mediapipe_hand 输入是 video —— 同属视频数据,应允许连接。
  const videoKeys = ['video', 'video_left', 'video_right'];
  if (videoKeys.includes(inp.key) && videoKeys.includes(out.key)) return true;

  return out.key === inp.key;                   // typed ports must match
}

export const useWorkflowStore = create<WorkflowState>((set, get) => ({
  nodes: [],
  edges: [],
  workflowId: null,
  workflowProjectId: null,
  workflowName: 'Untitled',
  workflowStatus: 'draft',
  isDirty: false,
  isSaving: false,
  userRole: null,
  isPresetWorkflow: false,
  referenceProjectId: null,
  referenceInputs: null,
  projectBindings: null,
  currentWorkflowBindings: {},
  globalInputs: null,
  inputsLoading: false,
  inputsError: null,
  workflows: [],
  listLoading: false,
  history: [],
  historyIndex: -1,

  pushHistory: () => {
    const s = get();
    // Don't push if we're in the middle of undo/redo
    const h = [...s.history.slice(0, s.historyIndex + 1)];
    h.push({ nodes: _clone(s.nodes), edges: _clone(s.edges) });
    if (h.length > MAX_HISTORY) h.shift();
    set({ history: h, historyIndex: h.length - 1 });
  },

  undo: () => {
    const s = get();
    if (s.historyIndex < 0) return;
    const entry = s.history[s.historyIndex];
    if (!entry) return;
    set({
      nodes: _clone(entry.nodes),
      edges: _clone(entry.edges),
      historyIndex: s.historyIndex - 1,
      isDirty: true,
    });
  },

  redo: () => {
    const s = get();
    if (s.historyIndex >= s.history.length - 1) return;
    const entry = s.history[s.historyIndex + 1];
    if (!entry) return;
    set({
      nodes: _clone(entry.nodes),
      edges: _clone(entry.edges),
      historyIndex: s.historyIndex + 1,
      isDirty: true,
    });
  },

  onNodesChange: (changes) => set((s) => {
    // React Flow can emit non-drag position updates while measuring or
    // remounting controlled nodes. Only a position change with dragging=true
    // is a user edit; remove/add are always user edits.
    const userChange = changes.some((c) =>
      c.type === 'remove'
      || c.type === 'add'
      || (c.type === 'position' && c.dragging === true),
    );
    if (userChange) {
      // Push snapshot before applying change
      const h = [...s.history.slice(0, s.historyIndex + 1)];
      h.push({ nodes: _clone(s.nodes), edges: _clone(s.edges) });
      if (h.length > MAX_HISTORY) h.shift();
      const nodes = applyNodeChanges(changes, s.nodes) as Node<WorkflowNodeData>[];
      saveDraft({ ...s, nodes });
      return { nodes, isDirty: true, history: h, historyIndex: h.length - 1 };
    }
    const nodes = applyNodeChanges(changes, s.nodes) as Node<WorkflowNodeData>[];
    return { nodes, isDirty: s.isDirty };
  }),
  onEdgesChange: (changes) => set((s) => {
    const userChange = changes.some((c) => c.type === 'remove' || c.type === 'add');
    if (userChange) {
      const h = [...s.history.slice(0, s.historyIndex + 1)];
      h.push({ nodes: _clone(s.nodes), edges: _clone(s.edges) });
      if (h.length > MAX_HISTORY) h.shift();
      const edges = applyEdgeChanges(changes, s.edges);
      saveDraft({ ...s, edges });
      return { edges, isDirty: true, history: h, historyIndex: h.length - 1 };
    }
    const edges = applyEdgeChanges(changes, s.edges);
    return { edges, isDirty: s.isDirty };
  }),
  onConnect: (conn) => set((s) => {
    // Type check: the source output must feed a compatible input. A
    // generic "data" input accepts anything; a typed input (e.g. "video")
    // only accepts the matching output kind (e.g. camera video, not glove
    // sensor_data). Self-connections are rejected as well.
    if (conn.source === conn.target || !isConnectionValid(conn, s.nodes)) {
      const src = s.nodes.find((n) => n.id === conn.source);
      const tgt = s.nodes.find((n) => n.id === conn.target);
      const inpLabel = tgt?.data?.inputs?.find((i) => i.key === (conn.targetHandle || tgt?.data?.inputs?.[0]?.key))?.label
        || tgt?.data?.inputs?.[0]?.label || 'input';
      const srcOut = src?.data?.outputs?.find((o) => o.key === (conn.sourceHandle || src?.data?.outputs?.[0]?.key));
      const hint = srcOut && srcOut.key !== 'data'
        ? `Port mismatch: this input only accepts "${inpLabel}" data, but "${srcOut.label || srcOut.key}" outputs "${srcOut.key}"`
        : 'Self-connection is not allowed';
      useUiStore.getState().pushToast(`Cannot connect: ${hint}`, 'error');
      return s;
    }
    // 重复连线检查:同一 源端口→目标端口 已存在时阻止再次添加。
    // React Flow 同边 id 冲突会导致后加的连线不显示(看起来"没连上")。
    const isDup = (c: typeof conn) => s.edges.some((e) =>
      e.source === c.source
      && e.target === c.target
      && (e.sourceHandle || null) === (c.sourceHandle || null)
      && (e.targetHandle || null) === (c.targetHandle || null));
    if (isDup(conn)) {
      useUiStore.getState().pushToast('Already connected: this port pair is already linked.', 'error');
      return s;
    }
    let edges = addEdge(conn, s.edges);

    // 双目打包连线:Stereo Video 卡(输出 video_left + video_right 双端口)
    // 任一端口连到目标时,自动把另一端口也连到同一目标输入 ——
    // 避免"只拉了左目、忘记右目"导致处理节点只收到一路视频。
    const srcNode = s.nodes.find((n) => n.id === conn.source);
    const srcOuts = srcNode?.data?.outputs || [];
    const isStereoPair = srcNode?.data?.nodeType === 'stereo_camera'
      || (srcOuts.some((o) => o.key === 'video_left')
          && srcOuts.some((o) => o.key === 'video_right'));
    if (isStereoPair) {
      const mainHandle = conn.sourceHandle || 'video_left';
      const sibling = mainHandle === 'video_left' ? 'video_right' : 'video_left';
      const siblingConn = {
        ...conn,
        sourceHandle: sibling,
      };
      if (!isDup(siblingConn) && isConnectionValid(siblingConn, s.nodes)) {
        edges = addEdge(siblingConn, edges);
        useUiStore.getState().pushToast(
          `Stereo pair connected: ${mainHandle} + ${sibling} both linked.`, 'info');
      }
    }
    saveDraft({ ...s, edges });
    return { edges, isDirty: true };
  }),
  setNodes: (nodes) => {
    const s = get();
    saveDraft({ ...s, nodes });
    set({ nodes, isDirty: true });
  },
  setEdges: (edges) => {
    const s = get();
    saveDraft({ ...s, edges });
    set({ edges, isDirty: true });
  },

  setUserRole: (role: string) => set({ userRole: role }),

  loadReferenceInputs: async (projectId) => {
    const requestToken = ++inputsLoadToken;
    set({ inputsLoading: true, inputsError: null });
    // These are independent read-only endpoints. Start both together so a
    // remote-mounted session scan does not make the binding request wait in
    // a second network round trip.
    const bindingsPromise = getProjectBindings(projectId).catch(() => null);
    try {
      const data = await getProjectInputSources(projectId);
      // A late response from a previously selected workflow must not replace
      // the current project's real device list.
      if (requestToken !== inputsLoadToken) return;
      const enrichedNodes = enrichInputNodeDeviceMetadata(get().nodes, data);
      set({
        nodes: enrichedNodes,
        workflowProjectId: projectId,
        referenceProjectId: projectId,
        referenceInputs: data,
        globalInputs: null,
        inputsLoading: false,
        inputsError: null,
      });
      // Opening or switching workflows is read-only. Project × workflow
      // input snapshots must only be written by an explicit project/device
      // update, never merely by viewing the canvas.
    } catch (error) {
      if (requestToken !== inputsLoadToken) return;
      // Do not fall back to a previous project's/global device list after a
      // failed project lookup; that would expose cards from the wrong context.
      set({
        referenceInputs: null,
        referenceProjectId: null,
        globalInputs: null,
        inputsLoading: false,
        inputsError: error instanceof Error ? error.message : 'Failed to load project inputs',
      });
      return;
    }
    if (requestToken !== inputsLoadToken) return;
    // 顺带加载项目设备命名映射(工作流共享,本项目按节点覆盖)
    // bindings 可能晚于 loadWorkflow 完成 —— 加载完要同步当前工作流绑定,
    // 否则卡片不显示 P 徽标/绑定值
    try {
      const b = await bindingsPromise;
      if (requestToken !== inputsLoadToken) return;
      if (!b) return;
      const s = get();
      const nextBindings = b.workflow_bindings || null;
      set({
        projectBindings: nextBindings,
        currentWorkflowBindings: (nextBindings || {})[s.workflowId || ''] || {},
      });
    } catch { /* bindings 加载失败不阻断画布 */ }
  },

  loadInputsForWorkflow: async (projectId) => {
    // Project entry supplies ?project=. A direct workflow URL may omit it,
    // so use the project_id resolved by loadWorkflow(). Concrete device names
    // are project-scoped; the fixed input-type palette itself is independent
    // of this request.
    const state = get();
    const resolved = projectId || state.workflowProjectId;
    if (resolved) {
      set({ workflowProjectId: resolved });
      await get().loadReferenceInputs(resolved);
    } else {
      // A workflow can be designed before any project upload exists. Never
      // fall back to the global online-device directory here: it belongs to
      // the collector/heartbeat view, not to this workflow's input contract.
      // NodePalette still exposes the four fixed input categories; a concrete
      // source becomes available only after this project reports real data.
      ++inputsLoadToken;
      set({
        referenceInputs: {
          has_episodes: false,
          camera_names: [],
          devices: [],
          device_sources: [],
          sensors: [],
          modules: [],
        },
        globalInputs: null,
        referenceProjectId: null,
        inputsLoading: false,
        inputsError: null,
      });
    }
  },

  /** 写入/清除当前工作流某个节点的本项目绑定(画布绑定值立即生效)。 */
  setProjectBinding: async (nodeId, sourceKey) => {
    const s = get();
    const projectId = s.referenceProjectId;
    const workflowId = s.workflowId;
    if (!projectId || !workflowId) return;
    try {
      const res = await putProjectBinding(projectId, {
        workflow_id: workflowId,
        node_id: nodeId,
        source_key: sourceKey,
      });
      const nextBindings = res.workflow_bindings || null;
      set({
        projectBindings: nextBindings,
        currentWorkflowBindings: (nextBindings || {})[workflowId] || {},
      });
    } catch (e) { console.error('set binding failed', e); throw e; }
  },

  loadGlobalInputs: async () => {
    const requestToken = ++inputsLoadToken;
    set({ inputsLoading: true, inputsError: null });
    try {
      const data = await getDeviceInputSources();
      if (requestToken !== inputsLoadToken) return;
      const enrichedNodes = enrichInputNodeDeviceMetadata(get().nodes, data);
      set({
        nodes: enrichedNodes,
        globalInputs: data,
        referenceInputs: null,
        referenceProjectId: null,
        inputsLoading: false,
        inputsError: null,
      });
    } catch (error) {
      if (requestToken !== inputsLoadToken) return;
      set({
        globalInputs: null,
        referenceInputs: null,
        referenceProjectId: null,
        inputsLoading: false,
        inputsError: error instanceof Error ? error.message : 'Failed to load device inputs',
      });
    }
  },

  newWorkflow: () => {
    ++workflowLoadToken;
    ++inputsLoadToken;
    clearDraft();
    set({
      nodes: [],
      edges: [],
      workflowId: null,
      workflowProjectId: null,
      workflowName: 'Untitled',
      workflowStatus: 'draft',
      isDirty: false,
      isPresetWorkflow: false,
      currentWorkflowBindings: {},
      // A new workflow must not inherit the previous workflow's project/device
      // context. The input palette is driven only by the active context.
      referenceProjectId: null,
      referenceInputs: null,
      globalInputs: null,
      projectBindings: null,
      inputsLoading: false,
      inputsError: null,
      history: [],
      historyIndex: -1,
    });
  },

  /** 应用模板到当前工作流:模板内容(剔除输入设备节点)替换当前画布,
   *  但保留当前工作流的 ID 与名称 —— 用户在自己的工作流里搭流程,
   *  保存时仍是更新当前工作流,而不是顶替成模板的 "(copy)"。 */
  newWorkflowFromTemplate: async (template) => {
    const wf = await api.getWorkflow(template.id);
    const src = wf.graph || { nodes: [], edges: [] };
    // 模板样式 = 只含后处理链:剔除全部 input 设备节点及其关联边,
    // 悬空输入保留 —— 设备卡片由用户在采集端上报后按实际设备手动添加。
    const inputIds = new Set(
      (src.nodes || [])
        .filter((n) => (n.data as WorkflowNodeData)?.category === 'input')
        .map((n) => n.id),
    );
    const migrated = _migrateWorkflowGraph(src);
    const nodes = (migrated.nodes || [])
      .filter((n) => !inputIds.has(n.id))
      .map(_hydrateNode)
      .map((n) => ({ ...n, selected: false, dragging: false }));
    const edges = (migrated.edges || []).filter((e) => !inputIds.has(e.source) && !inputIds.has(e.target));
    const cur = get();
    set({
      nodes, edges,
      workflowId: cur.workflowId,
      // 当前工作流保留原名;从未打开任何工作流时用模板名
      workflowName: cur.workflowId ? cur.workflowName : (template.name || 'Workflow'),
      workflowStatus: cur.workflowStatus,
      isDirty: true, isPresetWorkflow: false,
      currentWorkflowBindings: {},
    });
    // 立即落盘草稿:应用模板后立刻刷新页面也不会丢失画布内容
    saveDraft(get());
  },

  loadWorkflowList: async () => {
    set({ listLoading: true });
    const cached = readWorkflowListCache();
    if (cached) set({ workflows: cached });
    try {
      const d = await api.listWorkflows(100);
      saveWorkflowListCache(d.workflows);
      set({ workflows: d.workflows });
    } catch {}
    set({ listLoading: false });
  },

  loadWorkflow: async (id) => {
    const loadToken = ++workflowLoadToken;
    // Invalidate input/binding requests belonging to the previous workflow.
    ++inputsLoadToken;
    set({ inputsLoading: true, inputsError: null });
    let wf;
    try {
      wf = await api.getWorkflow(id);
    } catch (error) {
      if (loadToken === workflowLoadToken) {
        set({
          inputsLoading: false,
          inputsError: error instanceof Error ? error.message : 'Failed to load workflow',
        });
      }
      throw error;
    }
    if (loadToken !== workflowLoadToken) return;
    const migrated = normalizeWorkflowGraphForEditor(wf.graph || { nodes: [], edges: [] });
    const nodes: Node<WorkflowNodeData>[] = migrated.nodes;
    // Sync idCounter to avoid duplicate IDs when new nodes are added
    if (nodes.length > 0) {
      let maxNum = 0;
      for (const n of nodes) {
        const m = n.id.match(/^node_(\d+)$/);
        if (m) maxNum = Math.max(maxNum, parseInt(m[1], 10));
      }
      if (maxNum >= idCounter) idCounter = maxNum + 1;
    }
    // Loading a saved workflow is the discard boundary. Remove any draft
    // belonging to the previous workflow so it cannot resurrect on reload.
    clearDraft();
    set({
      nodes,
      edges: migrated.edges,
      workflowId: wf.id,
      workflowProjectId: wf.project_id || null,
      workflowName: wf.name,
      workflowStatus: wf.status,
      isDirty: false,
      isPresetWorkflow: !!wf.is_preset,
      // Input sources and bindings are loaded for this workflow below. Do
      // not keep the previous project's list during the transition.
      referenceProjectId: null,
      referenceInputs: null,
      globalInputs: null,
      projectBindings: null,
      currentWorkflowBindings: {},
      inputsLoading: true,
      inputsError: null,
      history: [],
      historyIndex: -1,
    });
  },

  saveWorkflow: async () => {
    const s = get();
    if (s.isSaving) return;
    set({ isSaving: true });
    const graph = { nodes: s.nodes, edges: s.edges };
    try {
      if (s.workflowId) {
        await api.updateWorkflow(s.workflowId, { name: s.workflowName, project_id: s.workflowProjectId, graph, status: 'active' });
        set({ workflowStatus: 'active' });
      } else {
        const c = await api.createWorkflow({ name: s.workflowName, project_id: s.workflowProjectId, graph, status: 'active' });
        set({ workflowId: c.id, workflowProjectId: c.project_id || s.workflowProjectId, workflowStatus: c.status });
      }
      clearDraft();
      set({ isSaving: false, isDirty: false });
    } catch (e) { console.error(e); set({ isSaving: false }); }
  },

  saveWorkflowSafe: async () => {
    const s = get();
    // 预设对非 admin 只读:返回 needs-save-as,由 UI 弹"另存为"。
    // userRole 为 null(未获取到)→ 按非 admin 处理(安全默认)。
    if (s.isPresetWorkflow && s.userRole !== 'admin') return 'needs-save-as';
    if (s.isSaving) return 'saved';
    // 内联保存逻辑(不复用 saveWorkflow —— 后者吞错误):
    // 后端 403(权限实时变更)在这里捕获并转另存为。
    set({ isSaving: true });
    const graph = { nodes: s.nodes, edges: s.edges };
    try {
      if (s.workflowId) {
        await api.updateWorkflow(s.workflowId, { name: s.workflowName, project_id: s.workflowProjectId, graph, status: 'active' });
        set({ workflowStatus: 'active' });
      } else {
        const c = await api.createWorkflow({ name: s.workflowName, project_id: s.workflowProjectId, graph, status: 'active' });
        set({ workflowId: c.id, workflowProjectId: c.project_id || s.workflowProjectId, workflowStatus: c.status });
      }
      clearDraft();
      set({ isSaving: false, isDirty: false });
      return 'saved';
    } catch (e) {
      set({ isSaving: false });
      const msg = e instanceof Error ? e.message : String(e);
      if (/preset/i.test(msg) || /403/.test(msg)) return 'needs-save-as';
      throw e;
    }
  },

  saveWorkflowAs: async (name) => {
    const s = get();
    set({ isSaving: true, workflowName: name, workflowId: null });
    const graph = { nodes: s.nodes, edges: s.edges };
    try {
      // “Save as copy” creates a standalone workflow; it must not silently
      // become a second workflow of the current project.
      const c = await api.createWorkflow({ name, project_id: null, graph, status: 'active' });
      clearDraft();
      // 副本不是预设
      set({ workflowId: c.id, workflowProjectId: c.project_id || null, workflowStatus: c.status, isSaving: false, isDirty: false, isPresetWorkflow: false });
    } catch (e) { console.error(e); set({ isSaving: false }); }
  },
}));

/** Allocate unique node IDs, guaranteed not to collide with existing nodes. */
export function nextNodeId() {
  const existing = useWorkflowStore.getState().nodes;
  let id: string;
  do {
    id = `node_${idCounter++}`;
  } while (existing.some(n => n.id === id));
  return id;
}
