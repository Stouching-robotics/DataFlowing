import type { NodeTypeDescriptor } from '../../../types/workflow';
import { MODULE_COLORS } from './theme';

const registry = new Map<string, NodeTypeDescriptor>();

/** Persisted slugs renamed with the semantic RGB/RGB-D contract. */
export const LEGACY_NODE_TYPE_ALIASES: Record<string, string> = {
  rgb_hand_3d: 'rgb_to_2d_bare_hand',
  black_hand_rgb_3d: 'rgb_to_2d_black_glove',
  stereo_triangulate: 'rgbd_to_3d_bare_hand',
  black_glove_hand: 'rgbd_to_3d_black_glove',
};

/** Canonical labels are part of the editor contract, not server-provided copy. */
export const CANONICAL_NODE_LABELS: Record<string, string> = {
  annotation: 'Human Annotation',
  human_review: 'Human Review',
  rgb_to_2d_bare_hand: 'RGB_TO_2D_BareHand',
  rgb_to_2d_black_glove: 'RGB_TO_2D_BlackGlove',
  rgbd_to_3d_bare_hand: 'RGB-D_3D_BareHand',
  rgbd_to_3d_black_glove: 'RGB-D_3D_BlackGlove',
};

// These ports are part of the persisted workflow contract. Keep them fixed
// even while an older backend process is still serving a stale catalog during
// a rolling reload; otherwise an old Hand 3D port can reappear on RGB-only
// cards after the async catalog request completes.
const CANONICAL_PORTS: Record<string, Pick<NodeTypeDescriptor, 'inputs' | 'outputs'>> = {
  rgb_camera: {
    inputs: [], outputs: [{ key: 'video', label: 'RGB Video' }],
  },
  mono_camera: {
    inputs: [], outputs: [{ key: 'video', label: 'RGB Video' }],
  },
  fisheye_camera: {
    inputs: [], outputs: [{ key: 'video', label: 'RGB Video' }],
  },
  rgbd_camera: {
    inputs: [], outputs: [{ key: 'video', label: 'RGB Video' }, { key: 'depth', label: 'Depth' }],
  },
  stereo_camera: {
    inputs: [], outputs: [{ key: 'video_left', label: 'Left RGB Video' }, { key: 'video_right', label: 'Right RGB Video' }],
  },
  stereo_rgbd_camera: {
    inputs: [], outputs: [
      { key: 'video_left', label: 'Left RGB Video' },
      { key: 'video_right', label: 'Right RGB Video' },
      { key: 'depth', label: 'Depth' },
    ],
  },
  glove_sensor: {
    inputs: [], outputs: [{ key: 'sensor_data', label: 'Glove Sensor Data' }],
  },
  mediapipe_hand: {
    inputs: [{ key: 'video', label: 'RGB Video' }],
    outputs: [{ key: 'hand_keypoints', label: 'Hand 2D' }],
  },
  annotation: {
    inputs: [{ key: 'data', label: 'RGB Video' }],
    outputs: [{ key: 'annotation', label: 'Annotation' }],
  },
  ai_annotation: {
    inputs: [{ key: 'data', label: 'RGB Video' }],
    outputs: [{ key: 'annotation', label: 'Annotation' }],
  },
  rgb_to_2d_bare_hand: {
    inputs: [{ key: 'video', label: 'RGB Video' }],
    outputs: [{ key: 'hand_keypoints', label: 'Hand 2D' }],
  },
  rgb_to_2d_black_glove: {
    inputs: [{ key: 'video', label: 'RGB Video' }],
    outputs: [{ key: 'hand_keypoints', label: 'Hand 2D' }],
  },
  rgbd_to_3d_bare_hand: {
    inputs: [{ key: 'video', label: 'RGB Video' }, { key: 'depth', label: 'Depth' }],
    outputs: [{ key: 'hand_3d', label: 'Hand 3D' }],
  },
  rgbd_to_3d_black_glove: {
    inputs: [{ key: 'video', label: 'RGB Video' }, { key: 'depth', label: 'Depth' }],
    outputs: [{ key: 'hand_3d', label: 'Hand 3D' }],
  },
  human_review: {
    inputs: [{ key: 'data', label: 'Review Target' }],
    outputs: [{ key: 'reviewed', label: 'Reviewed Data' }],
  },
  ai_quality_review: {
    inputs: [{ key: 'data', label: 'Quality Review Target' }],
    outputs: [{ key: 'reviewed', label: 'Reviewed Data' }],
  },
  lerobot_export: {
    inputs: [{ key: 'data', label: 'Exportable Data' }],
    outputs: [{ key: 'dataset', label: 'Dataset' }],
  },
  hdf5_export: {
    inputs: [{ key: 'data', label: 'Exportable Data' }],
    outputs: [{ key: 'dataset', label: 'Dataset' }],
  },
};

export function canonicalNodeType(type: string): string {
  const raw = String(type || '').trim();
  return LEGACY_NODE_TYPE_ALIASES[raw.toLowerCase()] || raw;
}

export function registerNodeType(n: NodeTypeDescriptor) {
  const type = canonicalNodeType(n.type);
  const normalized = type === n.type ? n : { ...n, type, slug: type };
  registry.set(type, CANONICAL_NODE_LABELS[type]
    ? { ...normalized, label: CANONICAL_NODE_LABELS[type] }
    : normalized);
}
export function getNodeType(type: string) {
  return registry.get(canonicalNodeType(type));
}
export function getAllNodeTypes() {
  return Array.from(registry.values());
}
export function getNodeTypesByCategory(cat: string) {
  const order = cat === 'process'
    ? ['annotation', 'ai_annotation', 'rgb_to_2d_bare_hand',
       'rgb_to_2d_black_glove', 'rgbd_to_3d_bare_hand',
       'rgbd_to_3d_black_glove', 'mediapipe_hand']
    : [];
  return Array.from(registry.values())
    .filter((n) => n.category === cat)
    .sort((a, b) => {
      const ai = order.indexOf(a.type);
      const bi = order.indexOf(b.type);
      if (ai < 0 && bi < 0) return 0;
      if (ai < 0) return 1;
      if (bi < 0) return -1;
      return ai - bi;
    });
}

function mergeConfigSchemas(
  current?: NodeTypeDescriptor['configSchema'],
  fallback?: NodeTypeDescriptor['configSchema'],
): NodeTypeDescriptor['configSchema'] {
  if (!current?.length) return fallback || [];
  if (!fallback?.length) return current;
  const names = new Set(current.map((field) => field.name));
  return [...current, ...fallback.filter((field) => !names.has(field.name))];
}

/** Replace/extend the fallback catalog with descriptors returned by FastAPI. */
export function hydrateNodeTypes(items: Array<Partial<NodeTypeDescriptor> & { slug?: string; defaultConfig?: Record<string, unknown>; default_config?: Record<string, unknown> }>) {
  for (const item of items) {
    const type = canonicalNodeType(item.type || item.slug || '');
    if (!type || !item.category || !item.label) continue;
    const existing = registry.get(type);
    const ports = CANONICAL_PORTS[type];
    registry.set(type, {
      type,
      slug: type,
      version: item.version || '1.0',
      category: item.category as NodeTypeDescriptor['category'],
      // A stale backend process may still return a historical display label.
      // Once the type is canonical, the editor must never regress visually.
      label: CANONICAL_NODE_LABELS[type] || item.label,
      icon: item.icon || 'ant-design:appstore-outlined',
      color: item.color || '#64748b',
      // 后端未提供 description 时沿用内置说明(不因 hydrate 丢失悬停提示)
      description: item.description ?? existing?.description,
      // A stale backend may still return the old RGB Hand 3D descriptor.
      // Canonical RGB/RGB-D contracts always win over that response.
      inputs: ports?.inputs || item.inputs || existing?.inputs || [],
      outputs: ports?.outputs || item.outputs || existing?.outputs || [],
      defaultConfig: item.defaultConfig || item.default_config || {},
      // Preserve a newly added local field when an older backend catalog is
      // still being served during a rolling restart.
      configSchema: mergeConfigSchemas(item.configSchema, existing?.configSchema),
      executionTarget: item.executionTarget,
      capabilities: item.capabilities || [],
    });
  }
}

/** Max port rows across all registered nodes — used for uniform card height. */
export function getMaxPortRows(): number {
  let max = 1;
  for (const n of registry.values()) {
    const rows = (n.inputs?.length || 0) + (n.outputs?.length || 0);
    if (rows > max) max = rows;
  }
  return max;
}

// ── Built-in nodes ────────────────────────────────────
// All icons use ant-design: prefix — all preloaded offline, no CDN
const BUILTIN: NodeTypeDescriptor[] = [
  // ── Input: Cameras ──
  {
    type: 'rgb_camera', category: 'input', label: 'RGB Camera',
    icon: 'ant-design:video-camera-outlined', color: MODULE_COLORS.input,
    description: 'Single-view RGB camera input. Fisheye and other lens variants use this same device category.',
    inputs: [],
    outputs: [{ key: 'video', label: 'RGB Video' }],
    defaultConfig: { position: '', source_key: '', fps: 30 },
  },
  {
    type: 'mono_camera', category: 'input', label: 'RGB Camera',
    icon: 'ant-design:video-camera-outlined', color: MODULE_COLORS.input,
    description: 'Single-view RGB input (including fisheye and RGB-D color streams). Device name is optional; blank auto-detects the batch video.',
    inputs: [],
    outputs: [{ key: 'video', label: 'RGB Video' }],
    defaultConfig: { source_key: '', fps: 30 },
  },
  {
    type: 'rgbd_camera', category: 'input', label: 'RGB-D Camera',
    icon: 'ant-design:video-camera-outlined', color: MODULE_COLORS.input,
    description: 'RGB-D camera input. Select the physical device after its uploaded metadata is available; the depth stream is paired by the backend.',
    inputs: [],
    outputs: [{ key: 'video', label: 'RGB Video' },
              { key: 'depth', label: 'Depth' }],
    defaultConfig: { source_key: '', fps: 30 },
  },
  {
    type: 'fisheye_camera', category: 'input', label: 'RGB Camera',
    icon: 'ant-design:video-camera-outlined', color: MODULE_COLORS.input,
    description: 'Historical fisheye input alias. New workflows use Mono RGB with lens=fisheye.',
    inputs: [],
    outputs: [{ key: 'video', label: 'RGB Video' }],
    defaultConfig: { position: '', source_key: '', fps: 30 },
  },
  {
    type: 'stereo_camera', category: 'input', label: 'Stereo RGB Camera',
    icon: 'ant-design:video-camera-outlined', color: MODULE_COLORS.input,
    description: 'Stereo video capture (left + right). Device names optional — blank auto-detects the stereo pair. Connect either port once: both Left and Right edges are created together.',
    inputs: [],
    outputs: [{ key: 'video_left', label: 'Left RGB Video' },
              { key: 'video_right', label: 'Right RGB Video' }],
    defaultConfig: { source_keys: '', source_key: '', fps: 30 },
  },
  {
    type: 'stereo_rgbd_camera', category: 'input', label: 'Stereo RGB-D Camera',
    icon: 'ant-design:video-camera-outlined', color: MODULE_COLORS.input,
    description: 'Stereo RGB-D input with left/right RGB video and a real depth stream. Connect Depth to RGB-D 3D processing when metric 3D is required.',
    inputs: [],
    outputs: [{ key: 'video_left', label: 'Left RGB Video' },
              { key: 'video_right', label: 'Right RGB Video' },
              { key: 'depth', label: 'Depth' }],
    defaultConfig: { source_keys: '', source_key: '', fps: 30 },
  },
  // ── Input: Glove ──
  {
    type: 'glove_sensor', category: 'input', label: 'Glove Sensor',
    icon: 'ant-design:edit-outlined', color: MODULE_COLORS.input,
    description: 'Glove sensor (pressure/joint data, no video). Connect its sensor data directly to AI Quality Review for sensor checks.',
    inputs: [],
    outputs: [{ key: 'sensor_data', label: 'Glove Sensor Data' }],
    defaultConfig: { source_key: '', device: 'SenseGlove', hand: 'both', fps: 60 },
  },
  // ── Process ──
  {
    type: 'mediapipe_hand', category: 'process', label: 'MediaPipe Hand',
    icon: 'ant-design:aim-outlined', color: MODULE_COLORS.hand3d,
    description: 'Skeleton detection. Accepts VIDEO input only (mono or stereo camera); glove sensor data cannot connect here.',
    inputs: [{ key: 'video', label: 'RGB Video' }],
    outputs: [{ key: 'hand_keypoints', label: 'Hand 2D' }],
    defaultConfig: { model_complexity: 1, min_detection_conf: 0.5, min_tracking_conf: 0.5, max_hands: 2 },
  },
  {
    type: 'annotation', category: 'process', label: 'Human Annotation',
    icon: 'ant-design:field-time-outlined', color: MODULE_COLORS.annotation,
    description: 'Frame-level annotation on RGB video. Glove sensor data is not an annotation input.',
    inputs: [{ key: 'data', label: 'RGB Video' }],
    outputs: [{ key: 'annotation', label: 'Annotation' }],
    defaultConfig: { type: 'frame_level', auto_label: true },
  },
  {
    type: 'ai_annotation', category: 'process', label: 'AI Annotation',
    icon: 'ant-design:robot-outlined', color: MODULE_COLORS.annotation,
    description: 'AI-assisted annotation on RGB video. Glove sensor data is not an annotation input.',
    inputs: [{ key: 'data', label: 'RGB Video' }],
    outputs: [{ key: 'annotation', label: 'Annotation' }],
    defaultConfig: {
      mode: 'signal_vlm', min_confidence: 0.7, prompt_language: 'zh',
      max_segments: 50, vlm_provider: 'local', api_vendor: 'kimi',
      api_model: 'kimi-k3', api_key: '', api_base_url: '',
    },
    configSchema: [
      { name: 'prompt_language', type: 'select', label: 'Label language', default: 'zh', options: ['zh', 'en'] },
      { name: 'vlm_provider', type: 'select', label: 'VLM provider', default: 'local', options: ['local', 'api'] },
      { name: 'api_vendor', type: 'select', label: 'API vendor', default: 'kimi', options: ['kimi', 'qwen', 'siliconflow'] },
      { name: 'api_model', type: 'string', label: 'API model', default: 'kimi-k3' },
      { name: 'api_base_url', type: 'string', label: 'API base URL', default: '' },
    ],
  },
  {
    type: 'rgbd_to_3d_bare_hand', category: 'process', label: 'RGB-D_3D_BareHand',
    icon: 'ant-design:deployment-unit-outlined', color: MODULE_COLORS.hand3d,
    description: 'Metric 3D bare-hand keypoints from RGB video plus a matched depth stream.',
    inputs: [{ key: 'video', label: 'RGB Video' }, { key: 'depth', label: 'Depth' }],
    outputs: [{ key: 'hand_3d', label: 'Hand 3D' }],
    defaultConfig: { mode: 'auto' },
  },
  {
    type: 'rgb_to_2d_bare_hand', category: 'process', label: 'RGB_TO_2D_BareHand',
    icon: 'ant-design:deployment-unit-outlined', color: MODULE_COLORS.hand3d,
    description: '2D bare-hand keypoints from RGB video. The optional 3D view is display-only and is not exported as metric 3D.',
    inputs: [{ key: 'video', label: 'RGB Video' }],
    outputs: [{ key: 'hand_keypoints', label: 'Hand 2D' }],
    defaultConfig: { max_hands: 2, min_detection_conf: 0.1, min_presence_conf: 0.1, min_tracking_conf: 0.5, device: 'auto', smooth: true, preview_3d: true, freq_min: 5, beta: 0.05 },
  },
  {
    type: 'rgbd_to_3d_black_glove', category: 'process', label: 'RGB-D_3D_BlackGlove',
    icon: 'ant-design:aim-outlined', color: MODULE_COLORS.hand3d,
    description: 'Metric 3D black-glove keypoints from RGB video plus a matched depth stream.',
    inputs: [{ key: 'video', label: 'RGB Video' }, { key: 'depth', label: 'Depth' }],
    // 2D landmarks are internal to the detector; only Hand 3D is a graph
    // output and may be connected to downstream processing/review/export.
    outputs: [{ key: 'hand_3d', label: 'Hand 3D' }],
    defaultConfig: { mode: 'auto', max_hands: 2, det_conf: 0.05, device: 'auto', pose_device: 'auto', imgsz: 640, smooth: true, preview_3d: true, freq_min: 5, beta: 0.05, movement_thresh: 1.5, skip_timeout: 3, box_alpha: 0.7, pose_conf_thr: 0.15, new_track_conf: 0.1, lost_timeout: 8, hold_max: 12, spawn_confirm: 2, match_contain_thr: 0.7 },
  },
  {
    type: 'rgb_to_2d_black_glove', category: 'process', label: 'RGB_TO_2D_BlackGlove',
    icon: 'ant-design:deployment-unit-outlined', color: MODULE_COLORS.hand3d,
    description: '2D black-glove keypoints from RGB video. The optional 3D view is display-only and is not exported as metric 3D.',
    inputs: [{ key: 'video', label: 'RGB Video' }],
    outputs: [{ key: 'hand_keypoints', label: 'Hand 2D' }],
    defaultConfig: { mode: 'auto', max_hands: 2, det_conf: 0.05, device: 'auto', pose_device: 'auto', imgsz: 640, smooth: true, preview_3d: true, freq_min: 5, beta: 0.05, movement_thresh: 1.5, skip_timeout: 3, box_alpha: 0.7, pose_conf_thr: 0.15, new_track_conf: 0.1, lost_timeout: 8, hold_max: 12, spawn_confirm: 2, match_contain_thr: 0.7 },
  },
  // ── Review ──
  {
    type: 'human_review', category: 'review', label: 'Human Review',
    icon: 'ant-design:eye-outlined', color: MODULE_COLORS.review,
    description: 'Human review gate. Accepts any upstream data; required review before export.',
    inputs: [{ key: 'data', label: 'Review Target' }],
    outputs: [{ key: 'reviewed', label: 'Reviewed Data' }],
    defaultConfig: { required: true, reviewers: 1 },
  },
  {
    type: 'ai_quality_review', category: 'review', label: 'AI Quality Review',
    icon: 'ant-design:robot-outlined', color: MODULE_COLORS.review,
    description: 'Checks video decode, frame continuity, black screens, freezes, and AI annotation coverage before automatic approval and export.',
    inputs: [{ key: 'data', label: 'Quality Review Target' }],
    outputs: [{ key: 'reviewed', label: 'Reviewed Data' }],
    defaultConfig: { mode: 'gate' },
  },
  // ── Export ──
  {
    type: 'lerobot_export', category: 'export', label: 'LeRobot Export',
    icon: 'ant-design:cloud-server-outlined', color: MODULE_COLORS.export,
    description: 'Export to LeRobot dataset format (v3.0 / v2.1). Accepts any upstream data.',
    inputs: [{ key: 'data', label: 'Exportable Data' }],
    outputs: [{ key: 'dataset', label: 'Dataset' }],
    defaultConfig: { version: 'v3.0', split_ratio: 0.9, shard_size: 100000 },
  },
  {
    type: 'hdf5_export', category: 'export', label: 'HDF5 Export',
    icon: 'ant-design:cloud-server-outlined', color: MODULE_COLORS.export,
    description: 'Export to HDF5 dataset format. Accepts any upstream data.',
    inputs: [{ key: 'data', label: 'Exportable Data' }],
    outputs: [{ key: 'dataset', label: 'Dataset' }],
    defaultConfig: { compression: 'gzip', level: 4 },
  },
];
BUILTIN.forEach(registerNodeType);
