import { useState } from 'react';
import { IconifyIcon } from './IconifyIcon';
import { getAllNodeTypes, getNodeTypesByCategory } from './nodes/registry';
import type { NodeCategory, NodeTypeDescriptor } from '../../types/workflow';

// 旧单目卡只保留给历史工作流渲染/执行,不再出现在新工作流调色板。
const LEGACY_INPUT_TYPES = ['rgb_camera', 'fisheye_camera'];
// 新工作流始终从这组稳定的采集端分类开始。真实设备名称由工作流卡片
// 内部的 source_key 控件从项目上传/心跳数据中选择,不作为调色板卡片。
const FIXED_INPUT_TYPES = [
  'glove_sensor', 'mono_camera', 'rgbd_camera', 'stereo_camera', 'stereo_rgbd_camera',
];
// MediaPipe Hand is retained as a backend/old-workflow compatibility node;
// the current hand modules are exposed through the canonical process cards.
const LEGACY_PROCESS_TYPES = ['mediapipe_hand'];

const CATEGORIES: { key: NodeCategory; label: string; icon: string }[] = [
  { key: 'input', label: 'Input', icon: 'ant-design:video-camera-outlined' },
  { key: 'process', label: 'Process', icon: 'ant-design:desktop-outlined' },
  { key: 'review', label: 'Review', icon: 'ant-design:eye-outlined' },
  { key: 'export', label: 'Export', icon: 'ant-design:download-outlined' },
];

export function NodePalette() {
  const [search, setSearch] = useState('');
  const [collapsed, setCollapsed] = useState<Record<string, boolean>>({});
  // 悬停说明卡(fixed 定位,避免被滚动容器裁剪):null = 未悬停
  const [tip, setTip] = useState<{ text: string; x: number; y: number } | null>(null);

  const onDragStart = (e: React.DragEvent, nd: NodeTypeDescriptor) => {
    e.dataTransfer.setData('application/reactflow-type', nd.type);
    e.dataTransfer.effectAllowed = 'move';
  };

  const showTip = (text: string, e: React.MouseEvent) => {
    setTip({ text, x: Math.min(e.clientX + 14, window.innerWidth - 256), y: Math.min(e.clientY + 12, window.innerHeight - 100) });
  };
  const hideTip = () => setTip(null);

  const filtered = search.trim()
    ? getAllNodeTypes().filter((nd) => nd.label.toLowerCase().includes(search.toLowerCase()))
    : null;

  const shouldShow = (nd: NodeTypeDescriptor): boolean => {
    if (LEGACY_INPUT_TYPES.includes(nd.type)) return false;
    if (LEGACY_PROCESS_TYPES.includes(nd.type)) return false;
    if (nd.category === 'input') return FIXED_INPUT_TYPES.includes(nd.type);
    return true;
  };

  const renderNode = (nd: NodeTypeDescriptor) => {
    const cls = 'flex items-center gap-2 px-2 py-1.5 mx-1 mb-0.5 rounded cursor-grab hover:bg-gray-800 active:cursor-grabbing border border-transparent hover:border-gray-700 transition-colors';
    return (
      <div key={nd.type}
        onMouseEnter={(e) => nd.description && showTip(nd.description, e)}
        onMouseMove={(e) => nd.description && showTip(nd.description, e)}
        onMouseLeave={hideTip}>
        <div draggable onDragStart={(e) => onDragStart(e, nd)} className={cls}>
          <span className="w-2.5 h-2.5 rounded-full shrink-0" style={{ backgroundColor: nd.color }} />
          <span className="text-xs text-gray-300 truncate">{nd.label}</span>
        </div>
      </div>
    );
  };

  return (
    <div className="w-[220px] bg-gray-900 border-r border-gray-800 flex flex-col shrink-0 overflow-hidden relative">
      <div className="p-2">
        <input type="text" placeholder="Search nodes..." value={search} onChange={(e) => setSearch(e.target.value)}
          className="w-full bg-gray-800 border border-gray-700 rounded px-2 py-1.5 text-xs text-gray-200 placeholder-gray-500 focus:outline-none focus:border-blue-500" />
      </div>
      <div className="flex-1 overflow-y-auto px-1 pb-2">
        {filtered
          ? filtered.filter(shouldShow).map(renderNode)
          : CATEGORIES.map((cat) => {
              const nodes = getNodeTypesByCategory(cat.key).filter(shouldShow).sort((a, b) => {
                if (cat.key !== 'input') return 0;
                return FIXED_INPUT_TYPES.indexOf(a.type) - FIXED_INPUT_TYPES.indexOf(b.type);
              });
              const open = !collapsed[cat.key];
              return (
                <div key={cat.key}>
                  <button onClick={() => setCollapsed((p) => ({ ...p, [cat.key]: !p[cat.key] }))}
                    className="w-full flex items-center gap-1.5 px-2 py-1.5 text-[11px] font-semibold text-gray-500 uppercase tracking-wider hover:text-gray-300">
                    <IconifyIcon icon={cat.icon} className="text-[14px]" />
                    <span>{cat.label}</span>
                    <IconifyIcon icon={open ? 'ant-design:caret-down-filled' : 'ant-design:caret-right-outlined'} className="ml-auto text-[10px]" />
                  </button>
                  {open && nodes.map(renderNode)}
                </div>
              );
            })}
      </div>
      <div className="px-2 py-2 border-t border-gray-800 text-[10px] text-gray-600 text-center">Drag nodes onto canvas</div>
      {tip && (
        <div className="pointer-events-none fixed z-[200] w-60 rounded border border-gray-700 bg-gray-950/95 p-2 shadow-xl"
          style={{ left: tip.x, top: tip.y }}>
          <p className="text-[11px] leading-snug text-gray-300">{tip.text}</p>
        </div>
      )}
    </div>
  );
}
