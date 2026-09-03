import { useCallback, useMemo, useRef } from 'react';
import { ReactFlow, Background, MiniMap, Controls, useReactFlow, type NodeTypes, type EdgeTypes } from '@xyflow/react';
import '@xyflow/react/dist/style.css';
import { useWorkflowStore, nextNodeId } from '../../store/workflowStore';
import { getAllNodeTypes, getMaxPortRows } from './nodes/registry';
import { WorkflowNodeComponent } from './nodes/WorkflowNode';
import { DeletableEdge } from './edges/DeletableEdge';
import type { DeviceInputSource, WorkflowNodeData } from '../../types/workflow';

const nodeTypes: NodeTypes = { workflowNode: WorkflowNodeComponent };
// 自定义边:选中时显示 × 删除按钮(未选中时外观与默认一致)
const edgeTypes: EdgeTypes = { default: DeletableEdge };

export function WorkflowCanvas() {
  const { nodes, edges, onNodesChange, onEdgesChange, onConnect } = useWorkflowStore();
  // 切换工作流时强制 ReactFlow 整体重挂载:各工作流节点 id 重复
  // (都是 node_1..node_N),React Flow 按 id 复用内部实例,切换时会
  // 残留上一工作流的 handle 位置/边状态,导致连线有时显示有时丢失。
  const workflowId = useWorkflowStore((s) => s.workflowId);
  const rf = useReactFlow();
  const wrapperRef = useRef<HTMLDivElement>(null);

  // Uniform card height: compute from max port rows across all node types
  const maxRows = useMemo(() => getMaxPortRows(), []);
  const cardHeight = maxRows * 22 + 42; // 22px per row + header+padding

  const onDragOver = useCallback((e: React.DragEvent) => { e.preventDefault(); e.dataTransfer.dropEffect = 'move'; }, []);

  const onDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    const nodeTypeStr = e.dataTransfer.getData('application/reactflow-type');
    if (!nodeTypeStr || !wrapperRef.current) return;
    const desc = getAllNodeTypes().find((nd) => nd.type === nodeTypeStr);
    if (!desc) return;

    let deviceSource: DeviceInputSource | null = null;
    const rawDeviceSource = e.dataTransfer.getData('application/reactflow-device');
    if (rawDeviceSource) {
      try { deviceSource = JSON.parse(rawDeviceSource) as DeviceInputSource; } catch { deviceSource = null; }
    }
    const sourceKeys = deviceSource?.source_keys || [];
    const sourceConfig = deviceSource
      ? (deviceSource.input_type === 'stereo_camera'
        || deviceSource.input_type === 'stereo_rgbd_camera')
        ? {
            source_keys: sourceKeys.join(','),
            source_key: sourceKeys[0] || '',
            position: sourceKeys[0] || '',
          }
        : {
            source_key: deviceSource.source_key || sourceKeys[0] || '',
            position: deviceSource.source_key || sourceKeys[0] || '',
          }
      : {};

    const newNodeData: WorkflowNodeData = {
      label: desc.label, category: desc.category, icon: desc.icon, color: desc.color,
      config: { ...desc.defaultConfig, ...sourceConfig }, nodeType: desc.type, inputs: desc.inputs, outputs: desc.outputs,
      configSchema: desc.configSchema, executionTarget: desc.executionTarget,
      ...(deviceSource ? {
        // Keep the raw name/id for processing and add the standardized name
        // for rendering even when the input metadata is later unavailable.
        device_name: deviceSource.name,
        device_id: deviceSource.id,
        device_display_name: deviceSource.display_name,
        // The category is fixed when the physical source is added. Editing
        // source_key later must never rename the card on the canvas.
        device_type: deviceSource.device_type,
        depth_keys: deviceSource.depth_keys,
      } : {}),
    };

    // Convert screen (viewport) coords to React Flow internal coords.
    // Apply centering offset (-100, -40) in SCREEN pixels BEFORE conversion,
    // so the offset stays consistent regardless of zoom level.
    // (Applying offset after conversion would scale it by 1/zoom.)
    const flowPos = rf.screenToFlowPosition({
      x: e.clientX - 100,
      y: e.clientY - 40,
    });

    // Use a single atomic state update to avoid React Flow sync issues
    useWorkflowStore.setState((state) => ({
      nodes: [...state.nodes, {
        id: nextNodeId(), type: 'workflowNode',
        position: flowPos,
        data: newNodeData,
      }],
      isDirty: true,
    }));
  }, [rf]);

  return (
    <div ref={wrapperRef} className="flex-1 h-full" onDragOver={onDragOver} onDrop={onDrop}
      style={{ '--node-card-height': `${cardHeight}px` } as React.CSSProperties}>
      <ReactFlow
        key={workflowId || 'blank'}
        nodes={nodes} edges={edges}
        onNodesChange={onNodesChange} onEdgesChange={onEdgesChange} onConnect={onConnect}
        nodeTypes={nodeTypes} edgeTypes={edgeTypes} fitView
        deleteKeyCode={['Delete', 'Backspace']}
        multiSelectionKeyCode="Shift"
        snapToGrid snapGrid={[16, 16]}
        defaultEdgeOptions={{ style: { stroke: '#475569', strokeWidth: 2 } }}
        style={{ background: '#0f172a' }}
      >
        <Background color="#1e293b" gap={32} size={1} />
        <MiniMap style={{ background: '#1e293b' }} maskColor="rgba(0,0,0,0.5)"
          nodeColor={(n) => (n.data as WorkflowNodeData)?.color || '#475569'} />
        <Controls className="!bg-gray-800 !border-gray-700" />
      </ReactFlow>
    </div>
  );
}
