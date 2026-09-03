import { BaseEdge, EdgeLabelRenderer, getBezierPath, useReactFlow, type EdgeProps } from '@xyflow/react';

/** 可删除连线:默认渲染与现状一致的 BaseEdge(样式/箭头不变);
 *  选中(selected)时在边中点显示红色 × 按钮,点击删除该连线。
 *  删除走 deleteElements → onEdgesChange(remove) → store 现有
 *  pushHistory + saveDraft 通道,与键盘 Delete 完全同一路径。 */
export function DeletableEdge(props: EdgeProps) {
  const {
    id, sourceX, sourceY, targetX, targetY,
    sourcePosition, targetPosition, selected, style, markerEnd,
  } = props;
  const [edgePath, labelX, labelY] = getBezierPath({
    sourceX, sourceY, sourcePosition, targetX, targetY, targetPosition,
  });
  const { deleteElements } = useReactFlow();

  return (
    <>
      <BaseEdge id={id} path={edgePath} markerEnd={markerEnd} style={style} />
      {selected && (
        <EdgeLabelRenderer>
          <button
            type="button"
            className="nodrag nopan"
            title="Delete connection"
            onClick={() => deleteElements({ edges: [{ id }] })}
            style={{
              position: 'absolute',
              transform: `translate(-50%, -50%) translate(${labelX}px, ${labelY}px)`,
              width: 20,
              height: 20,
              borderRadius: 10,
              background: '#dc2626',
              color: '#fff',
              border: 'none',
              cursor: 'pointer',
              fontSize: 13,
              lineHeight: '18px',
              textAlign: 'center',
              boxShadow: '0 1px 4px rgba(0,0,0,.5)',
              zIndex: 10,
            }}
          >
            ×
          </button>
        </EdgeLabelRenderer>
      )}
    </>
  );
}
