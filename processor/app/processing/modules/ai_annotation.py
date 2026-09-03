"""AI Annotation — 声明式模块:标记工作流启用 AI 辅助标注。

卡片本身**不参与自动执行**(透传上游,秒级完成);它的作用是:
  1. 声明该项目的工作流启用了 AI 辅助标注 → Review 页显示
     "✨ AI Annotate" 按钮(与"手套显示由工作流驱动"同哲学);
  2. 批次处理完成后后台异步跑一次 AI 标注(不阻塞管线,失败可见);
  3. 节点配置(mode/min_confidence)作为该项目的 AI 标注默认参数。

真正的切段 + VLM 标注逻辑在 app/ai_annotation.py(独立服务模块,
不进 DAG),产物写入 state/annotations/<批次>.json 的已确认段
(status="confirmed"),直接进入导出。
"""

from app.processing import ProcessingModule, JobContext, ArtifactRef, field
from app.processing.registry import register
from app.processing.theme import ANNOTATION_COLOR


@register
class AIAnnotationModule(ProcessingModule):
    slug = "ai_annotation"
    version = "1.0"
    category = "process"
    label = "AI Annotation"
    icon = "ant-design:robot-outlined"
    color = ANNOTATION_COLOR
    # Keep the persisted ``data`` handle for old workflows, but make the
    # semantic contract explicit: AI annotation consumes RGB video.
    inputs = ({"key": "data", "label": "RGB Video"},)
    outputs = ({"key": "annotation", "label": "Annotation"},)
    default_config = {"mode": "signal_vlm",
                      "min_confidence": 0.7, "prompt_language": "zh",
                      "max_segments": 50,
                      # VLM 供应商(local = 本地 vLLM;api = 云端厂商接口。
                      # 两模式严格分离互不回退,api 字段见下)
                      "vlm_provider": "local",
                      "api_vendor": "kimi",
                      "api_model": "kimi-k3",
                      "api_key": "",
                      "api_base_url": ""}
    config_schema = (
        field("mode", "select", "Mode", "signal_vlm",
              options=["signal_only", "signal_vlm", "vlm_only"]),
        field("min_confidence", "number", "Min confidence", 0.7,
              min=0, max=1, step=0.05),
        field("prompt_language", "select", "Label language", "zh",
              options=["zh", "en"]),
        # VLM 供应商(provider=api 时由卡片头部 ⚙ 设置弹窗编辑)
        field("vlm_provider", "select", "VLM provider", "local",
              options=["local", "api"]),
        field("api_vendor", "select", "API vendor", "kimi",
              options=["kimi", "qwen", "siliconflow"]),
        field("api_model", "string", "API model", "kimi-k3"),
        field("api_key", "string",
              "API key (enter in the frontend node settings)", ""),
        field("api_base_url", "string",
              "API base URL (blank = vendor default)", ""),
        # P2 切段参数(信号去抖 / 最小段长 / 段数上限)
        field("debounce_sec", "number", "Debounce (s)", 2.0,
              min=0.5, max=10, step=0.5),
        field("min_seg_sec", "number", "Min segment (s)", 0.8,
              min=0.2, max=30, step=0.2),
        field("max_segments", "number", "Max segments (long video)", 50,
              min=1, max=50, step=1),
    )
    execution_target = "server"
    capabilities = ("ai_annotation",)

    def run(self, ctx: JobContext) -> dict[str, ArtifactRef] | None:
        # The actual VLM job is scheduled separately; this node only marks
        # the video input and passes the artifact through. Sensor/keypoint
        # artifacts must not be accepted as the AI annotation source.
        if not any(ref.kind == "video" for ref in ctx.incoming.values()):
            ctx.skip("No RGB video input — skipped")
        return dict(ctx.incoming)
