"""Human Review — 审核门模块:透传上游 artifact。

实际审核在 Video Review 页面完成;模块在 worker 中透传数据,
状态流转由后端(localstore)与审核接口处理。
"""

from app.processing import ProcessingModule, JobContext, ArtifactRef, field
from app.processing.registry import register
from app.processing.theme import REVIEW_COLOR


@register
class HumanReviewModule(ProcessingModule):
    slug = "human_review"
    version = "1.0"
    category = "review"
    label = "Human Review"
    icon = "ant-design:eye-outlined"
    color = REVIEW_COLOR
    inputs = ({"key": "data", "label": "Review Target"},)
    outputs = ({"key": "reviewed", "label": "Reviewed Data"},)
    default_config = {"required": True, "reviewers": 1}
    execution_target = "server"
    capabilities = ("review_gate",)

    def run(self, ctx: JobContext) -> dict[str, ArtifactRef] | None:
        # 透传上游 artifact;审核状态由 Review 页面/API 管理
        if not ctx.incoming:
            ctx.skip("No upstream data — skipped")
        return dict(ctx.incoming)
