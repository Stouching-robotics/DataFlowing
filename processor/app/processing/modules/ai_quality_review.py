"""AI Quality Review — lightweight media and annotation quality gate.

The actual check runs after the workflow result is persisted.  It checks video
decode/frame continuity/black screens/freezes and, when connected after AI
Annotation, annotation coverage too.  This node keeps the normal artifact
contract so existing worker graphs remain compatible.  The service-level gate
moves failed/incomplete batches to the existing Reviewing queue and only
allows a passed result to become Approved.
"""

from app.processing import ProcessingModule, JobContext, ArtifactRef
from app.processing.registry import register
from app.processing.theme import REVIEW_COLOR


@register
class AIQualityReviewModule(ProcessingModule):
    slug = "ai_quality_review"
    version = "1.0"
    category = "review"
    label = "AI Quality Review"
    icon = "ant-design:robot-outlined"
    color = REVIEW_COLOR
    inputs = ({"key": "data", "label": "Quality Review Target"},)
    outputs = ({"key": "reviewed", "label": "Reviewed Data"},)
    default_config = {"mode": "gate"}
    execution_target = "server"
    capabilities = ("quality_review",)

    def run(self, ctx: JobContext) -> dict[str, ArtifactRef] | None:
        # The workflow executor still receives the upstream artifact.  The
        # episode-level gate is applied by the AI annotation service after its
        # persisted ranges have been validated.
        if not ctx.incoming:
            ctx.skip("No upstream data — skipped")
        return dict(ctx.incoming)
