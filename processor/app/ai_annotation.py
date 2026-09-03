"""AI 辅助标注服务 —— 信号切段 + VLM 标注 + 一致性打分。

独立于工作流 DAG:由 Review 页 "AI Annotate" 按钮按需触发(或
ai_annotation 卡片存在时后台触发)。产物 = 写入
    state/annotations/<episode>.json 的已确认段(status="confirmed",
    source="ai");AI 成功后按工作流中的 AI Quality Review 卡片决定是否
    自动进入 Approved/导出，不显示 Confirm。

信号源(全部来自批次已有产物,零模型):
  - hand_3d parquet:hand_0/1_fingers bitmask 跳变(fist↔open = 抓取/松开)
  - 主数据 parquet:observation.tactile.left/right 压力起落(接触事件)
VLM(可选):Qwen3-VL @ 127.0.0.1:8001,默认 MP4 video_url + FPS-aware sampling → 严格 JSON 标注。
一致性:信号段与 VLM 段时间交叠打分 → ai_score(低分 = 人工优先复核)。
"""

from __future__ import annotations

import asyncio
import base64
import glob
import json
import os
import re
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import httpx
from fastapi import APIRouter, HTTPException

from app.localstore import (
    get_episode,
    list_annotations,
    mutate_annotations,
    list_projects,
    get_workflow,
    list_runs,
    STATE_ROOT,
    read_episode_state,
    write_episode_state,
)
from app.routes.annotations import notify_annotations_changed
from app.video_quality import check_video_quality_async
from app.workflow_types import HAND_PROCESS_TYPES, canonical_node_type

router = APIRouter(prefix="/api/v1", tags=["ai-annotation"])

# 部署脚本(deploy.py)换端口避让时通过 EGODATA_VLLM_URL 让标注请求跟随
VLLM_URL = os.getenv(
    "EGODATA_VLLM_URL",
    "http://127.0.0.1:8001/v1/chat/completions",
)
VLLM_MODEL = os.getenv(
    "EGODATA_VLLM_MODEL",
    str(Path(__file__).resolve().parents[1]
        / "models" / "llm" / "Qwen3-VL-8B-Instruct-FP8"),
)
VLLM_MAX_FRAMES = 32       # 抽帧上限(显存/上下文约束)
VLLM_FRAME_W, VLLM_FRAME_H = 512, 320
# 本地 Qwen3-VL 默认直接接收 MP4。需要兼容旧版/非视频模型时可设置
# EGODATA_VLLM_MEDIA_MODE=frames 临时回退到旧的客户端抽帧链路。
VLLM_MEDIA_MODE = os.getenv("EGODATA_VLLM_MEDIA_MODE", "video").strip().lower()
try:
    VLLM_VIDEO_SAMPLE_FPS = min(
        30.0, max(0.1, float(os.getenv("EGODATA_VLLM_VIDEO_SAMPLE_FPS", "2"))))
except (TypeError, ValueError):
    VLLM_VIDEO_SAMPLE_FPS = 2.0
LOCAL_INLINE_VIDEO_MAX_BYTES = 120 * 1024 * 1024
DEFAULT_MAX_VLM_SEGMENTS = 50  # 长时第一人称视频的动作阶段上限
MERGE_WINDOW_SEC = 0.6     # 双手/触觉事件合并窗口
MIN_SEG_SEC = 0.8          # 短于该时长的段丢弃
DEBOUNCE_SEC = 2.0         # 状态翻转短于该时长的成对事件 = 抖动,丢弃
# 租约代理不一定实现 Kimi Files API。Kimi 的 OpenAI 兼容接口还支持
# data:video/mp4;base64,... 直接传视频；给 JSON 预留少量开销后限制在
# 80 MiB，超出时明确失败，不切换到本地模型。
API_INLINE_VIDEO_MAX_BYTES = 80 * 1024 * 1024

try:
    API_MIN_REQUEST_INTERVAL_SEC = max(
        0.0, float(os.getenv("EGODATA_API_MIN_INTERVAL_SEC", "10.0")))
except (TypeError, ValueError):
    API_MIN_REQUEST_INTERVAL_SEC = 10.0

# API 视频推理不能无限期占住一个后台任务。180 秒给云端模型留下足够
# 生成时间，同时让失败的分段可以尽快落成 pending_retry。local vLLM
# 仍使用原来的 600 秒超时，两个供应商路径互不影响。
try:
    API_REQUEST_TIMEOUT_SEC = min(
        180.0, max(120.0, float(os.getenv("EGODATA_API_TIMEOUT_SEC", "180"))))
except (TypeError, ValueError):
    API_REQUEST_TIMEOUT_SEC = 180.0

_API_RATE_LIMIT_LOCK = asyncio.Lock()
_API_LAST_REQUEST_AT = 0.0

# ── API 供应商(协议 = OpenAI 兼容 Chat Completions)。与本地 vLLM 严格分离:
#    provider=api 只打厂商接口,配置缺失/请求失败显式报错,绝不回退 local。
_API_VENDOR_BASE = {
    "kimi": "https://api.moonshot.ai/v1",
    # DashScope OpenAI-compatible endpoint. Users may override this with a
    # workspace-specific Beijing endpoint in the frontend.
    "qwen": "https://dashscope.aliyuncs.com/compatible-mode/v1",
    # SiliconFlow 硅基流动:OpenAI 兼容端点,托管 Qwen3-VL-30B-A3B-Instruct/
    # -Thinking 等开源多模态模型(Qwen/Qwen3-VL-*)
    "siliconflow": "https://api.siliconflow.cn/v1",
}


class VLMAnnotationError(RuntimeError):
    """A user-facing, non-secret failure from a VLM annotation request.

    The previous implementation collapsed transport failures and invalid model
    output into an empty list.  That made a stopped/not-ready model look the
    same as a model that answered with an invalid plan.  Keep a stable machine
    code for the task file and a short detail for the UI.
    """

    def __init__(self, code: str, detail: str):
        self.code = str(code or "vlm_failed")
        self.detail = str(detail or "VLM 标注失败")
        super().__init__(self.detail)


def _safe_error_text(value: object, limit: int = 240) -> str:
    """Shorten provider errors without exposing credentials in task state."""
    text = re.sub(r"\s+", " ", str(value or "")).strip()
    text = re.sub(
        r"(?i)(authorization|api[_ -]?key|token|secret)\s*[:=]\s*\S+",
        r"\1=<redacted>", text,
    )
    return text[:limit]


def _model_text_parts(value: object) -> list[str]:
    """Normalize OpenAI-compatible text content into plain strings.

    Kimi-compatible gateways may return content as a string, a list of text
    blocks, or leave ``content`` empty while putting the answer in
    ``reasoning_content``/``reasoning``. Keep this helper transport-agnostic
    and never include binary/media fields in the extracted text.
    """
    if isinstance(value, str):
        return [value] if value.strip() else []
    if isinstance(value, list):
        parts: list[str] = []
        for item in value:
            if isinstance(item, str):
                if item.strip():
                    parts.append(item)
            elif isinstance(item, dict):
                text = item.get("text")
                if isinstance(text, str) and text.strip():
                    parts.append(text)
        return parts
    if isinstance(value, dict):
        text = value.get("text") or value.get("content")
        return _model_text_parts(text)
    return []


def _model_text_candidates(message: object,
                           include_reasoning: bool = False) -> list[str]:
    """Return response text candidates, preferring final content.

    ``reasoning_content``/``reasoning`` are only used as compatibility
    fallbacks for API gateways. Callers still validate the extracted text as
    JSON before accepting a reasoning field as an annotation result.
    """
    if not isinstance(message, dict):
        return []
    candidates: list[str] = []
    keys = ["content", "reasoning_content"]
    if include_reasoning:
        # Some OpenAI-compatible rental gateways expose the model's final
        # stream under ``reasoning`` instead of ``reasoning_content``.
        keys.append("reasoning")
    for key in keys:
        for text in _model_text_parts(message.get(key)):
            if text not in candidates:
                candidates.append(text)
    return candidates


def _select_model_text(candidates: list[str], prefer_plan: bool = False) -> str:
    """Select a usable model response from content/reasoning candidates."""
    if prefer_plan:
        for text in candidates:
            if _has_plan_scene_contract(text):
                return text
    return next((text for text in candidates if text.strip()), "")


def _json_dict_candidates(raw: object) -> list[dict]:
    """Extract JSON objects from plain, fenced, or prose-wrapped output.

    ``JSONDecoder.raw_decode`` is used at every object start so nested
    ``scene``/``subtasks`` objects are handled without a fragile first/last
    brace regex.
    """
    text = str(raw or "").strip()
    if not text:
        return []
    variants = [text]
    without_thinking = re.sub(r"<think>.*?</think>", "", text,
                              flags=re.IGNORECASE | re.DOTALL).strip()
    if without_thinking and without_thinking != text:
        variants.insert(0, without_thinking)
    decoder = json.JSONDecoder()
    result: list[dict] = []
    seen: set[str] = set()
    for variant in variants:
        try:
            value = json.loads(variant)
            if isinstance(value, dict):
                key = json.dumps(value, ensure_ascii=False, sort_keys=True)
                if key not in seen:
                    result.append(value)
                    seen.add(key)
        except (TypeError, ValueError, json.JSONDecodeError):
            pass
        for index, char in enumerate(variant):
            if char != "{":
                continue
            try:
                value, _ = decoder.raw_decode(variant, index)
            except (TypeError, ValueError, json.JSONDecodeError):
                continue
            if isinstance(value, dict):
                key = json.dumps(value, ensure_ascii=False, sort_keys=True)
                if key not in seen:
                    result.append(value)
                    seen.add(key)
    return result


def _api_response_text(response: httpx.Response, *, prefer_plan: bool = False,
                       require_json: bool = False) -> tuple[str, str]:
    """Extract a validated candidate from an API chat response.

    Providers differ on the OpenAI-compatible response shape. The normal
    final answer remains ``message.content``; ``reasoning_content`` and the
    rental gateway's ``reasoning`` alias are compatibility fallbacks only.
    A fallback is accepted only when it contains a parseable JSON object, so
    hidden/free-form reasoning is never written as an annotation.
    """
    payload = response.json()
    choice = payload["choices"][0]
    message = choice["message"]
    finish_reason = str(choice.get("finish_reason") or "")
    candidates = _model_text_candidates(message, include_reasoning=True)
    if prefer_plan:
        return _select_model_text(candidates, prefer_plan=True), finish_reason
    if require_json:
        for candidate in candidates:
            if _json_dict_candidates(candidate):
                return candidate, finish_reason
        return "", finish_reason
    return _select_model_text(candidates), finish_reason


def _plan_payload(raw: object) -> dict | None:
    """Choose the JSON object that contains the full-video plan contract."""
    candidates = _json_dict_candidates(raw)
    for data in candidates:
        scene = data.get("scene")
        rows = _plan_rows(data)
        if (isinstance(scene, dict)
                and isinstance(scene.get("objects"), list)
                and isinstance(scene.get("locations"), list)
                and isinstance(rows, list) and rows):
            return data
    for data in candidates:
        scene = data.get("scene")
        if (isinstance(scene, dict)
                and isinstance(scene.get("objects"), list)
                and isinstance(scene.get("locations"), list)):
            return data
    return candidates[0] if candidates else None


def _plan_rows(data: dict) -> list:
    """Return action rows from common local-VLM naming variants.

    The prompt uses ``subtasks``, but local Qwen/vLLM deployments are often
    served with a generic instruction-following template and may emit
    ``actions``, ``steps`` or ``phases`` instead.  These are equivalent at
    this boundary; accepting them avoids throwing away an otherwise valid
    visual plan merely because of a field-name variation.
    """
    for key in ("subtasks", "segments", "actions", "steps", "phases",
                "timeline", "action_segments", "action_plan"):
        value = data.get(key)
        if isinstance(value, dict):
            value = value.get("items") or value.get("rows")
        if isinstance(value, list) and value:
            return value
    task = data.get("task")
    if isinstance(task, dict):
        for key in ("subtasks", "segments", "actions", "steps", "phases"):
            value = task.get(key)
            if isinstance(value, list) and value:
                return value
    return []


def _http_error_detail(response: httpx.Response | None) -> str:
    """Extract a bounded provider error message from an HTTP response."""
    if response is None:
        return ""
    detail = ""
    try:
        payload = response.json()
        if isinstance(payload, dict):
            error = payload.get("error")
            if isinstance(error, dict):
                detail = str(error.get("message") or error.get("detail") or "")
            elif isinstance(error, str):
                detail = error
            if not detail:
                detail = str(payload.get("message") or payload.get("detail") or "")
    except Exception:
        detail = ""
    return _safe_error_text(detail)


def _plan_contract_failure(raw: str) -> tuple[str, str]:
    """Return (error_code, detail) for an invalid full-video VLM plan."""
    data = _plan_payload(raw)
    if not isinstance(data, dict):
        if not str(raw or "").strip():
            return "invalid_plan_json", "VLM 返回内容为空"
        return "invalid_plan_json", "VLM 返回内容不是有效 JSON"
    scene = data.get("scene")
    if not isinstance(scene, dict):
        return "invalid_plan_contract", "VLM 返回缺少 scene 对象"
    if not isinstance(scene.get("objects"), list):
        return "invalid_plan_contract", "VLM 返回缺少 scene.objects 数组"
    if not isinstance(scene.get("locations"), list):
        return "invalid_plan_contract", "VLM 返回缺少 scene.locations 数组"
    return "empty_plan", "VLM 返回了场景信息，但没有有效 subtasks 动作计划"


def _vlm_endpoint(vlm_cfg: dict | None) -> tuple[str, str, str, dict, str | None]:
    """解析 VLM 调用目标 → (chat_url, upload_base, model, headers, error)。

    严格分离:provider=api 只打厂商接口,配置缺失直接返回 error 文案
    (调用方写进任务 detail),绝不回退本地 vLLM;provider≠api 返回本地
    vLLM 常量(现状,零行为变化)。
    """
    cfg = vlm_cfg or {}
    if str(cfg.get("vlm_provider") or "local") != "api":
        return VLLM_URL, "", VLLM_MODEL, {}, None
    vendor = str(cfg.get("api_vendor") or "kimi")
    base = str(cfg.get("api_base_url") or "").strip().rstrip("/") \
        or _API_VENDOR_BASE.get(vendor, "")
    if not base:
        return "", "", "", {}, (
            f"API 模式缺少 base URL(vendor={vendor} 无预设,请在卡片填 api_base_url)")
    model = str(cfg.get("api_model") or "").strip()
    if not model:
        return "", "", "", {}, "API 模式缺少模型名(卡片 api_model 为空)"
    # API Key must come from the API settings entered in the workflow node.
    # Do not fall back to a server-wide environment variable: different users
    # and workflows must be able to use their own provider credentials.
    key = str(cfg.get("api_key") or "").strip()
    if not key:
        return "", "", "", {}, "API 模式缺少 Key:请在前端 AI Annotation 节点中填写 API key"
    return (base + "/chat/completions", base, model,
            {"Authorization": f"Bearer {key}"}, None)


@router.post("/ai-annotation/test-connection")
async def test_ai_annotation_connection(body: dict):
    """Test a provider configuration without saving it.

    Prefer the provider's model-list endpoint, but fall back to a tiny text
    completion for providers (notably DashScope) that do not expose a useful
    ``/models`` endpoint. No video is uploaded by this check.
    """
    cfg = {
        "vlm_provider": "api",
        "api_vendor": body.get("api_vendor"),
        "api_model": body.get("api_model"),
        "api_key": body.get("api_key"),
        "api_base_url": body.get("api_base_url"),
    }
    chat_url, base_url, model, headers, error = _vlm_endpoint(cfg)
    if error:
        if "Key" in error:
            return {"ok": False, "message": "API key required"}
        if "model" in error.lower():
            return {"ok": False, "message": "Model required"}
        return {"ok": False, "message": "API settings invalid"}

    try:
        timeout = httpx.Timeout(15.0, connect=5.0)
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.get(f"{base_url}/models", headers=headers)
            # Some OpenAI-compatible gateways do not expose /models. A tiny
            # text request still validates DNS/TLS, the key, and the model.
            if response.status_code in (404, 405):
                ping_payload = {
                    "model": model,
                    "messages": [{"role": "user", "content": "Reply OK."}],
                    "max_tokens": 8,
                    "temperature": 0,
                }
                if str(body.get("api_vendor") or "").strip() == "qwen":
                    ping_payload["enable_thinking"] = False
                response = await client.post(
                    chat_url, headers=headers, json=ping_payload)
    except httpx.TimeoutException:
        return {"ok": False, "message": "Connection timeout"}
    except httpx.RequestError as exc:
        return {"ok": False, "message": f"Connection failed ({exc.__class__.__name__})"}

    if response.status_code >= 400:
        detail = ""
        try:
            payload = response.json()
            err_payload = payload.get("error") if isinstance(payload, dict) else None
            if isinstance(err_payload, dict):
                detail = str(err_payload.get("message") or "")
            elif isinstance(err_payload, str):
                detail = err_payload
            elif isinstance(payload, dict):
                detail = str(payload.get("message") or "")
        except Exception:
            pass
        # Never echo request headers or the API key in the response.
        suffix = f": {detail[:180]}" if detail else ""
        return {"ok": False,
                "message": f"API error ({response.status_code}){suffix}"}

    try:
        payload = response.json()
        rows = payload.get("data") if isinstance(payload, dict) else None
        model_ids = {
            str(row.get("id")) for row in (rows or [])
            if isinstance(row, dict) and row.get("id")
        }
        # A successful chat-completion fallback is valid even when the
        # provider has no model-list response.
        if not model_ids and isinstance(payload, dict):
            choices = payload.get("choices")
            if isinstance(choices, list) and choices:
                return {"ok": True, "message": f"Connected: {model}"}
    except Exception:
        return {"ok": False, "message": "Invalid API response"}

    if model_ids and model not in model_ids:
        return {"ok": False,
                "message": f"Model unavailable: {model}",
                "model_available": False}
    return {"ok": True,
            "message": f"Connected: {model}",
            "model_available": True}


def _tmp_dir(task_id: str) -> Path:
    """临时切片目录:项目 data/tmp 下(需求 §8,不写 C 盘),任务粒度。"""
    from app.config import settings
    d = Path(settings.storage_root) / "tmp" / f"ai_vlm_{task_id}"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _cleanup_tmp(task_id: str) -> None:
    """清理该任务的临时切片目录(只算路径、不创建;任务结束必调,
    成功/失败都清;目录不存在时静默)。"""
    from app.config import settings
    import shutil
    shutil.rmtree(Path(settings.storage_root) / "tmp" / f"ai_vlm_{task_id}",
                  ignore_errors=True)


_CANDIDATE_COLOR = "#22d3ee"

# ── 任务状态(内存 + 磁盘镜像;服务热重载后可从磁盘读到
#    最后一次状态,前端把"进行中被重启打断"显示为 interrupted)──
_tasks: dict[str, dict] = {}
_episode_ai_generation: dict[str, int] = {}
_TASKS_DIR = STATE_ROOT / "ai_tasks"

# ── VLM 调用记账 ─────────────────────────────────────────
# 每次 API/本地 VLM 调用的 usage(tokens)、耗时、成败原因追加到
# data/logs/vlm_calls.jsonl:基准排行与真实标注共用,按模型核算花费。
_VLM_CALLS_LOG = STATE_ROOT.parent / "logs" / "vlm_calls.jsonl"


def _usage_from_response(response: httpx.Response) -> dict | None:
    try:
        usage = (response.json() or {}).get("usage")
        return usage if isinstance(usage, dict) else None
    except Exception:
        return None


def _record_vlm_call(entry: dict) -> None:
    """Append one VLM call record. Never raises — logging must not break annotation."""
    try:
        entry["ts"] = datetime.now(timezone.utc).isoformat()
        _VLM_CALLS_LOG.parent.mkdir(parents=True, exist_ok=True)
        with _VLM_CALLS_LOG.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except Exception:
        pass


def _task_file(episode_id: str) -> Path:
    return _TASKS_DIR / f"{episode_id}.json"


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def _annotation_quality_report(episode_id: str, total_frames: int,
                               annotations: list[dict] | None = None) -> dict:
    """Validate AI annotation coverage without judging left/right identity.

    Annotation ranges use inclusive frame indexes.  The first quality gate is
    intentionally small: every frame in the episode must be covered once the
    AI pipeline has finished, and no segment may still be pending or carry an
    error.  Overlapping ranges are reported for diagnostics but do not fail a
    batch by themselves; a gap is the condition that makes the batch unsafe
    for automatic approval.
    """
    total = max(0, int(total_frames or 0))
    rows = annotations if annotations is not None else list_annotations(episode_id)
    if not isinstance(rows, list):
        rows = []
    report = {
        "passed": False,
        "total_frames": total,
        "covered_frames": 0,
        "coverage_ratio": 0.0,
        "missing_ranges": [],
        "invalid_ranges": [],
        "pending_segments": [],
        "overlap_ranges": [],
    }
    if total <= 0:
        report["reason"] = "frame_count_unknown"
        return report

    ranges: list[tuple[int, int, int]] = []
    covered: list[tuple[int, int]] = []
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            report["invalid_ranges"].append(index)
            continue
        segment_index = row.get("ai_segment_index", index)
        try:
            segment_index = int(segment_index)
        except (TypeError, ValueError):
            segment_index = index
        if (row.get("ai_retry_pending")
                or row.get("status") in {"pending", "pending_retry", "running"}
                or str(row.get("ai_error") or "").strip()):
            report["pending_segments"].append(segment_index)
        try:
            start = int(row["start_frame_index"])
            end = int(row["end_frame_index"])
        except (KeyError, TypeError, ValueError):
            report["invalid_ranges"].append(segment_index)
            continue
        if (not str(row.get("label") or "").strip()
                or start < 0 or end < start or end >= total):
            report["invalid_ranges"].append(segment_index)
            continue
        ranges.append((start, end, segment_index))

    ranges.sort(key=lambda item: (item[0], item[1], item[2]))
    expected = 0
    for start, end, segment_index in ranges:
        if start > expected:
            report["missing_ranges"].append([expected, start - 1])
        if start < expected:
            report["overlap_ranges"].append([start, min(end, expected - 1), segment_index])
        expected = max(expected, end + 1)
        covered.append((start, end))
    if expected < total:
        report["missing_ranges"].append([expected, total - 1])

    # Merge the union for an accurate coverage ratio, independent of overlap.
    union_end = -1
    covered_frames = 0
    for start, end in sorted(covered):
        if start > union_end + 1:
            covered_frames += end - start + 1
        elif end > union_end:
            covered_frames += end - union_end
        union_end = max(union_end, end)
    report["covered_frames"] = covered_frames
    report["coverage_ratio"] = round(covered_frames / total, 6)
    report["passed"] = bool(
        ranges
        and not report["missing_ranges"]
        and not report["invalid_ranges"]
        and not report["pending_segments"]
    )
    report["reason"] = "passed" if report["passed"] else "annotation_incomplete"
    return report


def _set_ai_quality_state(episode_id: str, report: dict, *,
                          auto_approve: bool = False) -> None:
    """Persist the internal quality result while exposing only old UI states.

    The web UI continues to use only Reviewing/Approved.  The detailed report
    is kept in episode state for diagnostics and future review tooling.
    """
    state = read_episode_state(episode_id)
    state["ai_quality_status"] = "passed" if report.get("passed") else "failed"
    state["ai_quality_report"] = report
    state["updated_at"] = _utcnow()
    current = str(state.get("status") or "")
    if report.get("passed") and auto_approve:
        if current in {"processing", "to_review", "completed", "failed"}:
            state["status"] = "reviewed"
            if not state.get("approved_at"):
                state["approved_at"] = _utcnow()
    elif not report.get("passed"):
        # Fail closed: an incomplete/failed AI result remains in Reviewing.
        if current in {"processing", "completed", "reviewed", "approved", "failed"}:
            state["status"] = "to_review"
            state["approved_at"] = None
    write_episode_state(episode_id, state)


def _set_ai_quality_pending(episode_id: str) -> None:
    """Move a batch back to Reviewing before a new AI attempt starts."""
    state = read_episode_state(episode_id)
    state["ai_quality_status"] = "running"
    state["ai_quality_report"] = {
        "passed": False,
        "reason": "ai_annotation_running",
    }
    state["updated_at"] = _utcnow()
    if state.get("status") in {"processing", "reviewed", "approved", "completed", "failed"}:
        state["status"] = "to_review"
        state["approved_at"] = None
    write_episode_state(episode_id, state)


def _record_ai_quality_failure(episode_id: str, total_frames: int,
                               reason: str) -> dict:
    """Record a failed gate without introducing a new user-facing status."""
    report = _annotation_quality_report(episode_id, total_frames)
    report["passed"] = False
    report["reason"] = str(reason or "annotation_incomplete")
    _set_ai_quality_state(episode_id, report)
    return report


# ═══════════════════════════════════════════════════════════
#  数据源定位
# ═══════════════════════════════════════════════════════════

_REAL_FPS_CACHE: dict[str, float] = {}


def _real_episode_fps(episode_id: str, episode_dir: Path) -> float:
    """实测视频 fps(带缓存;失败返回 0,调用方回退 ep 元数据)。"""
    if episode_id in _REAL_FPS_CACHE:
        return _REAL_FPS_CACHE[episode_id]
    fps = 0.0
    try:
        import cv2
        mp4 = _first_video(episode_dir, episode_id)
        if mp4 is not None:
            cap = cv2.VideoCapture(str(mp4))
            try:
                fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
            finally:
                cap.release()
    except Exception:
        fps = 0.0
    _REAL_FPS_CACHE[episode_id] = fps
    return fps


def _episode_dir(episode_id: str) -> Path | None:
    ep = get_episode(episode_id)
    if ep is None:
        return None
    # localstore already resolves canonical project-level datasets to the
    # project root.  Searching recursively for the episode ID would return
    # meta/collector/<episode_id> instead of the dataset root.
    path = Path(ep.get("path") or "")
    return path if path.is_dir() else None


def _hand3d_parquets(episode_dir: Path, episode_id: str | None = None) -> list[Path]:
    # Current processing results are merged into the canonical episode data
    # parquet.  Keep that as the first source; historical run sidecars are
    # only a compatibility fallback for episodes processed by older workers.
    canonical = None
    if episode_id:
        try:
            from app.artifact_resolver import _canonical_episode_data
            canonical = _canonical_episode_data(episode_dir, episode_id, "3d")
        except Exception:
            canonical = None
    if canonical is not None:
        return [canonical]
    return []


def _hand3d_parquet(episode_dir: Path, episode_id: str | None = None) -> Path | None:
    """Backward-compatible canonical accessor for older callers."""
    values = _hand3d_parquets(episode_dir, episode_id)
    return values[0] if values else None


def _episode_data_index(episode_dir: Path, episode_id: str | None = None) -> int | None:
    if not episode_id:
        return None
    try:
        from app.project_dataset import episode_row
        row = episode_row(episode_dir, str(episode_id))
        return int(row["episode_index"]) if row and row.get("episode_index") is not None else None
    except (TypeError, ValueError, OSError):
        return None


def _main_parquet(episode_dir: Path, episode_id: str | None = None) -> Path | None:
    index = _episode_data_index(episode_dir, episode_id)
    candidates = sorted(episode_dir.glob("data/**/*.parquet"), reverse=True)
    if index is not None:
        candidates = [p for p in candidates
                      if p.name == f"episode_{index:06d}.parquet"]
    for p in candidates:
        if p.is_file():
            return p
    return None


def _first_video(episode_dir: Path, episode_id: str | None = None) -> Path | None:
    """Choose a stable canonical raw video for the single-video VLM API.

    Mono episodes have one candidate. Stereo episodes are frame-aligned, so
    the left/primary view is selected explicitly instead of relying on path
    sorting. Generated skeleton videos are never selected as raw input.
    """
    videos = [p for p in episode_dir.glob("videos/**/*.mp4")
              if "skeleton" not in str(p).lower()]
    index = _episode_data_index(episode_dir, episode_id)
    if index is not None:
        videos = [p for p in videos if p.stem == f"episode_{index:06d}"]
    if not videos:
        return None

    def priority(path: Path) -> tuple[int, str]:
        text = str(path).lower().replace("-", "_")
        if re.search(r"(^|[_/])stereo_left([_/\.])", text) or \
                re.search(r"(^|[_/])left([_/\.])", text):
            return 0, text
        if any(token in text for token in ("primary", "main")):
            return 1, text
        if any(token in text for token in ("rgb", "color")):
            return 2, text
        return 3, text

    return min(videos, key=priority)


def _ai_video_streams(episode_dir: Path, episode_id: str | None = None) -> list[tuple[str, Path]]:
    """Return the RGB camera streams belonging to this episode.

    LeRobot v2.1 stores videos below ``videos/<source>/chunk-XXX``.  The
    previous directory walk treated ``chunk-XXX`` as the camera name and
    consequently selected only the first stereo file.  Use the canonical
    v2.1 iterator so both stereo views are discovered and synchronized.
    """
    root = episode_dir / "videos"
    if not root.is_dir():
        return []

    from app.lerobot_v21 import is_depth_source, iter_video_streams

    streams: list[tuple[str, Path]] = []
    index = _episode_data_index(episode_dir, episode_id)
    expected_stem = f"episode_{index:06d}" if index is not None else None
    seen: set[str] = set()
    for source, path in iter_video_streams(root):
        source = str(source)
        if source in seen:
            continue
        lower = source.lower()
        if expected_stem is not None and path.stem != expected_stem:
            continue
        if lower.endswith("_aux") or "_aux" in lower:
            continue
        if is_depth_source(source):
            continue
        if "skeleton" in lower or "skeleton" in str(path).lower():
            continue
        if path.is_file():
            streams.append((source, path))
            seen.add(source)
    return streams


def _build_multiview_video(episode_dir: Path, out_dir: Path,
                           fps: float, total_frames: int
                           , episode_id: str | None = None
                           ) -> tuple[Path | None, list[str], str]:
    """Build a time-synchronized contact video for multi-device AI labeling.

    The output is a temporary review input only.  Original videos stay
    separate, while the VLM receives all available views on one shared frame
    timeline and therefore produces one episode-level annotation set.
    """
    streams = _ai_video_streams(episode_dir, episode_id)
    if len(streams) <= 1:
        return (streams[0][1] if streams else None,
                [streams[0][0]] if streams else [], "")

    import cv2
    import math
    import numpy as np

    out_dir.mkdir(parents=True, exist_ok=True)
    output = out_dir / "ai_multiview.mp4"
    caps = []
    source_info = []
    writer = None
    try:
        for source, path in streams:
            cap = cv2.VideoCapture(str(path))
            if not cap.isOpened():
                cap.release()
                continue
            source_fps = float(cap.get(cv2.CAP_PROP_FPS) or fps or 30.0)
            count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            caps.append(cap)
            source_info.append({"source": source, "path": path, "fps": source_fps,
                                "count": count, "index": -1, "frame": None})
        if len(caps) <= 1:
            return (source_info[0]["path"] if source_info else None,
                    [source_info[0]["source"]] if source_info else [], "")

        n_frames = int(total_frames or 0)
        if n_frames <= 0:
            n_frames = max((int(item["count"]) for item in source_info), default=0)
        if n_frames <= 0:
            return None, [], ""
        cols = 2 if len(source_info) <= 4 else 3
        rows = int(math.ceil(len(source_info) / cols))
        tile_w, tile_h = 512, 320
        canvas_w, canvas_h = cols * tile_w, rows * tile_h
        writer = cv2.VideoWriter(
            str(output), cv2.VideoWriter_fourcc(*"mp4v"),
            float(fps or 30.0), (canvas_w, canvas_h))
        if not writer.isOpened():
            return None, [], ""

        for frame_index in range(n_frames):
            timestamp = frame_index / max(1.0, float(fps or 30.0))
            canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
            for slot, (cap, info) in enumerate(zip(caps, source_info)):
                target_index = int(round(timestamp * info["fps"]))
                while info["index"] < target_index:
                    ok, frame = cap.read()
                    info["index"] += 1
                    if not ok:
                        info["frame"] = None
                        break
                    info["frame"] = frame
                tile = np.zeros((tile_h, tile_w, 3), dtype=np.uint8)
                if info["frame"] is not None:
                    tile = cv2.resize(info["frame"], (tile_w, tile_h),
                                      interpolation=cv2.INTER_AREA)
                cv2.rectangle(tile, (0, 0), (tile_w, 28), (0, 0, 0), -1)
                cv2.putText(tile, str(info["source"]), (8, 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255),
                            1, cv2.LINE_AA)
                row, col = divmod(slot, cols)
                canvas[row * tile_h:(row + 1) * tile_h,
                       col * tile_w:(col + 1) * tile_w] = tile
            writer.write(canvas)
        if not output.is_file() or output.stat().st_size < 1024:
            return None, [], ""
        labels = [str(info["source"]) for info in source_info]
        note = ("这是同一批次多个设备的同步多视角画面。每个画面顶部有设备名；"
                "请综合所有视角判断同一个动作，只输出一套连续的 episode 动作时间段，"
                "不要因为多个摄像头重复同一个动作而生成重复段。视角包括："
                + ", ".join(labels))
        return output, labels, note
    finally:
        if writer is not None:
            writer.release()
        for cap in caps:
            cap.release()


# ═══════════════════════════════════════════════════════════
#  ① 信号切段(零模型)
# ═══════════════════════════════════════════════════════════

def _signal_events(episode_dir: Path, fps: float,
                   debounce_sec: float = 2.0,
                   episode_id: str | None = None) -> list[dict]:
    """返回事件列表 [{frame, kind, reason}],按帧排序。

    kind: "grasp"(手指收拢/压力上升) / "open"(手指张开/压力下降)。
    """
    import numpy as np
    import pandas as pd

    events: list[dict] = []

    def add(fi: float, kind: str, reason: str) -> None:
        events.append({"frame": max(0, int(round(fi))), "kind": kind,
                       "reason": reason})

    # a) 手势 bitmask 跳变:0(fist) ↔ >0(伸指)
    for hp in _hand3d_parquets(episode_dir, episode_id):
        try:
            df = pd.read_parquet(hp)
        except Exception:
            df = None
        if df is not None and len(df):
            source_name = hp.stem
            for slot in ("hand_0", "hand_1"):
                col = f"{slot}_fingers"
                if col not in df.columns:
                    continue
                mask = df[col].fillna(-1).astype(int).values
                prev = None
                for fi, v in zip(df["frame_index"].values, mask):
                    v = 0 if v == 0 else (1 if v > 0 else None)
                    if v is None or prev is None:
                        prev = v
                        continue
                    if v != prev:
                        if v == 0:
                            add(fi, "grasp", f"{source_name}/{slot} fingers closed (fist)")
                        else:
                            add(fi, "open", f"{source_name}/{slot} fingers extended")
                    prev = v

    # b) 触觉压力起落(16×16 压力阵列均值)
    mp = _main_parquet(episode_dir, episode_id)
    if mp is not None:
        try:
            mdf = pd.read_parquet(mp)
        except Exception:
            mdf = None
        if mdf is not None and len(mdf):
            for side in ("left", "right"):
                col = f"observation.tactile.{side}"
                if col not in mdf.columns:
                    continue
                series = mdf[col].apply(
                    lambda v: float(np.nanmean(np.asarray(v, dtype=np.float64)))
                    if v is not None else np.nan)
                series = series.interpolate(limit_direction="both").fillna(0.0)
                th = float(series.max()) * 0.15 + 1e-6
                frames = (mdf.index if "frame_index" not in mdf.columns
                          else mdf["frame_index"]).values
                pressed = series.values > th
                prev = None
                for fi, p in zip(frames, pressed):
                    if prev is None:
                        prev = p
                        continue
                    if p != prev:
                        add(fi, "grasp" if p else "open",
                            f"tactile.{side} pressure {'on' if p else 'off'}")
                    prev = p

    # 合并窗口内同类事件;最终按帧排序
    events.sort(key=lambda e: e["frame"])
    merged: list[dict] = []
    win = max(1, int(MERGE_WINDOW_SEC * fps))
    for e in events:
        if merged and e["frame"] - merged[-1]["frame"] <= win:
            if e["kind"] != merged[-1]["kind"]:
                # 窗口内方向冲突 → 取后到者(最新状态)
                merged[-1] = e
            # 同类事件窗口内合并:保留首个事件的 reason(避免噪音)
        else:
            merged.append(dict(e))

    # 去抖:方向相反且间隔 < DEBOUNCE_SEC 的事件对 = 手部小动作抖动,
    # 成对丢弃(否则 60 分钟视频会切出几百个 1-3 秒碎段)
    debounce = max(1, int(max(0.5, debounce_sec) * fps))
    out: list[dict] = []
    i = 0
    while i < len(merged):
        if (i + 1 < len(merged)
                and merged[i + 1]["kind"] != merged[i]["kind"]
                and merged[i + 1]["frame"] - merged[i]["frame"] < debounce):
            i += 2  # 抖动对,双双丢弃
            continue
        out.append(merged[i])
        i += 1
    return out


def _cap_segments(segs: list[dict], max_n: int) -> list[dict]:
    """段数超过上限 → 反复合并最短的相邻段(只合不拆)。"""
    out = [dict(s) for s in segs]
    while len(out) > max_n:
        best_i, best_len = 0, None
        for i in range(len(out) - 1):
            length = out[i + 1]["end_frame_index"] - out[i]["start_frame_index"]
            if best_len is None or length < best_len:
                best_i, best_len = i, length
        a, b = out[best_i], out[best_i + 1]
        a["end_frame_index"] = b["end_frame_index"]
        a["ai_reason"] = (a.get("ai_reason") or "") + " · 上限合并"
        out.pop(best_i + 1)
    return out


def _events_to_segments(events: list[dict], total_frames: int, fps: float,
                        min_seg_sec: float = 0.8,
                        max_segments: int = 0) -> list[dict]:
    """事件 → 初始时间段(帧区间 + 临时标签 + 理由)。

    长视频(≥10 分钟)使用更长的最小段长(5s):60 分钟级视频的手部
    小动作密度高,1-3 秒碎段对人工审核没有意义,并入前段。
    max_segments > 0 时合并最短的相邻段,控制人工审核工作量。
    """
    segs: list[dict] = []
    bounds = [0] + [e["frame"] for e in events] + [max(1, total_frames - 1)]
    min_sec = max(min_seg_sec, 5.0) if total_frames / max(1.0, fps) >= 600 else min_seg_sec
    min_frames = max(2, int(min_sec * fps))
    for i in range(len(bounds) - 1):
        # 半开区间:段 = [bounds[i], bounds[i+1]-1],下一段从 bounds[i+1]
        # 开始 —— 相邻段不共享边界帧(修复"两个切片叠加一帧"的数据层
        # 问题);最后一段收尾到 total-1。
        start = bounds[i]
        end = (bounds[i + 1] - 1) if i < len(bounds) - 2 else max(1, total_frames - 1)
        if end - start + 1 < min_frames:
            # 碎段并入下一段(而不是丢弃,保证时间轴全覆盖)
            continue
        # 段内主导事件类型 → 临时标签(仅用于 VLM 上下文)
        seg_events = [e for e in events if start <= e["frame"] <= end]
        if not seg_events:
            label, reason = "phase", "no signal events"
        else:
            kinds = [e["kind"] for e in seg_events]
            if kinds[0] == "grasp":
                label, reason = "hold", seg_events[0]["reason"]
            else:
                label, reason = "open_phase", seg_events[0]["reason"]
        if segs and start > segs[-1]["end_frame_index"] + 1:
            # 前段与当前段之间有被跳过的碎段 → 空隙并入前段
            segs[-1]["end_frame_index"] = end
            continue
        segs.append({
            "start_frame_index": start,
            "end_frame_index": end,
            "label": label,
            "ai_reason": reason,
        })
    # 末尾碎段并入最后一段(视频尾部无事件 = 状态延续)
    if segs:
        segs[-1]["end_frame_index"] = max(segs[-1]["end_frame_index"],
                                          max(1, total_frames - 1))
    if max_segments > 0 and len(segs) > max_segments:
        segs = _cap_segments(segs, max_segments)
    return segs


# ═══════════════════════════════════════════════════════════
#  ② VLM 标注
# ═══════════════════════════════════════════════════════════

_VLM_PROMPT = (
    "You are labeling an egocentric hand-manipulation video from an available "
    "mono or stereo camera view. The {n} frames below are sampled uniformly "
    "across the whole video. Output ONLY a JSON object:\n"
    '{{"task": {{"goal": "one-sentence task goal"}}, "subtasks": '
    '[{{"index": 0, "start_s": 0.0, "end_s": 1.0, '
    '"action": "short action", "object": "object", "target": "target", '
    '"hand": "left|right|both|unknown", '
    '"status": "completed|failed|partial|uncertain", '
    '"instruction": "one-sentence instruction", '
    '"confidence": 0.0, "boundary_confidence": 0.0}}]}}\n'
    "Rules: first understand the complete task from the operator's first-person "
    "(egocentric) view, then split it into ordered observable action phases. "
    "Use 2-50 meaningful subtasks; do not split idle "
    "time or tiny hand micro-motions. Cover the video timeline with adjacent, "
    "non-overlapping intervals. Each interval must describe action, object, "
    "target, and completion status. Use an empty string when an object or "
    "target is not visually identifiable; do not hallucinate names. "
    "Focus only on hands and directly manipulated objects."
)

# 原生视频整片标注提示词(API vlm_only 路径;视频直接交给模型)
_VLM_VIDEO_PROPOSE_PROMPT = _VLM_PROMPT

# 按段标注提示词(长视频方案:信号段边界已定,VLM 只负责"这段叫什么")
_VLM_SEGMENT_PROMPTS = {
    "zh": (
        "你是手部操作视频的标注专家。下面是**同一段动作**中均匀采样的 {n} 帧"
        "(按时间顺序,第一帧为该段起点)。请判断这段动作的类别,只输出 JSON:\n"
        '{{"label": "简短中文动作标签(只写动作,如:抓取)", '
        '"object": "被操作物体(2-6字,无则空字符串)", '
        '"instruction": "一句话中文自然语言指令(须包含动作和对象,如:抓取水杯)", '
        '"confidence": 0到1的置信度}}\n'
        "规则:注意力放在手部及其附近的操作物体上,忽略背景环境;label 只写"
        "动作、object 单写物体;相邻段必须首尾相接(上一段终点=下一段起点,"
        "无缝隙、无重叠);不要输出其他文字,只输出 JSON。"
    ),
    "en": (
        "You are labeling one hand-action segment. The {n} frames below are "
        "sampled uniformly within this single segment (first frame = segment "
        "start). Output ONLY a JSON object:\n"
        '{{"label": "short action label, ACTION ONLY (e.g. grasp)", '
        '"object": "the object being manipulated (empty string if none)", '
        '"instruction": "one-sentence instruction mentioning BOTH action and object", '
        '"confidence": float 0-1}} '
        "Adjacent segments must connect end-to-end (prev end == next start, "
        "no gap, no overlap). Focus ONLY on the hand(s) and nearby "
        "manipulated objects; ignore background environment."
    ),
}

# ═══════════════════════════════════════════════════════════
#  提示词封装(app/prompts/ai_annotation_prompts.json):
#  受控词表 + 接触页模板 + 边界自检字段 + VLM 采样参数。
#  改提示词/词表 = 改 JSON 文件,无需动代码、无需重启(每次任务重载)。
# ═══════════════════════════════════════════════════════════

_PROMPTS_FILE = Path(__file__).resolve().parent / "prompts" / "ai_annotation_prompts.json"
_prompts_cache: dict | None = None
_prompts_cache_ts: float = 0.0


def _prompts() -> dict:
    """加载提示词配置(带 mtime 缓存:改 JSON 后下个任务自动生效)。"""
    global _prompts_cache, _prompts_cache_ts
    try:
        ts = _PROMPTS_FILE.stat().st_mtime
        if _prompts_cache is None or ts != _prompts_cache_ts:
            data = json.loads(_PROMPTS_FILE.read_text(encoding="utf-8"))
            data["vocab_zh"] = "; ".join(
                f"{k}={v}" for k, v in (data.get("label_definitions_zh") or {}).items())
            data["vocab_en"] = "; ".join(
                f"{k}={v}" for k, v in (data.get("label_definitions_en") or {}).items())
            data["label_whitelist_zh"] = list((data.get("label_definitions_zh") or {}).keys())
            data["label_whitelist_en"] = list((data.get("label_definitions_en") or {}).keys())
            # Existing prompt files may still say stereo-only.  The same
            # annotation path is valid for mono and stereo, so normalize the
            # wording at load time without invalidating user prompt files.
            for key in ("segment_prompt_zh", "segment_prompt_en",
                        "segment_prompt_video_zh", "segment_prompt_video_en"):
                if isinstance(data.get(key), str):
                    data[key] = data[key].replace(
                        "头戴双目相机拍摄的", "单目或双目相机拍摄的")
                    data[key] = data[key].replace(
                        "head-mounted stereo camera", "available mono or stereo camera")
            _prompts_cache, _prompts_cache_ts = data, ts
        return _prompts_cache
    except Exception:
        return {}


def _scene_context(seg: dict, lang: str = "zh") -> str:
    """Build a compact, per-video context for segment-level VLM labeling.

    Object and location names come from the model's full-video inventory.  No
    global object whitelist is applied; the refs only keep repeated mentions
    in the same episode consistent.
    """
    objects = seg.get("scene_objects") or seg.get("task_objects") or []
    locations = seg.get("scene_locations") or seg.get("task_locations") or []
    if not objects and not locations:
        return ""
    if lang == "zh":
        parts = ["本视频前一阶段识别出的场景上下文（仅作同一物体/位置的指代参考）："]
        if objects:
            parts.append("对象：" + "; ".join(
                f"{o.get('ref', '')}={o.get('name', '')}"
                f"({o.get('attributes', '')})"
                for o in objects if isinstance(o, dict)))
        if locations:
            parts.append("位置：" + "; ".join(
                f"{p.get('ref', '')}={p.get('name', '')}"
                f"({p.get('type', '')})"
                for p in locations if isinstance(p, dict)))
        return "\n".join(parts)
    parts = ["Scene context from the full-video pass (use only to keep object/location references consistent):"]
    if objects:
        parts.append("Objects: " + "; ".join(
            f"{o.get('ref', '')}={o.get('name', '')}"
            f"({o.get('attributes', '')})"
            for o in objects if isinstance(o, dict)))
    if locations:
        parts.append("Locations: " + "; ".join(
            f"{p.get('ref', '')}={p.get('name', '')}"
            f"({p.get('type', '')})"
            for p in locations if isinstance(p, dict)))
    return "\n".join(parts)


def _parse_bool(value) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() == "true"
    return None


def _parse_plan_result(raw: str, lang: str = "zh", fps: float = 30.0,
                       total_frames: int | None = None) -> list[dict]:
    """Parse the shared full-video plan contract used by local and API VLMs.

    ``segments`` is accepted as a legacy alias so old providers/prompts do not
    break.  The result is normalized to the fields consumed by the rest of the
    pipeline; frame conversion and continuity checks happen separately.
    """
    raw = str(raw or "").strip()
    data = _plan_payload(raw)
    if isinstance(data, dict):
        rows = _plan_rows(data)
    else:
        rows = []
    # Some VLM responses are cut off at the token limit or wrap valid JSON in
    # additional prose. Recover complete flat subtask objects instead of
    # silently falling back to one full-video segment.
    if not isinstance(rows, list):
        rows = []
    if not rows:
        recovered: list[dict] = []
        for match in re.finditer(
                r"\{(?=[^{}]*\"(?:start_s|start)\"\s*:)[^{}]*\}", raw,
                flags=re.DOTALL):
            try:
                row = json.loads(match.group(0))
            except Exception:
                continue
            if isinstance(row, dict):
                recovered.append(row)
        rows = recovered
    if not rows:
        return []
    if not isinstance(rows, list):
        return []

    def _has_time(row: dict) -> bool:
        return any(key in row for key in (
            "start_s", "end_s", "start_sec", "end_sec", "start_seconds",
            "end_seconds", "start_time_s", "end_time_s", "start_time",
            "end_time", "start", "end", "start_frame_index",
            "end_frame_index", "start_frame", "end_frame"))

    # Compact retry prompts used by local Qwen deployments sometimes return a
    # valid scene plus ``subtasks: [{steps: [...]}]`` without time boundaries.
    # Recover those steps as contiguous phases instead of discarding the
    # entire model answer. The duration is only used to assign boundaries;
    # action text still comes from the model.
    expanded: list[dict] = []
    for row in rows:
        if not isinstance(row, dict):
            if isinstance(row, str) and row.strip():
                expanded.append({"action": row.strip()})
            continue
        if _has_time(row):
            expanded.append(row)
            continue
        nested = row.get("steps") or row.get("actions") or row.get("phases")
        if isinstance(nested, list) and nested:
            for child in nested:
                if isinstance(child, dict):
                    item = dict(child)
                    if not any(item.get(k) for k in ("action", "label", "skill_id")):
                        item["action"] = str(
                            item.get("description") or item.get("task") or
                            item.get("name") or "phase")
                    expanded.append(item)
                elif str(child).strip():
                    expanded.append({"action": str(child).strip()})
        else:
            expanded.append(row)
    rows = expanded
    if not rows:
        return []

    # Give an untimed compact plan an honest contiguous partition. A missing
    # boundary is not a reason to lose all annotations, but no timing is
    # invented when the provider supplied usable boundaries for the row.
    duration_s = (float(total_frames) / max(1.0, float(fps))
                  if total_frames and int(total_frames) > 0 else None)
    untimed = [row for row in rows if isinstance(row, dict) and not _has_time(row)]
    if untimed and duration_s is not None:
        step_s = duration_s / max(1, len(untimed))
        untimed_index = 0
        for row in rows:
            if isinstance(row, dict) and not _has_time(row):
                row["start_s"] = untimed_index * step_s
                row["end_s"] = ((untimed_index + 1) * step_s
                                if untimed_index < len(untimed) - 1
                                else duration_s)
                untimed_index += 1
    task = data.get("task") if isinstance(data, dict) and isinstance(
        data.get("task"), dict) else {}
    task_goal = str(task.get("goal") or (
        data.get("goal") if isinstance(data, dict) else "") or "").strip()
    scene = data.get("scene") if isinstance(data, dict) else {}
    if not isinstance(scene, dict):
        scene = {}
    task_objects = scene.get("objects") or (
        task.get("objects") if isinstance(task, dict) else [])
    task_locations = scene.get("locations") or (
        task.get("locations") if isinstance(task, dict) else [])
    if not isinstance(task_objects, list):
        task_objects = []
    if not isinstance(task_locations, list):
        task_locations = []
    task_objects = [obj for obj in task_objects if isinstance(obj, dict)]
    task_locations = [loc for loc in task_locations if isinstance(loc, dict)]
    out: list[dict] = []
    for i, row in enumerate(rows):
        if not isinstance(row, dict):
            continue
        def _first(*keys, default=None):
            for key in keys:
                if key in row and row.get(key) is not None:
                    return row.get(key)
            return default

        try:
            start_s = float(_first(
                "start_s", "start_sec", "start_seconds", "start_time_s",
                "start_time", "start", default=0))
            end_s = float(_first(
                "end_s", "end_sec", "end_seconds", "end_time_s",
                "end_time", "end", default=0))
            # Some local templates return frame boundaries instead of seconds.
            # Convert them before the common frame-partition stage.
            if end_s <= start_s:
                start_frame = _first("start_frame_index", "start_frame")
                end_frame = _first("end_frame_index", "end_frame")
                if start_frame is not None and end_frame is not None:
                    start_s = float(start_frame) / max(1.0, float(fps))
                    end_s = (float(end_frame) + 1.0) / max(1.0, float(fps))
            confidence = max(0.0, min(1.0, float(row.get("confidence", 0.5))))
            boundary_confidence = max(
                0.0, min(1.0, float(row.get("boundary_confidence", confidence))))
        except (TypeError, ValueError):
            continue
        action = str(row.get("action") or row.get("label") or "phase").strip()
        status = str(row.get("status") or "uncertain").strip().lower()
        if status not in {"completed", "failed", "partial", "uncertain"}:
            status = "uncertain"
        out.append({
            "index": int(row.get("index", i) or i),
            "start_s": start_s,
            "end_s": end_s,
            "label": action,
            "skill_id": str(row.get("skill_id") or "").strip(),
            "object": str(row.get("object") or "").strip(),
            "object_ref": str(row.get("object_ref") or "").strip(),
            "object_attributes": str(row.get("object_attributes") or "").strip(),
            "source": str(row.get("source") or "").strip(),
            "source_ref": str(row.get("source_ref") or "").strip(),
            "target": str(row.get("target") or "").strip(),
            "target_ref": str(row.get("target_ref") or "").strip(),
            "hand": str(row.get("hand") or "unknown").strip().lower(),
            "status": status,
            "instruction": str(row.get("instruction") or "").strip(),
            "confidence": confidence,
            "boundary_confidence": boundary_confidence,
            "task_goal": task_goal,
            "task_objects": task_objects,
            "task_locations": task_locations,
            "scene_objects": task_objects,
            "scene_locations": task_locations,
        })
    return out


def _has_plan_scene_contract(raw: str) -> bool:
    """Return whether a VLM plan includes the required scene inventory.

    A legacy ``segments`` response can still be parsed by the compatibility
    parser, but it must not be accepted by the new full-video annotation
    pipeline: without ``scene.objects`` and ``scene.locations`` the following
    segment pass cannot keep object references stable.
    """
    data = _plan_payload(raw)
    if not isinstance(data, dict):
        return False
    scene = data.get("scene")
    return (isinstance(scene, dict)
            and isinstance(scene.get("objects"), list)
            and isinstance(scene.get("locations"), list))


def _plan_to_candidates(plan: list[dict], total_frames: int,
                        fps: float,
                        max_segments: int = DEFAULT_MAX_VLM_SEGMENTS) -> list[dict]:
    """Convert a VLM plan into contiguous, bounded frame candidates."""
    raw: list[dict] = []
    for row in sorted(plan, key=lambda x: (float(x.get("start_s", 0)),
                                           int(x.get("index", 0)))):
        try:
            start_s, end_s = float(row.get("start_s", 0)), float(row.get("end_s", 0))
        except (TypeError, ValueError):
            continue
        if end_s <= start_s:
            continue
        start = max(0, min(total_frames - 1, int(round(start_s * fps))))
        end = max(start, min(total_frames - 1, int(round(end_s * fps)) - 1))
        item = dict(row)
        item.update(start_frame_index=start, end_frame_index=end,
                    ai_object=str(row.get("object") or ""),
                    ai_object_ref=str(row.get("object_ref") or ""),
                    ai_object_attributes=str(row.get("object_attributes") or ""),
                    ai_source=str(row.get("source") or ""),
                    ai_source_ref=str(row.get("source_ref") or ""),
                    ai_target=str(row.get("target") or ""),
                    ai_target_ref=str(row.get("target_ref") or ""),
                    ai_hand=str(row.get("hand") or "unknown"),
                    ai_status=str(row.get("status") or "uncertain"),
                    ai_instruction=str(row.get("instruction") or ""),
                    ai_reason="VLM full-video plan")
        raw.append(item)
    if not raw:
        return []
    cap = max_segments if max_segments > 0 else DEFAULT_MAX_VLM_SEGMENTS
    if len(raw) > cap:
        raw = _cap_segments(raw, cap)
    raw[0]["start_frame_index"] = 0
    for i in range(len(raw) - 1):
        # Make the model's intervals a true partition.  A gap belongs to the
        # preceding phase; an overlap is trimmed at the next phase boundary.
        next_start = max(raw[i]["start_frame_index"],
                         raw[i + 1]["start_frame_index"])
        raw[i]["end_frame_index"] = max(raw[i]["start_frame_index"], next_start - 1)
        raw[i + 1]["start_frame_index"] = raw[i]["end_frame_index"] + 1
    raw[-1]["end_frame_index"] = total_frames - 1
    return [r for r in raw if r["end_frame_index"] > r["start_frame_index"]]


_GLOBAL_CHECK_PROMPTS = {
    "zh": (
        "你是机器人操作数据集的质检员。下面是一个视频切分并标注后的动作段列表"
        "(序号/起止秒/标签/置信度):\n{list}\n"
        "请检查:1) 相邻段标签是否明显错序或矛盾 2) 标签相同或过短的相邻段"
        "是否应该合并 3) 是否有明显重复或幻觉标签 4) 相邻段首尾是否相接"
        "(有无缝隙或重叠)。\n"
        "只输出 JSON:{{\"issues\": [{{\"index\": 段序号, \"problem\": \"问题简述\"}}], "
        "\"merges\": [[i, i+1], ...]}}\n"
        "没有问题输出空数组;不要输出其他文字。"
    ),
    "en": (
        "You are a QA reviewer for a robot dataset. Below is a video's segment "
        "list (index / start-end seconds / label / confidence):\n{list}\n"
        "Check: 1) adjacent labels clearly out of order or contradictory "
        "2) adjacent segments with same label or too short that should merge "
        "3) obvious duplicate or hallucinated labels 4) adjacent segments "
        "connect end-to-end (no gaps or overlaps).\n"
        "Output ONLY JSON: {{\"issues\": [{{\"index\": i, \"problem\": \"...\"}}], "
        "\"merges\": [[i, i+1], ...]}}\nEmpty arrays if no problems."
    ),
}


async def _vlm_global_check(segments: list[dict], fps: float, lang: str,
                            client: httpx.AsyncClient,
                            vlm_cfg: dict | None = None) -> dict:
    """全局校验遍(DenseStep2M 思路):把完整段列表交给 LLM 查错序/
    应合并/幻觉标签。返回 {"issues": [...], "merges": [[i,i+1]...]};
    请求失败 → 空(不阻塞标注流程)。local/API 按 vlm_cfg 分派。"""
    rows = [
        f"{i}: {s['start_frame_index'] / max(1.0, fps):.1f}-"
        f"{s['end_frame_index'] / max(1.0, fps):.1f}s "
        f"label={s.get('label') or '(空)'} obj={s.get('ai_object') or ''} "
        f"conf={s.get('ai_confidence', 0):.2f}"
        for i, s in enumerate(segments)
    ]
    template = _GLOBAL_CHECK_PROMPTS.get(lang, _GLOBAL_CHECK_PROMPTS["zh"])
    chat_url, _, model, headers, err = _vlm_endpoint(vlm_cfg)
    if err:
        print(f"[ai_annotation] global check skipped: {err}")
        return {"issues": [], "merges": []}
    payload = {
        "model": model,
        "messages": [{"role": "user",
                      "content": template.format(list="\n".join(rows))}],
        "max_tokens": 800,
        "temperature": 0.1,
    }
    if (vlm_cfg or {}).get("vlm_provider") == "api":
        payload.update(_api_generation_options(vlm_cfg))
    try:
        if (vlm_cfg or {}).get("vlm_provider") == "api":
            r = await _vlm_post_api(client, chat_url, payload, headers)
        else:
            r = await client.post(chat_url, json=payload, timeout=600)
            r.raise_for_status()
        text, _finish_reason = _api_response_text(
            r, require_json=True)
        if not text:
            raise TypeError("API response has no valid JSON content")
    except Exception as exc:
        print(f"[ai_annotation] global check failed: {exc}")
        return {"issues": [], "merges": []}
    raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", text)
    try:
        data = json.loads(raw)
    except Exception:
        m = re.search(r"\{[^{}]*\}", raw)
        if not m:
            return {"issues": [], "merges": []}
        try:
            data = json.loads(m.group(0))
        except Exception:
            return {"issues": [], "merges": []}
    return {"issues": data.get("issues") or [],
            "merges": data.get("merges") or []}


def _vlm_propose(video_path: Path, fps: float, total_frames: int,
                 lang: str = "zh", media_note: str = "",
                 max_segments: int = DEFAULT_MAX_VLM_SEGMENTS
                 ) -> list[dict]:
    """本地 VLM 规划。

    默认是 MP4 data URL + vLLM 的 FPS-aware 视频采样；设置
    ``EGODATA_VLLM_MEDIA_MODE=frames`` 时保留旧的均匀抽帧实现。

    Transport/response failures are raised as :class:`VLMAnnotationError`
    so the task state can tell an unavailable model from an invalid plan.
    """
    if VLLM_MEDIA_MODE == "video":
        return _vlm_propose_video(
            video_path, fps, total_frames, lang, media_note, max_segments)

    import cv2

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise VLMAnnotationError(
            "input_video_unreadable",
            f"无法打开 VLM 输入视频: {video_path.name}",
        )
    step = max(1, total_frames // VLLM_MAX_FRAMES)
    content: list[dict] = []
    fi = 0
    count = 0
    while fi < total_frames and count < VLLM_MAX_FRAMES:
        cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.resize(frame, (VLLM_FRAME_W, VLLM_FRAME_H))
        ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if not ok:
            break
        b64 = base64.b64encode(buf.tobytes()).decode()
        content.append({
            "type": "text",
            "text": f"[frame {fi}, t={fi / max(1.0, fps):.1f}s]",
        })
        content.append({"type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{b64}"}})
        fi += step
        count += 1
    cap.release()
    if not count:
        raise VLMAnnotationError(
            "input_video_decode_failed",
            f"VLM 输入视频无法读取有效帧: {video_path.name}",
        )

    prompts = _prompts()
    template = prompts.get("plan_prompt_zh" if lang == "zh" else "plan_prompt_en")
    vocab = prompts.get("vocab_zh" if lang == "zh" else "vocab_en", "")
    prompt = (template or _VLM_PROMPT).format(
        n=count, duration_s=total_frames / max(1.0, fps),
        total_frames=total_frames, fps=fps, vocab=vocab)
    if media_note:
        prompt = media_note + "\n" + prompt
    content.append({"type": "text", "text": prompt})
    vlm_options = _prompts().get("vlm") or {}
    configured_budget = max(1, int(vlm_options.get("plan_max_tokens", 2400)))
    duration_s = total_frames / max(1.0, fps)
    # A fixed 2400-token cap truncates otherwise valid plans for long videos.
    # Scale the first request conservatively, then keep a bounded retry below
    # the model context limit.  max_segments is an upper bound, not a request
    # to manufacture that many segments, so duration is the safer predictor.
    duration_budget = 2400 + (max(0, int(duration_s - 30)) // 15) * 400
    # A plan contains the scene inventory as well as every time-bounded
    # subtask.  2400 tokens is frequently enough for short clips but can cut
    # the JSON in the middle of a long object/plan list.  Reserve headroom on
    # the first local request so the normal response is valid; the retry is
    # still bounded by the vLLM context window.
    first_budget = min(6000, max(configured_budget, duration_budget + 2800))
    retry_budget = min(6000, max(first_budget + 800, 5200))
    temperature = float(vlm_options.get("plan_temperature", 0.1))

    def _request(request_content: list[dict], budget: int) -> tuple[str, str]:
        try:
            response = httpx.post(VLLM_URL, json={
                "model": VLLM_MODEL,
                "messages": [{"role": "user", "content": request_content}],
                "max_tokens": budget,
                "temperature": temperature,
            }, timeout=600)
            response.raise_for_status()
            payload = response.json()
            choice = payload["choices"][0]
            message = choice["message"]
            raw_content = _select_model_text(
                _model_text_candidates(message, include_reasoning=True),
                prefer_plan=True)
            if not raw_content:
                raise TypeError("message.content and reasoning_content are empty")
            return raw_content, str(choice.get("finish_reason") or "")
        except httpx.ConnectError as exc:
            raise VLMAnnotationError(
                "model_unavailable",
                f"本地 VLM 未启动或无法连接 ({_safe_error_text(exc.__class__.__name__)})",
            ) from exc
        except httpx.TimeoutException as exc:
            raise VLMAnnotationError(
                "model_timeout",
                "本地 VLM 响应超时，模型可能仍在加载或当前负载过高",
            ) from exc
        except httpx.HTTPStatusError as exc:
            status = exc.response.status_code if exc.response is not None else 0
            provider_detail = _http_error_detail(exc.response)
            if status in (502, 503, 504):
                code = "model_not_ready"
                detail = "本地 VLM 服务未就绪或正在加载模型"
            else:
                code = "model_http_error"
                detail = f"本地 VLM 返回 HTTP {status}"
            if provider_detail:
                detail += f": {provider_detail}"
            raise VLMAnnotationError(code, detail) from exc
        except httpx.RequestError as exc:
            raise VLMAnnotationError(
                "model_connection_failed",
                f"本地 VLM 请求失败 ({_safe_error_text(exc.__class__.__name__)})",
            ) from exc
        except (KeyError, IndexError, TypeError, ValueError,
                json.JSONDecodeError) as exc:
            raise VLMAnnotationError(
                "invalid_model_response",
                f"本地 VLM 返回格式无效 ({_safe_error_text(exc)})",
            ) from exc
        except Exception as exc:
            raise VLMAnnotationError(
                "model_request_failed",
                f"本地 VLM 调用异常 ({_safe_error_text(exc)})",
            ) from exc

    text, finish_reason = _request(content, first_budget)
    plan = (_parse_plan_result(text, lang, fps=fps, total_frames=total_frames)
            if _has_plan_scene_contract(text) else [])

    # Qwen/vLLM returns HTTP 200 even when generation stops at the token limit.
    # Retry once with a compact contract instead of treating that partial JSON
    # as a hard model failure.  We still reject the result if the retry is not
    # a complete, contract-valid plan; no mechanical fallback is introduced.
    if finish_reason == "length" or not plan:
        compact_prompt = (
            "上一条输出可能因长度限制被截断。请重新观看同一组画面并只输出一个"
            "完整、可解析的 JSON 对象，不要输出 Markdown、解释或思考过程。"
            "为了保证完整性，请保持字段简短：scene.objects 只保留 ref/name/"
            "attributes/confidence，scene.locations 只保留 ref/name/type/confidence；"
            "每个 subtasks 只保留 index/start_s/end_s/skill_id/action/object_ref/"
            "object/object_attributes/source_ref/source/target_ref/target/hand/"
            "status/instruction/confidence/boundary_confidence。字符串尽量简短，"
            "必须输出完整的 scene.objects、scene.locations 和 subtasks，所有动作"
            "阶段首尾相接覆盖整段视频。"
        )
        retry_content = list(content)
        retry_content[-1] = {"type": "text", "text": compact_prompt}
        text, finish_reason = _request(retry_content, retry_budget)
        plan = (_parse_plan_result(text, lang, fps=fps, total_frames=total_frames)
                if _has_plan_scene_contract(text) else [])

    if not _has_plan_scene_contract(text):
        code, detail = _plan_contract_failure(text)
        if finish_reason == "length":
            code = "plan_output_truncated"
            detail = "VLM 输出仍被长度限制截断，未写入 AI 标注"
        raise VLMAnnotationError(code, detail)
    if not plan:
        raise VLMAnnotationError(
            "empty_plan",
            "VLM 返回了完整场景信息，但没有可用 subtasks 动作计划",
        )
    return plan


def _local_video_media_options(source_fps: float,
                               vlm_options: dict) -> dict:
    """Build request-level vLLM video sampling options.

    ``fps`` here is the target sampling rate, not a rewrite of the source
    video's metadata. vLLM reads the original FPS from the MP4 and Qwen3-VL
    uses this target rate to choose temporal frames. Keeping the source FPS in
    the prompt makes the frame/time mapping explicit to the model as well.
    """
    try:
        sample_fps = float(vlm_options.get(
            "video_sample_fps", VLLM_VIDEO_SAMPLE_FPS))
    except (TypeError, ValueError):
        sample_fps = VLLM_VIDEO_SAMPLE_FPS
    sample_fps = min(
        max(0.1, sample_fps),
        max(0.1, float(source_fps or sample_fps)),
    )
    try:
        max_frames = max(4, int(vlm_options.get("max_frames", VLLM_MAX_FRAMES)))
    except (TypeError, ValueError):
        max_frames = VLLM_MAX_FRAMES
    return {
        "video": {
            "fps": sample_fps,
            "min_frames": min(4, max_frames),
            "max_frames": max_frames,
        }
    }


def _local_video_request_parts(clip: Path, source_fps: float,
                               vlm_options: dict) -> tuple[dict, dict]:
    """Return a vLLM ``video_url`` part and per-request media options."""
    video_ref = _inline_video_data_url(
        clip,
        max_bytes=LOCAL_INLINE_VIDEO_MAX_BYTES,
        label="本地 VLM",
    )
    return (
        {"type": "video_url", "video_url": {"url": video_ref}},
        _local_video_media_options(source_fps, vlm_options),
    )


def _vlm_propose_video(video_path: Path, fps: float, total_frames: int,
                       lang: str, media_note: str,
                       max_segments: int) -> list[dict]:
    """本地 vLLM 原生视频规划:MP4 data URL + FPS-aware sampling。"""
    import math

    whole = {"start_frame_index": 0,
             "end_frame_index": max(0, total_frames - 1)}
    with tempfile.TemporaryDirectory(prefix="egodata-local-vlm-") as tmp:
        clip = _slice_segment_clip(
            video_path, fps, whole, None, None, Path(tmp))
        if clip is None:
            raise VLMAnnotationError(
                "video_slice_failed", "本地 VLM 输入视频切片失败")
        try:
            prompts = _prompts()
            vlm_options = prompts.get("vlm") or {}
            video_part, media_options = _local_video_request_parts(
                clip, fps, vlm_options)
        except (OSError, ValueError) as exc:
            raise VLMAnnotationError("local_video_read_failed", str(exc)) from exc

        template = prompts.get("plan_prompt_zh" if lang == "zh" else "plan_prompt_en")
        vocab = prompts.get("vocab_zh" if lang == "zh" else "vocab_en", "")
        sample_fps = float(media_options["video"]["fps"])
        sampled_n = min(
            int(media_options["video"]["max_frames"]),
            max(1, int(math.ceil(total_frames / max(1.0, fps) * sample_fps))),
        )
        prompt = (template or _VLM_PROMPT).format(
            n=sampled_n, duration_s=total_frames / max(1.0, fps),
            total_frames=total_frames, fps=fps, vocab=vocab)
        video_note = (
            f"输入媒体是原始 MP4 视频，源帧率约 {float(fps):.3f} FPS；"
            f"服务端按 {sample_fps:.3f} FPS 采样，但请按源视频时间理解动作。"
        )
        prompt = video_note + ("\n" + media_note if media_note else "") + "\n" + prompt
        prompt += (
            "\n\n输出要求：最终答案只放在 assistant content 中，"
            "只输出一个完整 JSON 对象，不要输出 Markdown、解释或思考过程。")
        content = [video_part, {"type": "text", "text": prompt}]

        configured_budget = max(1, int(vlm_options.get("plan_max_tokens", 2400)))
        duration_s = total_frames / max(1.0, fps)
        duration_budget = 2400 + (max(0, int(duration_s - 30)) // 15) * 400
        first_budget = min(6000, max(configured_budget, duration_budget + 2800))
        retry_budget = min(6000, max(first_budget + 800, 5200))
        temperature = float(vlm_options.get("plan_temperature", 0.1))

        def _request(request_content: list[dict], budget: int) -> tuple[str, str]:
            try:
                response = httpx.post(VLLM_URL, json={
                    "model": VLLM_MODEL,
                    "messages": [{"role": "user", "content": request_content}],
                    "max_tokens": budget,
                    "temperature": temperature,
                    "media_io_kwargs": media_options,
                }, timeout=600)
                response.raise_for_status()
                payload = response.json()
                choice = payload["choices"][0]
                message = choice["message"]
                raw = _select_model_text(
                    _model_text_candidates(message, include_reasoning=True),
                    prefer_plan=True)
                if not raw:
                    raise TypeError("message.content and reasoning_content are empty")
                return raw, str(choice.get("finish_reason") or "")
            except httpx.ConnectError as exc:
                raise VLMAnnotationError(
                    "model_unavailable", "本地 VLM 未启动或无法连接") from exc
            except httpx.TimeoutException as exc:
                raise VLMAnnotationError(
                    "model_timeout", "本地 VLM 视频推理响应超时") from exc
            except httpx.HTTPStatusError as exc:
                status = exc.response.status_code if exc.response is not None else 0
                detail = _http_error_detail(exc.response)
                suffix = f": {detail}" if detail else ""
                raise VLMAnnotationError(
                    "model_http_error", f"本地 VLM 返回 HTTP {status}{suffix}") from exc
            except httpx.RequestError as exc:
                raise VLMAnnotationError(
                    "model_connection_failed",
                    f"本地 VLM 请求失败 ({_safe_error_text(exc.__class__.__name__)})",
                ) from exc
            except (KeyError, IndexError, TypeError, ValueError,
                    json.JSONDecodeError) as exc:
                raise VLMAnnotationError(
                    "invalid_model_response",
                    f"本地 VLM 返回格式无效 ({_safe_error_text(exc)})",
                ) from exc

        text, finish_reason = _request(content, first_budget)
        plan = (_parse_plan_result(text, lang, fps=fps, total_frames=total_frames)
                if _has_plan_scene_contract(text) else [])
        if finish_reason == "length" or not plan:
            retry_content = list(content)
            retry_content[-1] = {"type": "text", "text": (
                "请重新观看同一 MP4，只输出一个完整、可解析的 JSON 对象，"
                "不要输出 Markdown、解释或思考过程；必须包含 scene.objects、"
                "scene.locations 和 subtasks，所有阶段首尾相接覆盖整段视频。")}
            text, finish_reason = _request(retry_content, retry_budget)
            plan = (_parse_plan_result(text, lang, fps=fps, total_frames=total_frames)
                    if _has_plan_scene_contract(text) else [])
        if not _has_plan_scene_contract(text):
            code, detail = _plan_contract_failure(text)
            if finish_reason == "length":
                code, detail = "plan_output_truncated", "VLM 输出被长度限制截断"
            raise VLMAnnotationError(code, detail)
        if not plan:
            raise VLMAnnotationError("empty_plan", "VLM 没有返回可用动作计划")
        return plan


async def _vlm_label_segment(video_path: Path, fps: float, seg: dict, lang: str,
                             client: httpx.AsyncClient,
                             prev_seg: dict | None = None,
                             next_seg: dict | None = None,
                             media_note: str = "") -> dict | None:
    """对单个信号段做 VLM 标注(接触页方案,WGO-Bench 验证做法):

    - 默认发送本段 MP4(≤ max_frames,由 vLLM 按 FPS 采样)
    - 兼容模式下本段均匀抽帧(间隔 frame_interval_s)
    - 附带上一段末 2 帧 + 下一段首 2 帧做消歧上下文(判边界合理性)
    - 标签受控词表(app/prompts/ai_annotation_prompts.json),未命中
      词表 → 模糊匹配 + confidence × 0.7
    - 输出 boundary_ok/boundary_confidence(本段与下一段的边界自检)

    帧数不足或请求失败 → None(该段不写入 AI 结果)。
    """
    if VLLM_MEDIA_MODE == "video":
        return await _vlm_label_segment_video(
            video_path, fps, seg, lang, client, prev_seg, next_seg, media_note)

    import cv2

    prompts = _prompts()
    vlm_cfg = prompts.get("vlm") or {}
    ctx_frames = int(vlm_cfg.get("context_frames", 2))

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None

    def _read_frame(fi: int) -> str | None:
        cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
        ok, frame = cap.read()
        if not ok:
            return None
        frame = cv2.resize(frame, (VLLM_FRAME_W, VLLM_FRAME_H))
        ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if not ok:
            return None
        return base64.b64encode(buf.tobytes()).decode()

    content: list[dict] = []
    try:
        # ① 上一段末帧(消歧上下文)
        if prev_seg is not None:
            end_prev = int(prev_seg.get("end_frame_index", 0))
            for fi in range(max(0, end_prev - ctx_frames + 1), end_prev + 1):
                b64 = _read_frame(fi)
                if b64:
                    content.append({"type": "text",
                                    "text": f"[上一段末帧 t={fi / max(1.0, fps):.1f}s]"})
                    content.append({"type": "image_url",
                                    "image_url": {"url": f"data:image/jpeg;base64,{b64}"}})
        # ② 本段均匀抽帧
        start_f, end_f = int(seg["start_frame_index"]), int(seg["end_frame_index"])
        max_frames = int(vlm_cfg.get("max_frames", VLLM_MAX_FRAMES))
        interval_s = float(vlm_cfg.get("frame_interval_s", 2))
        n_want = min(max_frames, max(4, (end_f - start_f) // max(1, int(fps * interval_s))))
        step = max(1, (end_f - start_f) // n_want)
        count = 0
        for fi in range(start_f, end_f + 1, step):
            if count >= n_want:
                break
            b64 = _read_frame(fi)
            if b64 is None:
                break
            content.append({"type": "text",
                            "text": f"[本段 t={(fi - start_f) / max(1.0, fps):.1f}s]"})
            content.append({"type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{b64}"}})
            count += 1
        # ③ 下一段首帧(消歧上下文)
        if next_seg is not None:
            start_next = int(next_seg.get("start_frame_index", 0))
            for fi in range(start_next, start_next + ctx_frames):
                b64 = _read_frame(fi)
                if b64:
                    content.append({"type": "text",
                                    "text": f"[下一段首帧 t={fi / max(1.0, fps):.1f}s]"})
                    content.append({"type": "image_url",
                                    "image_url": {"url": f"data:image/jpeg;base64,{b64}"}})
    finally:
        cap.release()
    if not count:
        return None

    # 模板与词表:优先封装配置,缺失回退旧模板(自由标签)
    template = prompts.get("segment_prompt_zh" if lang == "zh" else "segment_prompt_en")
    vocab = prompts.get("vocab_zh" if lang == "zh" else "vocab_en")
    if template and vocab:
        segment_prompt = template.format(
            n=count, vocab=vocab, scene_context=_scene_context(seg, lang))
    else:
        segment_prompt = _VLM_SEGMENT_PROMPTS.get(
            lang, _VLM_SEGMENT_PROMPTS["zh"]).format(n=count)
    if media_note:
        segment_prompt = media_note + "\n" + segment_prompt
    content.append({"type": "text", "text": segment_prompt})
    try:
        r = await client.post(VLLM_URL, json={
            "model": VLLM_MODEL,
            "messages": [{"role": "user", "content": content}],
            "max_tokens": int(vlm_cfg.get("max_tokens", 420)),
            "temperature": float(vlm_cfg.get("temperature", 0.1)),
        }, timeout=600)
        r.raise_for_status()
        text = r.json()["choices"][0]["message"]["content"].strip()
    except Exception as exc:
        print(f"[ai_annotation] segment VLM failed: {exc}")
        return None
    return _parse_segment_result(text, lang)


def _parse_segment_result(raw: str, lang: str) -> dict | None:
    """VLM 段标注响应 → 严格 JSON 解析(剥围栏 → json.loads → 正则兜底)
    → 词表软约束(未命中保留新词,label_matched=false 前端 ⚠ 提示)。
    本地抽帧图与 API 原生视频两条路径共用。"""
    prompts = _prompts()
    raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw)
    try:
        data = json.loads(raw)
    except Exception:
        m = re.search(r"\{[^{}]*\}", raw)
        if not m:
            return None
        try:
            data = json.loads(m.group(0))
        except Exception:
            return None
    label = str(data.get("label") or "").strip()
    if not label:
        return None
    try:
        conf = max(0.0, min(1.0, float(data.get("confidence", 0.5))))
    except (TypeError, ValueError):
        conf = 0.5
    # 词表 = 软约束(每个视频动作不固定):命中原样;未命中保留 VLM 的
    # 新词,不改词、不打折 —— 仅 label_matched=false 标记,前端 ⚠
    # 提示"新词,建议人工确认"。
    label_matched = True
    final_label = label
    whitelist = prompts.get("label_whitelist_zh" if lang == "zh" else "label_whitelist_en")
    if whitelist and label not in whitelist:
        label_matched = False
    # 模型偶发输出非数值字符串 → 单段解析失败不应拖垮整个任务
    try:
        bconf = max(0.0, min(1.0, float(data.get("boundary_confidence") or 0.0)))
    except (TypeError, ValueError):
        bconf = 0.0
    status = str(data.get("status") or "uncertain").strip().lower()
    if status not in {"completed", "failed", "partial", "uncertain"}:
        status = "uncertain"
    return {
        "label": final_label,
        "skill_id": str(data.get("skill_id") or "").strip(),
        "object": str(data.get("object") or "").strip(),
        "object_ref": str(data.get("object_ref") or "").strip(),
        "object_attributes": str(data.get("object_attributes") or "").strip(),
        "source": str(data.get("source") or "").strip(),
        "source_ref": str(data.get("source_ref") or "").strip(),
        "target": str(data.get("target") or "").strip(),
        "target_ref": str(data.get("target_ref") or "").strip(),
        "hand": str(data.get("hand") or "unknown").strip().lower(),
        "status": status,
        "instruction": str(data.get("instruction") or "").strip(),
        "confidence": conf,
        "boundary_ok": _parse_bool(data.get("boundary_ok")),
        "boundary_confidence": bconf,
        "label_matched": label_matched,
    }


async def _vlm_label_segment_video(video_path: Path, fps: float, seg: dict,
                                   lang: str, client: httpx.AsyncClient,
                                   prev_seg: dict | None = None,
                                   next_seg: dict | None = None,
                                   media_note: str = "") -> dict | None:
    """本地 vLLM 原生视频逐段标注:MP4 data URL + FPS-aware sampling。"""
    import math

    start_f = int(seg["start_frame_index"])
    end_f = int(seg["end_frame_index"])
    ctx_frames = int((_prompts().get("vlm") or {}).get("context_frames", 2))
    if prev_seg is not None:
        start_f = max(0, int(prev_seg["end_frame_index"]) - ctx_frames + 1)
    if next_seg is not None:
        end_f = int(next_seg["start_frame_index"]) + ctx_frames - 1
    clip_seg = {"start_frame_index": start_f, "end_frame_index": end_f}

    with tempfile.TemporaryDirectory(prefix="egodata-local-vlm-") as tmp:
        clip = _slice_segment_clip(
            video_path, fps, clip_seg, None, None, Path(tmp))
        if clip is None:
            return None
        prompts = _prompts()
        vlm_options = prompts.get("vlm") or {}
        try:
            video_part, media_options = _local_video_request_parts(
                clip, fps, vlm_options)
        except (OSError, ValueError) as exc:
            print(f"[ai_annotation] local video encoding failed: {exc}")
            return None

        template = prompts.get("segment_prompt_zh" if lang == "zh" else "segment_prompt_en")
        vocab = prompts.get("vocab_zh" if lang == "zh" else "vocab_en")
        if not (template and vocab):
            return None
        sample_fps = float(media_options["video"]["fps"])
        clip_duration = (end_f - start_f + 1) / max(1.0, fps)
        sampled_n = min(
            int(media_options["video"]["max_frames"]),
            max(1, int(math.ceil(clip_duration * sample_fps))),
        )
        segment_prompt = template.format(
            n=sampled_n, vocab=vocab, scene_context=_scene_context(seg, lang))
        video_note = (
            f"输入媒体是原始 MP4 视频，源帧率约 {float(fps):.3f} FPS；"
            f"服务端按 {sample_fps:.3f} FPS 采样，请按源视频时间理解本段。"
        )
        segment_prompt = video_note + ("\n" + media_note if media_note else "") \
            + "\n" + segment_prompt
        segment_prompt += (
            "\n\n输出要求：最终答案只放在 assistant content 中，"
            "只输出一个完整 JSON 对象，不要输出 Markdown、解释或思考过程。")
        content = [video_part, {"type": "text", "text": segment_prompt}]
        configured_budget = max(1, int(vlm_options.get("max_tokens", 420)))
        first_budget = max(configured_budget, 1200)
        retry_budget = min(4000, max(first_budget + 1000, 2400))
        request_payload = {
            "model": VLLM_MODEL,
            "messages": [{"role": "user", "content": content}],
            "max_tokens": first_budget,
            "temperature": float(vlm_options.get("temperature", 0.1)),
            "media_io_kwargs": media_options,
        }
        retry_content = list(content)
        retry_content[-1] = {"type": "text", "text": (
            "请重新输出本段标注，只输出一个完整 JSON 对象，不要输出"
            "Markdown、解释或思考过程；最终 JSON 必须放在 assistant content 中。")}
        retry_payload = dict(request_payload)
        retry_payload["messages"] = [{"role": "user", "content": retry_content}]
        retry_payload["max_tokens"] = retry_budget

        last_error: Exception | None = None
        for attempt, payload in enumerate((request_payload, retry_payload)):
            try:
                response = await client.post(VLLM_URL, json=payload, timeout=600)
                response.raise_for_status()
                text, finish_reason = _api_response_text(
                    response, require_json=True)
                parsed = _parse_segment_result(text, lang) if text else None
                if parsed is not None and (attempt == 1 or finish_reason != "length"):
                    return parsed
                last_error = TypeError("本地 VLM 视频响应不是完整段标注 JSON")
            except Exception as exc:
                last_error = exc
        print(f"[ai_annotation] local video segment failed: {last_error}")
        return None


# ═══════════════════════════════════════════════════════════
#  API 视频路径:本地切段 → video_url → Chat Completions。
#  不同厂商共用 OpenAI-compatible 协议；Qwen 额外接收 fps，
#  使视频时间戳与动作分段保持一致。
# ═══════════════════════════════════════════════════════════

async def _vlm_post_api(client: httpx.AsyncClient, url: str, payload: dict,
                        headers: dict, retries: int = 1,
                        stage: str = "",
                        meta: dict | None = None) -> httpx.Response:
    """API 请求:限速 + 429/5xx/网络错误重试,最终失败抛出。

    The provider is a shared rental gateway. Requests are serialized at the
    process level and respect Retry-After, preventing a burst of segment calls
    from turning one temporary 429 into a failed annotation generation.

    stage/meta 用于调用记账(vlm_calls.jsonl):标注段 = label_segment,
    整片规划 = propose。仅日志用途,不影响请求本身。
    """
    global _API_LAST_REQUEST_AT
    t0 = time.monotonic()
    last_err: Exception | None = None
    for attempt in range(retries + 1):
        try:
            async with _API_RATE_LIMIT_LOCK:
                now = time.monotonic()
                wait = (API_MIN_REQUEST_INTERVAL_SEC
                        - (now - _API_LAST_REQUEST_AT))
                if wait > 0:
                    await asyncio.sleep(wait)
                _API_LAST_REQUEST_AT = time.monotonic()
            r = await client.post(
                url, json=payload, headers=headers,
                timeout=httpx.Timeout(API_REQUEST_TIMEOUT_SEC, connect=20.0))
            if r.status_code in (429, 500, 502, 503, 504) and attempt < retries:
                retry_after = r.headers.get("retry-after")
                try:
                    delay = min(60.0, max(10.0, float(retry_after)))
                except (TypeError, ValueError):
                    delay = min(60.0, 15.0 * (attempt + 1))
                await asyncio.sleep(delay)
                continue
            r.raise_for_status()
            _record_vlm_call({
                **(meta or {}),
                "provider": "api",
                "stage": stage,
                "model": str(payload.get("model") or ""),
                "ok": True,
                "seconds": round(time.monotonic() - t0, 2),
                "status_code": r.status_code,
                "usage": _usage_from_response(r),
            })
            return r
        except httpx.HTTPError as exc:
            last_err = exc
            if attempt >= retries:
                break
            await asyncio.sleep(2 * (attempt + 1))
    _record_vlm_call({
        **(meta or {}),
        "provider": "api",
        "stage": stage,
        "model": str(payload.get("model") or ""),
        "ok": False,
        "seconds": round(time.monotonic() - t0, 2),
        "error": _safe_error_text(last_err) if isinstance(last_err, Exception)
                  else "API VLM request failed",
    })
    if isinstance(last_err, Exception):
        raise last_err
    raise RuntimeError("API VLM request failed")


def _slice_segment_clip(video_path: Path, fps: float, seg: dict,
                        prev_seg: dict | None, next_seg: dict | None,
                        out_dir: Path, ctx_frames: int = 2) -> Path | None:
    """ffmpeg 把 [上一段末 ctx 帧 + 本段 + 下一段首 ctx 帧] 切成 mp4
    (与抽帧图模式的上下文结构一致)。-ss 放 -i 前(快定位,起点允许
    提前到最近关键帧,上下文语义不变);缩放 ≤1080p(官方建议);
    H.264 yuv420p faststart,浏览器/云端都能读。失败返回 None。"""
    import subprocess

    start_f = int(seg["start_frame_index"])
    end_f = int(seg["end_frame_index"])
    if prev_seg is not None:
        start_f = max(0, int(prev_seg["end_frame_index"]) - ctx_frames + 1)
    if next_seg is not None:
        end_f = int(next_seg["start_frame_index"]) + ctx_frames - 1
    if end_f < start_f:
        return None
    t0 = start_f / max(1.0, fps)
    dur = (end_f - start_f + 1) / max(1.0, fps) + 0.5  # 0.5s 容差,超 EOF ffmpeg 自截
    out = out_dir / f"seg_{start_f}_{end_f}.mp4"
    cmd = ["ffmpeg", "-y", "-loglevel", "error",
           "-ss", str(t0), "-i", str(video_path), "-t", str(dur),
           "-vf", "scale='min(1920,iw)':-2",
           "-c:v", "libx264", "-pix_fmt", "yuv420p",
           "-movflags", "+faststart", "-an", str(out)]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    except (OSError, subprocess.TimeoutExpired) as exc:
        print(f"[ai_annotation] ffmpeg slice error: {exc}")
        return None
    if r.returncode != 0 or not out.is_file() or out.stat().st_size < 1024:
        # <1KB = 空输出(-ss 超出视频长度等),同样视为失败
        print(f"[ai_annotation] ffmpeg slice failed: {r.stderr[:300]}")
        return None
    return out


async def _upload_video_file(client: httpx.AsyncClient, upload_base: str,
                             clip: Path, headers: dict) -> str | None:
    """Kimi Files API 上传视频(purpose=video,官方文档协议)→ 返回
    文件 id(ms:// 引用用);失败返回 None。"""
    for attempt in range(2):
        try:
            async with _API_RATE_LIMIT_LOCK:
                global _API_LAST_REQUEST_AT
                now = time.monotonic()
                wait = (API_MIN_REQUEST_INTERVAL_SEC
                        - (now - _API_LAST_REQUEST_AT))
                if wait > 0:
                    await asyncio.sleep(wait)
                _API_LAST_REQUEST_AT = time.monotonic()
            with open(clip, "rb") as f:
                r = await client.post(
                    f"{upload_base}/files", headers=headers,
                    files={"file": (clip.name, f, "video/mp4")},
                    data={"purpose": "video"},
                    timeout=httpx.Timeout(API_REQUEST_TIMEOUT_SEC, connect=20.0))
            if r.status_code in (429, 500, 502, 503, 504) and attempt < 1:
                await asyncio.sleep(2 * (attempt + 1))
                continue
            r.raise_for_status()
            file_id = r.json().get("id")
            if not file_id:
                print(f"[ai_annotation] upload response missing id: {r.text[:200]}")
                return None
            return str(file_id)
        except Exception as exc:
            if attempt >= 1:
                print(f"[ai_annotation] video upload failed: {exc}")
                return None
            await asyncio.sleep(2 * (attempt + 1))
    return None


def _inline_video_data_url(clip: Path, *,
                           max_bytes: int = API_INLINE_VIDEO_MAX_BYTES,
                           label: str = "API VLM") -> str:
    """Encode one video clip as a data URL without requiring a Files API.

    This is still an API VLM request; it is only a transport alternative for
    OpenAI-compatible proxies that expose ``/chat/completions`` but not
    ``/files``.  The caller turns the size error into a visible failed AI
    task, never into a local-model fallback.
    """
    size = clip.stat().st_size
    if size <= 0:
        raise ValueError("API VLM 视频片段为空")
    if size > max_bytes:
        raise ValueError(
            f"{label} 视频片段过大({size / 1024 / 1024:.1f} MiB)，"
            "请缩短片段或改用支持 Files API 的代理")
    encoded = base64.b64encode(clip.read_bytes()).decode("ascii")
    return f"data:video/mp4;base64,{encoded}"


def _api_video_content(vlm_cfg: dict, video_ref: str,
                       source_fps: float) -> dict:
    """Build provider-compatible video content for Chat Completions."""
    video_url = {"url": video_ref}
    if str((vlm_cfg or {}).get("api_vendor") or "").strip().lower() == "qwen":
        try:
            # Qwen uses fps both for sampling and temporal alignment; cap it
            # to a practical value for hand-action videos.
            video_url["fps"] = min(5.0, max(0.1, float(source_fps)))
        except (TypeError, ValueError):
            video_url["fps"] = 2.0
    return {"type": "video_url", "video_url": video_url}


def _api_generation_options(vlm_cfg: dict | None) -> dict:
    """Return provider-specific OpenAI-compatible generation options.

    Qwen reasoning models can put the entire response in the reasoning field
    when thinking is enabled, leaving ``message.content`` empty. Annotation
    requires a machine-readable final JSON, so disable thinking for Qwen API
    calls. Other providers keep the existing payload unchanged.
    """
    if str((vlm_cfg or {}).get("api_vendor") or "").strip().lower() == "qwen":
        return {"enable_thinking": False}
    return {}


async def _vlm_label_segment_api(video_path: Path, fps: float, seg: dict,
                                 lang: str, client: httpx.AsyncClient,
                                 vlm_cfg: dict, prev_seg: dict | None = None,
                                 next_seg: dict | None = None,
                                 tmp_dir: Path | None = None,
                                 media_note: str = "") -> dict | None:
    """API 原生视频逐段标注:切片 → 上传 → video_url 推理 → 严格 JSON。
    请求失败/解析失败 → None(该段不写入 AI 结果)。"""
    chat_url, _upload_base, model, headers, err = _vlm_endpoint(vlm_cfg)
    if err:
        print(f"[ai_annotation] segment VLM (api) skipped: {err}")
        return None
    clip = _slice_segment_clip(video_path, fps, seg, prev_seg, next_seg,
                               tmp_dir or _tmp_dir("orphan"))
    if clip is None:
        return None
    try:
        video_ref = _inline_video_data_url(clip)
    except (OSError, ValueError) as exc:
        print(f"[ai_annotation] inline video encoding failed: {exc}")
        return None
    prompts = _prompts()
    # Local image sampling and API video input deliberately use the same
    # semantic prompt contract.  Inline video keeps this path compatible with
    # OpenAI-compatible gateways that expose Chat Completions but not Files.
    template = prompts.get("segment_prompt_zh" if lang == "zh" else "segment_prompt_en")
    vocab = prompts.get("vocab_zh" if lang == "zh" else "vocab_en")
    if not (template and vocab):
        print("[ai_annotation] video prompt template missing")
        return None
    segment_prompt = template.format(
        n=0, vocab=vocab, scene_context=_scene_context(seg, lang))
    if media_note:
        segment_prompt = media_note + "\n" + segment_prompt
    segment_prompt += (
        "\n\n输出要求：最终答案必须放在 assistant 的 content 字段中，"
        "只输出一个完整 JSON 对象，不要把答案只放在 reasoning 或思考内容中。")
    content = [
        _api_video_content(vlm_cfg, video_ref, fps),
        {"type": "text", "text": segment_prompt},
    ]
    vlm_cfg_p = prompts.get("vlm") or {}
    configured_budget = max(1, int(vlm_cfg_p.get("max_tokens", 420)))
    first_budget = max(configured_budget, 1200)
    retry_budget = min(4000, max(first_budget + 1000, 2400))
    request_payload = {
        "model": model,
        "messages": [{"role": "user", "content": content}],
        "max_tokens": first_budget,
        "temperature": float(vlm_cfg_p.get("temperature", 0.1)),
    }
    request_payload.update(_api_generation_options(vlm_cfg))
    retry_content = list(content)
    retry_content[-1] = {
        "type": "text",
        "text": (
            "请重新输出本段标注。上一次输出不完整或不是有效 JSON。"
            "现在只输出一个完整 JSON 对象，所有字段都要结束，"
            "不要输出 Markdown、解释或思考过程；最终 JSON 必须放在"
            "assistant content 中。")
    }
    retry_payload = dict(request_payload)
    retry_payload["messages"] = [{"role": "user", "content": retry_content}]
    retry_payload["max_tokens"] = retry_budget

    # 一个分段严格只有两次机会：首次请求 + 一次重试。传输异常、429、
    # 空 content、JSON 无效和 finish_reason=length 都消耗同一个重试机会。
    parsed = None
    last_error: Exception | None = None
    for attempt, payload in enumerate((request_payload, retry_payload)):
        try:
            r = await _vlm_post_api(
                client, chat_url, payload, headers, retries=0,
                stage="label_segment",
                meta={"vendor": str((vlm_cfg or {}).get("api_vendor") or "")})
            text, finish_reason = _api_response_text(r, require_json=True)
            parsed = _parse_segment_result(text, lang) if text else None
            if parsed is not None and (attempt == 1 or finish_reason != "length"):
                break
            last_error = TypeError(
                "API response is incomplete or is not a valid segment JSON")
        except Exception as exc:
            last_error = exc
        parsed = None if attempt == 0 else parsed
    if parsed is None:
        print(f"[ai_annotation] segment VLM (api) failed: {last_error}")
        return None
    return parsed


async def _vlm_propose_api(video_path: Path, fps: float, total_frames: int,
                           vlm_cfg: dict, client: httpx.AsyncClient,
                           tmp_dir: Path, lang: str = "zh",
                           media_note: str = "") -> list[dict]:
    """API 原生视频整片标注(vlm_only 短视频场景):整片切片上传,
    video_url 推理 → 段列表 JSON。失败抛出带分类的 VLMAnnotationError。"""
    chat_url, _upload_base, model, headers, err = _vlm_endpoint(vlm_cfg)
    if err:
        raise VLMAnnotationError("configuration_error", err)
    whole = {"start_frame_index": 0,
             "end_frame_index": max(0, total_frames - 1)}
    clip = _slice_segment_clip(video_path, fps, whole, None, None, tmp_dir)
    if clip is None:
        raise VLMAnnotationError("video_slice_failed", "API VLM 输入视频切片失败")
    try:
        video_ref = _inline_video_data_url(clip)
    except OSError as exc:
        raise VLMAnnotationError(
            "api_video_read_failed", "API VLM 视频读取失败") from exc
    except ValueError as exc:
        raise VLMAnnotationError(
            "api_video_too_large", str(exc)) from exc
    prompts = _prompts()
    template = prompts.get("plan_prompt_zh" if lang == "zh" else "plan_prompt_en")
    vocab = prompts.get("vocab_zh" if lang == "zh" else "vocab_en", "")
    prompt = (template or _VLM_VIDEO_PROPOSE_PROMPT).format(
        n=1, duration_s=total_frames / max(1.0, fps),
        total_frames=total_frames, fps=fps, vocab=vocab)
    if media_note:
        prompt = media_note + "\n" + prompt
    prompt += (
        "\n\n输出要求：最终答案必须放在 assistant 的 content 字段中，"
        "只输出完整 JSON，不要把答案只放在 reasoning 或思考内容中。")
    content = [
        _api_video_content(vlm_cfg, video_ref, fps),
        {"type": "text", "text": prompt},
    ]
    try:
        vlm_options = _prompts().get("vlm") or {}
        configured_budget = max(
            1, int(vlm_options.get("plan_max_tokens", 2400)))
        # Reasoning models spend part of max_tokens on hidden reasoning. Give
        # the final structured plan enough room for scene data and subtasks.
        duration_s = total_frames / max(1.0, fps)
        duration_budget = 2400 + (max(0, int(duration_s - 30)) // 15) * 400
        plan_budget = min(8000, max(configured_budget, duration_budget, 4800))
        request_payload = {
            "model": model,
            "messages": [{"role": "user", "content": content}],
            "max_tokens": plan_budget,
            "temperature": float(vlm_options.get("plan_temperature", 0.1)),
        }
        request_payload.update(_api_generation_options(vlm_cfg))
        r = await _vlm_post_api(
            client, chat_url, request_payload, headers,
            stage="propose",
            meta={"vendor": str((vlm_cfg or {}).get("api_vendor") or "")})
        raw, finish_reason = _api_response_text(r, prefer_plan=True)

        # Reasoning models may spend the whole first budget on analysis and
        # leave no final content. Retry once with a compact contract and more
        # room. A reasoning-field fallback is accepted only when it contains
        # a complete plan contract; free-form reasoning is rejected.
        if finish_reason == "length" or not _has_plan_scene_contract(raw):
            compact_content = list(content)
            compact_content[-1] = {
                "type": "text",
                "text": (
                    "请重新观看视频并只输出一个完整、可解析的 JSON 对象。"
                    "不要输出 Markdown、解释或思考过程。必须包含 scene.objects、"
                    "scene.locations 和 subtasks；每个 subtasks 只保留 index、"
                    "start_s、end_s、action、object、target、hand、status、"
                    "instruction、confidence、boundary_confidence。"
                    "所有动作阶段必须首尾相接覆盖视频，最终 JSON 必须放在"
                    "assistant content 中。")
            }
            retry_payload = dict(request_payload)
            retry_payload["messages"] = [{"role": "user",
                                          "content": compact_content}]
            retry_payload["max_tokens"] = min(
                12000, max(plan_budget + 2000, 8000))
            r = await _vlm_post_api(
                client, chat_url, retry_payload, headers,
                stage="propose_retry",
                meta={"vendor": str((vlm_cfg or {}).get("api_vendor") or "")})
            raw, finish_reason = _api_response_text(r, prefer_plan=True)
        if not raw:
            raise TypeError("API response has no usable JSON content")
    except httpx.ConnectError as exc:
        raise VLMAnnotationError(
            "model_unavailable",
            "API VLM 服务无法连接",
        ) from exc
    except httpx.TimeoutException as exc:
        raise VLMAnnotationError(
            "model_timeout",
            "API VLM 响应超时",
        ) from exc
    except httpx.HTTPStatusError as exc:
        status = exc.response.status_code if exc.response is not None else 0
        if status == 429:
            detail = "API VLM 请求被限流(HTTP 429)，请稍后重试或降低并发"
        else:
            detail = f"API VLM 返回 HTTP {status}"
        provider_detail = _http_error_detail(exc.response)
        if provider_detail:
            detail += f": {provider_detail}"
        code = "api_rate_limited" if status == 429 else "model_http_error"
        raise VLMAnnotationError(code, detail) from exc
    except httpx.RequestError as exc:
        raise VLMAnnotationError(
            "model_connection_failed",
            f"API VLM 请求失败 ({_safe_error_text(exc.__class__.__name__)})",
        ) from exc
    except (KeyError, IndexError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise VLMAnnotationError(
            "invalid_model_response",
            f"API VLM 返回格式无效 ({_safe_error_text(exc)})",
        ) from exc
    except Exception as exc:
        raise VLMAnnotationError(
            "model_request_failed",
            f"API VLM 调用异常 ({_safe_error_text(exc)})",
        ) from exc
    if not _has_plan_scene_contract(raw):
        code, detail = _plan_contract_failure(raw)
        raise VLMAnnotationError(code, detail)
    plan = _parse_plan_result(raw, lang, fps=fps, total_frames=total_frames)
    if not plan:
        raise VLMAnnotationError(
            "empty_plan",
            "API VLM 返回了完整场景信息，但没有可用 subtasks 动作计划",
        )
    return plan


def _match_vlm(seg: dict, vlm_segs: list[dict], fps: float) -> tuple[dict, float]:
    """信号段 ↔ VLM 段时间交叠打分,返回 (最佳匹配, 交叠分 0-1)。"""
    a0, a1 = seg["start_frame_index"] / fps, seg["end_frame_index"] / fps
    best, best_score = None, 0.0
    for v in vlm_segs:
        b0, b1 = v["start_s"], v["end_s"]
        ov = min(a1, b1) - max(a0, b0)
        if ov <= 0:
            continue
        score = 2.0 * ov / ((a1 - a0) + (b1 - b0) + 1e-6)
        if score > best_score:
            best, best_score = v, score
    return best, best_score


# ═══════════════════════════════════════════════════════════
#  主流程 + 任务管理
# ═══════════════════════════════════════════════════════════

def _set_task(task_id: str, **kw) -> None:
    t = _tasks.get(task_id)
    if t is not None:
        t.update(kw)
        # 磁盘镜像:dev 环境 uvicorn 热重载会丢内存任务,落盘后
        # 状态接口仍能读到最后状态(标记 interrupted),前端不再干等。
        try:
            ep = t.get("episode_id")
            if ep:
                _TASKS_DIR.mkdir(parents=True, exist_ok=True)
                _task_file(ep).write_text(
                    json.dumps(t, ensure_ascii=False), encoding="utf-8")
        except OSError:
            pass


def invalidate_ai_annotation_tasks(episode_id: str) -> None:
    """Invalidate in-flight AI work before a workflow generation replaces it."""
    episode_id = str(episode_id)
    _episode_ai_generation[episode_id] = (
        _episode_ai_generation.get(episode_id, 0) + 1)
    for task_id, task in list(_tasks.items()):
        if (str(task.get("episode_id")) == episode_id
                and task.get("status") not in {"done", "failed", "superseded"}):
            _set_task(task_id, status="superseded",
                      detail="工作流已重跑，旧 AI 标注任务已作废",
                      segments_added=0)


def purge_episode_ai_annotation(episode_id: str) -> None:
    """Cancel in-flight AI work and drop the state/ai_tasks mirror.

    Order matters: invalidate first (bumps the generation that gates every
    segment/quality-state write and rewrites the mirror), then pop the
    in-memory tasks and unlink the mirror so nothing recreates it.
    """
    episode_id = str(episode_id)
    invalidate_ai_annotation_tasks(episode_id)
    for task_id, task in list(_tasks.items()):
        if str(task.get("episode_id")) == episode_id:
            _tasks.pop(task_id, None)
    try:
        _task_file(episode_id).unlink(missing_ok=True)
    except OSError:
        pass


def _ai_generation_is_current(episode_id: str, generation: int) -> bool:
    return _episode_ai_generation.get(str(episode_id)) == generation


def _ai_annotation_record(episode_id: str, candidate: dict, index: int,
                          media_sources: list[str], *, pending: bool = False,
                          error: str = "", existing: dict | None = None) -> dict:
    """Build one durable AI segment record.

    ``ai_segment_index`` is stable for the current generation, so a segment
    can be replaced in-place when its request finishes.  Failed segments stay
    visible as ``pending_retry`` but are excluded by the dataset exporter.
    """
    now = _utcnow()
    record = {
        "id": str((existing or {}).get("id") or uuid4()),
        "episode_id": episode_id,
        "label": "" if pending else str(candidate.get("label") or ""),
        "start_frame_index": int(candidate.get("start_frame_index", 0)),
        "end_frame_index": int(candidate.get("end_frame_index", 0)),
        "color": _CANDIDATE_COLOR,
        "sort_order": index,
        "notes": None,
        "source_scope": ["episode"],
        "ai_media_sources": list(media_sources),
        "created_at": str((existing or {}).get("created_at") or now),
        "updated_at": now,
        "keyframes": [],
        "status": "pending_retry" if pending else "confirmed",
        "source": "ai",
        "ai_segment_index": index,
        "ai_retry_pending": pending,
        "ai_error": str(error or ""),
        "ai_score": float(candidate.get("ai_score", 0.0 if pending else 0.5)),
        "ai_reason": (str(candidate.get("ai_reason") or "")
                      + (" · VLM 分段失败，等待重试" if pending
                         else " · 自动确认")),
        "ai_instruction": str(candidate.get("ai_instruction") or ""),
        "skill_id": str(candidate.get("skill_id") or ""),
        "ai_object": str(candidate.get("ai_object") or ""),
        "ai_object_ref": str(candidate.get("ai_object_ref") or ""),
        "ai_object_attributes": str(candidate.get("ai_object_attributes") or ""),
        "ai_source": str(candidate.get("ai_source") or ""),
        "ai_source_ref": str(candidate.get("ai_source_ref") or ""),
        "ai_target": str(candidate.get("ai_target") or ""),
        "ai_target_ref": str(candidate.get("ai_target_ref") or ""),
        "ai_hand": str(candidate.get("ai_hand") or "unknown"),
        "ai_status": str(candidate.get("ai_status") or "uncertain"),
        "task_instruction": str(candidate.get("task_goal") or ""),
        "task_objects": list(candidate.get("task_objects") or []),
        "task_locations": list(candidate.get("task_locations") or []),
        "ai_confidence": float(candidate.get("ai_confidence", 0.0)),
        "boundary_ok": candidate.get("boundary_ok"),
        "boundary_confidence": float(candidate.get("boundary_confidence", 0.0)),
        "label_matched": bool(candidate.get("label_matched", True)),
    }
    return record


def _persist_ai_segment(episode_id: str, generation: int, candidate: dict,
                        index: int, media_sources: list[str], *,
                        pending: bool = False, error: str = "") -> dict:
    """Atomically insert/update one segment without waiting for its siblings."""
    record = _ai_annotation_record(
        episode_id, candidate, index, media_sources,
        pending=pending, error=error)

    def mutator(segs: list[dict]) -> dict:
        if not _ai_generation_is_current(episode_id, generation):
            return {"count": 0, "superseded": True}
        for pos, old in enumerate(segs):
            if old.get("ai_segment_index") == index:
                record["id"] = old.get("id") or record["id"]
                record["created_at"] = old.get("created_at") or record["created_at"]
                segs[pos] = record
                return {"count": 1}
        segs.append(record)
        return {"count": 1}

    result = mutate_annotations(episode_id, mutator)
    if not result.get("superseded"):
        notify_annotations_changed(episode_id, "ai_segment")
    return result


async def _rebuild_lerobot_after_ai(episode_id: str) -> bool:
    """AI 标注自动确认后,重建系统级导出缓存中的 LeRobot 数据集。
    项目源目录仍只保留 data/meta/videos；失败静默 —— 下次重跑工作流时
    导出节点仍会按新标注导出。"""
    try:
        from app.lerobot_export import build_lerobot_dataset
        runs = sorted(
            (r for r in list_runs()
             if r.get("episode_id") == episode_id
             and r.get("status") == "completed"
             and r.get("created_at")),
            key=lambda r: r.get("created_at") or "",
        )
        if not runs:
            return True
        # 有新 run 在排队/执行 → 其导出节点会读到刚确认的标注,无需重建
        newest = max(
            (r for r in list_runs() if r.get("episode_id") == episode_id),
            key=lambda r: r.get("created_at") or "", default=None)
        if newest is not None and newest.get("status") in ("queued", "running"):
            return True
        run = runs[-1]
        graph = run.get("graph") or {}
        export_cfg = None
        for node in graph.get("nodes", []):
            data = node.get("data") or {}
            if data.get("nodeType") == "lerobot_export":
                export_cfg = dict(data.get("config") or {})
                export_cfg.update((run.get("node_configs") or {}).get(node.get("id"), {}))
                break
        if export_cfg is None:
            return True  # 该工作流没有 LeRobot 导出节点,无需重建数据集
        ep = get_episode(episode_id)
        if ep is None:
            return False
        from app.project_dataset import episode_chunk_for_index, episode_row
        project_root = Path(ep["path"])
        source_row = episode_row(project_root, episode_id)
        if source_row is None:
            return False
        source_index = int(source_row.get("episode_index", 0))
        source_data = (project_root / "data"
                       / f"chunk-{episode_chunk_for_index(source_index):03d}"
                       / f"episode_{source_index:06d}.parquet")
        if not source_data.is_file():
            return False
        out_dir = STATE_ROOT / "exports" / str(run["id"]) / "dataset"
        import shutil
        shutil.rmtree(out_dir, ignore_errors=True)
        out_dir.parent.mkdir(parents=True, exist_ok=True)
        import pandas as pd
        columns = set(pd.read_parquet(source_data, engine="pyarrow").columns)
        has_2d = any("keypoints" in str(name) or "2d_present" in str(name)
                      for name in columns)
        has_3d = any("landmarks_3d" in str(name) or "world_position" in str(name)
                      for name in columns)
        version = str(export_cfg.get("version") or "v3.0")
        split_ratio = float(export_cfg.get("split_ratio", 0.9))
        hand_3d_paths = [str(source_data)] if has_3d else None
        hand_3d_right_paths = [str(source_data)] if has_3d else None
        hand_keypoints_paths = [str(source_data)] if has_2d else None
        await asyncio.to_thread(
            build_lerobot_dataset, episode_id, [episode_id], out_dir,
            split_ratio, None,
            hand_keypoints_paths or None,
            hand_3d_paths or None,
            hand_3d_right_paths or None,
            version,
            None,  # hand_3d_unit:重建时不改导出语义(沿用数据集默认)
        )
        run.setdefault("outputs", {}).setdefault("artifacts", {})[
            "ai_rebuild"] = {"dataset": {
                "kind": "dataset", "path": str(out_dir),
                "metadata": {"root": ".", "version": version},
            }}
        from app.localstore import save_run
        save_run(run)
        print(f"[ai_annotation] rebuilt LeRobot dataset: {out_dir}")
        return True
    except Exception as exc:
        print(f"[ai_annotation] LeRobot rebuild skipped: {exc}")
        return False


def run_ai_annotation(episode_id: str, mode: str, min_confidence: float,
                      lang: str = "zh", debounce_sec: float = 2.0,
                      min_seg_sec: float = 0.8,
                      max_segments: int = DEFAULT_MAX_VLM_SEGMENTS,
                      auto_confirm: bool = True,
                      quality_gate: bool = False,
                      video_quality_gate: bool = False,
                      vlm_cfg: dict | None = None) -> str:
    """异步执行 AI 标注,返回 task_id(状态经 /ai-annotate/status 查)。

    AI 成功后直接以 confirmed 写入(无需人工 Confirm)。启用
    AI Quality Review 卡片时，会先检查已落盘帧区间和视频质量，只有完整
    连续且视频无异常才重建 LeRobot 并自动进入 Approved；失败/缺失结果
    留在 Reviewing。视频质检可以独立于 AI 标注完整性检查启用。
    API 分段失败会保留成功结果，并把失败段写成 pending_retry。

    vlm_cfg:VLM 供应商配置(ai_annotation 卡片)。provider=local(缺省)
    走本地 vLLM 抽帧图;provider=api 走 Kimi 原生视频 —— 两模式严格
    分离,配置缺失/请求失败显式报错(任务 detail 可见),绝不互回退。
    """
    episode_id = str(episode_id)
    quality_review_enabled = bool(quality_gate or video_quality_gate)
    if quality_review_enabled:
        # Replacing annotations must immediately remove an old Approved state.
        # If the service is interrupted before the new result is complete, the
        # batch remains in Reviewing and cannot be exported.
        _set_ai_quality_pending(episode_id)
    generation = _episode_ai_generation.get(episode_id, 0) + 1
    _episode_ai_generation[episode_id] = generation
    task_id = str(uuid4())
    _tasks[task_id] = {"episode_id": episode_id, "status": "queued",
                       "progress": 0, "detail": "", "segments_added": 0,
                       "segments_total": 0, "segments_succeeded": 0,
                       "segments_pending": 0, "current_segment": 0,
                       "auto_confirm": True,
                       "quality_gate": bool(quality_gate),
                       "video_quality_gate": bool(video_quality_gate),
                       "generation": generation,
                       "error": "",
                       "error_code": ""}

    async def _run() -> None:
        # AI annotations are always final/confirmed in this application. A
        # failed AI pass writes no fallback candidates and leaves the file
        # empty, so the review page never needs a Confirm action for AI data.
        auto_confirm_eff = True
        try:
            _set_task(task_id, status="loading", detail="定位批次数据")
            await asyncio.sleep(0)
            ep = get_episode(episode_id)
            if ep is None:
                raise RuntimeError("Episode not found")
            ep_dir = _episode_dir(episode_id)
            if ep_dir is None:
                raise RuntimeError("Episode directory not found")
            # Manual AI invocation is also a new generation. Clear every old
            # annotation; a failed run must leave this file empty.
            mutate_annotations(episode_id, lambda segs: segs.clear())
            notify_annotations_changed(episode_id, "reset")
            # fps 实测优先:元数据可能不准(如 30 vs 实际 25),否则去抖
            # 窗口/最小段长/VLM 时间标签全部按错帧率计算 → 段帧数不对应
            fps = _real_episode_fps(episode_id, ep_dir) or float(ep.get("fps") or 30.0)
            total = int(ep.get("frame_count") or 0)
            if total <= 0:
                raise RuntimeError("frame_count unknown")

            # ① 信号切段(或整段兜底)
            candidates: list[dict] = []
            events: list[dict] = []
            if mode in ("signal_only", "signal_vlm"):
                _set_task(task_id, status="signal_segmenting",
                          detail="信号变化点检测中")
                await asyncio.sleep(0)
                events = _signal_events(ep_dir, fps, debounce_sec=debounce_sec,
                                        episode_id=episode_id)
                candidates = _events_to_segments(events, total, fps,
                                                 min_seg_sec=min_seg_sec,
                                                 max_segments=max_segments)
                if not candidates:
                    candidates = [{"start_frame_index": 0,
                                   "end_frame_index": max(1, total - 1),
                                   "label": "phase",
                                   "ai_reason": "no signal events"}]

            # ② VLM 标注
            vlm_err_detail: str | None = None
            vlm_error_code = ""
            vlm_fail = 0
            partial_err_detail: str | None = None
            provider_api = False
            media_sources: list[str] = []
            media_note = ""
            if mode in ("signal_vlm", "vlm_only"):
                # 统一的两阶段管线:
                # 1) VLM 观看完整视频,提出 task/subtasks 和时间边界;
                # 2) VLM 逐段识别 action/object/target/status。
                # signal_events 只用于确定段内上下文;AI 失败时不写机械候选。
                # 多设备时先生成同步多视角临时视频,最终仍只写一套 episode 标注。
                tmp_dir = _tmp_dir(task_id)
                video, media_sources, media_note = _build_multiview_video(
                    ep_dir, tmp_dir, fps, total, episode_id)
                if video is None:
                    vlm_error_code = "input_video_missing"
                    vlm_err_detail = "没有可供 VLM 读取的原始视频"
                    auto_confirm_eff = False
                    candidates = []
                else:
                    _c_url, _u_base, _m, _h, vlm_err_detail = \
                        _vlm_endpoint(vlm_cfg)
                    if vlm_err_detail:
                        # API 模式配置缺失 → 绝不回退本地 vLLM，保持空标注。
                        vlm_error_code = "configuration_error"
                        auto_confirm_eff = False
                        _set_task(task_id, detail=vlm_err_detail)
                        candidates = []
                    else:
                        _set_task(task_id, status="vlm_analyzing",
                                  detail="VLM 正在观看完整视频并规划子任务")
                        await asyncio.sleep(0)
                        plan: list[dict] = []
                        if (vlm_cfg or {}).get("vlm_provider") == "api":
                            try:
                                async with httpx.AsyncClient(
                                        timeout=API_REQUEST_TIMEOUT_SEC) as pclient:
                                    plan = await _vlm_propose_api(
                                        video, fps, total, vlm_cfg, pclient,
                                        tmp_dir, lang=lang, media_note=media_note)
                            except VLMAnnotationError as exc:
                                vlm_error_code = exc.code
                                vlm_err_detail = exc.detail
                        else:
                            try:
                                plan = await asyncio.to_thread(
                                    _vlm_propose, video, fps, total, lang,
                                    media_note, max_segments)
                            except VLMAnnotationError as exc:
                                vlm_error_code = exc.code
                                vlm_err_detail = exc.detail
                        if not vlm_err_detail:
                            planned = _plan_to_candidates(
                                plan, total, fps,
                                max_segments=max_segments or DEFAULT_MAX_VLM_SEGMENTS)
                            if planned:
                                candidates = planned
                            else:
                                # 合法场景契约但没有可用动作段时保持空标注。
                                vlm_error_code = "empty_plan"
                                vlm_err_detail = (
                                    "VLM 返回了完整场景信息，但没有可用 subtasks 动作计划"
                                )
                                auto_confirm_eff = False
                                candidates = []
                        if vlm_err_detail:
                            auto_confirm_eff = False
                            candidates = []
                        _set_task(
                            task_id,
                            status="vlm_analyzing",
                            detail=(f"VLM 失败：{vlm_err_detail}"
                                    if vlm_err_detail
                                    else f"VLM 逐段识别中({len(candidates)} 段)"),
                        )
                        tmp_dir = _tmp_dir(task_id)
                        provider_api = (vlm_cfg or {}).get("vlm_provider") == "api"
                        if candidates and not vlm_err_detail:
                            segment_total = len(candidates)
                            _set_task(task_id, segments_total=segment_total,
                                      segments_succeeded=0, segments_pending=segment_total,
                                      current_segment=0, progress=0.45)

                            # 先建立可恢复的分段索引；随后每个成功结果会原位
                            # 替换对应 pending_retry 记录。
                            if provider_api:
                                initial_records = [
                                    _ai_annotation_record(
                                        episode_id, seg, i, media_sources,
                                        pending=True)
                                    for i, seg in enumerate(candidates)]

                                def _seed_pending(segs: list[dict]) -> dict:
                                    if not _ai_generation_is_current(episode_id, generation):
                                        return {"count": 0, "superseded": True}
                                    segs.extend(initial_records)
                                    return {"count": len(initial_records)}

                                seeded = mutate_annotations(episode_id, _seed_pending)
                                if seeded.get("superseded"):
                                    return
                                notify_annotations_changed(episode_id, "ai_segments_pending")

                            def _apply_result(i: int, res: dict | None,
                                              error: str = "") -> None:
                                nonlocal vlm_fail
                                candidate = candidates[i]
                                if res is None:
                                    vlm_fail += 1
                                    candidate.update(
                                        _vlm_ok=False, ai_score=0.0,
                                        ai_instruction="", ai_confidence=0.0,
                                        ai_error=error or "VLM 分段请求失败")
                                    _persist_ai_segment(
                                        episode_id, generation, candidate, i,
                                        media_sources, pending=True,
                                        error=error or "VLM 分段请求失败")
                                    return
                                candidate.update({
                                    "_vlm_ok": True,
                                    "label": res.get("label") or "",
                                    "skill_id": res.get("skill_id") or "",
                                    "ai_object": res.get("object") or "",
                                    "ai_object_ref": res.get("object_ref") or "",
                                    "ai_object_attributes": res.get("object_attributes") or "",
                                    "ai_source": res.get("source") or "",
                                    "ai_source_ref": res.get("source_ref") or "",
                                    "ai_target": res.get("target") or "",
                                    "ai_target_ref": res.get("target_ref") or "",
                                    "ai_hand": res.get("hand") or "unknown",
                                    "ai_status": res.get("status") or "uncertain",
                                    "ai_instruction": res.get("instruction") or "",
                                    "ai_confidence": res.get("confidence", 0.0),
                                    "ai_score": res.get("confidence", 0.0),
                                    "ai_reason": (candidate.get("ai_reason") or "") + " · VLM",
                                    "boundary_ok": res.get("boundary_ok"),
                                    "boundary_confidence": res.get("boundary_confidence", 0.0),
                                    "label_matched": res.get("label_matched", True),
                                })
                                _persist_ai_segment(
                                    episode_id, generation, candidate, i,
                                    media_sources)

                            if provider_api:
                                # API 视频生成严格串行：限速、超时和重试均只
                                # 作用于 API 路径，单段结束立即落盘。
                                async with httpx.AsyncClient(
                                        timeout=API_REQUEST_TIMEOUT_SEC) as client:
                                    for i, seg in enumerate(candidates):
                                        _set_task(
                                            task_id, current_segment=i + 1,
                                            detail=f"API 分段识别中({i + 1}/{segment_total})")
                                        try:
                                            res = await _vlm_label_segment_api(
                                                video, fps, seg, lang, client,
                                                vlm_cfg,
                                                prev_seg=(candidates[i - 1]
                                                          if i > 0 else None),
                                                next_seg=(candidates[i + 1]
                                                          if i + 1 < segment_total else None),
                                                tmp_dir=tmp_dir,
                                                media_note=media_note)
                                            error = ""
                                        except Exception as exc:
                                            res = None
                                            error = _safe_error_text(exc)
                                        _apply_result(
                                            i, res, error or "API 分段返回无效 JSON")
                                        done = i + 1
                                        success = done - vlm_fail
                                        _set_task(
                                            task_id,
                                            progress=min(0.90, 0.45 + 0.45 * done / segment_total),
                                            segments_succeeded=success,
                                            segments_pending=vlm_fail,
                                            detail=(f"API 分段识别 {done}/{segment_total}，"
                                                    f"成功 {success}，待重试 {vlm_fail}"))
                            else:
                                # 本地 vLLM 保留独立的视频/抽帧并发实现；它不会
                                # 经过 API 限速或 API 超时设置。
                                async with httpx.AsyncClient(timeout=600) as client:
                                    sem = asyncio.Semaphore(4)

                                    async def _label_local(i: int, seg: dict):
                                        async with sem:
                                            try:
                                                result = await _vlm_label_segment(
                                                    video, fps, seg, lang, client,
                                                    prev_seg=(candidates[i - 1]
                                                              if i > 0 else None),
                                                    next_seg=(candidates[i + 1]
                                                              if i + 1 < segment_total else None),
                                                    media_note=media_note)
                                                return i, result, ""
                                            except Exception as exc:
                                                return i, None, _safe_error_text(exc)

                                    results = await asyncio.gather(
                                        *(_label_local(i, s)
                                          for i, s in enumerate(candidates)))
                                for i, res, error in results:
                                    _apply_result(
                                        i, res, error or "本地 VLM 分段返回无效 JSON")

                            if provider_api:
                                # API 的成功段已经逐段持久化；这里只保留成功
                                # 段进入后续合并/导出，失败段继续留在文件中。
                                candidates = [c for c in candidates
                                              if c.get("_vlm_ok")]
                                if vlm_fail:
                                    partial_err_detail = (
                                        f"{vlm_fail} 个 API 分段失败，已保留成功段，"
                                        "失败段标记为待重试")
                                    vlm_error_code = "partial_segment_failure"

                        if not provider_api:
                            # local vLLM 才做相邻合并和全局质检；API 结果已经
                            # 逐段写入，保留原始段索引，避免质检合并造成已落盘
                            # 的 segment 记录与内存列表错位。
                            merged: list[dict] = []
                            for seg in candidates:
                                if (merged
                                        and merged[-1].get("label")
                                        and merged[-1].get("label") == seg.get("label")
                                        and merged[-1].get("boundary_ok") is not True):
                                    merged[-1]["end_frame_index"] = seg["end_frame_index"]
                                    merged[-1]["boundary_ok"] = seg.get("boundary_ok")
                                    merged[-1]["boundary_confidence"] = seg.get(
                                        "boundary_confidence", 0.0)
                                    merged[-1]["ai_reason"] = (
                                        merged[-1].get("ai_reason") or "") + " · 合并同标签"
                                    continue
                                merged.append(seg)
                            candidates = merged
                            gcheck = {"issues": [], "merges": []}
                            if not vlm_err_detail:
                                async with httpx.AsyncClient(timeout=600) as gclient:
                                    gcheck = await _vlm_global_check(
                                        candidates, fps, lang, gclient, vlm_cfg)
                            for pair in sorted(gcheck.get("merges") or [],
                                               reverse=True):
                                if not (isinstance(pair, (list, tuple)) and len(pair) >= 2):
                                    continue
                                try:
                                    i = int(pair[0])
                                except (TypeError, ValueError):
                                    continue
                                if 0 <= i < len(candidates) - 1:
                                    a, b = candidates[i], candidates[i + 1]
                                    if (not a.get("label") or not b.get("label")
                                            or a.get("label") == b.get("label")):
                                        a["end_frame_index"] = b["end_frame_index"]
                                        a["boundary_ok"] = b.get("boundary_ok")
                                        a["ai_reason"] = (a.get("ai_reason") or "") \
                                            + " · 全局校验合并"
                                        candidates.pop(i + 1)
                            for issue in gcheck.get("issues") or []:
                                if not isinstance(issue, dict):
                                    continue
                                try:
                                    idx = int(issue.get("index", -1))
                                except (TypeError, ValueError):
                                    continue
                                if 0 <= idx < len(candidates):
                                    candidates[idx]["ai_reason"] = (
                                        candidates[idx].get("ai_reason") or "") + \
                                        f" · 全局校验:{issue.get('problem', '')}"
                                    candidates[idx]["ai_confidence"] = (
                                        float(candidates[idx].get("ai_confidence", 1.0))
                                        * 0.8)
            else:  # signal_only
                for seg in candidates:
                    seg.setdefault("ai_score", 0.5)
                    seg.setdefault("ai_instruction", "")
                    seg.setdefault("ai_confidence", 0.0)

            # ④ 过滤 + 写回。工作流重跑和 AI 任务开始时均已清空旧标注；
            # 这里仅允许成功的 AI 结果落库，且直接以 confirmed 写入。
            candidates = [c for c in candidates
                          if c["end_frame_index"] > c["start_frame_index"]
                          and (mode == "signal_only"
                               or c.get("boundary_ok") is False
                               or c.get("ai_confidence", 1.0)
                               >= min_confidence)]
            if not auto_confirm_eff:
                # Failure is an empty-result terminal state.  This guard is
                # deliberately independent of the current candidate list so
                # a future fallback path cannot leak mechanical segments.
                candidates = []
                detail = vlm_err_detail or (
                    "AI 执行失败；当前批次保持空标注，请检查模型后重试")
                if quality_review_enabled and _ai_generation_is_current(episode_id, generation):
                    _record_ai_quality_failure(
                        episode_id, total, vlm_error_code or "ai_annotation_failed")
                _set_task(task_id, status="failed", detail=detail,
                          segments_added=0,
                          error=vlm_error_code or "ai_annotation_failed",
                          error_code=vlm_error_code or "ai_annotation_failed")
                return
            if not candidates:
                detail = (partial_err_detail
                          or "AI 未生成有效标注；当前批次保持空标注，请检查模型后重试")
                if quality_review_enabled and _ai_generation_is_current(episode_id, generation):
                    _record_ai_quality_failure(
                        episode_id, total, vlm_error_code or "no_valid_segments")
                _set_task(task_id, status="failed", detail=detail,
                          segments_added=0,
                          segments_pending=vlm_fail,
                          error="no_valid_segments",
                          error_code="no_valid_segments")
                return
            _set_task(task_id, status="writing", detail="写入 AI 标注")
            await asyncio.sleep(0)

            def mutator(segs: list[dict]) -> dict:
                if not _ai_generation_is_current(episode_id, generation):
                    return {"count": 0, "superseded": True}
                # This task is a complete new generation.  Do not merge with
                # annotations created/edited while it was running: rerunning
                # a workflow intentionally replaces the whole episode set.
                # On failure candidates is empty, so this also guarantees an
                # empty result rather than a stale or manual fallback.
                # API 段已经在请求完成时逐段写入；这里不能清掉其中的
                # pending_retry 记录，也不能等待所有段才首次落盘。
                if provider_api:
                    return {"count": len(candidates)}
                segs[:] = []
                now = _utcnow()
                for i, c in enumerate(candidates):
                    # 规则校验结果仍写入 boundary_ok/boundary_confidence，供
                    # 审核页只读展示；AI 结果不再进入候选/Confirm 状态。
                    status = "confirmed"
                    segs.append({
                        "id": str(uuid4()),
                        "episode_id": episode_id,
                        "label": c["label"],
                        "start_frame_index": int(c["start_frame_index"]),
                        "end_frame_index": int(c["end_frame_index"]),
                        "color": _CANDIDATE_COLOR,
                        "sort_order": i,
                        "notes": None,
                        "source_scope": ["episode"],
                        "ai_media_sources": list(media_sources),
                        "created_at": now,
                        "updated_at": now,
                        "keyframes": [],
                        "status": status,
                        "source": "ai",
                        "ai_score": float(c.get("ai_score", 0.5)),
                        "ai_reason": str(c.get("ai_reason") or "")
                                     + " · 自动确认",
                        "ai_instruction": str(c.get("ai_instruction") or ""),
                        "skill_id": str(c.get("skill_id") or ""),
                        "ai_object": str(c.get("ai_object") or ""),
                        "ai_object_ref": str(c.get("ai_object_ref") or ""),
                        "ai_object_attributes": str(
                            c.get("ai_object_attributes") or ""),
                        "ai_source": str(c.get("ai_source") or ""),
                        "ai_source_ref": str(c.get("ai_source_ref") or ""),
                        "ai_target": str(c.get("ai_target") or ""),
                        "ai_target_ref": str(c.get("ai_target_ref") or ""),
                        "ai_hand": str(c.get("ai_hand") or "unknown"),
                        "ai_status": str(c.get("ai_status") or "uncertain"),
                        "task_instruction": str(c.get("task_goal") or ""),
                        "task_objects": list(c.get("task_objects") or []),
                        "task_locations": list(c.get("task_locations") or []),
                        "ai_confidence": float(c.get("ai_confidence", 0.0)),
                        "boundary_ok": c.get("boundary_ok"),
                        "boundary_confidence": float(c.get("boundary_confidence", 0.0)),
                        "label_matched": bool(c.get("label_matched", True)),
                    })
                return {"count": len(candidates)}

            result = mutate_annotations(episode_id, mutator)
            if result.get("superseded"):
                _set_task(task_id, status="superseded",
                          detail="工作流已重跑，旧 AI 标注任务已作废",
                          segments_added=0)
                return
            notify_annotations_changed(episode_id, "ai_suggest")

            quality_report = None
            if quality_review_enabled and _ai_generation_is_current(episode_id, generation):
                # Validate the persisted file, not the in-memory candidates:
                # API failures are represented by pending_retry records and
                # local failures may have been filtered before writing.
                if quality_gate:
                    quality_report = _annotation_quality_report(
                        episode_id, total, list_annotations(episode_id))
                else:
                    # A video-only quality card must not unexpectedly impose
                    # an AI annotation coverage rule on an unrelated branch.
                    quality_report = {
                        "passed": True,
                        "reason": "annotation_check_skipped",
                        "total_frames": total,
                    }
                if video_quality_gate:
                    # Video inspection is CPU/IO work.  Keep it away from the
                    # event loop so status polling and page navigation remain
                    # responsive while a long MP4 is being sampled.
                    quality_report["video"] = await check_video_quality_async(
                        ep_dir, total, fps)
                    if not quality_report["video"].get("passed"):
                        quality_report["passed"] = False
                        quality_report["reason"] = "video_quality_failed"
                if not quality_report.get("passed"):
                    if not _ai_generation_is_current(episode_id, generation):
                        _set_task(task_id, status="superseded",
                                  detail="工作流已重跑，旧 AI 标注任务已作废",
                                  segments_added=0)
                        return
                    _set_ai_quality_state(episode_id, quality_report)
                    detail = ("视频质检未通过，批次保留在审核页面"
                              if quality_report.get("reason") == "video_quality_failed"
                              else "AI 标注未完整，批次保留在审核页面")
                    _set_task(
                        task_id,
                        status="failed",
                        progress=1.0,
                        detail=detail,
                        quality_status="failed",
                        quality_report=quality_report,
                        error=("video_quality_failed"
                               if quality_report.get("reason") == "video_quality_failed"
                               else "annotation_quality_failed"),
                        error_code=("video_quality_failed"
                                    if quality_report.get("reason") == "video_quality_failed"
                                    else "annotation_quality_failed"),
                        segments_added=len(candidates),
                        segments_pending=vlm_fail,
                    )
                    # Do not rebuild/export a partial or damaged result.
                    return

            if auto_confirm_eff:
                if not _ai_generation_is_current(episode_id, generation):
                    _set_task(task_id, status="superseded",
                              detail="工作流已重跑，旧 AI 标注任务已作废",
                              segments_added=0)
                    return
                # 直接写入数据集:重建最新 run 的 LeRobot 产物
                # (parquet 标注列),无需人工确认；不生成独立的
                # meta/annotations.jsonl。
                _set_task(task_id, status="exporting",
                          detail="标注已自动确认,重建 LeRobot 数据集中…")
                await asyncio.sleep(0)
                rebuilt = await _rebuild_lerobot_after_ai(episode_id)
                if quality_review_enabled and not rebuilt:
                    if not _ai_generation_is_current(episode_id, generation):
                        _set_task(task_id, status="superseded",
                                  detail="工作流已重跑，旧 AI 标注任务已作废",
                                  segments_added=0)
                        return
                    failed_report = dict(quality_report or {})
                    failed_report.update({
                        "passed": False,
                        "reason": "export_rebuild_failed",
                    })
                    _set_ai_quality_state(episode_id, failed_report)
                    _set_task(
                        task_id,
                        status="failed",
                        progress=1.0,
                        detail="导出产物未生成，批次保留在审核页面",
                        quality_status="failed",
                        quality_report=failed_report,
                        error="export_rebuild_failed",
                        error_code="export_rebuild_failed",
                    )
                    return

            if quality_review_enabled and _ai_generation_is_current(episode_id, generation):
                # Only after all enabled checks pass and the optional LeRobot
                # rebuild succeeds may the old UI move to Approved.
                _set_ai_quality_state(
                    episode_id, quality_report or {}, auto_approve=True)
            if vlm_err_detail:
                # VLM 阶段失败(配置缺失/全部请求失败):保持空标注，任务
                # 标记 failed 并提供错误文案；绝不回退到机械候选。
                _set_task(task_id, status="failed", detail=vlm_err_detail,
                          segments_added=len(candidates),
                          error=vlm_error_code or "vlm_failed",
                          error_code=vlm_error_code or "vlm_failed")
            elif partial_err_detail:
                _set_task(
                    task_id, status="done", progress=1.0,
                    detail=(f"完成:{len(candidates)} 个 AI 标注(已确认并写入数据集)；"
                            f"{vlm_fail} 段失败，已标记为待重试"),
                    segments_added=len(candidates),
                    segments_pending=vlm_fail,
                    error=vlm_error_code or "partial_segment_failure",
                    error_code=vlm_error_code or "partial_segment_failure")
            else:
                _set_task(task_id, status="done", progress=1.0,
                          detail=f"完成:{len(candidates)} 个 AI 标注"
                                 + ("(已确认并写入数据集)"
                                    if auto_confirm_eff else "")
                                 + (f";{vlm_fail} 段 VLM 失败，已标记待重试"
                                    if vlm_fail else ""),
                          segments_added=len(candidates),
                          error="", error_code="")
        except VLMAnnotationError as exc:
            if quality_review_enabled and _ai_generation_is_current(episode_id, generation):
                try:
                    failed_ep = get_episode(episode_id) or {}
                    _record_ai_quality_failure(
                        episode_id,
                        int(failed_ep.get("frame_count") or 0),
                        exc.code,
                    )
                except Exception:
                    pass
            if _ai_generation_is_current(episode_id, generation):
                _set_task(task_id, status="failed", detail=exc.detail,
                          error=exc.code, error_code=exc.code,
                          segments_added=0)
            else:
                _set_task(task_id, status="superseded",
                          detail="工作流已重跑，旧 AI 标注任务已作废",
                          segments_added=0)
        except Exception as exc:
            if quality_review_enabled and _ai_generation_is_current(episode_id, generation):
                try:
                    failed_ep = get_episode(episode_id) or {}
                    _record_ai_quality_failure(
                        episode_id,
                        int(failed_ep.get("frame_count") or 0),
                        "unexpected_error",
                    )
                except Exception:
                    pass
            if _ai_generation_is_current(episode_id, generation):
                detail = _safe_error_text(exc) or "AI 标注执行异常"
                _set_task(task_id, status="failed", detail=detail,
                          error="unexpected_error", error_code="unexpected_error",
                          segments_added=0)
            else:
                _set_task(task_id, status="superseded",
                          detail="工作流已重跑，旧 AI 标注任务已作废",
                          segments_added=0)
        finally:
            # 临时切片目录清理:成功/失败都清(目录不存在时静默),
            # 防止 data/tmp/ai_vlm_* 累积。
            _cleanup_tmp(task_id)

    asyncio.get_running_loop().create_task(_run())
    return task_id


# ═══════════════════════════════════════════════════════════
#  API
# ═══════════════════════════════════════════════════════════

@router.post("/episode/{episode_id}/ai-annotate")
async def ai_annotate(episode_id: str, body: dict):
    if get_episode(episode_id) is None:
        raise HTTPException(status_code=404, detail="Episode not found")
    # 配置来源 = 工作流卡片(AI Annotation 节点 config),后端自己读:
    # 前端只管在卡片填好保存,body 旧参数仍可覆盖(兼容现前端)。
    cfg = _workflow_ai_cfg(episode_id) or {}
    mode = str(body.get("mode") or cfg.get("mode") or "signal_vlm")
    if mode not in ("signal_only", "signal_vlm", "vlm_only"):
        raise HTTPException(status_code=400, detail="invalid mode")
    min_conf = float(body.get("min_confidence", cfg.get("min_confidence", 0.0)))
    requested_lang = str(body.get("lang") or cfg.get("prompt_language") or "zh").lower()
    # Keep the public/manual trigger aligned with the workflow editor: the
    # only supported output languages are Chinese and English.
    lang = "en" if requested_lang in {"en", "english"} else "zh"
    # P2 参数化:切段参数由 AI Annotation 卡片 config 驱动(前端透传)
    debounce_sec = float(body.get("debounce_sec") or cfg.get("debounce_sec") or 2.0)
    min_seg_sec = float(body.get("min_seg_sec") or cfg.get("min_seg_sec") or 0.8)
    max_segments = int(body.get("max_segments") or cfg.get("max_segments")
                       or DEFAULT_MAX_VLM_SEGMENTS)
    # 默认自动确认:AI 标注完直接进数据集,无需人工 Confirm
    # AI results in this product are always final/confirmed. The old
    # candidate/Confirm flow is intentionally disabled.
    auto_confirm = True
    # VLM 供应商配置(卡片 vlm_provider/api_vendor/api_model/api_key/
    # api_base_url);body 同名字段可覆盖
    vlm_cfg = {k: (body.get(k) if body.get(k) is not None
                   else cfg.get(k, ""))
               for k in ("vlm_provider", "api_vendor", "api_model",
                         "api_key", "api_base_url")}
    # The browser may select one of several configured providers without
    # receiving any secret. Resolve that selection server-side.
    if not vlm_cfg.get("api_key") and isinstance(cfg.get("api_providers"), list):
        selected = next((item for item in cfg["api_providers"]
                         if isinstance(item, dict)
                         and str(item.get("vendor") or "") == str(vlm_cfg.get("api_vendor") or "")
                         and str(item.get("model") or "") == str(vlm_cfg.get("api_model") or "")), None)
        if selected:
            vlm_cfg["api_key"] = selected.get("key") or ""
            vlm_cfg["api_base_url"] = selected.get("base_url") or vlm_cfg.get("api_base_url")
    task_id = run_ai_annotation(episode_id, mode, min_conf, lang,
                                debounce_sec=debounce_sec,
                                min_seg_sec=min_seg_sec,
                                max_segments=max_segments,
                                auto_confirm=auto_confirm,
                                quality_gate=_workflow_ai_quality_enabled(episode_id),
                                video_quality_gate=_workflow_video_quality_enabled(episode_id),
                                vlm_cfg=vlm_cfg)
    return {"task_id": task_id, "status": "queued"}


@router.get("/episode/{episode_id}/ai-annotate/status")
async def ai_annotate_status(episode_id: str):
    # 返回该批次最近一次任务(插入序倒序 = 最新)
    for t in reversed(list(_tasks.values())):
        if t.get("episode_id") == episode_id:
            return t
    # 内存无任务 → 读磁盘镜像:若最后状态是进行中,说明服务重启
    # 打断了任务,标记 interrupted 让前端停止轮询并提示重新触发。
    try:
        disk = json.loads(_task_file(episode_id).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {"episode_id": episode_id, "status": "idle"}
    if disk.get("status") in ("queued", "loading", "signal_segmenting",
                              "vlm_analyzing", "writing", "exporting"):
        disk["status"] = "interrupted"
        disk["detail"] = str(disk.get("detail") or "") + "(服务重启,任务中断,请重新触发)"
    return disk


def ai_annotation_node_config(graph: dict, node_configs: dict) -> dict | None:
    """从工作流图(快照)提取 ai_annotation 节点配置:节点 data.config
    与工作流级 node_configs 覆盖合并;没有该节点返回 None。"""
    for node in (graph or {}).get("nodes", []) or []:
        data = node.get("data") or {}
        if canonical_node_type(data.get("nodeType")) == "ai_annotation":
            cfg = dict(data.get("config") or {})
            cfg.update((node_configs or {}).get(node.get("id"), {}))
            return cfg
    return None


def _connected_quality_node_config(graph: dict, node_configs: dict,
                                   source_types: set[str]) -> dict | None:
    """Return a quality-card config when a supported source reaches it."""
    nodes = (graph or {}).get("nodes", []) or []
    edges = (graph or {}).get("edges", []) or []
    source_ids = {
        str(node.get("id")) for node in nodes
        if canonical_node_type((node.get("data") or {}).get("nodeType")) in source_types
        and node.get("id") is not None
    }
    quality_nodes = {
        str(node.get("id")): node for node in nodes
        if canonical_node_type((node.get("data") or {}).get("nodeType")) == "ai_quality_review"
        and node.get("id") is not None
    }
    if not source_ids or not quality_nodes:
        return None

    outgoing: dict[str, set[str]] = {}
    for edge in edges:
        source, target = edge.get("source"), edge.get("target")
        if source is None or target is None:
            continue
        outgoing.setdefault(str(source), set()).add(str(target))

    reachable = set(source_ids)
    queue = list(source_ids)
    while queue:
        current = queue.pop(0)
        for target in outgoing.get(current, ()):
            if target not in reachable:
                reachable.add(target)
                queue.append(target)

    for node_id, node in quality_nodes.items():
        if node_id not in reachable:
            continue
        data = node.get("data") or {}
        cfg = dict(data.get("config") or {})
        cfg.update((node_configs or {}).get(node.get("id"), {}))
        return cfg
    return None


def ai_quality_review_node_config(graph: dict, node_configs: dict) -> dict | None:
    """Return the quality card only when it is connected to AI Annotation.

    A card sitting alone in the canvas is configuration only.  The gate is
    enabled for a directed path ``AI Annotation -> ... -> AI Quality Review``;
    allowing pass-through nodes in the middle keeps existing workflow layouts
    compatible while making a dangling quality card inert.
    """
    return _connected_quality_node_config(
        graph, node_configs, {"ai_annotation"})


def video_quality_review_node_config(graph: dict, node_configs: dict) -> dict | None:
    """Return the quality card when it is connected to a media/data source.

    This enables a video-only workflow to run the post-processing media gate,
    while an unconnected ``AI Quality Review`` card remains inert.  The list
    intentionally contains existing input/process node types only; export
    nodes cannot accidentally activate a review gate.
    """
    return _connected_quality_node_config(
        graph,
        node_configs,
        {
            "rgb_camera", "mono_camera", "rgbd_camera", "fisheye_camera", "stereo_camera",
            "stereo_rgbd_camera",
            "glove_sensor", "mediapipe_hand", "annotation",
            *HAND_PROCESS_TYPES,
            "human_review", "ai_annotation",
        },
    )


def _workflow_ai_cfg(episode_id: str) -> dict | None:
    """批次所属项目的绑定工作流里,第一个含 ai_annotation 节点的配置
    (卡片填的 mode/vlm_provider/api_* 都在这里)。手动触发标注时后端
    自己读 —— 前端只管在卡片填好保存。"""
    ep = get_episode(episode_id)
    if ep is None:
        return None
    project_name = ep.get("project")
    project = None
    if project_name:
        project = next((p for p in list_projects()
                        if p.get("name") == project_name), None)
    if project is None:
        return None
    # 新项目是一对一 workflow_id；workflow_ids 仅作为旧状态兼容。
    workflow_id = project.get("workflow_id")
    workflow_ids = project.get("workflow_ids") or []
    if workflow_id:
        workflow_ids = [workflow_id]
    elif not isinstance(workflow_ids, list):
        workflow_ids = [workflow_ids] if workflow_ids else []
    for wf_id in workflow_ids:
        wf = get_workflow(wf_id)
        if wf is None:
            continue
        cfg = ai_annotation_node_config(wf.get("graph") or {},
                                        wf.get("node_configs") or {})
        if cfg is not None:
            return cfg
    return None


def _workflow_ai_quality_enabled(episode_id: str) -> bool:
    """Whether the current project workflow explicitly enables the quality gate."""
    ep = get_episode(episode_id)
    if ep is None:
        return False
    project = next(
        (p for p in list_projects() if p.get("name") == ep.get("project")),
        None,
    )
    if project is None:
        return False
    workflow_id = project.get("workflow_id")
    workflow_ids = project.get("workflow_ids") or []
    if workflow_id:
        workflow_ids = [workflow_id]
    elif not isinstance(workflow_ids, list):
        workflow_ids = [workflow_ids] if workflow_ids else []
    for wf_id in workflow_ids:
        wf = get_workflow(wf_id)
        if wf is None:
            continue
        if ai_quality_review_node_config(
                wf.get("graph") or {}, wf.get("node_configs") or {}) is not None:
            return True
    return False


def _workflow_video_quality_enabled(episode_id: str) -> bool:
    """Whether the bound workflow connects quality review to a source."""
    ep = get_episode(episode_id)
    if ep is None:
        return False
    project = next(
        (p for p in list_projects() if p.get("name") == ep.get("project")),
        None,
    )
    if project is None:
        return False
    workflow_id = project.get("workflow_id")
    workflow_ids = project.get("workflow_ids") or []
    if workflow_id:
        workflow_ids = [workflow_id]
    elif not isinstance(workflow_ids, list):
        workflow_ids = [workflow_ids] if workflow_ids else []
    for wf_id in workflow_ids:
        wf = get_workflow(wf_id)
        if wf is None:
            continue
        if video_quality_review_node_config(
                wf.get("graph") or {}, wf.get("node_configs") or {}) is not None:
            return True
    return False


def _public_ai_config(cfg: dict | None) -> dict:
    """Return workflow AI settings without exposing provider credentials."""
    result = dict(cfg or {})
    result["api_key_present"] = bool(result.get("api_key"))
    result.pop("api_key", None)
    providers = result.get("api_providers")
    if isinstance(providers, list):
        safe = []
        for provider in providers:
            if not isinstance(provider, dict):
                continue
            item = {key: provider.get(key) for key in ("vendor", "model", "base_url")}
            item["key_present"] = bool(provider.get("key"))
            safe.append(item)
        result["api_providers"] = safe
    return result


@router.get("/episode/{episode_id}/ai-annotation-enabled")
async def ai_annotation_enabled(episode_id: str):
    """Return workflow AI configuration plus a safe last-run summary.

    The browser needs to know whether the API node is configured and whether
    the last run actually produced annotations. Never return API credentials.
    """
    if get_episode(episode_id) is None:
        raise HTTPException(status_code=404, detail="Episode not found")
    cfg = _workflow_ai_cfg(episode_id)
    task = None
    for candidate in reversed(list(_tasks.values())):
        if str(candidate.get("episode_id")) == str(episode_id):
            task = dict(candidate)
            break
    if task is None:
        try:
            task = json.loads(_task_file(episode_id).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            task = None
    if isinstance(task, dict):
        task = {key: task.get(key) for key in (
            "status", "progress", "detail", "segments_added",
            "segments_total", "segments_succeeded", "segments_pending",
            "current_segment", "error", "error_code", "generation")}
    return {
        "enabled": cfg is not None,
        "config": _public_ai_config(cfg),
        "api": {
            "configured": bool(cfg and str(cfg.get("vlm_provider") or "") == "api"),
            "vendor": (cfg or {}).get("api_vendor") or "",
            "model": (cfg or {}).get("api_model") or "",
            "base_url": (cfg or {}).get("api_base_url") or "",
            "key_present": bool((cfg or {}).get("api_key")),
        },
        "last_task": task,
        "annotation_count": len(list_annotations(episode_id)),
    }
