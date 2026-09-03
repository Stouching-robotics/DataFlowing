"""Standardized, model-independent names for workflow input devices.

The collector's ``name`` and video ``source_key`` values are compatibility
identifiers: workers and historical batches still use them to locate files.
This module provides a separate UI-facing classification/name so a D435,
DECXIN, or a future camera model follows the same naming rules.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Iterable


DEVICE_LABELS = {
    "rgbd_camera": "RGB-D Camera",
    "stereo_rgbd_camera": "Stereo RGB-D Camera",
    "mono_rgb": "RGB Camera",
    "stereo_rgb": "Stereo RGB Camera",
    "glove_sensor": "Glove Sensor",
}


def _keys(values: Iterable[object] | None) -> list[str]:
    if isinstance(values, str):
        return [values.strip()] if values.strip() else []
    return [str(value).strip() for value in (values or []) if str(value).strip()]


def _is_left(value: str) -> bool:
    low = value.lower()
    return low.endswith("_left") or "_left_" in low


def _is_right(value: str) -> bool:
    low = value.lower()
    return low.endswith("_right") or "_right_" in low


def is_depth_only_key(value: object) -> bool:
    """Whether a source key denotes depth only, rather than RGB video."""
    low = str(value or "").strip().lower()
    if "depth" not in low:
        return False
    return not any(token in low for token in ("rgb", "color", "image", "video"))


def camera_profile(kind: object = "", name: object = "",
                   slots: Iterable[object] | None = None) -> tuple[str, str | None]:
    """Return ``(generic device type, lens)`` without depending on a model.

    ``mono/stereo`` describes camera layout while ``rgb/depth`` describes
    stream modalities.  An RGB + depth pair is therefore classified as
    RGB-D, even though only the RGB stream is connected to a video input
    node.  Fisheye remains a lens attribute of Mono RGB.
    """
    kind_text = str(kind or "").strip().lower()
    name_text = str(name or "").strip().lower()
    slot_keys = _keys(slots)
    text = " ".join([kind_text, name_text, *[key.lower() for key in slot_keys]])

    # Collectors do not always spell RGB slots with ``rgb``/``color``;
    # ``stereo_left`` and ``stereo_right`` are also color video streams. Any
    # non-depth-only slot is therefore a color/video slot unless metadata
    # explicitly says otherwise.
    has_rgb = (any(not is_depth_only_key(key) for key in slot_keys)
               or "rgb" in kind_text or "color" in kind_text)
    has_depth = any("depth" in key or "depth" in kind_text for key in slot_keys)
    has_pair = any(_is_left(key) for key in slot_keys) and any(
        _is_right(key) for key in slot_keys
    )
    if has_rgb and has_depth and has_pair:
        profile = "stereo_rgbd_camera"
    elif has_rgb and has_depth:
        profile = "rgbd_camera"
    elif has_pair or "stereo" in text:
        profile = "stereo_rgb"
    else:
        profile = "mono_rgb"

    lens = "fisheye" if "fisheye" in text else None
    return profile, lens


def generic_label(profile: str, ordinal: int = 1) -> str:
    """Return the fixed, human-facing device category name.

    ``ordinal`` is intentionally kept in the signature for compatibility with
    callers that still calculate deterministic group numbers.  The number is
    not part of the UI name: the editable source/device field carries the
    concrete physical device identity.
    """
    return DEVICE_LABELS.get(profile, "Input Device")


def decorate_device_sources(sources: list[dict]) -> list[dict]:
    """Add standardized display metadata while preserving raw identifiers.

    Classification is deterministic by source id. ``name`` and
    ``source_keys`` are deliberately left unchanged because they are used for
    compatibility matching and file resolution.
    """
    groups: dict[str, list[dict]] = defaultdict(list)
    for source in sources:
        input_type = str(source.get("input_type") or "")
        if input_type == "glove_sensor":
            profile, lens = "glove_sensor", None
        else:
            profile, lens = camera_profile(
                source.get("kind"), source.get("name"),
                [*(source.get("slots") or []), *(source.get("source_keys") or [])],
            )
        source["device_type"] = profile
        if lens:
            source["lens"] = lens
        else:
            source.pop("lens", None)
        groups[profile].append(source)

    for profile, items in groups.items():
        for ordinal, source in enumerate(
            sorted(items, key=lambda item: str(item.get("id") or "")), start=1
        ):
            display = generic_label(profile, ordinal)
            source["display_name"] = display
            source["label"] = display
    return sources


def display_names_for_sources(sources: Iterable[dict] | None) -> dict[str, str]:
    """Build a stream-key → standardized display-name map for the UI."""
    result: dict[str, str] = {}
    for source in sources or []:
        display = str(source.get("display_name") or "").strip()
        if not display:
            continue
        for key in [*(source.get("source_keys") or []),
                    *(source.get("depth_keys") or [])]:
            key = str(key).strip()
            if key:
                result[key] = display
    return result
