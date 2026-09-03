import cv2
import numpy as np

from app.lerobot_v21 import (
    DEPTH_MAX_MM,
    DEPTH_MIN_MM,
    DEPTH_QMAX,
    DEPTH_QP,
    codes_to_heatmap_bgr,
    dequantize_depth,
    depth_video_encoder_args,
    depth_to_heatmap_bgr,
    quantize_depth,
)


def test_canonical_quantization_boundaries_and_invalid_value():
    depths = np.array([0, DEPTH_MIN_MM, DEPTH_MAX_MM, 6000], dtype=np.uint16)
    codes = quantize_depth(depths)

    assert codes.dtype == np.dtype("<u2")
    assert codes.tolist() == [0, 0, DEPTH_QMAX, DEPTH_QMAX]
    assert dequantize_depth(codes).dtype == np.dtype("<u2")


def test_quantization_is_logarithmic_and_round_trips_reasonably():
    depths = np.array([100, 200, 600, 1000, 2500, 5000], dtype=np.uint16)
    restored = dequantize_depth(quantize_depth(depths)).astype(np.int32)
    relative_error = np.abs(restored - depths) / depths

    assert np.all(np.diff(quantize_depth(depths)) > 0)
    assert float(relative_error.max()) < 0.002


def test_heatmap_uses_exact_code_to_opencv_jet_mapping_including_zero():
    codes = np.array([[0, 1, 1799, DEPTH_QMAX]], dtype=np.uint16)
    expected_8 = ((codes.astype(np.int32) * 255) // DEPTH_QMAX).astype(np.uint8)
    expected = cv2.applyColorMap(expected_8, cv2.COLORMAP_JET)

    np.testing.assert_array_equal(codes_to_heatmap_bgr(codes), expected)
    np.testing.assert_array_equal(
        depth_to_heatmap_bgr(dequantize_depth(codes)),
        codes_to_heatmap_bgr(quantize_depth(dequantize_depth(codes))),
    )
    np.testing.assert_array_equal(
        codes_to_heatmap_bgr(codes)[0, 0],
        cv2.applyColorMap(np.array([[0]], dtype=np.uint8), cv2.COLORMAP_JET)[0, 0],
    )


def test_encoder_contract_has_no_profile_or_crf():
    args = depth_video_encoder_args()
    assert args == [
        "-pix_fmt", "gray12le",
        "-tag:v", "hvc1",
        "-x265-params", f"qp={DEPTH_QP}:range=full:keyint=2:min-keyint=2",
    ]
    assert "-profile:v" not in args
    assert not any("crf" in value.lower() for value in args)
