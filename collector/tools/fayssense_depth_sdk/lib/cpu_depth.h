#pragma once

/**
 * @file cpu_depth.h
 * @brief CPU-only stereo fisheye depth pipeline (C++ port of the benchmark/ Python variants).
 *
 * Implements five alternative techniques for stereo fisheye depth estimation,
 * all running on the CPU with OpenCV (no SDK depth engine involved):
 *
 *   V1  fisheye -> pinhole rectification (cv::fisheye::stereoRectify) + SGBM
 *       (the classic approach, same rectification path as the SDK engine)
 *   V2v vertical-polar ERP spherical rectification + SGBM (rows approx. epipolar)
 *   V2b baseline-polar ERP rectification + SGBM (rows exactly epipolar)
 *   V3  multi-pinhole bands at several yaw angles + SGBM + ERP fusion
 *       (largest FOV coverage; the winner on low-texture scenes)
 *   V4  fisheye -> pinhole + StereoBM (fastest)
 *   V5  SDK-engine-rectified input + own SGBM (isolates the matcher)
 *
 * All variants output a float32 depth map (meters, 0 = invalid), a rectified
 * stacked BGR image for display, and a rectified-frame point cloud. Depth for
 * V2/V3 comes from per-pixel ray triangulation; V1/V4/V5 from z = fx*B/d.
 *
 * The calibration is a kalibr-format yaml (pinhole + equidistant / KB4), the
 * same format as config/calib/calib.yaml.
 */

#include <opencv2/core.hpp>

#include <string>
#include <vector>

namespace cpu_depth {

// ---------------------------------------------------------------------------
// Stereo fisheye calibration (kalibr: pinhole + equidistant distortion)
// ---------------------------------------------------------------------------
struct StereoCalib {
    cv::Matx33d K[2];      // per-camera intrinsics, scaled to the frame size
    cv::Vec4d D[2];        // equidistant distortion coefficients (k1..k4)
    int calibW = 0, calibH = 0;  // resolution as written in the yaml
    int frameW = 0, frameH = 0;  // current per-camera frame size (K scaled to this)

    cv::Matx33d R;         // cam0 -> cam1 rotation:  p1 = R*p0 + t
    cv::Vec3d t;           // cam0 origin in cam1 frame (m)
    cv::Matx33d R_rect;    // rectified -> cam0 (rows = rectified axes in cam0 frame)
    cv::Matx33d R1_rect;   // rectified -> cam1
    cv::Vec3d t_rect;      // right camera origin in the rectified frame (~(B,0,0))
    double baseline = 0.0; // m
    double lensMaxTheta = 0.0; // max off-axis angle covered by the sensor (rad)
};

/// Parse a kalibr cam0/cam1 yaml (line-based, no yaml-cpp dependency).
bool loadKalibr(const std::string& path, StereoCalib& out);

// ---------------------------------------------------------------------------
// Options
// ---------------------------------------------------------------------------
enum Variant {
    V1_PINHOLE_SGBM = 0,
    V2V_ERP_SGBM    = 1,
    V2B_ERP_SGBM    = 2,
    V3_BANDS_SGBM   = 3,
    V4_PINHOLE_BM   = 4,
    V5_ENGINE_SGBM  = 5,
};

struct CpuOptions {
    int variant = V1_PINHOLE_SGBM;

    // SGBM (all SGBM variants; P1/P2 are recomputed from the block size when
    // v3Relaxed is set, mirroring the benchmark's relaxed band preset)
    int minDisp = 0, numDisp = 128, blockSize = 11;
    int P1 = 1904, P2 = 11616, disp12MaxDiff = 4, preFilterCap = 51;
    int uniquenessRatio = 10, speckleWindowSize = 150, speckleRange = 1;
    int sgbmMode = 1;    // cv::StereoSGBM::MODE_SGBM_3WAY

    // StereoBM (V4)
    int bmBlockSize = 19;

    // V1/V4 rectification
    double balance = 0.0, fovScale = 1.0;

    // V2 rectification (ERP span in degrees)
    double lonDeg = 120.0, latDeg = 110.0;

    // V3 bands
    double bandHalfDeg = 45.0;
    double yaws[3] = {-35.0, 0.0, 35.0};
    int nYaws = 3;
    double v3OutScale = 0.5;   // band resolution = frame size * scale
    bool v3Relaxed = true;     // relaxed SGBM for band images (benchmark preset)
    double erpFillGap = 8.0;   // px, fused-ERP gap fill radius
    double erpDensity = 4.0;   // px per deg of the fused ERP grid

    // common
    double maxZ = 20.0;        // m
    int maxPoints = 262144;    // point cloud cap
};

// ---------------------------------------------------------------------------
// Result
// ---------------------------------------------------------------------------
struct CpuResult {
    cv::Mat depth;       // CV_32FC1 meters, 0 = invalid
    cv::Mat rectStereo;  // CV_8UC3 stacked BGR (top = left), for display
    cv::Mat disparity;   // CV_32FC1 px (empty for V3, which has no single grid)
    cv::Mat pts;         // V2 only: N x 3 CV_32FC1 rectified-frame points
    std::vector<float> pcXYZ;    // N*3 rectified-frame points (m)
    std::vector<uint8_t> pcRGB;  // N*3
    double runtimeMs = 0.0;
};

// ---------------------------------------------------------------------------
// Processor
// ---------------------------------------------------------------------------

/// One rectification bundle (complete here so std::vector<Maps> works in any TU).
struct CpuMaps {
    int kind = 0;          // 0 = V1/V4 pinhole, 1 = V2 ERP, 2 = V3 band
    cv::Mat mapx0, mapy0, mapx1, mapy1;   // CV_32FC1
    cv::Mat valid0, valid1;               // CV_8UC1
    // pinhole meta (kind 0)
    double fx = 0, fy = 0, cx = 0, cy = 0, baseline = 0;
    // ERP/band meta (kind 1/2)
    cv::Mat raysRect;   // N x 3 CV_32F
    cv::Vec3d tRect;
    double yaw = 0, f_v = 0;
};

class CpuDepthProcessor {
public:
    CpuDepthProcessor() = default;
    ~CpuDepthProcessor() = default;

    bool loadCalib(const std::string& calibPath);

    /// Per-camera frame size of the input (stacked frames are 2*frameH tall).
    /// Intrinsics are rescaled accordingly; maps rebuild on change.
    void setFrameSize(int w, int h);

    bool isReady() const { return mReady; }

    void setOptions(const CpuOptions& opts) { mOpts = opts; }
    const CpuOptions& options() const { return mOpts; }

    /// Process a stacked stereo frame (top = left, bottom = right).
    /// For V5 pass the engine-rectified stacked image and its intrinsics
    /// (engineP0 = {fx, fy, cx, cy}) and baseline.
    bool process(const cv::Mat& stacked,
                 const cv::Mat* engineRect = nullptr,
                 const double* engineP0 = nullptr,
                 double engineBaseline = 0.0,
                 CpuResult* out = nullptr);

    double lastRuntimeMs() const { return mLastRuntimeMs; }

private:
    bool buildMaps(const StereoCalib& calib);
    bool matchV3(const cv::Mat& left, const cv::Mat& right, CpuResult* out);

    StereoCalib mCalib;
    cv::Matx33d mKCalib[2];  // K at the yaml resolution (scaling baseline)
    bool mReady = false;
    CpuOptions mOpts;

    // cached maps (rebuilt when variant/geometry/frame size changes)
    std::string mMapsKey;
    std::vector<CpuMaps> mMaps;

    double mLastRuntimeMs = 0.0;
};

} // namespace cpu_depth
