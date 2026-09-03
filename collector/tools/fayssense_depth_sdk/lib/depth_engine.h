#pragma once

#include <string>
#include <vector>
#include <cstdint>
#include <opencv2/opencv.hpp>

// SDK types (C++ structs, no extern "C" needed)
#include "fays_atrak/fays_atrak_types.h"
#include "fays_atrak/fays_atrak_depth.h"

/**
 * @brief Wrapper around libfayssense_aikit_depth.so via the C API.
 *
 * Provides a simple C++ interface for feeding stereo (and optional RGB)
 * images and retrieving depth, disparity, rectified images, rectify info,
 * and 3D point clouds.
 *
 * All output buffers are pre-allocated in the constructor and reused.
 */
class DepthEngine {
public:
    /**
     * @param configPath  Path to stereo_depth.yaml
     */
    explicit DepthEngine(const std::string& configPath);
    ~DepthEngine();

    bool isInitialized() const { return mInitialized; }

    /// Bind a ViKit sensor handle (live mode)
    bool bindViKit(void* viKitHandle);

    // ---- Input ----

    /**
     * @brief Feed a vertically-stacked stereo image.
     *
     * @param stackedStereo  1-channel grayscale (AIE_RAW8) or 3-channel BGR.
     *                       Left on top half, right on bottom half.
     * @param timestampNs    Frame timestamp in nanoseconds.
     * @return 0 on success, non-zero on error.
     */
    int feedStereoImage(const cv::Mat& stackedStereo, uint64_t timestampNs);

    /**
     * @brief Set depth algorithm parameters at runtime (key-value batch).
     *
     * @param backend    Target algorithm backend (e.g. ATRAK_DEPTH_BACKEND_SGBM).
     * @param kv_array   KV array of parameter ID/value pairs.
     * @param kv_count   Number of KVs in the array.
     * @return 0 on success, non-zero on error.
     */
    int setDepthParam(ATRAK_DepthBackendType backend, const ATRAK_DepthParamKv* kv_array, int kv_count);

    /**
     * @brief Feed the middle RGB camera image for depth-to-RGB alignment.
     *
     * Only meaningful when the stereo_depth.yaml has align_cam_id != -1.
     *
     * @param rgbBgr        3-channel BGR image from the middle camera.
     * @param timestampNs    Frame timestamp in nanoseconds.
     * @return 0 on success, non-zero on error.
     */
    int feedRgbImage(const cv::Mat& rgbBgr, uint64_t timestampNs);

    // ---- Output ----

    /**
     * @brief Get the latest depth image.
     *
     * @param depthMeters     [out] CV_32FC1, values in meters.
     * @param disparity       [out] CV_32FC1 raw disparity (required, pass empty mat to skip).
     * @param pairedImage     [out] Aligned image if alignment enabled (required, pass empty mat to skip).
     * @return true if new data was available, false if stale/empty.
     */
    bool getDepthImage(cv::Mat& depthMeters,
                       cv::Mat& disparity,
                       cv::Mat& pairedImage);

    /**
     * @brief Get the latest rectified stereo image.
     *
     * @param rectifiedStereo  [out] Vertically stacked, 3-channel BGR.
     * @return true if new data was available.
     */
    bool getRectifiedImage(cv::Mat& rectifiedStereo);

    /**
     * @brief Get rectification info (R0, R1, P0, P1 matrices).
     */
    bool getRectifyInfo(AtrakRectifyInfo& rectifyInfo);

    /**
     * @brief Get the 3D point cloud.
     *
     * @param xyz         [out] N*3 interleaved float coordinates (meters).
     * @param rgb         [out] N*3 uint8 BGR colors.
     * @param maxPoints   Maximum number of points to retrieve.
     * @return true if new data was available.
     */
    bool getPointCloud(std::vector<float>& xyz,
                       std::vector<uint8_t>& rgb,
                       int maxPoints = 262144);

    /**
     * @brief Set point cloud generation parameters at runtime (key-value batch).
     *
     * @param backend    Target point cloud generation backend (e.g. ATRAK_DEPTHPOINT_BACK_CPU).
     * @param kv_array   KV array of parameter ID/value pairs.
     * @param kv_count   Number of KVs in the array.
     * @return 0 on success, non-zero on error.
     */
    int setPointCloudParam(ATRAK_DepthPointBackType backend, const ATRAK_DepthPointParamKv* kv_array, int kv_count);

private:
    void* mHandle = nullptr;
    bool mInitialized = false;

    // Pre-allocated output buffers
    AtrakDepthImage* mDepthBuf = nullptr;
    AtrakDepthImage* mDispBuf = nullptr;
    AtrakImage* mRectBuf = nullptr;
    AtrakImage* mPairedBuf = nullptr;

    // Point cloud output buffer
    AtrakMap* mCloudBuf = nullptr;
    int mCloudMaxPoints = 0;

    // Maximum buffer sizes — use SDK defines to prevent overflow
    static constexpr size_t kMaxDepthFloats = FAYS_ATRAK_IMG_MAX_BYTES;
    static constexpr size_t kMaxMonoBytes   = FAYS_ATRAK_RGB_MAX_BYTES;
    static constexpr size_t kMaxRgbBytes    = FAYS_ATRAK_RGB_MAX_BYTES;
};
