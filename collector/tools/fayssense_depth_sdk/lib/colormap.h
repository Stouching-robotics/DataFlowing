#pragma once

#include <opencv2/opencv.hpp>

/**
 * @brief Generate a RealSense/Orbbec-style depth colormap using HSV interpolation.
 *
 * The RealSense viewer uses a color ramp where:
 *   Near depth  → White (255,255,255)
 *   Near-mid    → Red → Yellow → Green → Cyan → Blue
 *   Far depth   → Dark (0,0,0)
 *
 * This matches the librealsense colorizer default scheme.
 */

/**
 * @brief Apply RealSense-style depth colormap to a float32 depth image.
 *
 * @param depthMeters  CV_32FC1 depth image, values in meters
 * @param minDepth     Depths at or below this map to white (near)
 * @param maxDepth     Depths at or above this map to black (far)
 * @return             CV_8UC3 BGR color image. Invalid pixels (<= 0) are black.
 */
cv::Mat applyDepthColormap(const cv::Mat& depthMeters,
                            float minDepth = 0.1f,
                            float maxDepth = 2.0f,
                            bool equalize = true);

/**
 * @brief Attach a vertical color bar with depth value labels to the depth image.
 *
 * @param depthColored   CV_8UC3 BGR colorized depth image
 * @param minDepth       Minimum depth value (meters)
 * @param maxDepth       Maximum depth value (meters)
 * @param barWidth       Width of the color bar in pixels (default 30)
 * @return               Combined image (depth + color bar on the right)
 */
cv::Mat attachColorBar(const cv::Mat& depthColored,
                        float minDepth, float maxDepth,
                        int barWidth = 30);

/**
 * @brief Build the 256-entry colormap LUT (BGR).
 *
 * @param lut  [out] 256x1x3 uint8 array (BGR)
 */
void buildRealSenseLUT(cv::Mat& lut);
