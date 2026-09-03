#include "colormap.h"
#include <cmath>
#include <algorithm>
#include <sstream>
#include <iomanip>
#include <vector>

// ============================================================================
// Histogram-equalized depth visualization with JET colormap.
//
// RealSense-style: equalize the depth histogram so different depth layers
// are clearly separated, then apply JET (blue=near, red=far).
// ============================================================================

static void histogramEqualize(const cv::Mat& depthMeters, cv::Mat& equalized,
                               const float presetMin, const float presetMax)
{
    const int rows = depthMeters.rows;
    const int cols = depthMeters.cols;
    const int total = rows * cols;

    // Collect valid depth values within preset range
    std::vector<float> valid;
    valid.reserve(total);
    for (int r = 0; r < rows; ++r) {
        const float* row = depthMeters.ptr<float>(r);
        for (int c = 0; c < cols; ++c) {
            float d = row[c];
            if (d > 0.0f && d >= presetMin && d <= presetMax)
                valid.push_back(d);
        }
    }

    if (valid.size() < 100) {
        // Not enough data — fall back to linear normalization
        equalized.create(rows, cols, CV_8UC1);
        float range = presetMax - presetMin;
        float inv = (range > 0) ? (255.0f / range) : 0.0f;
        for (int r = 0; r < rows; ++r) {
            const float* src = depthMeters.ptr<float>(r);
            uchar* dst = equalized.ptr<uchar>(r);
            for (int c = 0; c < cols; ++c) {
                float d = src[c];
                if (d <= 0.0f) { dst[c] = 0; continue; }
                float t = (d - presetMin) * inv;
                dst[c] = static_cast<uchar>(std::max(0.0f, std::min(255.0f, t)));
            }
        }
        return;
    }

    // Sort and build CDF lookup
    std::sort(valid.begin(), valid.end());
    int n = static_cast<int>(valid.size());

    // Build a LUT: depth value → equalized 0-255
    // Use 4096 bins for 0-presetMax range (~1mm resolution at 4m)
    const int kLutSize = 4096;
    std::vector<uchar> lut(kLutSize, 0);
    float binWidth = presetMax / kLutSize;

    int cumCount = 0;
    size_t validIdx = 0;
    for (int bin = 0; bin < kLutSize; ++bin) {
        float binMax = (bin + 1) * binWidth;
        while (validIdx < valid.size() && valid[validIdx] <= binMax) {
            cumCount++;
            validIdx++;
        }
        // CDF → 0-255 (exclude 0 for valid depths to distinguish from invalid)
        lut[bin] = static_cast<uchar>(1 + (cumCount * 254) / n);
    }

    // Apply LUT
    equalized.create(rows, cols, CV_8UC1);
    float invBin = 1.0f / binWidth;
    for (int r = 0; r < rows; ++r) {
        const float* src = depthMeters.ptr<float>(r);
        uchar* dst = equalized.ptr<uchar>(r);
        for (int c = 0; c < cols; ++c) {
            float d = src[c];
            if (d <= 0.0f || d > presetMax) { dst[c] = 0; continue; }
            int bin = static_cast<int>(d * invBin);
            if (bin < 0) bin = 0;
            if (bin >= kLutSize) bin = kLutSize - 1;
            dst[c] = lut[bin];
        }
    }
}

cv::Mat applyDepthColormap(const cv::Mat& depthMeters, float minDepth, float maxDepth, bool equalize)
{
    if (depthMeters.empty()) return cv::Mat();
    CV_Assert(depthMeters.type() == CV_32FC1);

    cv::Mat indexed;
    if (equalize) {
        // Histogram equalization for dynamic range
        histogramEqualize(depthMeters, indexed, minDepth, maxDepth);
    } else {
        // Fixed linear mapping within preset range
        float range = maxDepth - minDepth;
        if (range <= 0.0f) return cv::Mat::zeros(depthMeters.size(), CV_8UC3);
        float invRange = 255.0f / range;
        int rows = depthMeters.rows, cols = depthMeters.cols;
        indexed.create(rows, cols, CV_8UC1);
        for (int r = 0; r < rows; ++r) {
            const float* src = depthMeters.ptr<float>(r);
            uchar* dst = indexed.ptr<uchar>(r);
            for (int c = 0; c < cols; ++c) {
                float d = src[c];
                if (d <= 0.0f) { dst[c] = 0; continue; }
                float t = (d - minDepth) * invRange / 255.0f;
                // t = (d-min)/(max-min), then *255
                int v = static_cast<int>((d - minDepth) * invRange);
                dst[c] = static_cast<uchar>(std::max(0, std::min(255, v)));
            }
        }
    }

    // Apply JET colormap
    cv::Mat color;
    cv::applyColorMap(indexed, color, cv::COLORMAP_JET);

    // Set invalid pixels (index 0 = depth <= 0) to black
    const int rows = depthMeters.rows;
    for (int r = 0; r < rows; ++r) {
        const float* src = depthMeters.ptr<float>(r);
        cv::Vec3b* dst = color.ptr<cv::Vec3b>(r);
        for (int c = 0; c < depthMeters.cols; ++c) {
            if (src[c] <= 0.0f) dst[c] = cv::Vec3b(0, 0, 0);
        }
    }

    return color;
}

// ============================================================================
// Color bar (JET LUT for vertical bar display)
// ============================================================================

void buildRealSenseLUT(cv::Mat& lut)
{
    // i=0 → grayscale=0 → JET blue (near/bottom)
    // i=255 → grayscale=255 → JET red (far/top)
    cv::Mat src(256, 1, CV_8UC1);
    for (int i = 0; i < 256; ++i)
        src.at<uchar>(i) = static_cast<uchar>(i);
    cv::applyColorMap(src, lut, cv::COLORMAP_JET);
}

cv::Mat attachColorBar(const cv::Mat& depthColored,
                        float minDepth, float maxDepth,
                        int barWidth)
{
    if (depthColored.empty()) return depthColored;

    static cv::Mat sLUT;
    if (sLUT.empty()) buildRealSenseLUT(sLUT);

    int imgH = depthColored.rows, imgW = depthColored.cols;
    int totalW = imgW + barWidth + 30;

    cv::Mat combined(imgH, totalW, CV_8UC3, cv::Scalar(20, 20, 30));
    depthColored.copyTo(combined(cv::Rect(0, 0, imgW, imgH)));

    int barX = imgW + 15;
    for (int y = 0; y < imgH; ++y) {
        int idx = (imgH - 1 - y) * 255 / (imgH - 1);
        if (idx < 0) idx = 0;
        if (idx > 255) idx = 255;
        cv::Vec3b c = sLUT.at<cv::Vec3b>(idx);
        cv::line(combined, cv::Point(barX, y), cv::Point(barX + barWidth, y),
                 cv::Scalar(c[0], c[1], c[2]));
    }

    cv::rectangle(combined, cv::Point(barX, 0), cv::Point(barX + barWidth, imgH - 1),
                  cv::Scalar(100, 100, 100), 1);

    int font = cv::FONT_HERSHEY_SIMPLEX;
    double fs = 0.4;
    cv::Scalar tc(200, 200, 200);
    int tx = barX + barWidth + 5;

    std::ostringstream ss;
    ss << std::fixed << std::setprecision(1) << maxDepth << "m";
    cv::putText(combined, ss.str(), cv::Point(tx, 12), font, fs, tc, 1);
    ss.str(""); ss << std::fixed << std::setprecision(1) << ((minDepth+maxDepth)*0.5f) << "m";
    cv::putText(combined, ss.str(), cv::Point(tx, imgH/2+4), font, fs, tc, 1);
    ss.str(""); ss << std::fixed << std::setprecision(1) << minDepth << "m";
    cv::putText(combined, ss.str(), cv::Point(tx, imgH-4), font, fs, tc, 1);

    return combined;
}
