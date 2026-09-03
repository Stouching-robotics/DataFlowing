//
// S80M 双目深度 Demo —— 严格按 FaysSense VI Kit SDK 官方实现
//
// 图像读取参考: SDK example/fays_vikit_example.cpp
//   FAYS_VIK_CreateHandleWithConfig -> FAYS_VIK_GetStereoFrames (轮询, 预分配缓冲)
// 深度引擎参考: SDK stereo_depth_gui/core/depth_engine.cpp
//   FAYS_ATRAK_D_CreateHandleWithConfig -> FAYS_ATRAK_D_BindViKit
//   -> FAYS_ATRAK_D_FeedStereoImage -> FAYS_ATRAK_D_GetDepthImage
//   -> FAYS_ATRAK_D_GetRectifiedImage -> FAYS_ATRAK_D_GetRectifyInfo
//   -> FAYS_ATRAK_D_GetPointcloud
//
// 通道顺序注意 (fays_atrak_types.h):
//   - AtrakImage 3 通道为 BGR-packed
//   - 点云 AtrakMap::rgb 为 ARGB8 (0x00RRGGBB: R 在高字节, alpha 固定 0)
//
// 构建: ./build.sh    运行: ./run.sh [viKitConfig] [depthConfig]
//
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <deque>
#include <fstream>
#include <iostream>
#include <string>
#include <thread>
#include <vector>

#include <opencv2/opencv.hpp>

#include "fays_atrak/fays_atrak_types.h"
#include "fays_atrak/fays_vikit.h"
#include "fays_atrak/fays_atrak_depth.h"

namespace {

// 点云颜色 ARGB8 (0x00RRGGBB) -> BGR (cv 通道序)
cv::Vec3b argb8ToBgr(uint32_t c)
{
    return cv::Vec3b(static_cast<uchar>(c & 0xFF),           // B
                     static_cast<uchar>((c >> 8) & 0xFF),    // G
                     static_cast<uchar>((c >> 16) & 0xFF));  // R
}

void savePly(const std::string& path, const AtrakMap& cloud)
{
    std::ofstream f(path);
    if (!f) { std::cerr << "[PLY] 无法写入 " << path << std::endl; return; }
    const uint32_t n = cloud.point_num;
    f << "ply\nformat ascii 1.0\n"
      << "element vertex " << n << "\n"
      << "property float x\nproperty float y\nproperty float z\n"
      << "property uchar red\nproperty uchar green\nproperty uchar blue\n"
      << "end_header\n";
    for (uint32_t i = 0; i < n; ++i) {
        const float* p = cloud.pos + 3 * i;
        const cv::Vec3b bgr = argb8ToBgr(cloud.rgb[i]);
        f << p[0] << ' ' << p[1] << ' ' << p[2] << ' '
          << static_cast<int>(bgr[2]) << ' '   // R
          << static_cast<int>(bgr[1]) << ' '   // G
          << static_cast<int>(bgr[0]) << '\n'; // B
    }
    std::cout << "[PLY] 已保存 " << path << " (" << n << " 点)" << std::endl;
}

// ═══════════════════════════════════════════════════════════════
//  官方 GUI 同款深度着色 (stereo_depth_gui/core/colormap.cpp)
//  深度数据原样, 仅做直方图均衡映射 + JET: 按像素数量分配色阶,
//  近距离层次分明; 无效(<=0)与超出范围(>max)置黑
// ═══════════════════════════════════════════════════════════════
static void officialHistogramEqualize(const cv::Mat& depthMeters, cv::Mat& equalized,
                                      const float presetMin, const float presetMax)
{
    const int rows = depthMeters.rows;
    const int cols = depthMeters.cols;

    std::vector<float> valid;
    valid.reserve(rows * cols);
    for (int r = 0; r < rows; ++r) {
        const float* row = depthMeters.ptr<float>(r);
        for (int c = 0; c < cols; ++c) {
            float d = row[c];
            if (d > 0.0f && d >= presetMin && d <= presetMax)
                valid.push_back(d);
        }
    }

    if (valid.size() < 100) {   // 有效像素太少 → 线性归一化兜底
        equalized.create(rows, cols, CV_8UC1);
        float range = presetMax - presetMin;
        float inv = (range > 0) ? (255.0f / range) : 0.0f;
        for (int r = 0; r < rows; ++r) {
            const float* src = depthMeters.ptr<float>(r);
            uchar* dst = equalized.ptr<uchar>(r);
            for (int c = 0; c < cols; ++c) {
                float d = src[c];
                if (d <= 0.0f) { dst[c] = 0; continue; }
                dst[c] = static_cast<uchar>(
                    std::max(0.0f, std::min(255.0f, (d - presetMin) * inv)));
            }
        }
        return;
    }

    std::sort(valid.begin(), valid.end());
    int n = static_cast<int>(valid.size());

    // CDF → LUT (官方同款: 4096 bin, 有效深度 1-255, 0 留黑)
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
        lut[bin] = static_cast<uchar>(1 + (cumCount * 254) / n);
    }

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

// 官方范围预设 (control_panel.cpp 同款, 按 M 循环)
struct RangePreset { float minD, maxD; };
static const RangePreset kRangePresets[] = {
    {0.1f, 1.0f}, {0.1f, 2.0f}, {0.1f, 4.0f}, {0.1f, 8.0f},
    {0.3f, 1.0f}, {0.3f, 2.0f}, {0.3f, 4.0f}, {0.3f, 8.0f},
};

} // namespace

// ═══════════════════════════════════════════════════════════════
//  Demo: 严格按 SDK 官方调用序列
// ═══════════════════════════════════════════════════════════════
class StereoDepthDemo {
public:
    StereoDepthDemo(const std::string& viKitConfig, const std::string& depthConfig);
    ~StereoDepthDemo();

    bool ok() const { return viHandle_ != nullptr && depthHandle_ != nullptr; }
    void run();

private:
    void* viHandle_ = nullptr;      // ViKit 传感器句柄
    void* depthHandle_ = nullptr;   // 深度引擎句柄
    std::atomic<bool> feedBgr_{true};  // 送引擎通道序: true=BGR (SDK 文档约定), 按 C 切换

    // ── 线程分离 (官方 GUI 同构): 采集/引擎线程 → 显示线程 ──
    // 采集线程: GetStereoFrames → FeedStereoImage → GetDepthImage → 入队
    // 显示线程: 取最新帧 → 后处理 → imshow → waitKey
    // 单线程串行时任何一帧处理慢都会拖住采集 (局部卡顿根因).
    std::mutex              qmtx_;
    std::condition_variable qcv_;
    std::deque<cv::Mat>     depthQ_;    // 640x400 float32 深度 (后处理前)
    std::deque<cv::Mat>     rectQ_;     // 640x800 矫正显示图
    std::deque<cv::Mat>     stereoQ_;   // 640x800 双目显示图
    std::atomic<bool>       running_{true};
    std::atomic<bool>       saveReq_{false};   // 主线程按 s → 采集线程存 .bin/_rect.png
    std::atomic<bool>       plyReq_{false};    // 主线程按 p → 采集线程存点云
    std::atomic<long long>  lastTs_{0};        // 采集线程最近帧时间戳 (命名用)
    std::atomic<double>     lumMean_{0.0};     // 采集线程算的亮度 (统计输出用)
    std::atomic<double>     lumOver_{0.0};

    // 显示模式 (仅主显示线程使用):
    //   officialMode_=true : 官方 GUI 同款 —— 深度数据原样 + 直方图均衡 + JET
    //   officialMode_=false: 增强模式 —— 后处理链 + 分位数映射 (历史迭代产物)
    bool officialMode_ = true;
    int  presetIdx_ = 6;   // 官方范围预设: 默认 "0.3 – 4.0 m" (官方 kDefaultPreset=6)

    // ── 输出缓冲: 按 SDK 头文件宏预先分配, 调用方管理内存 ──
    std::vector<uchar>  stereoBuf_;   // FAYS_ATRAK_MONO_MAX_BYTES*3 (官方示例同款)
    std::vector<uchar>  rectBuf_;     // FAYS_ATRAK_IMG_MAX_BYTES
    std::vector<uchar>  pairedBuf_;   // FAYS_ATRAK_RGB_MAX_BYTES
    std::vector<float>  depthBuf_;    // FAYS_ATRAK_DEPTH_IMG_MAX_BYTES/sizeof(float)
    std::vector<float>  dispBuf_;
    std::vector<float>  cloudPos_;    // FAYS_ATRAK_MAX_DENSE_MAP_POINTS_NUM * 3
    std::vector<uint32_t> cloudRgb_;  // FAYS_ATRAK_MAX_DENSE_MAP_POINTS_NUM

    AtrakImage      stereoImg_{};   // GetStereoFrames 输出
    AtrakImage      rectImg_{};     // GetRectifiedImage 输出
    AtrakImage      pairedImg_{};   // GetDepthImage 对齐图像输出
    AtrakDepthImage depthImg_{};    // GetDepthImage 深度输出 (float 米)
    AtrakDepthImage dispImg_{};     // GetDepthImage 视差输出
    AtrakMap        cloud_{};       // GetPointcloud 输出
};

// 深度显示映射范围 (EMA 平滑, 跨帧持久; 显示线程写, 采集线程统计时读)
static std::atomic<float> gMapLo{0.3f}, gMapHi{1.5f};

StereoDepthDemo::StereoDepthDemo(const std::string& viKitConfig,
                                 const std::string& depthConfig)
{
    // 1. 打开相机 (官方示例: FAYS_VIK_CreateHandleWithConfig)
    if (FAYS_VIK_CreateHandleWithConfig(&viHandle_, viKitConfig.c_str()) != 0) {
        std::cerr << "[ERROR] FAYS_VIK_CreateHandleWithConfig 失败: "
                  << viKitConfig << std::endl;
        return;
    }
    std::cout << "Fays VIKit Version: " << FAYS_VIK_GetVersion(viHandle_) << std::endl;

    // 2. 创建深度引擎 (官方 GUI: FAYS_ATRAK_D_CreateHandleWithConfig)
    if (FAYS_ATRAK_D_CreateHandleWithConfig(&depthHandle_, depthConfig.c_str()) != 0) {
        std::cerr << "[ERROR] FAYS_ATRAK_D_CreateHandleWithConfig 失败: "
                  << depthConfig << std::endl;
        return;
    }

    // 3. 绑定传感器 (官方 GUI: FAYS_ATRAK_D_BindViKit)
    if (FAYS_ATRAK_D_BindViKit(depthHandle_, viHandle_) != 0) {
        std::cerr << "[ERROR] FAYS_ATRAK_D_BindViKit 失败" << std::endl;
        return;
    }
    std::cout << "[DepthEngine] 初始化成功, 已绑定 ViKit." << std::endl;

    // 4. 预先分配输出缓冲
    stereoBuf_.resize(FAYS_ATRAK_MONO_MAX_BYTES * 3);
    rectBuf_.resize(FAYS_ATRAK_IMG_MAX_BYTES);
    pairedBuf_.resize(FAYS_ATRAK_RGB_MAX_BYTES);
    const size_t depthFloats = FAYS_ATRAK_DEPTH_IMG_MAX_BYTES / sizeof(float);
    depthBuf_.resize(depthFloats);
    dispBuf_.resize(depthFloats);
    cloudPos_.resize(FAYS_ATRAK_MAX_DENSE_MAP_POINTS_NUM * 3);
    cloudRgb_.resize(FAYS_ATRAK_MAX_DENSE_MAP_POINTS_NUM);

    stereoImg_.data = stereoBuf_.data();
    rectImg_.data   = rectBuf_.data();
    pairedImg_.data = pairedBuf_.data();
    depthImg_.data  = depthBuf_.data();
    dispImg_.data   = dispBuf_.data();
    cloud_.pos      = cloudPos_.data();
    cloud_.rgb      = cloudRgb_.data();
}

StereoDepthDemo::~StereoDepthDemo()
{
    if (depthHandle_) {
        FAYS_ATRAK_D_DestroyHandle(depthHandle_);
        depthHandle_ = nullptr;
    }
    if (viHandle_) {
        FAYS_VIK_DestroyHandle(viHandle_);
        viHandle_ = nullptr;
    }
}

void StereoDepthDemo::run()
{
    cv::namedWindow("Stereo Input (L/R)", cv::WINDOW_NORMAL);
    cv::namedWindow("Rectified (engine)", cv::WINDOW_NORMAL);
    cv::namedWindow("Depth", cv::WINDOW_NORMAL);
    cv::resizeWindow("Stereo Input (L/R)", 640, 800);
    cv::resizeWindow("Rectified (engine)", 640, 800);
    cv::resizeWindow("Depth", 640, 400);

    // ── 采集/引擎线程 (官方 GUI 同构) ──
    // 单线程串行时任何一帧处理慢都会拖住采集 (局部卡顿根因);
    // 分离后采集线程恒定拉帧, 显示线程消费最新帧, 慢帧只丢不拖.
    bool rectifyInfoShown = false;
    std::thread capThread([this, &rectifyInfoShown]() {
        int frames = 0;
        auto lastStats = std::chrono::steady_clock::now();
        while (running_.load()) {
            // 图像读取 (stereo_swap_lr=1 时输出 上=左目/下=右目)
            if (FAYS_VIK_GetStereoFrames(viHandle_, &stereoImg_) != 0) {
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
                continue;
            }
            lastTs_.store(stereoImg_.timestamp);
            cv::Mat stereo(stereoImg_.height, stereoImg_.width,
                           stereoImg_.channel == 1 ? CV_8UC1 : CV_8UC3,
                           stereoImg_.data);
            cv::Mat stereoBgr;
            if (stereo.channels() == 3)
                cv::cvtColor(stereo, stereoBgr, cv::COLOR_RGB2BGR);
            else
                stereoBgr = stereo;

            // 显示用缩略图 + 亮度统计 (显示线程直接取)
            cv::Mat stereoView;
            cv::resize(stereoBgr, stereoView, cv::Size(640, 800));
            cv::Mat graySmall;
            cv::cvtColor(stereoView, graySmall, cv::COLOR_BGR2GRAY);
            lumMean_.store(cv::mean(graySmall)[0]);
            lumOver_.store(100.0 * cv::countNonZero(graySmall > 245) / graySmall.total());
            {
                std::lock_guard<std::mutex> lk(qmtx_);
                stereoQ_.push_back(stereoView);
                if (stereoQ_.size() > 2) stereoQ_.pop_front();
            }

            // 送引擎 (feedBgr_: BGR=SDK 文档约定 / RGB=固件原始, 按 C 切换)
            const cv::Mat& feedMat = feedBgr_.load() ? stereoBgr : stereo;
            AtrakImage feed{};
            feed.device_id  = 0;                              // ATRAK_DEV_STEREO
            feed.timestamp  = stereoImg_.timestamp;
            feed.width      = feedMat.cols;
            feed.height     = feedMat.rows;
            feed.channel    = feedMat.channels();
            feed.step       = static_cast<int>(feedMat.step);
            feed.bytes      = static_cast<uint>(feedMat.total() * feedMat.elemSize());
            feed.data       = feedMat.data;
            feed.encoding   = (feedMat.channels() == 1) ? 2 /*AIE_RAW8*/ : 0 /*AIE_BGR8*/;
            FAYS_ATRAK_D_FeedStereoImage(depthHandle_, &feed);

            // 取深度 (async_mode=1: 未就绪时下一轮重试)
            depthImg_.width = depthImg_.height = depthImg_.channel = 0;
            depthImg_.step = depthImg_.bytes = 0;
            dispImg_.width = dispImg_.height = dispImg_.channel = 0;
            dispImg_.step = dispImg_.bytes = 0;
            pairedImg_.width = pairedImg_.height = pairedImg_.channel = 0;
            pairedImg_.step = pairedImg_.bytes = 0;
            if (FAYS_ATRAK_D_GetDepthImage(depthHandle_, &depthImg_, &dispImg_, &pairedImg_) != 0 ||
                depthImg_.width == 0 || depthImg_.height == 0) {
                continue;
            }
            cv::Mat depthM(depthImg_.height, depthImg_.width, CV_32FC1, depthImg_.data);

            // 矫正图 (显示用)
            rectImg_.width = rectImg_.height = rectImg_.channel = 0;
            rectImg_.step = rectImg_.bytes = 0;
            bool hasRect = (FAYS_ATRAK_D_GetRectifiedImage(depthHandle_, &rectImg_) == 0 &&
                            rectImg_.width > 0 && rectImg_.height > 0);
            if (hasRect) {
                cv::Mat rectM(rectImg_.height, rectImg_.width,
                              rectImg_.channel == 1 ? CV_8UC1 : CV_8UC3, rectImg_.data);
                cv::Mat rectView;
                cv::resize(rectM, rectView, cv::Size(640, 800));
                std::lock_guard<std::mutex> lk(qmtx_);
                rectQ_.push_back(rectView);
                if (rectQ_.size() > 2) rectQ_.pop_front();
            }
            if (!rectifyInfoShown && hasRect) {
                AtrakRectifyInfo info;
                std::memset(&info, 0, sizeof(info));
                if (FAYS_ATRAK_D_GetRectifyInfo(depthHandle_, &info) == 0) {
                    double fx = info.P0[0];
                    double baseline = std::fabs(info.P1[3]) / fx;   // P1[3] = -fx*T_x
                    std::cout << "[RectifyInfo] fx' = " << fx
                              << "  baseline = " << baseline * 1000.0 << " mm" << std::endl;
                }
                rectifyInfoShown = true;
            }

            // 深度入队 (640x400, 后处理前)
            cv::Mat small;
            cv::resize(depthM, small, cv::Size(640, 400));
            {
                std::lock_guard<std::mutex> lk(qmtx_);
                if (depthQ_.size() >= 3) depthQ_.pop_front();   // 显示跟不上的帧直接丢
                depthQ_.push_back(small);
                qcv_.notify_one();
            }

            // ── 每秒统计 (原始深度, 全尺寸) ──
            ++frames;
            auto now = std::chrono::steady_clock::now();
            if (now - lastStats >= std::chrono::seconds(1)) {
                double elapsed = std::chrono::duration<double>(now - lastStats).count();
                double fps = frames / elapsed;
                const float* dp = depthImg_.data;
                const size_t npx = depthImg_.width * depthImg_.height;
                size_t valid = 0;
                float mn = 1e9f, mx = 0.0f;
                std::vector<float> vals;
                vals.reserve(npx / 2);
                for (size_t i = 0; i < npx; ++i) {
                    float v = dp[i];
                    if (v > 0.1f && v < 10.0f) {
                        ++valid; vals.push_back(v);
                        if (v < mn) mn = v;
                        if (v > mx) mx = v;
                    }
                }
                std::sort(vals.begin(), vals.end());
                float median = vals.empty() ? 0.0f : vals[vals.size() / 2];
                std::cout << "[Stats] " << fps << " FPS | 有效深度 "
                          << 100.0 * valid / npx << "% | median "
                          << median << " m | [" << mn << ", " << mx << "] m | 亮度 "
                          << lumMean_.load() << " (过曝 " << lumOver_.load()
                          << "%) | 映射 [" << gMapLo.load() << ", "
                          << gMapHi.load() << "] m" << std::endl;
                // ── 深度分布诊断: 看深度值是否真的铺开 ──
                {
                    int hb[9] = {0};   // 0.2-0.4, 0.4-0.6, ..., 1.8-2.0
                    int hNear = 0, hFar = 0;
                    for (size_t i = 0; i < npx; ++i) {
                        float v = dp[i];
                        if (v <= 0.1f || v >= 10.0f) continue;
                        if (v < 0.2f) hNear++;
                        else if (v >= 2.0f) hFar++;
                        else hb[static_cast<int>((v - 0.2f) / 0.2f)]++;
                    }
                    std::cout << "[DepthHist] <0.2m:" << 100.0 * hNear / valid
                              << "% 0.2-0.4:" << 100.0 * hb[0] / valid
                              << "% 0.4-0.6:" << 100.0 * hb[1] / valid
                              << "% 0.6-0.8:" << 100.0 * hb[2] / valid
                              << "% 0.8-1.0:" << 100.0 * hb[3] / valid
                              << "% 1.0-1.2:" << 100.0 * hb[4] / valid
                              << "% 1.2-1.4:" << 100.0 * hb[5] / valid
                              << "% 1.4-1.6:" << 100.0 * hb[6] / valid
                              << "% 1.6-1.8:" << 100.0 * hb[7] / valid
                              << "% 1.8-2.0:" << 100.0 * hb[8] / valid
                              << "% >2.0m:" << 100.0 * hFar / valid << "%" << std::endl;
                }
                frames = 0;
                lastStats = now;
            }

            // 主线程按键请求的执行端 (数据都在本线程, 文件操作也在这里):
            // s: 全尺寸 .bin + 矫正图 (主线程已存 disp8 的 _depth.png)
            if (saveReq_.load()) {
                long long ts = lastTs_.load();
                char base[64];
                std::snprintf(base, sizeof(base), "depth_%lld",
                              static_cast<long long>(ts / 1000000));
                std::ofstream bin(base + std::string(".bin"), std::ios::binary);
                bin.write(reinterpret_cast<const char*>(depthImg_.data),
                          static_cast<std::streamsize>(depthImg_.width *
                                                       depthImg_.height * sizeof(float)));
                bin.close();
                std::string note = "[Save] " + std::string(base) + ".bin";
                if (hasRect) {
                    cv::Mat rectM(rectImg_.height, rectImg_.width,
                                  rectImg_.channel == 1 ? CV_8UC1 : CV_8UC3, rectImg_.data);
                    cv::imwrite(std::string(base) + "_rect.png", rectM);
                    note += " / _rect.png";
                }
                std::cout << note << std::endl;
                saveReq_.store(false);
            }
            // p: 点云 PLY
            if (plyReq_.load()) {
                cloud_.point_num = 0;
                int ret = FAYS_ATRAK_D_GetPointcloud(depthHandle_, &cloud_,
                                                     FAYS_ATRAK_MAX_DENSE_MAP_POINTS_NUM);
                if (ret == 0 && cloud_.point_num > 0) {
                    char base[64];
                    std::snprintf(base, sizeof(base), "cloud_%lld.ply",
                                  static_cast<long long>(lastTs_.load() / 1000000));
                    savePly(base, cloud_);
                } else {
                    std::cout << "[PLY] 无点云 (ret=" << ret << ")" << std::endl;
                }
                plyReq_.store(false);
            }
        }
    });

    // ── 主显示线程: 取最新帧 → 后处理 → 显示 ──
    int64_t prevFrameMs = 0;
    int64_t maxGapMs = 0, sumGapMs = 0;
    int gapCount = 0;
    while (running_.load()) {
        cv::Mat normSmall, stereoView, rectView;
        {
            std::unique_lock<std::mutex> lk(qmtx_);
            qcv_.wait(lk, [this] { return !depthQ_.empty() || !running_.load(); });
            if (depthQ_.empty()) break;
            normSmall = depthQ_.back();      // 取最新, 积压帧丢弃 (不拖采集)
            depthQ_.clear();
            if (!stereoQ_.empty()) { stereoView = stereoQ_.back(); stereoQ_.clear(); }
            if (!rectQ_.empty()) { rectView = rectQ_.back(); rectQ_.clear(); }
        }

        if (!stereoView.empty()) {
            int hh = stereoView.rows / 2;
            cv::putText(stereoView, "LEFT (cam0)", cv::Point(15, 30),
                        cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 255, 0), 2);
            cv::putText(stereoView, "RIGHT (cam1)", cv::Point(15, hh + 30),
                        cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 255, 255), 2);
            cv::imshow("Stereo Input (L/R)", stereoView);
        }
        if (!rectView.empty())
            cv::imshow("Rectified (engine)", rectView);

        // ── 深度显示 ──
        // 官方模式 (默认): 官方 GUI 同款 —— 深度数据原样 + 直方图均衡 + JET.
        //   范围预设按 M 循环 (0.1-1m ~ 0.3-8m, 默认 0.3-4m 同官方 GUI).
        //   这是 SDK 的"正常效果": 引擎输出什么样, 显示就是什么样.
        // 增强模式 (按 F): 历史迭代的后处理链 (去噪/填洞/CLAHE), 深度被改值.
        cv::Mat disp8;   // 保存用 (官方模式=indexed, 增强模式=分位数映射图)
        cv::Mat depthColor;
        if (officialMode_) {
            const float pmin = kRangePresets[presetIdx_].minD;
            const float pmax = kRangePresets[presetIdx_].maxD;
            officialHistogramEqualize(normSmall, disp8, pmin, pmax);
            cv::applyColorMap(disp8, depthColor, cv::COLORMAP_JET);
            // JET 的索引 0 是深蓝, 显式把无效像素置黑 (官方 colormap.cpp 同款)
            depthColor.setTo(cv::Scalar(0, 0, 0), normSmall <= 0.0f);
        } else {
        // ── 增强模式: D435 风格深度后处理 (显示侧, 640x400) ──
        // 1) speckle: 小于 15px 的孤立有效区是飞点, 置无效
        {
            cv::Mat validMask = (normSmall > 0.1f) & (normSmall < 1.5f);
            cv::Mat labels, stats, centroids;
            cv::connectedComponentsWithStats(validMask, labels, stats, centroids);
            for (int i = 1; i < stats.rows; ++i) {
                if (stats.at<int>(i, cv::CC_STAT_AREA) < 15)
                    normSmall.setTo(0.0f, labels == i);
            }
        }
        // 2) 单帧空间一致性剔除 (先于时间滤波!):
        //    像素偏离 5x5 邻域中值 > 0.5m 判为错配置无效.
        //    顺序关键: 若先做时间中值, 墙面 11 帧里的错配近值互相支持,
        //    会把 2.5m 真值拉进近距带 → 整面墙稳定成蓝色 (回归根因).
        {
            const float kInvalid = 1000.0f;
            cv::Mat tmp = normSmall.clone();
            tmp.setTo(kInvalid, normSmall <= 0.1f);
            cv::Mat base;
            cv::medianBlur(tmp, base, 5);
            cv::Mat diff = cv::abs(normSmall - base);
            normSmall.setTo(0.0f, diff > 0.5f);
        }
        // 3) 时间中值 (带远距保护): 5 帧环形缓冲 (11 帧抹边严重, 减负保轮廓;
        //    近距候选过半取中值, 近值不足一半但出现过远值 → 1.6m 远档红,
        //    全无效 → 0)
        {
            static std::vector<cv::Mat> tbuf(
                5, cv::Mat::zeros(normSmall.size(), CV_32FC1));
            static int slot = 0;
            normSmall.copyTo(tbuf[slot]);
            slot = (slot + 1) % 5;
            cv::Mat medT(normSmall.size(), CV_32FC1, cv::Scalar(0.0f));
            const float kFar = 1.6f;
            for (int r = 0; r < normSmall.rows; ++r) {
                for (int c = 0; c < normSmall.cols; ++c) {
                    float v[5];
                    int near = 0, far = 0;
                    for (int t = 0; t < 5; ++t) {
                        float x = tbuf[t].at<float>(r, c);
                        if (x > 0.1f && x < 1.5f) v[near++] = x;
                        else if (x >= 1.5f) ++far;
                    }
                    if (near >= 3) {
                        std::nth_element(v, v + near / 2, v + near);
                        medT.at<float>(r, c) = v[near / 2];
                    } else if (far >= 1) {
                        medT.at<float>(r, c) = kFar;
                    }
                }
            }
            normSmall = medT;
        }
        // ── 深度着色: 自适应分位数映射 (D435 Viewer 风格) ──
        // 直方图均衡按像素数量分色阶, 远景占多数时近景被压缩;
        // 改为每帧取有效深度 [2%,98%] 分位作映射范围, 线性拉伸到全色域.
        // 分位范围用 EMA 平滑, 颜色不闪烁.
        disp8.create(normSmall.size(), CV_8UC1);
        disp8.setTo(cv::Scalar(0));
        {
            const int BINS = 4096;
            const float dmin = 0.1f, dmax = 1.5f;
            const float binW = (dmax - dmin) / BINS;
            std::vector<int> hist(BINS, 0);
            long long total = 0;
            for (int r = 0; r < normSmall.rows; ++r) {
                const float* p = normSmall.ptr<float>(r);
                for (int c = 0; c < normSmall.cols; ++c) {
                    float d = p[c];
                    if (d > dmin && d < dmax) {
                        int b = static_cast<int>((d - dmin) / binW);
                        if (b >= BINS) b = BINS - 1;
                        hist[b]++; total++;
                    }
                }
            }
            float lo = dmin, hi = dmax;
            if (total >= 1000) {   // 有效像素太少时退回固定范围
                long long tLo = total * 2 / 100, tHi = total * 98 / 100;
                long long cum = 0;
                bool loFound = false;
                for (int b = 0; b < BINS; ++b) {
                    cum += hist[b];
                    if (!loFound && cum >= tLo) { lo = dmin + (b + 0.5f) * binW; loFound = true; }
                    if (cum >= tHi) { hi = dmin + (b + 0.5f) * binW; break; }
                }
                if (hi - lo < 0.15f) hi = lo + 0.15f;   // 最小范围防抖动
            }
            gMapLo.store(gMapLo.load() * 0.8f + lo * 0.2f);   // 慢速 EMA, 场景切换跟得上
            gMapHi.store(gMapHi.load() * 0.8f + hi * 0.2f);
            const float mapLo = gMapLo.load(), mapHi = gMapHi.load();
            const float inv = 255.0f / (mapHi - mapLo);
            for (int r = 0; r < normSmall.rows; ++r) {
                const float* p = normSmall.ptr<float>(r);
                uchar* q = disp8.ptr<uchar>(r);
                for (int c = 0; c < normSmall.cols; ++c) {
                    float d = p[c];
                    if (d > dmin) {   // 1.5m 以上自动 clamp 到 255 (最远档红色)
                        float v = (d - mapLo) * inv;
                        q[c] = static_cast<uchar>(std::max(0.0f, std::min(255.0f, v)));
                    }
                }
            }
        }
        // 4) 轮廓增强: 近景掩膜形态学闭运算 → 补掉物体边缘遮挡区窄缝/缺口
        //    (洞像素用 3x3 邻域中值填充, 不改变已有深度层次)
        {
            cv::Mat mask = (normSmall > 0.1f) & (normSmall < 1.6f);
            cv::Mat kern = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(5, 5));
            cv::morphologyEx(mask, mask, cv::MORPH_CLOSE, kern);
            cv::Mat hole = mask & (normSmall <= 0.1f);
            const float kInvalid = 1000.0f;
            cv::Mat tmp = normSmall.clone();
            tmp.setTo(kInvalid, normSmall <= 0.1f);
            cv::medianBlur(tmp, tmp, 3);
            tmp.setTo(0.0f, tmp > 10.0f);
            tmp.copyTo(normSmall, hole);
        }
        // 5) CLAHE: 局部对比度拉伸, 相邻色阶拉开 → 轮廓视觉上更锐
        {
            cv::Mat invMask = disp8 == 0;
            cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(2.5, cv::Size(8, 8));
            clahe->apply(disp8, disp8);
            disp8.setTo(0, invMask);   // 无效区保持黑
        }
            cv::applyColorMap(disp8, depthColor, cv::COLORMAP_JET);
        }
        cv::imshow("Depth", depthColor);

        // ── 显示帧间隔抖动 (局部卡顿的直接度量) ──
        auto now = std::chrono::steady_clock::now();
        int64_t nowMs = std::chrono::duration_cast<std::chrono::milliseconds>(
                            now.time_since_epoch()).count();
        if (prevFrameMs != 0) {
            int64_t gap = nowMs - prevFrameMs;
            if (gap > maxGapMs) maxGapMs = gap;
            sumGapMs += gap;
            ++gapCount;
        }
        prevFrameMs = nowMs;
        static auto lastDispStats = std::chrono::steady_clock::now();
        if (now - lastDispStats >= std::chrono::seconds(1)) {
            double avgGap = gapCount ? static_cast<double>(sumGapMs) / gapCount : 0.0;
            std::cout << "[Disp] 显示帧间隔均值 " << avgGap << " ms 最大 "
                      << maxGapMs << " ms" << std::endl;
            lastDispStats = now;
            maxGapMs = 0; sumGapMs = 0; gapCount = 0;
        }

        // ── 按键 (显示线程; e/g 直调 viKit, 与官方 GUI 同构) ──
        int key = cv::waitKey(1);
        if (key == 27 || key == 'q' || key == 'Q') {
            running_.store(false);
        } else if (key == 's' || key == 'S') {
            long long ts = lastTs_.load();
            char base[64];
            std::snprintf(base, sizeof(base), "depth_%lld",
                          static_cast<long long>(ts / 1000000));
            cv::imwrite(std::string(base) + "_depth.png", disp8);
            saveReq_.store(true);   // 采集线程存 .bin + _rect.png
        } else if (key == 'c' || key == 'C') {
            feedBgr_.store(!feedBgr_.load());
            std::cout << "[Feed] 送引擎通道序切换: "
                      << (feedBgr_.load() ? "BGR (SDK 文档约定)" : "RGB (固件原始)")
                      << std::endl;
        } else if (key == 'e' || key == 'E') {
            // 曝光预设循环 (文档范围 1.0~507.0; -1 = 自动)
            static const double kExposurePresets[] = {-1.0, 500.0, 400.0, 300.0,
                                                      200.0, 100.0, 50.0};
            static int expIdx = 0;
            expIdx = (expIdx + 1) %
                     static_cast<int>(sizeof(kExposurePresets) / sizeof(kExposurePresets[0]));
            double ev = kExposurePresets[expIdx];
            int ret = FAYS_VIK_SetStereoExposure(viHandle_, ev);
            std::cout << "[Exposure] " << (ev < 0 ? "AUTO" : std::to_string(ev))
                      << " (ret=" << ret << ")" << std::endl;
        } else if (key == 'g' || key == 'G') {
            // 增益预设循环 (文档范围 1.0~15.0; -1 = 自动)
            static const float kGainPresets[] = {-1.0f, 2.0f, 4.0f, 8.0f, 12.0f};
            static int gainIdx = 0;
            gainIdx = (gainIdx + 1) %
                      static_cast<int>(sizeof(kGainPresets) / sizeof(kGainPresets[0]));
            float gv = kGainPresets[gainIdx];
            int ret = FAYS_VIK_SetStereoGain(viHandle_, gv);
            std::cout << "[Gain] " << (gv < 0 ? "AUTO" : std::to_string(gv))
                      << " (ret=" << ret << ")" << std::endl;
        } else if (key == 'p' || key == 'P') {
            plyReq_.store(true);
        } else if (key == 'f' || key == 'F') {
            officialMode_ = !officialMode_;
            std::cout << "[Mode] " << (officialMode_ ? "官方模式 (深度原样 + 直方图均衡)"
                                                     : "增强模式 (后处理链 + 分位数映射)")
                      << std::endl;
        } else if (key == 'm' || key == 'M') {
            presetIdx_ = (presetIdx_ + 1) %
                         static_cast<int>(sizeof(kRangePresets) / sizeof(kRangePresets[0]));
            std::cout << "[Range] " << kRangePresets[presetIdx_].minD << " – "
                      << kRangePresets[presetIdx_].maxD << " m" << std::endl;
        }
    }

    running_.store(false);
    capThread.join();
    cv::destroyAllWindows();
}

int main(int argc, char** argv)
{
    const char* env = std::getenv("SDK_DIR");
    std::string base = env ? env : "/home/REDACTED/FaysSense_VI_Kit_Release";
    std::string viKitConfig = base + "/config/fays_vikit.yaml";
    std::string depthConfig = base + "/config/perception/stereo_depth/stereo_depth.yaml";
    if (argc > 1) viKitConfig = argv[1];
    if (argc > 2) depthConfig = argv[2];

    std::cout << "== S80M 双目深度 Demo (严格按 SDK 实现) ==" << std::endl;
    // 限制 OpenCV 并行线程数 (默认吃满所有核):
    // 深度引擎(WLS/SGBM) + 本 demo 后处理全开并行时把系统顶到高负载,
    // 与录屏/浏览器抢 CPU → 系统级卡顿. 限 2 线程后引擎仍 25fps,
    // CPU 占用大降. (引擎内部线程不受此控制, 见 ps 线程分布)
    cv::setNumThreads(2);
    StereoDepthDemo demo(viKitConfig, depthConfig);
    if (!demo.ok()) return 1;
    demo.run();
    return 0;
}
