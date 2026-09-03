#include "depth_engine.h"
#include <iostream>
#include <cstring>
#include <cstdlib>

#include "fays_atrak/fays_atrak_depth.h"

DepthEngine::DepthEngine(const std::string& configPath)
{
    // Create depth engine handle
    int ret = FAYS_ATRAK_D_CreateHandleWithConfig(&mHandle, configPath.c_str());
    if (ret != 0) {
        std::cerr << "[DepthEngine] Failed to create handle with config: "
                  << configPath << " (error " << ret << ")" << std::endl;
        return;
    }

    // Pre-allocate output buffers
    size_t depthBytes = kMaxDepthFloats * sizeof(float);
    size_t imgBytes   = kMaxMonoBytes;

    mDepthBuf = new AtrakDepthImage{};
    mDepthBuf->data = new float[kMaxDepthFloats];
    std::memset(mDepthBuf->data, 0, depthBytes);

    mDispBuf = new AtrakDepthImage{};
    mDispBuf->data = new float[kMaxDepthFloats];
    std::memset(mDispBuf->data, 0, depthBytes);

    mRectBuf = new AtrakImage{};
    mRectBuf->data = new unsigned char[imgBytes * 3];  // worst-case 3-channel
    std::memset(mRectBuf->data, 0, imgBytes * 3);

    mPairedBuf = new AtrakImage{};
    mPairedBuf->data = new unsigned char[kMaxRgbBytes];
    std::memset(mPairedBuf->data, 0, kMaxRgbBytes);

    // Pre-allocate point cloud buffer
    mCloudMaxPoints = 262144;  // FAYS_ATRAK_MAX_DENSE_MAP_POINTS_NUM
    mCloudBuf = new AtrakMap{};
    mCloudBuf->pos = new float[mCloudMaxPoints * 3];
    mCloudBuf->rgb = new uint32_t[mCloudMaxPoints];
    std::memset(mCloudBuf->pos, 0, mCloudMaxPoints * 3 * sizeof(float));
    std::memset(mCloudBuf->rgb, 0, mCloudMaxPoints * sizeof(uint32_t));

    mInitialized = true;
    std::cout << "[DepthEngine] Initialized successfully." << std::endl;
}

bool DepthEngine::bindViKit(void* viKitHandle)
{
    if (!mInitialized || !viKitHandle) return false;
    int ret = FAYS_ATRAK_D_BindViKit(mHandle, viKitHandle);
    if (ret == 0) {
        std::cout << "[DepthEngine] ViKit bound successfully." << std::endl;
        return true;
    }
    std::cerr << "[DepthEngine] BindViKit failed (error " << ret << ")" << std::endl;
    return false;
}

DepthEngine::~DepthEngine()
{
    if (mHandle) {
        FAYS_ATRAK_D_DestroyHandle(mHandle);
        mHandle = nullptr;
    }

    if (mDepthBuf) {
        delete[] mDepthBuf->data;
        delete mDepthBuf;
    }
    if (mDispBuf) {
        delete[] mDispBuf->data;
        delete mDispBuf;
    }
    if (mRectBuf) {
        delete[] mRectBuf->data;
        delete mRectBuf;
    }
    if (mPairedBuf) {
        delete[] mPairedBuf->data;
        delete mPairedBuf;
    }
    if (mCloudBuf) {
        delete[] mCloudBuf->pos;
        delete[] mCloudBuf->rgb;
        delete mCloudBuf;
    }
}

int DepthEngine::feedStereoImage(const cv::Mat& stackedStereo, uint64_t timestampNs)
{
    if (!mInitialized)
        return -1;

    // We need a non-const data pointer for the C API.
    // Create a local copy to ensure contiguous memory owned by us.
    cv::Mat temp;
    if (!stackedStereo.isContinuous()) {
        temp = stackedStereo.clone();
    } else {
        temp = stackedStereo;
    }

    AtrakImage img{};
    img.device_id = 0;  // ATRAK_DEV_STEREO
    img.timestamp = timestampNs;
    img.width     = temp.cols;
    img.height    = temp.rows;
    img.channel   = temp.channels();
    img.step      = static_cast<int>(temp.step);
    img.bytes     = static_cast<unsigned int>(temp.total() * temp.elemSize());
    img.data      = temp.data;

    if (temp.channels() == 1)
        img.encoding = 2;  // AIE_RAW8
    else
        img.encoding = 0;  // AIE_BGR8

    return FAYS_ATRAK_D_FeedStereoImage(mHandle, &img);
}

int DepthEngine::feedRgbImage(const cv::Mat& rgbBgr, uint64_t timestampNs)
{
    if (!mInitialized)
        return -1;

    cv::Mat temp;
    if (!rgbBgr.isContinuous()) {
        temp = rgbBgr.clone();
    } else {
        temp = rgbBgr;
    }

    AtrakImage img{};
    img.device_id = 1;  // ATRAK_DEV_MIDDLE
    img.timestamp = timestampNs;
    img.width     = temp.cols;
    img.height    = temp.rows;
    img.channel   = temp.channels();
    img.step      = static_cast<int>(temp.step);
    img.bytes     = static_cast<unsigned int>(temp.total() * temp.elemSize());
    img.data      = temp.data;
    img.encoding  = 0;  // AIE_BGR8

    return FAYS_ATRAK_D_FeedRGBImage(mHandle, &img);
}

bool DepthEngine::getDepthImage(cv::Mat& depthMeters, cv::Mat& disparity, cv::Mat& pairedImage)
{
    if (!mInitialized)
        return false;

    // Reset buffer fields before calling
    mDepthBuf->width = mDepthBuf->height = mDepthBuf->channel = 0;
    mDepthBuf->step = mDepthBuf->bytes = 0;
    mDispBuf->width = mDispBuf->height = mDispBuf->channel = 0;
    mDispBuf->step = mDispBuf->bytes = 0;
    mPairedBuf->width = mPairedBuf->height = mPairedBuf->channel = 0;
    mPairedBuf->step = mPairedBuf->bytes = 0;

    int ret = FAYS_ATRAK_D_GetDepthImage(mHandle, mDepthBuf, mDispBuf, mPairedBuf);

    if (ret != 0 || mDepthBuf->width == 0 || mDepthBuf->height == 0)
        return false;

    // Wrap the SDK's float* buffer into a cv::Mat (no copy — shares the pre-allocated buffer)
    depthMeters = cv::Mat(mDepthBuf->height, mDepthBuf->width, CV_32FC1, mDepthBuf->data).clone();

    // Disparity (optional)
    if (!disparity.empty() || mDispBuf->width > 0) {
        if (mDispBuf->width > 0 && mDispBuf->height > 0) {
            disparity = cv::Mat(mDispBuf->height, mDispBuf->width, CV_32FC1, mDispBuf->data).clone();
        }
    }

    // Paired image (optional — only valid when alignment is enabled)
    if (!pairedImage.empty() || mPairedBuf->width > 0) {
        if (mPairedBuf->width > 0 && mPairedBuf->height > 0 && mPairedBuf->data) {
            int type = (mPairedBuf->channel == 1) ? CV_8UC1 : CV_8UC3;
            pairedImage = cv::Mat(mPairedBuf->height, mPairedBuf->width, type, mPairedBuf->data).clone();
        }
    }

    return true;
}

bool DepthEngine::getRectifiedImage(cv::Mat& rectifiedStereo)
{
    if (!mInitialized)
        return false;

    mRectBuf->width = mRectBuf->height = mRectBuf->channel = 0;
    mRectBuf->step = mRectBuf->bytes = 0;

    int ret = FAYS_ATRAK_D_GetRectifiedImage(mHandle, mRectBuf);

    if (ret != 0 || mRectBuf->width == 0 || mRectBuf->height == 0)
        return false;

    int type = (mRectBuf->channel == 1) ? CV_8UC1 : CV_8UC3;
    rectifiedStereo = cv::Mat(mRectBuf->height, mRectBuf->width, type, mRectBuf->data).clone();

    return true;
}

bool DepthEngine::getRectifyInfo(AtrakRectifyInfo& rectifyInfo)
{
    if (!mInitialized)
        return false;

    std::memset(&rectifyInfo, 0, sizeof(AtrakRectifyInfo));
    int ret = FAYS_ATRAK_D_GetRectifyInfo(mHandle, &rectifyInfo);

    return ret == 0;
}

bool DepthEngine::getPointCloud(std::vector<float>& xyz,
                                 std::vector<uint8_t>& rgb,
                                 int maxPoints)
{
    if (!mInitialized)
        return false;

    if (maxPoints > mCloudMaxPoints) {
        maxPoints = mCloudMaxPoints;
    }

    mCloudBuf->point_num = 0;
    mCloudBuf->pos = mCloudBuf->pos;  // retain buffer pointer
    mCloudBuf->rgb = mCloudBuf->rgb;

    int ret = FAYS_ATRAK_D_GetPointcloud(mHandle, mCloudBuf, maxPoints);

    if (ret != 0 || mCloudBuf->point_num == 0)
        return false;

    int n = static_cast<int>(mCloudBuf->point_num);
    if (n > maxPoints) n = maxPoints;

    xyz.resize(n * 3);
    rgb.resize(n * 3);

    std::memcpy(xyz.data(), mCloudBuf->pos, n * 3 * sizeof(float));

    // Set all point cloud points to white
    for (int i = 0; i < n; ++i) {
        rgb[i * 3 + 0] = 255;  // B
        rgb[i * 3 + 1] = 255;  // G
        rgb[i * 3 + 2] = 255;  // R
    }

    return true;
}

int DepthEngine::setDepthParam(ATRAK_DepthBackendType backend, const ATRAK_DepthParamKv* kv_array, int kv_count)
{
      if (!mInitialized || !kv_array || kv_count <= 0) return -1;
      return FAYS_ATRAK_D_SetDepthParam(mHandle, backend, kv_array, kv_count);   //
}

int DepthEngine::setPointCloudParam(ATRAK_DepthPointBackType backend, const ATRAK_DepthPointParamKv* kv_array, int kv_count)
{
    if (!mInitialized || !kv_array || kv_count <= 0) return -1;
    return FAYS_ATRAK_D_SetPointCloudParam(mHandle, backend, kv_array, kv_count);
}