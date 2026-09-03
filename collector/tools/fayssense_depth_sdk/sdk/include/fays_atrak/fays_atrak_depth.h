#pragma once


#include "fays_atrak_types.h"


#if (defined (_WIN32) || defined(WIN64))
	#define FAYS_ATRAK_API __declspec(dllexport)
#else
    #define FAYS_ATRAK_API __attribute__((visibility("default")))
#endif


#ifdef __cplusplus
extern "C" {
#endif 



/**
 * @brief Maximum number of depth parameter KVs in a single
 *        FAYS_ATRAK_D_SetDepthParam call.
 */
#define FAYS_ATRAK_DEPTH_PARAM_MAX_COUNT 32

/**
 * @brief Total number of depth parameter slots in the depth engine.
 *
 * Each ATRAK_DepthParamId value is used directly as the slot index, so this
 * value must always be larger than the greatest defined parameter ID.
 */
#define FAYS_ATRAK_DEPTH_PARAM_SLOT_COUNT 128


/**
 * @brief Maximum number of 3D point cloud parameter KVs in a single
 *        FAYS_ATRAK_D_SetPointCloudParam call.
 */
#define FAYS_ATRAK_PC_PARAM_MAX_COUNT 32

/**
 * @brief Total number of 3D point cloud parameter slots in the depth engine.
 *
 * Each ATRAK_DepthPointParamId value is used directly as the slot index, so this
 * value must always be larger than the greatest defined parameter ID.
 */
#define FAYS_ATRAK_DEPTHPOINT_PARAM_SLOT_COUNT 256


/****************************************************************************
 *  standard stereo-depth SDK
 ****************************************************************************/

/**
 * @brief Depth algorithm backends.
 */
typedef enum {
    ATRAK_DEPTH_BACKEND_SGBM   = 0,   //< OpenCV SGBM (CPU mode)
    ATRAK_DEPTH_BACKEND_CUSTOM = 1,   //< custom depth model (e.g. NPU)
    ATRAK_DEPTH_BACKEND_OTHER  = 2,   //< reserved for future backends
} ATRAK_DepthBackendType;

/**
 * @brief Depth parameter IDs, used as KV keys.
 *
 * The ID space is partitioned per backend:
 *   [101, 120) = SGBM parameters (including the WLS post-filter)
 */
typedef enum {
    ATRAK_DP_SGBM_MIN_DISP             = 101,  //< min_disp
    ATRAK_DP_SGBM_NUM_DISP             = 102,  //< num_disp (multiple of 16, [16, 128])
    ATRAK_DP_SGBM_WINDOW_SIZE          = 103,  //< window_size / block size (odd, [1, 31])
    ATRAK_DP_SGBM_P1                   = 104,  //< P1 penalty
    ATRAK_DP_SGBM_P2                   = 105,  //< P2 penalty
    ATRAK_DP_SGBM_DISP12_MAX_DIFF      = 106,  //< disp12MaxDiff
    ATRAK_DP_SGBM_PRE_FILTER_CAP       = 107,  //< preFilterCap
    ATRAK_DP_SGBM_UNIQUENESS_RATIO     = 108,  //< uniquenessRatio
    ATRAK_DP_SGBM_SPECKLE_WINDOW_SIZE  = 109,  //< speckleWindowSize
    ATRAK_DP_SGBM_SPECKLE_RANGE        = 110,  //< speckleRange
    ATRAK_DP_SGBM_MODE                 = 111,  //< SGBM mode (0=SGBM_3WAY, 1=SGBM, 2=HH4)
    ATRAK_DP_SGBM_WLS_ENABLE           = 112,  //< WLS filter toggle (0=off, 1=on)
    ATRAK_DP_SGBM_WLS_LAMBDA           = 113,  //< WLS lambda
    ATRAK_DP_SGBM_WLS_SIGMA_COLOR      = 114,  //< WLS sigmaColor
    ATRAK_DP_SGBM_WLS_DISC_RADIUS      = 115,  //< WLS depthDiscontinuityRadius
} ATRAK_DepthParamId;

/**
 * @brief A single depth parameter, addressed by ID with a float value.
 */
typedef struct {
    ATRAK_DepthParamId param_id;    //< parameter ID
    float value;                    //< parameter value
} ATRAK_DepthParamKv;

/**
 * @brief 3D point cloud generation backends.
 */
typedef enum {
    ATRAK_DEPTHPOINT_BACK_CPU   = 0,   //< CPU point cloud generation
    ATRAK_DEPTHPOINT_BACK_OTHER = 1,   //< reserved for future backends
} ATRAK_DepthPointBackType;

/**
 * @brief 3D point cloud parameter IDs, used as KV keys.
 *
 * The ID space is partitioned per backend:
 *   [201, 220) = point cloud generation parameters
 */
typedef enum {
    ATRAK_DPP_POINTCLOUD_TYPE  = 201,  //< pointcloud_type
    ATRAK_DPP_POINTCLOUD_ROI   = 202,  //< pointcloud_ROI (%) 0~1
    ATRAK_DPP_SKIP_POINT       = 203,  //< skip_point (stride for point sampling)
    ATRAK_DPP_FARTHEST_DIST    = 204,  //< farthest_dist (m)
    ATRAK_DPP_CLOSEST_DIST     = 205,  //< closest_dist (m)
    ATRAK_DPP_VOXEL_SIZE       = 206,  //< voxelSize (m)
    ATRAK_DPP_MIN_POINTS       = 207,  //< minPoints (minimum points threshold)
    ATRAK_DPP_ENABLE_VOXEL     = 208,  //< enable_voxel (0=off, 1=on)
} ATRAK_DepthPointParamId;

/**
 * @brief A single 3D point cloud parameter, addressed by ID with a float value.
 */
typedef struct {
    ATRAK_DepthPointParamId param_id;    //< parameter ID
    float value;                    //< parameter value
} ATRAK_DepthPointParamKv;

/**
 * @brief Create a depth engine handle with configuration.
 *
 * @param[out] handle      Depth engine handle.
 * @param[in]  configPath  Path to stereo_depth.yaml configuration file.
 * @return 0 on success, non-zero on failure.
 */
FAYS_ATRAK_API     int   FAYS_ATRAK_D_CreateHandleWithConfig         (void** handle, const char* configPath);

/**
 * @brief Destroy the depth engine handle and release all resources.
 *
 * @param[in] handle  Depth engine handle.
 * @return 0 on success, non-zero on failure.
 */
FAYS_ATRAK_API     int   FAYS_ATRAK_D_DestroyHandle                  (void* handle);

/**
 * @brief Feed a vertically-stacked stereo image to the depth engine.
 *
 * The image is stacked top-bottom: upper half = left camera, lower half = right camera.
 *
 * @param[in] handle        Depth engine handle.
 * @param[in] pStereoImage  Stereo image (AIE_RAW8 or AIE_BGR8).
 * @return 0 on success, non-zero on failure.
 */
FAYS_ATRAK_API     int   FAYS_ATRAK_D_FeedStereoImage                (void* handle, AtrakImage* pStereoImgae);

/**
 * @brief Feed the middle RGB camera image for depth-to-RGB alignment.
 *
 * Only meaningful when align_cam_id is set in stereo_depth.yaml.
 *
 * @param[in] handle      Depth engine handle.
 * @param[in] pRgbImage   RGB image (AIE_BGR8).
 * @return 0 on success, non-zero on failure.
 */
FAYS_ATRAK_API     int   FAYS_ATRAK_D_FeedRGBImage                   (void* handle, AtrakImage* pStereoImgae);

/**
 * @brief Feed cam0 pose (T_w_cam0) for pointcloud transformation.
 * 
 * @param handle 
 * @param pPose 
 * @return Status code: 0 = Success, non-zero = Failure
 */
FAYS_ATRAK_API     int   FAYS_ATRAK_D_FeedPose                       (void* handle, AtrakPose* pPose);

/**
 * @brief Get depth results.
 *
 * Caller must pre-allocate data buffers and is responsible for freeing them.
 *
 * @param[in]  handle        Depth engine handle.
 * @param[out] pDepthImage   Depth image (float, in meters). Caller allocates data buffer.
 * @param[out] pDispImage    Rectified raw disparity image (non-aligned, possibly downsampled). Caller allocates data buffer.
 * @param[out] pairedImg     Paired image aligned to depth (only valid if alignment enabled). Caller allocates data buffer.
 * @return 0 on success, non-zero on failure.
 */
FAYS_ATRAK_API     int   FAYS_ATRAK_D_GetDepthImage                  (void* handle, AtrakDepthImage* pDepthImage,  AtrakDepthImage* pDispImage, AtrakImage* pairedImg);

/**
 * @brief Get the rectified stereo image.
 *
 * Returns a vertically-stacked (top-bottom) rectified stereo image.
 * Caller must pre-allocate the data buffer and is responsible for freeing it.
 *
 * @param[in]  handle     Depth engine handle.
 * @param[out] pRectImage Rectified stereo image (BGR8). Caller allocates data buffer.
 * @return 0 on success, non-zero on failure.
 */
FAYS_ATRAK_API     int   FAYS_ATRAK_D_GetRectifiedImage              (void* handle, AtrakImage* pRectImage);

/**
 * @brief Get the rectification matrices for the stereo pair.
 *
 * Caller allocates the AtrakRectifyInfo struct and is responsible for its lifetime.
 *
 * @param[in]  handle    Depth engine handle.
 * @param[out] pRectInfo Rectification info (R0, R1, P0, P1 matrices). Caller allocates.
 * @return 0 on success, non-zero on failure.
 */
FAYS_ATRAK_API     int   FAYS_ATRAK_D_GetRectifyInfo                 (void* handle, AtrakRectifyInfo* pRectInfo);

/**
 * @brief Get the 3D point cloud from the latest depth frame.
 *
 * @param[in]  handle    Depth engine handle.
 * @param[out] pCloud    Point cloud buffer. Caller must pre-allocate pos (float[3N]) and rgb (uint32_t[N]) arrays.
 * @param[in]  maxPoints Maximum number of points to retrieve.
 * @return 0 on success, non-zero on failure.
 */
FAYS_ATRAK_API     int   FAYS_ATRAK_D_GetPointcloud                  (void* handle, AtrakMap* pCloud, int maxPoints);

/**
 * @brief Bind a ViKit sensor handle to the depth engine.
 *
 * Call after FAYS_ATRAK_D_CreateHandleWithConfig
 *
 * @param depthHandle  Depth engine handle.
 * @param viKitHandle  ViKit sensor handle (from FAYS_VIK_CreateHandleWithConfig).
 * @return 0 on success, non-zero on failure.
 */
FAYS_ATRAK_API     int   FAYS_ATRAK_D_BindViKit                     (void* depthHandle, void* viKitHandle);



/**
 * @brief Set depth algorithm parameters at runtime in a batch of key-value pairs.
 *
 * Only the passed KVs are updated; all other parameters keep their previous
 * values. The parameters take effect on the next processed frame.
 *
 * SGBM parameters are only applied when backend is ATRAK_DEPTH_BACKEND_SGBM;
 * when another backend is active, they are ignored (and the engine keeps
 * running with its current parameters).
 *
 * @param[in] handle    Depth engine handle (from FAYS_ATRAK_D_CreateHandleWithConfig).
 * @param[in] backend   Target algorithm backend the KVs belong to.
 * @param[in] kv_array  KV array (may be NULL when kv_count is 0).
 * @param[in] kv_count  Number of KVs in the array (must be >= 0 and <= FAYS_ATRAK_DEPTH_PARAM_MAX_COUNT).
 * @return 0 on success, non-zero on failure (invalid handle / null kv_array with non-zero count / count out of range).
 */
FAYS_ATRAK_API int FAYS_ATRAK_D_SetDepthParam(void* handle,
                                              ATRAK_DepthBackendType backend,
                                              const ATRAK_DepthParamKv* kv_array,
                                              int kv_count);


/**
 * @brief Set the point cloud generation parameters at runtime in a batch of key-value pairs.
 *
 * Only the passed KVs are updated; all other parameters keep their previous
 * values. The parameters take effect on the next processed frame.
 *
 * 3D point parameters are only applied when backend is ATRAK_DEPTHPOINT_BACK_CPU;
 * when another backend is active, they are ignored (and the engine keeps
 * generating point clouds with its current parameters).
 *
 * @param[in] handle    Depth engine handle (from FAYS_ATRAK_D_CreateHandleWithConfig).
 * @param[in] backend   Target point cloud generation backend the KVs belong to.
 * @param[in] kv_array  KV array (may be NULL when kv_count is 0).
 * @param[in] kv_count  Number of KVs in the array (must be >= 0 and <= FAYS_ATRAK_PC_PARAM_MAX_COUNT).
 * @return 0 on success, non-zero on failure (invalid handle / null kv_array with non-zero count / count out of range).
 */
FAYS_ATRAK_API int FAYS_ATRAK_D_SetPointCloudParam(void* handle,
                                                   ATRAK_DepthPointBackType backend,
                                                   const ATRAK_DepthPointParamKv* kv_array,
                                                   int kv_count);

#ifdef __cplusplus
}
#endif

#ifdef __cplusplus

/**
 * @brief Convenience helper for building a depth parameter KV pair.
 */
inline ATRAK_DepthParamKv makeKv(ATRAK_DepthParamId id, float val)
{
    ATRAK_DepthParamKv kv;
    kv.param_id = id;
    kv.value = val;
    return kv;
}

/**
 * @brief Convenience helper for building a 3D point cloud parameter KV pair.
 */
inline ATRAK_DepthPointParamKv makeKv(ATRAK_DepthPointParamId id, float val)
{
    ATRAK_DepthPointParamKv kv;
    kv.param_id = id;
    kv.value = val;
    return kv;
}

#endif
