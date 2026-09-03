#pragma once

#include <stdint.h>

#define FAYS_ATRAK_MAX_CAMERAS 3
#define FAYS_ATRAK_MAX_TRACK_INSTANCES 1
#define FAYS_ATRAK_MAX_APRIL_INSTANCES 10
#define FAYS_ATRAK_MONO_MAX_BYTES (1280 * 800 * 2)
#define FAYS_ATRAK_RGB_MAX_BYTES (3840 * 2160 * 3)
#define FAYS_ATRAK_MAX_DENSE_MAP_POINTS_NUM 262144
#define FAYS_ATRAK_MAX_SPARSE_MAP_POINTS_NUM 4096
#if FAYS_ATRAK_MONO_MAX_BYTES > FAYS_ATRAK_RGB_MAX_BYTES
#   define FAYS_ATRAK_IMG_MAX_BYTES FAYS_ATRAK_MONO_MAX_BYTES
#else
#   define FAYS_ATRAK_IMG_MAX_BYTES FAYS_ATRAK_RGB_MAX_BYTES
#endif
#define FAYS_ATRAK_DEPTH_IMG_MAX_BYTES (FAYS_ATRAK_IMG_MAX_BYTES * sizeof(float))

/*********************** Atrak device defines *************************/
#define ATRAK_DEV_STEREO           0
#define ATRAK_DEV_MIDDLE           1
#define ATRAK_DEV_STEREO_RECT      2
#define ATRAK_DEV_DENSE_MAP        0
#define ATRAK_DEV_SPARSE_MAP       1

typedef unsigned char        uchar;
typedef unsigned short       ushort;
typedef unsigned int         uint;
typedef unsigned long        ulong;
typedef unsigned long long   ull;

/**
 * @brief 
 * 6DOF State. 
 * 4 bits: bit0 - idle(0) / running(1), bit1 - normal(0) / error(1), 
 *         bit2 & bit3 holds pose type:
 *             01 - Initializing (external pose only in fusion mode / no pose in vio-only mode)
 *             10 - Vio pose only (external pose has been missing for a while if in fusion mode)
 *             11 - Fusion pose
 */
enum ATRAK_STATE : uint {
    AS_STOP = 0,
    AS_ERROR = 3,
    AS_INITIALIZING = 5,
    AS_VIO_POSE_ONLY = 9,
    AS_FUSION_POSE = 13
};

enum ATRAK_ERROR_CODE : int {
    AEC_NO_ERROR = 0
};

enum ATRAK_TYPE : int {
    AT_VIO = 0,
    AT_GPOSE_FUSION = 1
};

enum ATRAK_IMG_ENCODING : short
{
    AIE_BGR8 = 0,
    AIE_RGB8,
    AIE_RAW8,
    AIE_JPEG
};

enum ATRAK_CAM_MODEL : uint8_t
{
    ACM_UNKNOWN = 0,
    ACM_PINHOLE,
};

enum ATRAK_DISTORTION_MODEL : uint8_t
{
    ADM_NONE = 0,         // No distortion.
    ADM_KB4,              // KANNALA_BRANDT k1 k2 k3 k4
    ADM_RADTAN,           // k1, k2, p1, p2
    ADM_BROWN_CONRADY,     // Brown-Conrady k1, k2, p1, p2, k3
    ADM_CVBASIC,          // OpenCV basic 8-parameter model k1, k2, p1, p2, k3, k4, k5, k6
};



/**
 * @brief AI Kit device info
 */
#define FAYS_ATRAK_MAX_INFO_BUFFER_SIZE         64

typedef struct AtrakDeviceInfo {
    char vendor_name[FAYS_ATRAK_MAX_INFO_BUFFER_SIZE];
    char product_type[FAYS_ATRAK_MAX_INFO_BUFFER_SIZE];
    char product_name[FAYS_ATRAK_MAX_INFO_BUFFER_SIZE];
    char product_serial_number[FAYS_ATRAK_MAX_INFO_BUFFER_SIZE];
    char user_defined_name[FAYS_ATRAK_MAX_INFO_BUFFER_SIZE];
    char device_ip_address[FAYS_ATRAK_MAX_INFO_BUFFER_SIZE];
    char device_version[FAYS_ATRAK_MAX_INFO_BUFFER_SIZE];
    char driver_version[FAYS_ATRAK_MAX_INFO_BUFFER_SIZE];

    uint32_t device_id;
    uint32_t reserved[15];
} AtrakDeviceInfo;


/**
 * @brief 6DOF pose
 * 
 */
typedef struct AtrakPose
{
    ull timestamp;     // unit: nanosencond
    double pos[3];     // unit: meter
    double quat[4];    // Hamilton convention quaternion: w, x, y, z
    short confidence;  // -1~100
} AtrakPose;

typedef struct AtrakImage
{
    int device_id;
    int seq;
    ull timestamp;    // unit: nanosencond
    int width;
    int height;
    int channel;
    short encoding;
    int step;
    uint bytes;       // num of data bytes
    uchar* data;      // pointer to the pre-allocated image buffer (BGR-packed if 3-channels)
} AtrakImage;

typedef struct AtrakIMU
{
    ull timestamp;    // unit: nanosencond
    double gyro[3];   
    double acc[3];    
} AtrakIMU;

typedef struct AtrakGNSS
{
    ull timestamp;
    int seq;
    uchar status;
    short service;
    double latitude;
    double longitude;
    double altitude;
    double position_cov[9];
    uchar position_cov_type;
} AtrakGNSS;

typedef struct AtrakExtraPose
{
    int frame;
    int seq;
    ull timestamp;
    uchar status;
    double pos[3];
    double quat[4];  // Hamilton convention quaternion: w, x, y, z
    double cov[36];
} AtrakExtraPose;

typedef struct AtrakMap
{
    int device_id;  // ATRAK_DEV_DENSE_MAP | ATRAK_DEV_SPARSE_MAP
    ull timestamp;
    uint point_num;
    float* pos;     // total length: 3 * point_num (x1, y1, z1, x2, y2, z2, ...)
    uint32_t* rgb;  // total length: 1 * point_num (08R8G8B8, ...)
} AtrakMap;


/****************************************** template tracking *******************************************/ 
typedef struct AtrakTempTrackConfig
{
    ull timestamp;                    // unit: nanosencond
    int seq;
    uint cls = 0;                     // For multi templates tracking (future)
    AtrakImage template_img;          // BGR 3-channel image support only
    float physical_width = -1.F;      // physical width of the template if available. uint: meter
    float physical_height = -1.F;     // physical height of the template if available. uint: meter
} AtrakTempTrackConfig;

typedef struct AtrakObject2D
{
    int cls;                          // class id. 0~2^24-1: (customized class), others: reserved
    int id;                           // instance id if available
    float confidence;                 // detection confidence 0.~1.
    float bbox[4];                    // bounding box in image pixel coordinates: center x, center y, width, height
    uint32_t extra_info_mask;         // extra info mask. 0: no extra info; 1: position available. Bit(L-->H): position, orientation, angle in image 
    float pose[7];                    // px, py, pz, qw, qx, qy, qz
    float angle;                      // rotation angle in image if available. uint: degree
} AtrakObject2D;
                                    
typedef struct AtrakObject2DDets
{
    ull timestamp;                    // unit: nanosencond
    int seq;
    uint num_objs;
    AtrakObject2D objs[FAYS_ATRAK_MAX_TRACK_INSTANCES];
} AtrakObject2DDets;


/****************************************** AprilTag tracking *******************************************/ 

enum ATRAK_TAG_FAMILY : uint {
    ATF_36H11 = 0,
    ATF_25H9,
    ATF_16H5,
    ATF_CIRCLE21H7,
    ATF_CIRCLE49H12,
    ATF_CUSTOM48H12,
    ATF_STANDARD41H12,
    ATF_STANDARD52H13
};

typedef struct AtrakAprilTagTrackConfig
{
    ull timestamp;                     // unit: nanosencond
    int seq;
    ATRAK_TAG_FAMILY tag_family;
    float tag_size = -1.F;             // only support multiple same-size tags for now. unit: meter
} AtrakAprilTagTrackConfig;

typedef struct AtrakAprilTag
{
    ATRAK_TAG_FAMILY tag_family;       // tag family
    uint id;                           // tag id
    float corners[8];                  // tag corners in image pixel coordinates [u, v, u, v, ...]. These always wrap counter-clock wise around the tag
    bool pose_available;               // tag pose is available or not
    float pos[3];                      // tag position if available. uint: meter
    float quat[4];                     // tag orientation if available. Hamilton convention quaternion: w, x, y, z
} AtrakAprilTag;

typedef struct AtrakAprilTagDets
{
    ull timestamp;                     // unit: nanosencond
    int seq;
    uint num_tags;
    AtrakAprilTag tags[FAYS_ATRAK_MAX_APRIL_INSTANCES];
} AtrakAprilTagDets;


/****************************************** Stereo depth *******************************************/ 

typedef struct AtrakDepthImage
{
    int device_id;
    int seq;
    ull timestamp;    // unit: nanosencond
    int width;
    int height;
    int channel;
    short encoding;
    int step;
    uint bytes;       // bytes of data array (sizeof(float) * length)
    float* data;      // pointer to the pre-allocated depth image buffer 
} AtrakDepthImage;

typedef struct AtrakRectifyInfo
{
    // camera pair id
    uint8_t cam_pair[2];
    // brings points in the unrectified first camera frame to points in the rectified first camera frame
    float R0[9];
    // brings points in the unrectified second camera frame to points in the rectified second camera frame
    float R1[9];
    // projects points in the rectified first camera coordinate system into the rectified first camera's image
    float P0[12];
    // projects points in the rectified first camera coordinate system into the rectified second camera's image
    float P1[12];
} AtrakRectifyInfo;


/****************************************** Calibration info *******************************************/ 

typedef struct AtrakIntrinsics {
    ATRAK_CAM_MODEL cam_model;
    uint32_t width;
    uint32_t height;
    float fx;
    float fy;
    float cx;
    float cy;
    float intrinsic_extra[4];          // support more camera models
    ATRAK_DISTORTION_MODEL distortion_model;
    float distortion[8];
} AtrakIntrinsics;


/**
 * @brief 6-DoF transformation (rotation/translation)
 */
typedef struct AtrakExtrinsics {
    float rot[9];                                       // row-major rotation matrix 
    float trans[3];                                     // translation vector
} AtrakExtrinsics;

/**
 * @brief Camera parameters
 */
typedef struct AtrakCamParam {
    uint8_t cam_id;
    uint8_t available_mask;            // indicator for available info (Bit L to H): intrinsics, T_cn_cnm1, T_cn_imu, timeshift_cam_imu
    AtrakIntrinsics intrinsics;
    AtrakExtrinsics T_cn_cnm1;         // Bring points in {camera n-1} frame to {camera n} frame
    AtrakExtrinsics T_cn_imu;          // Bring points in {imu} frame to {camera n} frame
    double timeshift_cam_imu;          // timeshift_cam_imu = t_imu - t_cam
} AtrakCamParam;

/**
 * @brief Camera chain parameters
 */
typedef struct AtrakCamChainParam {
    uint32_t num_of_cams;
    uint32_t downsize_ratio;
    AtrakCamParam cameras[FAYS_ATRAK_MAX_CAMERAS];
} AtrakCamChainParam;

/**
 * @brief IMU calib parameters
 */
typedef struct AtrakImuParam{
    double accelerometer_noise_density;  
    double accelerometer_random_walk;    
    double gyroscope_noise_density;      
    double gyroscope_random_walk;       
    float update_rate;             
} AtrakImuParam;

/**
 * @brief Cams and IMU calib parameters
 */
typedef struct AtrakCalibrationParam{
    AtrakCamChainParam cameras;
    AtrakImuParam imu;
} AtrakCalibrationParam;

/**
 * @brief Device Info
 */
typedef struct ViKitDeviceInfo
{
    char firmware_version[16];
    char device_model[16];            
    char serial_number[32];         
    uint32_t camera_nums;
    uint32_t imu_nums;           
    uint32_t reserved[32];
} ViKitDeviceInfo;