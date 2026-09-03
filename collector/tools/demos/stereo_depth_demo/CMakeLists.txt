cmake_minimum_required(VERSION 3.12)
project(stereo_depth_demo)

# 导出 compile_commands.json (供 VSCode IntelliSense 解析 include 路径,
# 否则头文件在编辑器里报"无法打开", 编译器本身不受影响)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# SDK 路径 (与官方 GUI CMakeLists 相同的 SDK_INCLUDE_DIR/SDK_LIB_DIR 机制)
if(NOT DEFINED SDK_DIR)
    set(SDK_DIR "/home/REDACTED/FaysSense_VI_Kit_Release")
endif()
set(SDK_INCLUDE_DIR "${SDK_DIR}/include" CACHE PATH "FaysSense SDK headers")
set(SDK_LIB_DIR "${SDK_DIR}/lib/fays_atrak/${CMAKE_SYSTEM_PROCESSOR}/Release" CACHE PATH "FaysSense SDK libs")
message(STATUS "SDK_INCLUDE_DIR: ${SDK_INCLUDE_DIR}")
message(STATUS "SDK_LIB_DIR:     ${SDK_LIB_DIR}")

# OpenCV: 必须使用 SDK 自带 4.2 (深度引擎按其编译, 系统 4.6/4.13 会崩).
# 在 build.sh 中以 -DOpenCV_DIR 指向 SDK 自带 4.2 的 cmake 配置.
find_package(OpenCV REQUIRED)
message(STATUS "OpenCV ${OpenCV_VERSION}: ${OpenCV_DIR}")

find_package(Threads REQUIRED)

include_directories(${SDK_INCLUDE_DIR})
link_directories(${SDK_LIB_DIR})

add_executable(stereo_depth_demo main.cpp)

# 链接顺序至关重要 (GNU ld 按从左到右解析未定义符号):
#   1. SDK 库在前 (libfayssense_aikit_depth.so 的 cv::ximgproc 等未定义符号)
#   2. 之后是 SDK 自带 OpenCV 4.2 模块 (仅链接深度引擎实际用到的模块)
#   3. 最后是 lib406 外部依赖 shim (4.2 的 TIFF/WebP/TBB/av* 等符号)
set(SDK_CV406_DEPS "${SDK_DIR}/thirdparty/opencv-4.2.0-linux-x86_64/lib406")
target_link_libraries(stereo_depth_demo
    fayssense_aikit_depth
    fays_vikit
    opencv_core
    opencv_imgproc
    opencv_imgcodecs
    opencv_highgui
    opencv_calib3d
    opencv_ximgproc
    opencv_features2d
    opencv_flann
    opencv_video
    opencv_videoio
    opencv_photo
    opencv_objdetect
    opencv_ml
    ${SDK_CV406_DEPS}/libtbb.so.2
    ${SDK_CV406_DEPS}/libtiff.so.5
    ${SDK_CV406_DEPS}/libwebp.so.6
    ${SDK_CV406_DEPS}/libdc1394.so.22
    ${SDK_CV406_DEPS}/libavcodec.so.58
    ${SDK_CV406_DEPS}/libavformat.so.58
    ${SDK_CV406_DEPS}/libavutil.so.56
    ${SDK_CV406_DEPS}/libswscale.so.5
    Threads::Threads
)
