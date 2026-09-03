cmake_minimum_required(VERSION 3.10)
project(fayssense_depth_examples CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# ===========================================================================
# 方式A: read_depth —— CPU 深度算法库 (纯 OpenCV, 用系统自带的 OpenCV 4.x)
# ===========================================================================
find_package(OpenCV 4 REQUIRED COMPONENTS core imgproc calib3d imgcodecs)

add_executable(read_depth
    examples/read_depth.cpp
    lib/cpu_depth.cpp
    lib/colormap.cpp)
target_include_directories(read_depth PRIVATE lib)
target_link_libraries(read_depth PRIVATE ${OpenCV_LIBS})

# ===========================================================================
# 方式B: engine_depth —— FaysSense SDK 深度引擎
#   依赖: sdk/lib/<arch>/ 下两个 .so + 随包 OpenCV 4.2 (x86_64)
#   aarch64 (RK3588) 上请把 SDK_LIB_DIR / OPENCV42_DIR 改到设备上的对应路径,
#   或直接用设备上 SDK 自带的 OpenCV。
# ===========================================================================
if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64")
    set(SDK_LIB_DIR   ${CMAKE_CURRENT_SOURCE_DIR}/sdk/lib/aarch64)
    # 设备端 OpenCV 路径请按实际 rootfs 修改:
    set(OPENCV42_DIR  ${CMAKE_CURRENT_SOURCE_DIR}/sdk/opencv/x86_64)  # 占位, aarch64 需自行提供
else()
    set(SDK_LIB_DIR   ${CMAKE_CURRENT_SOURCE_DIR}/sdk/lib/x86_64)
    set(OPENCV42_DIR  ${CMAKE_CURRENT_SOURCE_DIR}/sdk/opencv/x86_64)
endif()

add_executable(engine_depth
    examples/engine_depth.cpp
    lib/depth_engine.cpp
    lib/colormap.cpp)

target_include_directories(engine_depth PRIVATE
    lib
    ${CMAKE_CURRENT_SOURCE_DIR}/sdk/include
    ${OPENCV42_DIR}/include/opencv4)

target_link_directories(engine_depth PRIVATE
    ${SDK_LIB_DIR}
    ${OPENCV42_DIR}/lib)

# 注意: 不能写裸库名 opencv_core 等 —— find_package(OpenCV) 生成的 imported target
# 会劫持这些名字, 导致链接到系统 OpenCV 而不是随包 4.2。必须传完整文件路径
# (与 lib406 垫片相同的处理方式)。
set(SHIM_DIR ${OPENCV42_DIR}/lib/shim)
set(OPENCV42_LIB_DIR ${OPENCV42_DIR}/lib)
target_link_libraries(engine_depth PRIVATE
    fayssense_aikit_depth
    fays_vikit
    ${OPENCV42_LIB_DIR}/libopencv_core.so
    ${OPENCV42_LIB_DIR}/libopencv_imgproc.so
    ${OPENCV42_LIB_DIR}/libopencv_imgcodecs.so
    ${OPENCV42_LIB_DIR}/libopencv_calib3d.so
    ${OPENCV42_LIB_DIR}/libopencv_ximgproc.so
    ${OPENCV42_LIB_DIR}/libopencv_video.so
    ${OPENCV42_LIB_DIR}/libopencv_features2d.so
    ${OPENCV42_LIB_DIR}/libopencv_flann.so
    ${SHIM_DIR}/libtbb.so.2
    ${SHIM_DIR}/libtiff.so.5
    ${SHIM_DIR}/libwebp.so.6
    ${SHIM_DIR}/libdc1394.so.22
    ${SHIM_DIR}/libavcodec.so.58
    ${SHIM_DIR}/libavformat.so.58
    ${SHIM_DIR}/libavutil.so.56
    ${SHIM_DIR}/libswscale.so.5)

# 运行时自动找到同包内的 .so ($ORIGIN 相对 rpath, 免设 LD_LIBRARY_PATH)
set_target_properties(engine_depth PROPERTIES
    BUILD_RPATH   "$ORIGIN/../sdk/lib/x86_64:$ORIGIN/../sdk/opencv/x86_64/lib:$ORIGIN/../sdk/opencv/x86_64/lib/shim"
    INSTALL_RPATH "$ORIGIN/../sdk/lib/x86_64:$ORIGIN/../sdk/opencv/x86_64/lib:$ORIGIN/../sdk/opencv/x86_64/lib/shim"
    BUILD_WITH_INSTALL_RPATH FALSE)

# --disable-new-dtags: 生成可继承的 DT_RPATH 而不是 DT_RUNPATH。
# 否则随包 opencv 的间接依赖 (libtbb.so.2 等垫片) 不会被搜索到。
set_property(TARGET engine_depth APPEND PROPERTY LINK_FLAGS "-Wl,--disable-new-dtags")
