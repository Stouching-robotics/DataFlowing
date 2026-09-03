# FaysSense 双目鱼眼深度 SDK — 外部程序集成包

本包从 **FaysSense VI Kit**（FS-VI80-S80C 双目鱼眼相机，VI Kit v3.9.1）工程中抽取深度估计能力，
供**外部程序**读取深度图处理结果。提供两种方式：

| 方式 | 说明 | 适用场景 |
|------|------|----------|
| **方式A：CPU 深度算法库** | 纯 OpenCV 实现的 5 种双目鱼眼深度算法（C++ 源码），不依赖 SDK 引擎 | 跨平台、可定制参数、x86 开发机直接可用 |
| **方式B：SDK 深度引擎** | 调用官方 `libfayssense_aikit_depth.so` 深度引擎 | 需要官方引擎行为（含 NPU 加速模型，RK3588） |

两种方式都输出**统一的深度图格式**（float32 米制 raw + 伪彩 PNG + 点云 PLY），外部程序
（C++ / Python / ROS / 点云库）读取方式一致。

---

## 1. 目录结构

```
fayssense_depth_sdk/
├── README.md                本文件
├── build.sh                 一键构建脚本
├── CMakeLists.txt           构建配置（两个示例）
├── lib/                     方式A 的 CPU 算法库源码（纯 OpenCV，无 SDK 依赖）
│   ├── cpu_depth.h / .cpp   五种算法变体 + 标定解析 + 点云生成
│   ├── colormap.h / .cpp    深度伪彩图（RealSense 同款色调）
│   └── depth_engine.h/.cpp  SDK 引擎的 C++ 封装（方式B 使用）
├── examples/
│   ├── read_depth.cpp       方式A 示例程序
│   ├── engine_depth.cpp     方式B 示例程序
│   └── read_depth.py        深度图读取脚本（numpy，可选 cv2）
├── sdk/                     SDK 官方二进制与头文件
│   ├── include/fays_atrak/  深度引擎头文件
│   ├── lib/x86_64/          libfays_vikit.so + libfayssense_aikit_depth.so（x86）
│   ├── lib/aarch64/         同上（RK3588 / ARM64）
│   └── opencv/x86_64/       随包 OpenCV 4.2 子集（方式B 在 x86 上链接用，含 lib406 垫片）
├── config/stereo_depth.yaml 方式B 引擎配置（模型/标定路径、CPU/NPU 模式等）
├── calib/calib.yaml         双目鱼眼标定（kalibr 格式：pinhole + equidistant/KB4）
├── models/rk3588/           NPU 深度模型（仅 RK3588 上使用）
└── data/sample_stacked.bmp  示例双目帧（1280×1600 上下堆叠，上半左/下半右）
```

---

## 2. 快速开始

### 2.1 构建

```bash
tar xzf fayssense_depth_sdk.tar.gz
cd fayssense_depth_sdk
./build.sh              # cmake + make，生成 build/read_depth 和 build/engine_depth
```

要求：C++17 编译器、CMake ≥ 3.10、**系统 OpenCV 4.x**（方式A 需要；
`sudo apt install libopencv-dev` 或 `pip`/conda 版本均可，见 FAQ）。

### 2.2 运行方式A（CPU 算法库）

```bash
./build.sh run-cpu
# 等价于:
./build/read_depth data/sample_stacked.bmp calib/calib.yaml 3 out/v3
```

参数：`<双目堆叠图> <标定yaml> <变体 0~4> <输出前缀>`

变体（五种双目鱼眼深度技术，可分别测试效果）：

| 变体 | 算法 | 特点 |
|------|------|------|
| 0 | V1 针孔校正 + SGBM | 经典方案（与 SDK 引擎相同的校正路径） |
| 1 | V2v 垂直极 ERP + SGBM | 球面校正，行近似极线，视场保留好 |
| 2 | V2b 基线极 ERP + SGBM | 球面校正，行严格极线 |
| 3 | V3 多针孔分带 + SGBM + ERP 融合 | 视场覆盖最大，低纹理场景效果最好（推荐首选测试） |
| 4 | V4 针孔 + StereoBM | 最快（约 15 ms/帧） |

### 2.3 运行方式B（SDK 引擎）

```bash
./build.sh run-engine
# 等价于:
./build/engine_depth data/sample_stacked.bmp config/stereo_depth.yaml out/engine
```

引擎参数通过 `config/stereo_depth.yaml` 控制（`depth_mode: 1` = CPU SGBM 后端；
`depth_mode: 0` = NPU 模型后端，仅 RK3588 可用）。yaml 中的 model_path / calib_path
为**相对 yaml 所在目录**的路径（引擎自行解析），因此整个包放到任意目录、从任意
工作目录运行都可以。

### 2.4 输出文件

每次运行生成（前缀自定义）：

| 文件 | 内容 |
|------|------|
| `<前缀>_depth.raw` | **float32 深度图**，行优先 H×W，单位米，0 = 无效像素 |
| `<前缀>_depth.png` | 伪彩深度图（预览用） |
| `<前缀>_rect.png`  | 校正后的双目图（方式A 为算法内部校正图，方式B 为引擎校正图） |
| `<前缀>_pc.ply`    | 点云（ASCII PLY，方式A 为校正坐标系，方式B 为引擎点云坐标系） |

示例帧（方式A V3）预期输出：有效像素约 57%，中位深度约 0.6 m（视场景而定）。
**注意**：V3 输出深度图尺寸为 ERP 网格（示例帧为 672×288），其余变体与引擎输出为
单目输入尺寸 1280×800；实际尺寸以程序打印的 `[out] ... (WxH ...)` 为准。

---

## 3. 深度图数据格式（外部程序读取接口）

`_depth.raw` 是**无头裸数据**，格式固定：

```
float32 小端数组，行优先（第 y 行第 x 列 = data[y*W + x]）
单位: 米
无效像素: 0.0f
尺寸: 与输入单目分辨率一致（默认 1280×800；restore_depth_dim=1 时引擎输出也为输入尺寸）
```

### 3.1 Python 读取

```python
import numpy as np
depth = np.fromfile("out/v3_depth.raw", dtype=np.float32).reshape(288, 672)  # 尺寸见 [out] 打印
print("valid:", (depth > 0).mean() * 100, "%  median:", np.median(depth[depth > 0]))
```

本包自带完整脚本：

```bash
python3 examples/read_depth.py out/v3_depth.raw 672 288 depth_view.png   # V3 为 672x288
python3 examples/read_depth.py out/engine_depth.raw 1280 800 view.png    # 引擎为 1280x800
# 或连同点云:  python3 examples/read_depth.py out/v3_depth.raw 672 288 view.png out_pc.ply
```

### 3.2 C++ 读取

```cpp
#include <opencv2/core.hpp>
#include <fstream>
cv::Mat loadDepthRaw(const std::string& path, int w, int h) {
    cv::Mat depth(h, w, CV_32FC1);
    std::ifstream f(path, std::ios::binary);
    f.read((char*)depth.data, (size_t)w * h * sizeof(float));
    return depth;   // 米, 0 = 无效
}
```

### 3.3 直接集成算法（不落盘）

**方式A** —— 把 `lib/cpu_depth.cpp`、`lib/colormap.cpp` 编进你的工程：

```cpp
#include "cpu_depth.h"
cpu_depth::CpuDepthProcessor proc;
proc.loadCalib("calib/calib.yaml");
proc.setFrameSize(1280, 800);            // 单目分辨率
cpu_depth::CpuOptions o; o.variant = cpu_depth::V3_BANDS_SGBM;
proc.setOptions(o);
cpu_depth::CpuResult res;
proc.process(stackedBgr, nullptr, nullptr, 0.0, &res);
// res.depth: CV_32FC1 深度(米); res.pcXYZ/pcRGB: 点云; res.rectStereo: 校正图
```

**方式B** —— 把 `lib/depth_engine.cpp` 编进你的工程，链接 SDK 的 .so：

```cpp
#include "depth_engine.h"
DepthEngine eng("config/stereo_depth.yaml");
eng.feedStereoImage(stacked, tsNs);
cv::Mat depth, disp, paired;
eng.getDepthImage(depth, disp, paired);  // 异步模式需先连续喂几帧预热
```

---

## 4. 方式B：SDK 引擎说明

### 4.1 两种后端

| 后端 | yaml 配置 | 说明 |
|------|-----------|------|
| SGBM（CPU） | `depth_mode: 1` | x86 / RK3588 均可运行；SGBM 参数可运行时调节 |
| NPU 模型 | `depth_mode: 0` | 加载 `models/rk3588/stereo_s_general.rknn`，**仅 RK3588（NPU）可用**，x86 上会初始化失败 |

运行时调参（示例中已使用与 GUI 默认一致的参数）：

```cpp
const ATRAK_DepthParamKv kvs[] = {
    makeKv(ATRAK_DP_SGBM_MIN_DISP, 0),  makeKv(ATRAK_DP_SGBM_NUM_DISP, 128),
    makeKv(ATRAK_DP_SGBM_WINDOW_SIZE, 11), /* ... 见 examples/engine_depth.cpp */
};
eng.setDepthParam(ATRAK_DEPTH_BACKEND_SGBM, kvs, 11);
```

### 4.2 关键 yaml 字段（config/stereo_depth.yaml）

| 字段 | 含义 |
|------|------|
| `depth_mode` | 0 = NPU 模型，1 = CPU SGBM |
| `async_mode` | 1 = 喂帧立即返回（**需预热**：连续喂 5 帧以上，getDepthImage 才有数据） |
| `restore_depth_dim` | 1 = 深度图恢复到输入分辨率（1280×800） |
| `align_cam_id` | -1 = 不对齐；0/1/2 = 对齐到左/右/RGB 相机 |
| `enable_filter` / `wls_filter` | 时序滤波 / WLS 视差滤波开关（CPU 后端） |
| `farthest_dist` / `cloest_dist` | 点云提取深度范围（米） |

### 4.3 依赖库

- `libfayssense_aikit_depth.so` —— 深度引擎本体
- `libfays_vikit.so` —— 相机/标定接口（引擎会调用其标定读取函数）
- OpenCV 4.2 子集 —— x86 上随包提供（`sdk/opencv/x86_64/`），构建时用 `$ORIGIN` 相对
  rpath 自动定位，**无需设置 LD_LIBRARY_PATH**（随包 opencv 的 RUNPATH 与 lib406
  垫片的 SONAME 已在打包时改写为相对路径，整个包可任意搬移）；aarch64 设备上使用
  设备系统里 SDK 自带的 OpenCV（修改 CMakeLists.txt 中 `OPENCV42_DIR`）。

### 4.4 连续帧用法（接入相机流）

```cpp
while (true) {
    cv::Mat stacked = grabFromViKit();          // 或用 libfays_vikit 取流
    eng.feedStereoImage(stacked, tsNs);
    cv::Mat depth, disp, paired;
    if (eng.getDepthImage(depth, disp, paired) && !depth.empty()) {
        // 处理最新深度图
    }
}
```

x86 上 CPU SGBM 后端单帧约 0.7 s；RK3588 上 NPU 后端可实时。

---

## 5. 性能参考（x86 开发机，1280×800 输入）

| 方式 | 变体 | 耗时（示例帧实测） | 有效像素 |
|------|------|------|----------|
| A | V1 针孔+SGBM | 70 ms | 23% |
| A | V2v ERP+SGBM | 111 ms | 24% |
| A | V2b ERP+SGBM | 112 ms | 24% |
| A | V3 多针孔分带融合 | 361 ms（672×288 输出） | 57% |
| A | V4 针孔+BM | 46 ms | 6% |
| B | 引擎 SGBM（CPU） | ~700 ms | 25%（中位 0.4 m） |
| B | 引擎 NPU（RK3588） | 实时 | — |

（数值随场景与机器不同而变化，仅作量级参考。）

---

## 6. FAQ

**Q: 运行时提示 `libopencv_video.so.4.2: cannot open shared object file`？**
A: 本包示例已内置 `$ORIGIN` rpath，正常从 `build/` 运行不会出现。若自行链接
SDK 引擎，请把 `sdk/opencv/x86_64/lib` 和 `sdk/opencv/x86_64/lib/shim` 加入
`LD_LIBRARY_PATH`。

**Q: `getDepthImage` 一直失败 / 深度图是空的？**
A: 引擎 `async_mode: 1`，需连续喂 ≥5 帧预热管线（示例已处理，自带 50×100ms 重试）。

**Q: `depth_mode: 0`（NPU）在 x86 上初始化失败？**
A: 正常现象。NPU 模型仅支持 RK3588 板端。x86 上请用 `depth_mode: 1`（CPU SGBM）。

**Q: 深度图尺寸是多少？**
A: 程序运行时会打印 `[out] xxx_depth.raw (WxH float32, meters, ...)`，以打印为准。
默认 1280×800（与单目输入一致）。

**Q: 深度图与哪个相机对齐？**
A: 默认左相机（cam0）坐标系；`align_cam_id` 可改为右相机或 RGB 相机。

**Q: 系统只有 OpenCV 3.x？**
A: 方式A 需要 OpenCV 4.x（`find_package(OpenCV 4 REQUIRED)`）。可以安装
`libopencv-dev`（Ubuntu 20.04+ 为 4.2/4.5）或 conda `opencv`；方式B 不受影响
（用随包 OpenCV 4.2）。

**Q: 在 RK3588 设备上如何使用？**
A: `sdk/lib/aarch64/` 内已含 ARM64 版 .so。设备镜像里通常已带 OpenCV；
把 CMakeLists.txt 中 `OPENCV42_DIR` 指向设备上 SDK 自带的 OpenCV 路径后按同样步骤
构建，`depth_mode: 0` 走 NPU 实时推理。

---

## 7. 版本与来源

- 来源工程：FaysSense VI Kit Release v3.9.1（stereo_depth_gui 深度算法库）
- 相机型号：FS-VI80-S80C 双目鱼眼（1280×800×2，基线约 80.5 mm）
- 标定模型：kalibr pinhole + equidistant（KB4）
- 示例数据：`data/sample_stacked.bmp`（一次真实采集的堆叠帧）
