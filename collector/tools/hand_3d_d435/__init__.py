"""hand_3d_d435 —— D435 RGB-D 单目 3D 手部关键点独立模块。

用 D435 录制的 RGB（1280×720）+ 原生深度（848×480 mm PNG）离线做：
MediaPipe 2D 检测（+ 手性投票）→ 深度抬升 3D（彩色相机系）→ 槽位跟踪
（遮挡传播）→ 离线平滑 → 旋转渲染 + RGB 叠显 + parquet。

只 import 复用 tools/stereo_s80m/hand_3d/ 组件（detector/identity/track3d/
postprocess/renderer_3d/video_writer/io），不改任何现有文件；
输出全部落在 keypoints_output/ 下。
"""
