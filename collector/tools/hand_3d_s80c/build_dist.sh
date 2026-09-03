#!/usr/bin/env bash
# 组装自包含分发包 dist/s80c_hands_demo_v1.0/。
# 从**当前工作区**复制（含未提交修改）；代码/依赖有改动后重跑本脚本即可。
#   ./tools/hand_3d_s80c/build_dist.sh
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
VER="1.0"
DIST="$REPO_ROOT/dist/s80c_hands_demo_v$VER"

rm -rf "$DIST"
mkdir -p "$DIST"/{tools/hand_3d_s80c,tools/hand_3d_d435,tools/glove_package,\
tools/stereo_s80m/config,tools/stereo_s80m/hand_3d,tools/hand_detection,\
tools/models,tools/fayssense_depth_sdk/calib}

copy() { cp -a "$REPO_ROOT/$1" "$DIST/$2"; }

# ── 模块本体（含 SDK/OpenCV 自包含 third_party ~52MB）──
copy tools/hand_3d_s80c/live_demo_s80c.py     tools/hand_3d_s80c/live_demo_s80c.py
copy tools/hand_3d_s80c/s80c_depth_worker.py  tools/hand_3d_s80c/s80c_depth_worker.py
copy tools/hand_3d_s80c/browse_tear_dump.py   tools/hand_3d_s80c/browse_tear_dump.py
copy tools/hand_3d_s80c/README.md             tools/hand_3d_s80c/README.md
copy tools/hand_3d_s80c/run_live_s80c.sh      tools/hand_3d_s80c/run_live_s80c.sh
cp -a "$REPO_ROOT/tools/hand_3d_s80c/third_party" "$DIST/tools/hand_3d_s80c/"

# ── 复用链（hand_3d_d435 的 _run_3d_chain 及其 9 个子模块）──
for f in __init__.py live_demo.py depth_align.py lift3d.py mono_assign.py \
         render_overlay.py replay_compat.py glove_detector.py pose_backends.py; do
    copy "tools/hand_3d_d435/$f" "tools/hand_3d_d435/$f"
done

# ── 黑手套模式（glove_package 3 文件 + YOLO-World 权重 ~57MB）──
copy tools/glove_package/hand_tracker.py      tools/glove_package/hand_tracker.py
copy tools/glove_package/world_detector.py    tools/glove_package/world_detector.py
copy tools/glove_package/yolov8m-worldv2.pt   tools/glove_package/yolov8m-worldv2.pt

# ── stereo_s80m 组件（三角化 + 相机配置 + hand_3d 7 个模块）──
# dist 布局镜像**当前仓库布局**（tools/ 下），路径解析零改动
copy tools/stereo_s80m/stereo_triangulate.py        tools/stereo_s80m/stereo_triangulate.py
copy tools/stereo_s80m/config/fays_vikit_50fps.yaml tools/stereo_s80m/config/fays_vikit_50fps.yaml
copy tools/stereo_s80m/config/fays_vikit.yaml      tools/stereo_s80m/config/fays_vikit.yaml
for f in detector.py identity.py track3d.py smoother.py renderer_3d.py \
         video_writer.py io.py; do
    copy "tools/stereo_s80m/hand_3d/$f" "tools/stereo_s80m/hand_3d/$f"
done
# 仓库版 __init__.py 会 import run_pipeline（离线管线，分发包不含）——
# 唯一对仓库文件的裁剪，S80C 实时链不引用该管线。
cat > "$DIST/tools/stereo_s80m/hand_3d/__init__.py" <<'EOF'
"""stereo_s80m.hand_3d —— 分发包裁剪版。

仅含 S80C 实时链需要的组件（detector/identity/track3d/smoother/
renderer_3d/video_writer/io）；仓库版的 run_pipeline 离线管线 import
已剥离（分发包不含该文件）。其余行为与仓库版一致。
"""
EOF

# ── 手检测 / 模型权重 / 标定回退 ──
copy tools/hand_detection/hand_pipeline_mediapipe.py tools/hand_detection/hand_pipeline_mediapipe.py
copy tools/models/hand_landmarker.task               tools/models/hand_landmarker.task
copy tools/fayssense_depth_sdk/calib/calib.yaml      tools/fayssense_depth_sdk/calib/calib.yaml

# ── 使用说明 + requirements + 分发包启动器 ──
copy tools/hand_3d_s80c/dist/使用说明.md         使用说明.md
copy tools/hand_3d_s80c/dist/requirements.txt    requirements.txt
copy tools/hand_3d_s80c/dist/run_s80c_hands_demo.sh run_s80c_hands_demo.sh
chmod +x "$DIST/run_s80c_hands_demo.sh" "$DIST/tools/hand_3d_s80c/run_live_s80c.sh"

# ── 冒烟：分发包内关键模块可编译 ──
"${PY:-python3}" -m py_compile \
    "$DIST/tools/hand_3d_s80c/live_demo_s80c.py" \
    "$DIST/tools/hand_3d_s80c/s80c_depth_worker.py" \
    "$DIST/tools/hand_3d_s80c/browse_tear_dump.py" \
    "$DIST/tools/hand_3d_d435/live_demo.py" \
    "$DIST/tools/stereo_s80m/hand_3d/renderer_3d.py"

cd "$REPO_ROOT/dist"
rm -f "s80c_hands_demo_v$VER.tar.gz"
tar -czf "s80c_hands_demo_v$VER.tar.gz" "s80c_hands_demo_v$VER"
echo "✓ dist/s80c_hands_demo_v$VER/（$(du -sh "$DIST" | cut -f1)）+ s80c_hands_demo_v$VER.tar.gz"
