#!/usr/bin/env bash
# vLLM 服务启动脚本 —— Qwen3-VL-8B-Instruct-FP8,OpenAI 兼容 API @ 127.0.0.1:8001
# (端口可用 EGODATA_VLLM_PORT 覆盖;deploy.py 避让端口占用时会设置它)
#
# 踩坑记录(2026-08-14 实测,勿删):
#  - VLLM_USE_DEEP_GEMM=0:RTX 5090(SM120)缺 DeepGEMM scale-factor 布局支持,
#    否则加载 FP8 权重报 "Unknown SF transformation" 断言
#  - VLLM_USE_FLASHINFER_SAMPLER=0:跳过 flashinfer 的 JIT CUDA 编译
#    (依赖 ninja + 项目路径带空格会破坏 ninja 依赖解析)
#  - 必须 cd 到无空格目录启动(项目目录名 "Data Acquisition" 含空格)
#  - venv bin 必须进 PATH(ninja 等工具在子进程里按 PATH 查找)
#  - 本地目录加载时 model 名 = 完整路径(客户端请求要用路径名)
#  - 视频输入用 base64 data URL 发送,不用 file:// 本地路径
#  - 应用默认用 MP4 + media_io_kwargs.video.fps；如需兼容旧版可设置
#    EGODATA_VLLM_MEDIA_MODE=frames 回退到客户端抽帧

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
VENV="$PROJECT_ROOT/.venv-llm"
MODEL="${EGODATA_VLLM_MODEL:-$PROJECT_ROOT/models/llm/Qwen3-VL-8B-Instruct-FP8}"

if [[ ! -x "$VENV/bin/vllm" ]]; then
  echo "vLLM 环境不存在:$VENV(先安装 vllm)" >&2
  exit 1
fi
if [[ ! -f "$MODEL/config.json" ]]; then
  echo "模型不存在:$MODEL" >&2
  exit 1
fi

# Leave GPU headroom for the hand-processing Worker to load its detector.
cd /tmp
exec env PATH="$VENV/bin:$PATH" \
  VLLM_USE_DEEP_GEMM=0 \
  VLLM_USE_FLASHINFER_SAMPLER=0 \
  "$VENV/bin/vllm" serve "$MODEL" \
  --host 127.0.0.1 --port "${EGODATA_VLLM_PORT:-8001}" \
  --gpu-memory-utilization 0.70 \
  --max-model-len 16384
