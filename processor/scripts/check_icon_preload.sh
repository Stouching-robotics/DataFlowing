#!/bin/bash
# 校验:代码中引用的 iconify 图标是否全部在预加载文件里。
# 缺失的图标会在运行时请求 api.iconify.design → 页面图标闪烁/空白。
# 用法: scripts/check_icon_preload.sh ;退出码 0 = 齐全,1 = 有缺失。
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PRE="$ROOT/web/static/iconify-preload.js"
[ -f "$PRE" ] || { echo "找不到 $PRE"; exit 1; }

# 代码里所有 "前缀:图标名" 引用(ant-design:xxx 等)
USED="$(grep -rhoE "[a-z0-9-]+:[a-z0-9-]+" \
  "$ROOT/web/workflow-studio/src" "$ROOT/web/static/js" "$ROOT/web/templates" 2>/dev/null \
  | sort -u)"

MISSING=""
while IFS= read -r ref; do
  # 只检查 ant-design 前缀(预加载文件目前只包含 ant-design)
  case "$ref" in
    ant-design:*) ;;
    *) continue ;;
  esac
  name="${ref#ant-design:}"
  grep -q "'${name}': { body" "$PRE" || MISSING="$MISSING"$'\n'"  $ref"
done <<< "$USED"

if [ -n "$MISSING" ]; then
  echo "以下图标未预加载(页面会出现闪烁/空白),请补进 $PRE :"
  echo "$MISSING"
  exit 1
fi
echo "全部引用图标均已预加载 ✓"
