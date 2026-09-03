#!/bin/bash
# 归档同步:把本地会话增量推送到 NAS(/vol2/egodata/sessions),只推不删。
# 用法:
#   archive_sync.sh                 # 全量:data/sessions/ → NAS
#   archive_sync.sh <批次目录>       # 单批次:推到 <项目>/<批次>/(处理完成钩子调用)
# 传输安全:rsync 逐块校验 + --partial 断点续传;推完 -rcn 二次校验;
# 失败自动重试一次;结果写 data/logs/archive.log。并发互斥(flock)。
set -u

NAS_USER="${EGODATA_NAS_USER:-}"
NAS_HOST="${EGODATA_NAS_HOST:-}"
NAS_ROOT="${EGODATA_NAS_ROOT:-/srv/egodata/sessions}"
SRC_ROOT="${EGODATA_SRC_ROOT:-data/sessions}"
LOG_DIR="${EGODATA_LOG_DIR:-data/logs}"
LOG_FILE="$LOG_DIR/archive.log"
SSH_OPTS="-o BatchMode=yes -o ConnectTimeout=10 -o ServerAliveInterval=15"

mkdir -p "$LOG_DIR"

log() { echo "[$(date '+%F %T')] $*" >> "$LOG_FILE"; }

do_sync() {  # $1=src(尾斜杠) $2=dst
  rsync -rltD --partial --info=stats2 --no-owner --no-group \
      -e "ssh $SSH_OPTS" "$1" "$2" >> "$LOG_FILE" 2>&1
}

verify_sync() {  # 输出非空 = 有差异(-rcn 逐文件 checksum dry-run)
  rsync -rcn --no-owner --no-group -e "ssh $SSH_OPTS" "$1" "$2" 2>&1 \
      | grep -v "setlocale" | grep -v '^$'
}

push_once() {
  local src="$1" dst="$2" t0 out
  t0=$(date +%s)
  if do_sync "$src" "$dst"; then
    out=$(verify_sync "$src" "$dst")
    if [ -z "$out" ]; then
      log "OK   $dst ($(( $(date +%s) - t0 ))s)"
      return 0
    fi
    log "校验不一致,重试: $dst"
  fi
  sleep 10
  if do_sync "$src" "$dst"; then
    out=$(verify_sync "$src" "$dst")
    if [ -z "$out" ]; then
      log "OK(重试) $dst ($(( $(date +%s) - t0 ))s)"
      return 0
    fi
  fi
  log "FAIL $dst"
  return 1
}

# 并发互斥:处理完成钩子与每晚 cron 可能同时触发
exec 9>"$LOG_DIR/archive.lock"
flock -n 9 || { log "SKIP 上一归档未结束"; exit 0; }

if [ $# -ge 1 ] && [ -n "$1" ]; then
  batch=$(realpath "$1")
  project=$(basename "$(dirname "$batch")")
  name=$(basename "$batch")
  push_once "$batch/" "$NAS_USER@$NAS_HOST:$NAS_ROOT/$project/$name/"
else
  push_once "$SRC_ROOT/" "$NAS_USER@$NAS_HOST:$NAS_ROOT/"
fi
