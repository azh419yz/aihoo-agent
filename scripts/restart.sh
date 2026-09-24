#!/usr/bin/env bash
# ============================================================
# AI 中医男科 Agent 重启脚本（ECS / 本地通用）
#
# 用法：
#   bash scripts/restart.sh        # 停旧进程 → 启动 → 健康检查
#   bash scripts/restart.sh stop   # 只停止，不启动
#
# 说明：
#   - 启动命令与手工一致：nohup uv run uvicorn app.main:app \
#       --host 0.0.0.0 --port 8000 >> logs/uvicorn-YYYY-MM-DD.log 2>&1 &
#   - 日志按天（2026-09-23 起）：
#       · 业务日志   logs/agent.log（当天）+ logs/agent-YYYY-MM-DD.log（历史，留 30 天，由 main.py 切分）
#       · 控制台输出 logs/uvicorn-YYYY-MM-DD.log（同一天内多次重启追加，留 30 天）
#   - CWD = 项目根目录（config.py 从 CWD 读 .env，main.py 基于根目录写 logs/）
#   - 进程 PID 记录在 logs/agent.pid（logs/ 已 gitignore，不污染代码）
#   - 先按 PID 文件杀，再按命令行 pkill 兜底（防 PID 文件丢失/进程残留）
# ============================================================

set -uo pipefail

# 定位项目根目录（脚本位于 scripts/ 下）
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PORT="${PORT:-8000}"
PID_FILE="${ROOT}/logs/agent.pid"
LOG_DIR="${ROOT}/logs"
UVICORN_LOG_RETAIN_DAYS="${UVICORN_LOG_RETAIN_DAYS:-30}"
UVICORN_PATTERN="uvicorn app.main:app"
# LOG_FILE 在 start_new 里按当天日期计算（形如 logs/uvicorn-2026-09-23.log）
LOG_FILE=""

stop_old() {
  echo "==> 停止旧进程"
  # 1) 按 PID 文件
  if [ -f "$PID_FILE" ]; then
    OLD_PID="$(cat "$PID_FILE" 2>/dev/null || true)"
    if [ -n "${OLD_PID:-}" ] && kill -0 "$OLD_PID" 2>/dev/null; then
      kill "$OLD_PID" 2>/dev/null && echo "    kill pid=$OLD_PID"
    fi
    rm -f "$PID_FILE"
  fi
  # 2) 命令行匹配兜底
  if pgrep -f "$UVICORN_PATTERN" >/dev/null 2>&1; then
    pkill -f "$UVICORN_PATTERN" 2>/dev/null || true
    echo "    pkill $UVICORN_PATTERN"
  fi
  # 3) 等待进程退出
  for _ in $(seq 1 20); do
    if ! pgrep -f "$UVICORN_PATTERN" >/dev/null 2>&1; then
      echo "    旧进程已退出"
      return 0
    fi
    sleep 0.5
  done
  echo "    ⚠ 等待超时，仍存在进程：$(pgrep -f "$UVICORN_PATTERN" | tr '\n' ' ')"
}

start_new() {
  echo "==> 启动新进程（port=${PORT}）"

  # uvicorn 的 stdout/stderr 按天落盘：logs/uvicorn-YYYY-MM-DD.log，同一天内多次重启**追加**到同一文件。
  # 保留这份重定向的独立价值：能捕获 app 日志系统初始化之前（import 期）的崩溃栈。
  # 2026-09-23 起改按天：旧做法是每次重启把 agent.log 重命名归档、留 5 份，
  # 在「频繁重启」的测试节奏下会堆出大量带时间戳的碎片文件，且文件名不含日期、不便按日期回溯
  #（2002 事故复盘时 09-18 访问日志正是这样丢的，见 docs/2026-09-20-IKGUP0-Agent侧2002证据.md）。
  mkdir -p "$LOG_DIR"
  LOG_FILE="${LOG_DIR}/uvicorn-$(date +%F).log"

  # 清理超期的按天归档（默认 30 天）
  OLD_LOGS="$(find "$LOG_DIR" -maxdepth 1 -type f -name 'uvicorn-*.log' \
    -mtime +"$UVICORN_LOG_RETAIN_DAYS" 2>/dev/null || true)"
  if [ -n "$OLD_LOGS" ]; then
    echo "$OLD_LOGS" | xargs rm -f
    echo "    已清理 ${UVICORN_LOG_RETAIN_DAYS} 天前的 uvicorn 日志"
  fi

  echo "    日志: $LOG_FILE"
  nohup uv run uvicorn app.main:app --host 0.0.0.0 --port "$PORT" \
    >> "$LOG_FILE" 2>&1 &
  NEW_PID=$!
  echo "$NEW_PID" > "$PID_FILE"
  echo "    已启动 pid=${NEW_PID}（PID 记录: ${PID_FILE}）"
}

health_check() {
  echo "==> 等待服务就绪"
  for i in $(seq 1 30); do
    sleep 1
    if curl -fsS "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then
      echo "    ✅ 服务已就绪（${i}s）"
      curl -s "http://127.0.0.1:$PORT/health"
      echo
      return 0
    fi
  done
  echo "    ⚠ 健康检查超时（30s），请查看 $LOG_FILE"
  tail -20 "$LOG_FILE"
  return 1
}

stop_old

if [ "${1:-}" = "stop" ]; then
  echo "==> 已停止（stop 模式），不启动"
  exit 0
fi

start_new
health_check
