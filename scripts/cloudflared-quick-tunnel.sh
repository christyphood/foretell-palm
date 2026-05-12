#!/usr/bin/env bash
# 把本机掌纹 API（默认 5050）暴露为公网 HTTPS，供 GitHub Pages 通过 ?api= 访问。
#
# 前置：
#   1) 安装 cloudflared：brew install cloudflare/cloudflare/cloudflared
#   2) 另开终端先启动后端，例如：
#        cd Palm-Astro-Application && python3 api_server.py
#
# 用法：
#   ./scripts/cloudflared-quick-tunnel.sh
#   或：CLOUDFLARE_TUNNEL_PORT=8080 ./scripts/cloudflared-quick-tunnel.sh
#
# 终端里会出现 https://xxxx.trycloudflare.com ，把它填进：
#   https://<你的用户名>.github.io/foretell-palm/?api=https://xxxx.trycloudflare.com
#（?api= 后面不要加 /analyze）
#
# 说明：Quick Tunnel 每次启动域名会变；关电脑或停进程后外网即不可访问。
#
# 若出现 ERR failed to dial a quic connection / timeout：
#   多为公司网、防火墙或运营商屏蔽 UDP(QUIC)。本脚本默认用 TCP 的 http2。
#   若仍失败，可换手机热点试，或改用 ngrok / localtunnel 等。

set -euo pipefail
PORT="${CLOUDFLARE_TUNNEL_PORT:-5050}"
PROTO="${CLOUDFLARE_TUNNEL_PROTOCOL:-http2}"
URL="http://127.0.0.1:${PORT}"

if ! command -v cloudflared >/dev/null 2>&1; then
  echo "未找到 cloudflared。请先安装："
  echo "  brew install cloudflare/cloudflare/cloudflared"
  exit 1
fi

echo "正在把 ${URL} 映射到公网 HTTPS（协议: ${PROTO}，按 Ctrl+C 停止）…"
exec cloudflared tunnel --url "${URL}" --protocol "${PROTO}"
