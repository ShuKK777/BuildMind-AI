#!/bin/bash
cd "$(dirname "$0")"
if command -v node >/dev/null 2>&1; then
  node server.mjs
elif command -v python3 >/dev/null 2>&1; then
  open http://127.0.0.1:4173
  python3 -m http.server 4173 --bind 127.0.0.1
else
  echo 'Please install Node.js 18+ once before going offline. 请在联网时安装 Node.js 18+。'
  read -r -p 'Press Enter to close...'
fi
