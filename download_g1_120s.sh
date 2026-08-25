#!/usr/bin/env bash

set -Eeuo pipefail

readonly REPO_ID="agibot-world/GenieSimAssets"
readonly REMOTE_DIR="robot/G1_120s"
readonly REVISION="${HF_REVISION:-main}"
readonly OUTPUT_DIR="${1:-./GenieSimAssets}"

if ! command -v hf >/dev/null 2>&1; then
    cat >&2 <<'EOF'
错误：未找到 Hugging Face CLI（hf）。
请先安装：
  python3 -m pip install -U huggingface_hub
EOF
    exit 1
fi

mkdir -p "$OUTPUT_DIR"

echo "正在下载 ${REPO_ID}/${REMOTE_DIR} ..."
hf download "$REPO_ID" \
    --repo-type dataset \
    --revision "$REVISION" \
    --include "${REMOTE_DIR}/**" \
    --local-dir "$OUTPUT_DIR"

echo "下载完成：${OUTPUT_DIR%/}/${REMOTE_DIR}"
