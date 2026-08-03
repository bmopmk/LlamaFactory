#!/bin/bash
# ============================================================
# Qwen3-4B LoRA 链式训练: pt -> sft -> dpo
# 每一步通过 adapter_name_or_path 接住上一步产物
# ============================================================
set -euo pipefail   # 任一步失败/未定义变量/管道错误立即终止

# 切到脚本所在目录(即项目根),保证相对路径正确
cd "$(dirname "$0")"

echo "==================== [1/3] 预训练 (pt) ===================="
llamafactory-cli train examples/chain/qwen3_pt.yaml

echo "==================== [2/3] 监督微调 (sft) ===================="
llamafactory-cli train examples/chain/qwen3_sft.yaml

echo "==================== [3/3] 偏好对齐 (dpo) ===================="
llamafactory-cli train examples/chain/qwen3_dpo.yaml

echo "✅ pt → sft → dpo 全部完成，最终产物在 saves/qwen3-4b/lora/dpo"
