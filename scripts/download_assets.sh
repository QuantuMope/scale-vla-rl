#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

if [[ ! -f "$TRANSFORMERS_CACHE/bridge_beta.pt" ]]; then
  bridge_checkpoint=bridge_beta_step19296_2024-12-26_22-30_42.pt
  hf download allenzren/open-pi-zero "$bridge_checkpoint" --local-dir "$TRANSFORMERS_CACHE"
  mv "$TRANSFORMERS_CACHE/$bridge_checkpoint" "$TRANSFORMERS_CACHE/bridge_beta.pt"
fi

hf download google/paligemma-3b-pt-224 \
  config.json tokenizer.json tokenizer.model tokenizer_config.json \
  special_tokens_map.json added_tokens.json \
  --local-dir "$TRANSFORMERS_CACHE/paligemma-3b-pt-224"

uv run --locked python -m mani_skill.utils.download_asset widowx250s -y

hf download quantumope/scale-vla-rl-scenes --repo-type dataset \
  --revision afdbd71c20b2abfbc1055daac5599c462f648309 \
  --include "${1:-task_*/*}" --exclude '*.mp4' \
  --local-dir "$EMBODIED_GEN_SCENES"
