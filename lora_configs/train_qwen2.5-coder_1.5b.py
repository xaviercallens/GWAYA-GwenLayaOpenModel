# The config-only scaffold was replaced by scripts/train_lora.py (QLoRA SFT/DPO, HF ids, resume, GCS sync).
# Continuity-anchor recipe: python scripts/train_lora.py --recipe qwen2.5-coder-1.5b-sft --gpu t4 \
#     --train-file sft.jsonl --out-dir RUN --print-plan
import runpy
import sys
from pathlib import Path

sys.argv[0] = str(Path(__file__).resolve().parents[1] / "scripts" / "train_lora.py")
runpy.run_path(sys.argv[0], run_name="__main__")
