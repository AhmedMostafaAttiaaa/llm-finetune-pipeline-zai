#!/bin/bash
# ================================================================
# LLM Fine-Tuning Pipeline - Quick Start Script
# ================================================================
# This script helps you get started with the pipeline quickly.
#
# Prerequisites:
#   - Python 3.10+
#   - CUDA-capable GPU with 30GB VRAM
#   - pip install -r requirements.txt
#
# Usage:
#   chmod +x quickstart.sh
#   ./quickstart.sh --data your_dataset.csv
# ================================================================

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}================================================${NC}"
echo -e "${BLUE}  LLM Fine-Tuning Pipeline - Quick Start${NC}"
echo -e "${BLUE}================================================${NC}"

# Parse arguments
DATA_FILE=""
MODEL="qwen2.5-coder-7b"
STAGE="all"
TEST_RUN=false

while [[ "$#" -gt 0 ]]; do
    case $1 in
        --data) DATA_FILE="$2"; shift ;;
        --model) MODEL="$2"; shift ;;
        --stage) STAGE="$2"; shift ;;
        --test) TEST_RUN=true ;;
        *) echo "Unknown parameter: $1"; exit 1 ;;
    esac
    shift
done

# Check prerequisites
echo -e "\n${YELLOW}Checking prerequisites...${NC}"

# Check Python
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}Python 3 not found. Please install Python 3.10+${NC}"
    exit 1
fi
echo -e "${GREEN}✓ Python 3 found${NC}"

# Check CUDA
if python3 -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
    GPU_NAME=$(python3 -c "import torch; print(torch.cuda.get_device_name(0))")
    GPU_VRAM=$(python3 -c "import torch; print(f'{torch.cuda.get_device_properties(0).total_mem / 1024**3:.1f} GB')")
    echo -e "${GREEN}✓ GPU: $GPU_NAME ($GPU_VRAM)${NC}"
else
    echo -e "${RED}✗ CUDA not available. GPU required for training.${NC}"
    exit 1
fi

# Check data file
if [ -z "$DATA_FILE" ]; then
    echo -e "${YELLOW}No data file specified. Use --data your_dataset.csv${NC}"
    echo -e "\nAvailable models:"
    python3 -c "
from pathlib import Path
import yaml
for f in sorted(Path('configs/models').glob('*.yaml')):
    config = yaml.safe_load(open(f))
    m = config['model']
    v = config['vram']['qlora_4bit_training']
    print(f\"  {m['short_name']:<30} {m['size_billions']}B  VRAM: {v}\")
"
    exit 0
fi

if [ ! -f "$DATA_FILE" ]; then
    echo -e "${RED}Data file not found: $DATA_FILE${NC}"
    exit 1
fi
echo -e "${GREEN}✓ Data file: $DATA_FILE${NC}"

# Run pipeline
echo -e "\n${BLUE}Starting pipeline...${NC}"
echo -e "  Model: $MODEL"
echo -e "  Data:  $DATA_FILE"
echo -e "  Stage: $STAGE"

CMD="python3 run_pipeline.py --model $MODEL --data $DATA_FILE --stage $STAGE"

if [ "$TEST_RUN" = true ]; then
    CMD="$CMD --test-run"
fi

echo -e "\n${GREEN}Running: $CMD${NC}\n"
eval $CMD

echo -e "\n${GREEN}================================================${NC}"
echo -e "${GREEN}  Pipeline Complete!${NC}"
echo -e "${GREEN}================================================${NC}"
