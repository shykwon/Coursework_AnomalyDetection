#!/bin/bash
# ============================================================
# Single Dataset Experiment Script
# ============================================================
#
# 단일 데이터셋 실험 실행 (백그라운드)
#
# 사용법:
#   # PSM 데이터셋, GPU 0에서 DLinear 실행
#   nohup bash scripts/run_single_experiment.sh PSM DLinear 0 &
#
#   # SWaT 데이터셋, 전체 모델, GPU 1에서 실행
#   nohup bash scripts/run_single_experiment.sh SWaT all 1 &
#
# 인자:
#   $1: 데이터셋 (PSM, SWaT)
#   $2: 모델 (DLinear, OmniAnomaly, all)
#   $3: GPU ID (0, 1, 2, ...)
#
# ============================================================

set -e

# 인자 확인
if [[ $# -lt 3 ]]; then
    echo "Usage: $0 <dataset> <model> <gpu_id>"
    echo "  dataset: PSM, SWaT"
    echo "  model: DLinear, OmniAnomaly, all"
    echo "  gpu_id: 0, 1, 2, ..."
    exit 1
fi

DATASET=$1
MODEL=$2
GPU_ID=$3

# 프로젝트 설정
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

LOG_DIR="$PROJECT_ROOT/logs"
mkdir -p "$LOG_DIR"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# 로그 함수
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1"
}

log "============================================================"
log "🚀 Single Experiment: $DATASET + $MODEL (GPU $GPU_ID)"
log "============================================================"

# Conda 활성화
source /opt/anaconda3/etc/profile.d/conda.sh 2>/dev/null || source ~/anaconda3/etc/profile.d/conda.sh 2>/dev/null || source ~/miniconda3/etc/profile.d/conda.sh 2>/dev/null || true
conda activate timeseries 2>/dev/null || true

# 전처리/후처리 설정
PREPROCESS="P_MM P_STD P_SM P_DT"
POSTPROCESS="T1 T2 T3 T4 T5"

# Step 1: 학습
log "────────────────────────────────────────────────────────"
log "📊 Step 1: Training"
log "────────────────────────────────────────────────────────"

if [[ "$MODEL" == "all" ]]; then
    MODELS_ARG="DLinear OmniAnomaly"
else
    MODELS_ARG="$MODEL"
fi

python scripts/run_step1.py \
    --dataset "$DATASET" \
    --models $MODELS_ARG \
    --preprocess $PREPROCESS \
    --gpu "$GPU_ID" \
    --log_dir "$LOG_DIR"

log "✅ Step 1 완료"

# Step 2: 평가
log "────────────────────────────────────────────────────────"
log "📊 Step 2: Evaluation"
log "────────────────────────────────────────────────────────"

python scripts/run_step2.py \
    --dataset "$DATASET" \
    --models $MODELS_ARG \
    --postprocess $POSTPROCESS \
    --log_dir "$LOG_DIR"

log "✅ Step 2 완료"

log "============================================================"
log "🎉 Experiment Complete: $DATASET + $MODEL"
log "============================================================"
