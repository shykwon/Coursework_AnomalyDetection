#!/bin/bash
# ============================================================
# Master Experiment Script - TEST VERSION (빠른 검증용)
# ============================================================
# epoch=2로 축소하여 스크립트 동작만 검증
# ============================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

LOG_DIR="$PROJECT_ROOT/logs"
mkdir -p "$LOG_DIR"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
MASTER_LOG="$LOG_DIR/${TIMESTAMP}_master_test.log"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$MASTER_LOG"
}

log "============================================================"
log "🧪 Master Experiment Script - TEST MODE"
log "============================================================"
log "Project Root: $PROJECT_ROOT"
log "Master Log: $MASTER_LOG"
log "============================================================"

# 테스트 설정 (최소 조합만)
DATASETS=("PSM")
MODELS=("DLinear" "OmniAnomaly")
PREPROCESS=("P_MM" "P_STD")
POSTPROCESS=("T1" "T2" "T3")
EPOCHS=2  # 빠른 테스트

# Conda 활성화
log "Activating conda environment..."
source /opt/anaconda3/etc/profile.d/conda.sh 2>/dev/null || source ~/anaconda3/etc/profile.d/conda.sh 2>/dev/null || true
conda activate timeseries || log "⚠️ Could not activate conda environment"

# Step 1: GPU 병렬 학습
run_step1_parallel() {
    local dataset=$1
    log "────────────────────────────────────────────────────────"
    log "📊 Step 1 시작: $dataset (TEST MODE - epochs=$EPOCHS)"
    log "────────────────────────────────────────────────────────"

    local pids=()

    # DLinear → GPU 0
    log "  [GPU 0] DLinear + $dataset"
    python scripts/run_step1.py \
        --dataset "$dataset" \
        --models DLinear \
        --preprocess ${PREPROCESS[@]} \
        --epochs $EPOCHS \
        --gpu 0 \
        --log_dir "$LOG_DIR" \
        >> "$LOG_DIR/${TIMESTAMP}_${dataset}_DLinear_test.log" 2>&1 &
    pids+=($!)

    # OmniAnomaly → GPU 1
    log "  [GPU 1] OmniAnomaly + $dataset"
    python scripts/run_step1.py \
        --dataset "$dataset" \
        --models OmniAnomaly \
        --preprocess ${PREPROCESS[@]} \
        --epochs $EPOCHS \
        --gpu 1 \
        --log_dir "$LOG_DIR" \
        >> "$LOG_DIR/${TIMESTAMP}_${dataset}_OmniAnomaly_test.log" 2>&1 &
    pids+=($!)

    # 완료 대기
    log "  Waiting for Step 1 jobs (PIDs: ${pids[*]})"
    for pid in "${pids[@]}"; do
        wait $pid
        local exit_code=$?
        if [[ $exit_code -eq 0 ]]; then
            log "  ✅ Job $pid completed"
        else
            log "  ❌ Job $pid failed (exit: $exit_code)"
        fi
    done

    log "  Step 1 완료: $dataset"
}

# Step 2: 평가
run_step2() {
    local dataset=$1
    log "────────────────────────────────────────────────────────"
    log "📊 Step 2 시작: $dataset"
    log "────────────────────────────────────────────────────────"

    python scripts/run_step2.py \
        --dataset "$dataset" \
        --postprocess ${POSTPROCESS[@]} \
        --log_dir "$LOG_DIR" \
        2>&1 | tee -a "$LOG_DIR/${TIMESTAMP}_${dataset}_step2_test.log"

    log "  Step 2 완료: $dataset"
}

# 메인 실행
START_TIME=$(date +%s)

for dataset in "${DATASETS[@]}"; do
    log ""
    log "============================================================"
    log "🔬 Dataset: $dataset"
    log "============================================================"

    run_step1_parallel "$dataset"
    run_step2 "$dataset"
done

END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))

log ""
log "============================================================"
log "✅ TEST Completed!"
log "============================================================"
log "Total Time: ${ELAPSED}s"
log "============================================================"

# 결과 요약
log ""
log "📊 Test Results:"
if [[ -f "$PROJECT_ROOT/outputs/results/evaluation_results.csv" ]]; then
    python -c "
import pandas as pd
df = pd.read_csv('$PROJECT_ROOT/outputs/results/evaluation_results.csv')
print(df[['model', 'preprocess', 'postprocess', 'dataset', 'pa_f1', 'point_f1']].to_string(index=False))
" 2>/dev/null | tee -a "$MASTER_LOG"
fi

log ""
log "🎉 Test complete!"
