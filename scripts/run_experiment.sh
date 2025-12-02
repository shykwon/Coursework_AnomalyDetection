#!/bin/bash
# -*- coding: utf-8 -*-
# ============================================================
# Master Experiment Script: Multi-GPU Parallel Execution
# ============================================================
#
# 실행 방법:
#   nohup bash scripts/run_experiment.sh > logs/master.log 2>&1 &
#
# GPU 분배 (3개 GPU 병렬):
#   GPU 0: DLinear + PSM
#   GPU 1: OmniAnomaly + PSM
#   GPU 2: DLinear + SWaT & OmniAnomaly + SWaT (순차)
#
# ============================================================

set -e  # 에러 발생시 중단

# 프로젝트 루트 설정
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

# 로그 디렉토리 생성
LOG_DIR="$PROJECT_ROOT/logs"
mkdir -p "$LOG_DIR"

# 타임스탬프
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
MASTER_LOG="$LOG_DIR/${TIMESTAMP}_master.log"

# 로그 함수
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$MASTER_LOG"
}

log "============================================================"
log "🚀 Master Experiment Script Started"
log "============================================================"
log "Project Root: $PROJECT_ROOT"
log "Log Directory: $LOG_DIR"
log "Master Log: $MASTER_LOG"
log "============================================================"

# ============================================================
# 설정
# ============================================================
PREPROCESS=("P_MM" "P_STD" "P_SM" "P_DT")
POSTPROCESS=("T1" "T2" "T3" "T4" "T5")

# Conda 환경 활성화
log "Activating conda environment: timeseries"
source /opt/anaconda3/etc/profile.d/conda.sh 2>/dev/null || source ~/anaconda3/etc/profile.d/conda.sh 2>/dev/null || source ~/miniconda3/etc/profile.d/conda.sh 2>/dev/null || true
conda activate timeseries || {
    log "⚠️ Could not activate conda environment. Using current environment."
}

# ============================================================
# Step 1: 4개 GPU 병렬 학습
# ============================================================
log ""
log "============================================================"
log "📊 Step 1: 전체 모델 병렬 학습 (4 GPUs)"
log "============================================================"

pids=()

# GPU 0: DLinear + PSM
log "  [GPU 0] DLinear + PSM"
nohup python scripts/run_step1.py \
    --dataset PSM \
    --models DLinear \
    --preprocess ${PREPROCESS[@]} \
    --gpu 0 \
    --log_dir "$LOG_DIR" \
    >> "$LOG_DIR/${TIMESTAMP}_PSM_DLinear_step1.log" 2>&1 &
pids+=($!)

# GPU 1: OmniAnomaly + PSM
log "  [GPU 1] OmniAnomaly + PSM"
nohup python scripts/run_step1.py \
    --dataset PSM \
    --models OmniAnomaly \
    --preprocess ${PREPROCESS[@]} \
    --gpu 1 \
    --log_dir "$LOG_DIR" \
    >> "$LOG_DIR/${TIMESTAMP}_PSM_OmniAnomaly_step1.log" 2>&1 &
pids+=($!)

# GPU 2: SWaT (DLinear + OmniAnomaly 순차)
log "  [GPU 2] DLinear + OmniAnomaly + SWaT (순차)"
nohup python scripts/run_step1.py \
    --dataset SWaT \
    --models DLinear OmniAnomaly \
    --preprocess ${PREPROCESS[@]} \
    --gpu 2 \
    --log_dir "$LOG_DIR" \
    >> "$LOG_DIR/${TIMESTAMP}_SWaT_all_step1.log" 2>&1 &
pids+=($!)

log ""
log "  Waiting for all Step 1 jobs (PIDs: ${pids[*]})"

# 모든 작업 완료 대기
for pid in "${pids[@]}"; do
    wait $pid
    exit_code=$?
    if [[ $exit_code -eq 0 ]]; then
        log "  ✅ Job $pid completed successfully"
    else
        log "  ❌ Job $pid failed with exit code $exit_code"
    fi
done

log ""
log "  Step 1 완료: 모든 모델 학습 완료"

# ============================================================
# Step 2: 평가 (PSM, SWaT 순차)
# ============================================================
log ""
log "============================================================"
log "📊 Step 2: 평가"
log "============================================================"

for dataset in "PSM" "SWaT"; do
    log "────────────────────────────────────────────────────────"
    log "  Step 2: $dataset"
    log "────────────────────────────────────────────────────────"

    python scripts/run_step2.py \
        --dataset "$dataset" \
        --postprocess ${POSTPROCESS[@]} \
        --log_dir "$LOG_DIR" \
        2>&1 | tee -a "$LOG_DIR/${TIMESTAMP}_${dataset}_step2.log"

    log "  Step 2 완료: $dataset"
done

# 종료 시간 및 소요 시간 계산
END_TIME=$(date +%s)
START_TIME=${START_TIME:-$END_TIME}
ELAPSED=$((END_TIME - START_TIME))
HOURS=$((ELAPSED / 3600))
MINUTES=$(((ELAPSED % 3600) / 60))
SECONDS=$((ELAPSED % 60))

log ""
log "============================================================"
log "✅ All Experiments Completed!"
log "============================================================"
log "Total Time: ${HOURS}h ${MINUTES}m ${SECONDS}s"
log "Results: $PROJECT_ROOT/outputs/results/"
log "Logs: $LOG_DIR/"
log "============================================================"

# 최종 결과 요약
log ""
log "📊 Final Results Summary:"
log "────────────────────────────────────────────────────────"

if [[ -f "$PROJECT_ROOT/outputs/results/evaluation_results.csv" ]]; then
    log "Top 5 by PA F1:"
    python -c "
import pandas as pd
df = pd.read_csv('$PROJECT_ROOT/outputs/results/evaluation_results.csv')
top5 = df.nlargest(5, 'pa_f1')[['model', 'preprocess', 'postprocess', 'dataset', 'pa_f1', 'point_f1']]
print(top5.to_string(index=False))
" 2>/dev/null | tee -a "$MASTER_LOG"
fi

log ""
log "🎉 Experiment complete. Check logs for details."
