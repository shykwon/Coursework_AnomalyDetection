# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Time Series Anomaly Detection (TSAD) project comparing **Prediction-based (DLinear)** and **Reconstruction-based (OmniAnomaly)** approaches on PSM/SWaT datasets.

## Development Commands

```bash
# Activate environment
conda activate timeseries

# Install dependencies
pip install -r requirements.txt

# Run Jupyter notebooks
jupyter notebook notebooks/
```

## Architecture

### Pipeline Flow
```
DataLoader → Preprocessor → Model → Postprocessor → Evaluator
                ↓              ↓           ↓
            (Scaler,      (DLinear,   (Threshold,
             Smoother,    OmniAnomaly) Relabeling)
             Detrender)
```

### Key Modules (`src/`)
- **models/cores/**: Open-source model implementations (DLinear, OmniAnomaly) with compatibility fixes
- **models/**: Wrapper classes implementing `BaseModel` interface
- **preprocessing/**: `BasePreprocessor` implementations (Scaler, Smoother, Detrender)
- **postprocessing/**: Thresholding (Fixed, POT, EWMA) and Point Adjustment
- **evaluation/**: Metrics (Point-wise F1, PA F1) and visualization
- **pipeline/**: `TSADPipeline` orchestrating end-to-end flow

### Model Integration Pattern
Models in `src/models/cores/` are adapted from open-source repositories with library compatibility fixes only. Wrapper classes in `src/models/` provide the `BaseModel` interface (fit, predict, get_anomaly_score).

## Development Principles

1. **Open-source based**: DLinear and OmniAnomaly use verified open-source code from `references/`
2. **Analyze first**: Study original code before implementing wrappers
3. **Functional equivalence**: Core algorithm logic must remain 100% identical to original
4. **Learning-focused**: Mark key learning points with `🎓` or `TODO(human)` for user implementation

## TODO(human) 학습 패턴

사용자가 직접 구현해볼 핵심 알고리즘에는 아래 패턴을 사용:

```python
# ============================================================
# TODO(human): [구현할 내용 설명]
# ============================================================
# [개념 설명]
# - 포인트 1
# - 포인트 2
#
# Hint: [사용할 함수/API 힌트]
# 반환값: [반환값 구조 설명]

variable1 = None  # TODO(human): 여기에 구현
variable2 = None  # TODO(human): 여기에 구현

# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# 📖 정답 (막히면 아래 주석을 해제하세요)
# ────────────────────────────────────────────────────────────
# result = some_function(args)
# variable1 = result[0]
# variable2 = result[1]
# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
```

**핵심 규칙:**
- `>>>>` / `<<<<` 로 정답 블록 경계 표시
- `📖 정답` 으로 명확하게 라벨링
- 주석 해제하면 바로 실행 가능하도록 들여쓰기 유지
- `tests/` 에 검증 스크립트 제공 (예: `python tests/test_xxx.py`)

## Code Style

- Object-oriented design with abstract base classes
- PEP 8 compliant
- Type hints required
- Update checklist in `docs/design_specification.md` after completing tasks

## Key Datasets
- **PSM**: Located in `data/raw/PSM/`
- **SWaT**: Located in `data/raw/SWaT/`

## Reference Repositories
- DLinear: https://github.com/cure-lab/LTSF-Linear
- OmniAnomaly: https://github.com/NetManAIOps/OmniAnomaly
