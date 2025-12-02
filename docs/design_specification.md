# 시계열 이상 탐지(TSAD) 프로젝트 상세 설계서

## 📋 문서 정보
| 항목 | 내용 |
|------|------|
| 버전 | 1.2 |
| 작성일 | 2025-11-29 |
| 수정일 | 2025-12-01 |
| 프로젝트명 | Time Series Anomaly Detection |

---

## 1. 프로젝트 개요

### 1.1 목적
PSM/SWaT 데이터셋을 활용하여 **Prediction-based (DLinear)**와 **Reconstruction-based (OmniAnomaly)** 기법을 비교 분석하는 End-to-End 이상 탐지 파이프라인 구현

### 1.2 핵심 요구사항
- 다변량(Multivariate) 시계열 데이터 처리
- 두 가지 방법론 비교 분석
- 전처리/후처리 실험을 통한 성능 개선
- Point-wise F1 및 PA F1-score 기반 평가

### 1.3 개발 원칙 ⚠️ 중요

#### 1.3.1 오픈소스 모델 활용 전략
| 원칙 | 설명 |
|------|------|
| **오픈소스 기반** | DLinear, OmniAnomaly는 검증된 오픈소스 코드를 기반으로 함 |
| **Framework 호환성** | 원본 코드를 우리 Framework의 `BaseModel` 인터페이스에 맞게 래핑 |
| **라이브러리 호환성** | deprecated/old 라이브러리는 현재 가상환경(timeseries)에 맞게 수정 |
| **기능 동일성** | 핵심 알고리즘 로직은 원본과 100% 동일하게 유지 |

#### 1.3.2 코드 분석 우선 접근법
```
[권장 개발 순서]
1. 오픈소스 코드 분석 (DLinear, OmniAnomaly 원본 구조 파악)
2. 공통 인터페이스 도출 (두 모델의 입출력 패턴 분석)
3. Framework 상세화 (BaseModel 인터페이스 확정)
4. Wrapper 클래스 구현 (원본 코드를 Framework에 통합)
```

#### 1.3.3 학습 중심 개발 🎓
| 구분 | 내용 |
|------|------|
| **목적** | 프로젝트를 통한 시계열 이상 탐지 기법 학습 |
| **방식** | 핵심 로직 구현 시 사용자 참여 유도 |
| **대상** | 설계 결정, 알고리즘 핵심부, 평가 로직 등 |

### 1.4 역할 분담
| 담당 | 작업 내용 |
|------|----------|
| **사용자** | 데이터셋 다운로드, 핵심 로직 구현 참여, 설계 결정 |
| **Assistant** | 코드 분석, Framework 설계, 보조 코드 구현, 코드 리뷰 |

---

## 2. 시스템 아키텍처

### 2.1 전체 디렉토리 구조
```
Anomaly-Detection/
├── docs/                           # 문서
│   ├── design_specification.md     # 상세 설계서 (본 문서)
│   └── experiment_results.md       # 실험 결과 기록
│
├── references/                     # 🆕 오픈소스 원본 코드 (사용자 제공)
│   ├── DLinear/                    # DLinear 원본 코드
│   │   └── (사용자가 오픈소스에서 복사)
│   └── OmniAnomaly/                # OmniAnomaly 원본 코드
│       └── (사용자가 오픈소스에서 복사)
│
├── src/                            # 소스 코드
│   ├── __init__.py
│   │
│   ├── data/                       # 데이터 관련 모듈
│   │   ├── __init__.py
│   │   ├── loader.py               # 데이터 로더 클래스
│   │   └── analyzer.py             # EDA 및 통계 분석 클래스
│   │
│   ├── preprocessing/              # 전처리 모듈
│   │   ├── __init__.py
│   │   ├── base.py                 # 전처리 기본 클래스
│   │   ├── scaler.py               # 정규화 클래스 (MinMax, Standard)
│   │   ├── smoother.py             # Smoothing 클래스 (EWMA)
│   │   └── detrend.py              # Detrending 클래스
│   │
│   ├── models/                     # 모델 모듈
│   │   ├── __init__.py
│   │   ├── base.py                 # 모델 기본 추상 클래스
│   │   ├── cores/                  # 🆕 오픈소스 기반 핵심 구현체
│   │   │   ├── __init__.py
│   │   │   ├── dlinear_core.py     # DLinear 핵심 (원본 기반, 호환성 수정)
│   │   │   └── omnianomaly_core.py # OmniAnomaly 핵심 (원본 기반, 호환성 수정)
│   │   ├── dlinear.py              # DLinear Wrapper (BaseModel 구현)
│   │   └── omnianomaly.py          # OmniAnomaly Wrapper (BaseModel 구현)
│   │
│   ├── postprocessing/             # 후처리 모듈
│   │   ├── __init__.py
│   │   ├── threshold.py            # Thresholding 클래스 (Fixed, Adaptive, POT)
│   │   └── relabeling.py           # Point Adjustment 클래스
│   │
│   ├── evaluation/                 # 평가 모듈
│   │   ├── __init__.py
│   │   ├── metrics.py              # F1-score, PA F1-score 등
│   │   └── visualizer.py           # 시각화 클래스
│   │
│   ├── pipeline/                   # 파이프라인 모듈
│   │   ├── __init__.py
│   │   └── tsad_pipeline.py        # End-to-End 파이프라인
│   │
│   └── utils/                      # 유틸리티
│       ├── __init__.py
│       └── config.py               # 설정 관리
│
├── notebooks/                      # Jupyter 노트북
│   └── 01_data_analysis.ipynb      # EDA 및 데이터 분석
│
├── scripts/                        # 🆕 실험 실행 스크립트
│   ├── config.py                   # 하이퍼파라미터 설정 (레퍼런스 기반)
│   ├── run_step1.py                # Step 1: 전처리 + 학습 + Score 저장
│   └── run_step2.py                # Step 2: 후처리 + 평가
│
├── data/                           # 데이터 디렉토리 (사용자가 다운로드)
│   ├── raw/                        # 원본 데이터
│   │   ├── PSM/
│   │   └── SWaT/
│   └── processed/                  # 전처리된 데이터
│
├── outputs/                        # 출력물
│   ├── models/                     # 학습된 모델
│   ├── figures/                    # 시각화 결과
│   └── logs/                       # 실험 로그
│
├── project_guide.md                # 프로젝트 가이드
└── requirements.txt                # 의존성 패키지
```

### 2.2 모듈 의존성 다이어그램
```
┌─────────────────────────────────────────────────────────────────┐
│                      TSADPipeline                               │
│  (파이프라인 오케스트레이션)                                       │
└─────────────────────────────────────────────────────────────────┘
         │              │              │              │
         ▼              ▼              ▼              ▼
┌──────────────┐ ┌────────────┐ ┌────────────┐ ┌────────────┐
│  DataLoader  │ │Preprocessor│ │   Model    │ │Postprocessor│
│  Analyzer    │ │  (Scaler,  │ │ (DLinear,  │ │(Threshold,  │
│              │ │ Smoother,  │ │OmniAnomaly)│ │ Relabeling) │
│              │ │ Detrend)   │ │            │ │             │
└──────────────┘ └────────────┘ └────────────┘ └────────────┘
         │              │              │              │
         └──────────────┴──────────────┴──────────────┘
                                 │
                                 ▼
                    ┌────────────────────────┐
                    │  Evaluator & Visualizer │
                    │  (Metrics, Plots)       │
                    └────────────────────────┘
```

---

## 3. 클래스 상세 설계

### 3.1 데이터 모듈 (`src/data/`)

#### 3.1.1 DataLoader 클래스
```python
class DataLoader:
    """PSM/SWaT 데이터셋 로딩 및 관리 클래스"""

    def __init__(self, dataset_name: str, data_path: str):
        """
        Args:
            dataset_name: 'PSM' 또는 'SWaT'
            data_path: 데이터 디렉토리 경로
        """

    def load_train(self) -> pd.DataFrame:
        """학습 데이터 로드"""

    def load_test(self) -> Tuple[pd.DataFrame, np.ndarray]:
        """테스트 데이터 및 라벨 로드"""

    def get_info(self) -> Dict:
        """데이터셋 정보 반환 (shape, columns, missing values 등)"""
```

#### 3.1.2 DataAnalyzer 클래스
```python
class DataAnalyzer:
    """EDA 및 시계열 분석 클래스"""

    def __init__(self, data: pd.DataFrame):
        """분석할 데이터프레임 초기화"""

    def basic_eda(self) -> Dict:
        """기본 통계량, 결측치, 분포 분석"""

    def stationarity_test(self, column: str) -> Dict:
        """ADF Test, KPSS Test를 통한 정상성 검정"""

    def decompose(self, column: str, period: int) -> Dict:
        """STL Decomposition (Trend, Seasonal, Residual)"""

    def correlation_analysis(self) -> pd.DataFrame:
        """변수 간 상관관계 분석"""

    def anomaly_distribution(self, labels: np.ndarray) -> Dict:
        """이상치 레이블 분포 분석"""
```

### 3.2 전처리 모듈 (`src/preprocessing/`)

#### 3.2.1 BasePreprocessor 추상 클래스
```python
from abc import ABC, abstractmethod

class BasePreprocessor(ABC):
    """전처리기 기본 추상 클래스"""

    @abstractmethod
    def fit(self, data: np.ndarray) -> 'BasePreprocessor':
        """학습 데이터에 맞춤"""
        pass

    @abstractmethod
    def transform(self, data: np.ndarray) -> np.ndarray:
        """데이터 변환"""
        pass

    def fit_transform(self, data: np.ndarray) -> np.ndarray:
        """fit + transform"""
        return self.fit(data).transform(data)

    @abstractmethod
    def inverse_transform(self, data: np.ndarray) -> np.ndarray:
        """역변환"""
        pass
```

#### 3.2.2 Scaler 클래스
```python
class MinMaxScaler(BasePreprocessor):
    """Min-Max 정규화 (0~1)"""

    def __init__(self, feature_range: Tuple[float, float] = (0, 1)):
        self.feature_range = feature_range
        self.min_ = None
        self.max_ = None

class StandardScaler(BasePreprocessor):
    """Z-Score 정규화"""

    def __init__(self):
        self.mean_ = None
        self.std_ = None
```

#### 3.2.3 Smoother 클래스
```python
class EWMASmoother(BasePreprocessor):
    """Exponentially Weighted Moving Average Smoothing"""

    def __init__(self, span: int = 10):
        self.span = span
```

#### 3.2.4 Detrender 클래스
```python
class MovingAverageDetrender(BasePreprocessor):
    """이동 평균 기반 Detrending"""

    def __init__(self, window: int = 24):
        self.window = window
        self.trend_ = None
```

### 3.3 모델 모듈 (`src/models/`)

> ⚠️ **개발 전략**: 오픈소스 코드를 먼저 분석한 후, 공통 인터페이스를 확정합니다.

#### 3.3.1 오픈소스 코드 분석 결과 (Step 0) ✅

##### DLinear 분석 결과
| 항목 | 내용 |
|------|------|
| **입력 형식** | `[Batch, seq_len, Channel]` (float32) |
| **출력 형식** | `[Batch, pred_len, Channel]` (예측값) |
| **핵심 구조** | Moving Average → Trend/Seasonal 분해 → 각각 Linear Layer |
| **필수 파라미터** | `seq_len`, `pred_len`, `enc_in`(채널 수), `individual` |
| **라이브러리** | PyTorch (호환성 양호) |
| **손실 함수** | MSELoss |
| **이상 점수** | `|x_actual - x_predicted|` (예측 오차) |

##### OmniAnomaly 분석 결과
| 항목 | 내용 |
|------|------|
| **입력 형식** | `[Batch, window_length, x_dim]` (float32) |
| **출력 형식** | `log_prob` (복원 확률), `z` (잠재 벡터) |
| **핵심 구조** | RNN(GRU) + VAE + Planar Normalizing Flow |
| **필수 파라미터** | `x_dim`, `z_dim`, `window_length`, `rnn_num_hidden`, `nf_layers` |
| **라이브러리** | TensorFlow 1.x + tfsnippet ⚠️ (PyTorch 변환 필요) |
| **손실 함수** | ELBO (SGVB) |
| **이상 점수** | `-log_prob(x|z)` (음의 복원 확률) |

##### 공통점/차이점 분석
| 구분 | DLinear | OmniAnomaly |
|------|---------|-------------|
| **방법론** | Prediction-based | Reconstruction-based |
| **시퀀스 처리** | 슬라이딩 윈도우 | 슬라이딩 윈도우 |
| **학습 루프** | 표준 PyTorch | TF Session 기반 |
| **이상 점수 방향** | 클수록 이상 | 클수록 이상 (음수 log_prob) |
| **변환 필요** | 없음 | TF→PyTorch 필요 |

#### 3.3.2 BaseModel 추상 클래스
```python
class BaseModel(ABC):
    """
    이상 탐지 모델 기본 추상 클래스
    - 오픈소스 모델을 래핑하기 위한 공통 인터페이스
    - 분석 결과에 따라 메서드 시그니처가 조정될 수 있음
    """

    def __init__(self, config: Dict):
        self.config = config
        self.model = None          # 실제 오픈소스 모델 인스턴스
        self.is_fitted = False
        self.device = None         # GPU/CPU 설정

    @abstractmethod
    def fit(self, train_data: np.ndarray) -> 'BaseModel':
        """모델 학습"""
        pass

    @abstractmethod
    def predict(self, test_data: np.ndarray) -> np.ndarray:
        """예측/복원 수행"""
        pass

    @abstractmethod
    def get_anomaly_score(self, test_data: np.ndarray) -> np.ndarray:
        """이상 점수 계산"""
        pass

    def save(self, path: str) -> None:
        """모델 저장"""

    def load(self, path: str) -> 'BaseModel':
        """모델 로드"""
```

#### 3.3.3 Core 모듈 구조 (`src/models/cores/`)

> 오픈소스 원본 코드를 기반으로 작성. **기능은 100% 동일**, 라이브러리만 호환성 수정

```python
# dlinear_core.py
"""
DLinear 핵심 구현체
- 원본: references/DLinear/models/DLinear.py
- 핵심 클래스: moving_avg, series_decomp, Model
- 수정 사항:
  - [x] 원본 그대로 사용 가능 (PyTorch 호환)
  - [ ] configs 객체 → Dict 변환 어댑터 필요
"""

# omnianomaly_core.py
"""
OmniAnomaly 핵심 구현체 (PyTorch 버전)
- 원본: references/OmniAnomaly/omni_anomaly/
- 핵심 클래스: OmniAnomaly, VAE, RecurrentDistribution
- 수정 사항:
  - [!] TensorFlow 1.x → PyTorch 완전 재작성 필요
  - [!] tfsnippet 의존성 제거
  - [!] Planar Normalizing Flow PyTorch 구현
  - [ ] 기능 동등성 검증 필요
"""
```

##### OmniAnomaly PyTorch 변환 계획
```
[변환 대상 모듈]
1. model.py       → omnianomaly_core.py (메인 모델)
2. vae.py         → omnianomaly_core.py (VAE 컴포넌트)
3. wrapper.py     → omnianomaly_core.py (RNN, Normalizing Flow)
4. training.py    → OmniAnomaly Wrapper에서 처리
5. prediction.py  → OmniAnomaly Wrapper에서 처리
```

#### 3.3.4 DLinear Wrapper 모델
```python
class DLinear(BaseModel):
    """
    Prediction-based 모델 Wrapper
    - 핵심 구현: cores/dlinear_core.py 사용
    - 역할: BaseModel 인터페이스 제공 + 이상 점수 계산
    """

    def __init__(self, config: Dict):
        super().__init__(config)
        # 원본 모델 인스턴스화 (cores에서 import)
        from .cores.dlinear_core import DLinearCore
        self.model = DLinearCore(**self._extract_core_params(config))

    def _extract_core_params(self, config: Dict) -> Dict:
        """Framework config → Core 모델 파라미터 변환"""
        # 🎓 [학습 포인트] 사용자가 구현

    def get_anomaly_score(self, test_data: np.ndarray) -> np.ndarray:
        """
        예측 오차 기반 이상 점수
        score = |x_actual - x_predicted|
        """
        # 🎓 [학습 포인트] 사용자가 구현
```

#### 3.3.5 OmniAnomaly Wrapper 모델
```python
class OmniAnomaly(BaseModel):
    """
    Reconstruction-based 모델 Wrapper
    - 핵심 구현: cores/omnianomaly_core.py 사용
    - 역할: BaseModel 인터페이스 제공 + 이상 점수 계산
    """

    def __init__(self, config: Dict):
        super().__init__(config)
        from .cores.omnianomaly_core import OmniAnomalyCore
        self.model = OmniAnomalyCore(**self._extract_core_params(config))

    def get_anomaly_score(self, test_data: np.ndarray) -> np.ndarray:
        """
        복원 오차 + 잠재 공간 확률 기반 이상 점수
        - 복원 오차: |x - x_reconstructed|
        - 확률 오차: -log p(z) from Normalizing Flow
        """
        # 🎓 [학습 포인트] 사용자가 구현
```

### 3.4 후처리 모듈 (`src/postprocessing/`)

#### 3.4.1 Threshold 클래스
```python
class BaseThreshold(ABC):
    """임계값 기본 클래스"""

    @abstractmethod
    def fit(self, scores: np.ndarray) -> 'BaseThreshold':
        pass

    @abstractmethod
    def apply(self, scores: np.ndarray) -> np.ndarray:
        """이진 라벨 반환"""
        pass

class FixedThreshold(BaseThreshold):
    """고정 임계값 (mu + n*sigma 또는 상위 k%)"""

    def __init__(self, method: str = 'sigma', n_sigma: float = 3.0, percentile: float = 95.0):
        self.method = method
        self.n_sigma = n_sigma
        self.percentile = percentile

class POTThreshold(BaseThreshold):
    """Peaks Over Threshold (극단값 이론 기반)"""

    def __init__(self, risk: float = 0.01, init_level: float = 0.98):
        self.risk = risk
        self.init_level = init_level

class EWMAThreshold(BaseThreshold):
    """EWMA 기반 동적 임계값"""

    def __init__(self, span: int = 20, n_sigma: float = 3.0):
        self.span = span
        self.n_sigma = n_sigma
```

#### 3.4.2 Relabeling 클래스
```python
class PointAdjustment:
    """Point Adjustment (PA) 적용 클래스"""

    def __init__(self):
        pass

    def adjust(self, pred: np.ndarray, true: np.ndarray) -> np.ndarray:
        """
        True Anomaly Window 내 최소 1개의 TP 포함 시
        해당 윈도우 전체를 TP로 간주
        """

class WindowAggregation:
    """윈도우 기반 집계"""

    def __init__(self, window_size: int, min_count: int):
        """
        window_size: 윈도우 크기
        min_count: 이상치로 판정할 최소 탐지 수
        """

    def aggregate(self, pred: np.ndarray) -> np.ndarray:
        """윈도우 내 min_count 이상 탐지 시 이상 이벤트로 확정"""
```

### 3.5 평가 모듈 (`src/evaluation/`)

#### 3.5.1 Metrics 클래스
```python
class AnomalyMetrics:
    """이상 탐지 평가 지표 클래스"""

    @staticmethod
    def point_wise_metrics(pred: np.ndarray, true: np.ndarray) -> Dict:
        """
        Point-wise Precision, Recall, F1-score
        """

    @staticmethod
    def pa_metrics(pred: np.ndarray, true: np.ndarray) -> Dict:
        """
        Point Adjustment F1-score
        이상 윈도우 단위 평가
        """

    @staticmethod
    def composite_metrics(pred: np.ndarray, true: np.ndarray) -> Dict:
        """모든 지표 종합"""
```

#### 3.5.2 Visualizer 클래스
```python
class AnomalyVisualizer:
    """이상 탐지 결과 시각화 클래스"""

    def __init__(self, figsize: Tuple[int, int] = (15, 8)):
        self.figsize = figsize

    def plot_anomaly_score(
        self,
        data: np.ndarray,
        scores: Dict[str, np.ndarray],
        labels: np.ndarray,
        columns: List[str] = None
    ) -> plt.Figure:
        """
        시간 축 Anomaly Score Plot
        - 실제 시계열과 이상 점수를 함께 시각화
        """

    def plot_binary_decision(
        self,
        true: np.ndarray,
        predictions: Dict[str, np.ndarray]
    ) -> plt.Figure:
        """
        Binary Decision Plot
        - Ground Truth vs 예측 결과 비교
        - FP/FN 발생 지점 표시
        """

    def plot_decomposition(
        self,
        trend: np.ndarray,
        seasonal: np.ndarray,
        residual: np.ndarray
    ) -> plt.Figure:
        """STL 분해 결과 시각화"""

    def plot_threshold_comparison(
        self,
        scores: np.ndarray,
        thresholds: Dict[str, float]
    ) -> plt.Figure:
        """임계값 비교 시각화"""
```

### 3.6 파이프라인 모듈 (`src/pipeline/`)

```python
class TSADPipeline:
    """End-to-End 이상 탐지 파이프라인"""

    def __init__(
        self,
        preprocessors: List[BasePreprocessor],
        model: BaseModel,
        threshold: BaseThreshold,
        relabeling: PointAdjustment = None
    ):
        self.preprocessors = preprocessors
        self.model = model
        self.threshold = threshold
        self.relabeling = relabeling

    def fit(self, train_data: np.ndarray) -> 'TSADPipeline':
        """
        1. 전처리기 학습 및 변환
        2. 모델 학습
        """

    def predict(self, test_data: np.ndarray) -> Dict:
        """
        1. 전처리 적용
        2. 이상 점수 계산
        3. 임계값 적용
        4. (선택) Relabeling 적용

        Returns:
            - anomaly_scores: 이상 점수
            - predictions: 이진 예측
            - adjusted_predictions: PA 적용 예측 (있는 경우)
        """

    def evaluate(
        self,
        test_data: np.ndarray,
        true_labels: np.ndarray
    ) -> Dict:
        """
        예측 수행 및 평가 지표 계산
        """

    def save(self, path: str) -> None:
        """파이프라인 전체 저장"""

    def load(self, path: str) -> 'TSADPipeline':
        """파이프라인 로드"""
```

---

## 4. 실험 설계

> **핵심 원칙**: 전처리 변경 시 재학습 필요 / 후처리 변경 시 재학습 불필요
> - Step 1에서 전처리 조합별로 모델 학습 후 **Anomaly Score 저장**
> - Step 2에서 저장된 Score를 재사용하여 후처리 전략 비교 (재학습 없음)

### 4.1 Step 1: 전처리 실험 (모델 학습 + Score 저장)

| 실험 ID | 정규화 | Smoothing | Detrending | 목적 |
|---------|--------|-----------|------------|------|
| **P_MM** | MinMaxScaler | - | - | MinMax 정규화 Baseline |
| **P_STD** | StandardScaler | - | - | Standard 정규화 Baseline |
| **P_SM** | Best of above | EWMA(span=10) | - | Smoothing 효과 분석 |
| **P_DT** | Best of above | - | MA(window=24) | Detrending 효과 분석 |

**학습 횟수:** 4개 전처리 × 2개 모델 = **8번 학습**
**산출물:** 각 조합별 Anomaly Score 파일 저장 (`outputs/scores/`)

### 4.2 Step 2: 후처리 실험 (Score 재사용, 재학습 없음)

| 실험 ID | Threshold | Score Smoothing | Relabeling | 비교 목적 |
|---------|-----------|-----------------|------------|-----------|
| **T1** | Fixed (3σ) | - | - | **Baseline** (순수 Score 성능) |
| **T2** | Adaptive (EWMA) | - | - | Fixed vs Adaptive Threshold |
| **T3** | Fixed (3σ) | EWMA | - | Score Smoothing 효과 |
| **T4** | Best of T1~T3 | Best | Window Aggregation | Collective Anomaly 탐지 개선 |
| **T5** | Best of T4 | Best | Point Adjustment | PA F1-score 극대화 |

**결과:** 8개 Score × 5개 후처리 = **40개 조합**

### 4.3 Step 3: 최적 파이프라인 선정 및 분석

| 분석 항목 | 내용 |
|-----------|------|
| **최적 파이프라인 선정** | 각 모델별 최고 PA F1 달성 조합 (P_best, T_best) 확정 |
| **성능 비교** | DLinear vs OmniAnomaly 최종 성능 비교 |
| **이상 유형 분석** | Prediction vs Reconstruction이 어떤 이상에 효과적인지 |
| **시각화** | Anomaly Score Plot, Binary Decision Plot |

### 4.4 Step 4: 시각화 및 산출물

| 시각화 ID | 내용 |
|-----------|------|
| **V1** | Anomaly Score Plot (시간 축 + Ground Truth) |
| **V2** | Binary Decision Plot (예측 vs 실제, FP/FN 표시) |
| **V3** | 모델 비교 Chart (DLinear vs OmniAnomaly) |
| **V4** | 전처리/후처리별 성능 Heatmap |

### 4.5 하이퍼파라미터 (레퍼런스 기반 + Early Stopping)

> 설정 파일: `scripts/config.py`

#### DLinear (레퍼런스: LTSF-Linear)
| 파라미터 | 값 | 레퍼런스 원본 | 설명 |
|---------|-----|-------------|------|
| seq_len | 100 | 336 | 입력 시퀀스 길이 (이상탐지용 조정) |
| pred_len | 1 | 96~720 | 예측 길이 (1-step ahead) |
| individual | False | False | 채널 공유 |
| learning_rate | 0.005 | 0.005 | 학습률 |
| batch_size | 32 | 32 | 배치 크기 |
| epochs | 50 | 10 | 최대 학습 에폭 (Early Stopping) |
| **early_stopping** | True | - | Early Stopping 활성화 |
| **patience** | 5 | - | 5회 연속 개선 없으면 종료 |
| **val_ratio** | 0.1 | - | 10% validation split |

#### OmniAnomaly (레퍼런스: OmniAnomaly)
| 파라미터 | 값 | 레퍼런스 원본 | 설명 |
|---------|-----|-------------|------|
| window_size | 100 | 100 | 입력 시퀀스 길이 |
| z_dim | 3 | 3 | 잠재 공간 차원 |
| hidden_size | 500 | 500 | GRU hidden 차원 |
| n_flows | 20 | 20 | Normalizing Flow 층 수 |
| learning_rate | 0.001 | 0.001 | 학습률 |
| batch_size | 50 | 50 | 배치 크기 |
| epochs | 50 | 10 | 최대 학습 에폭 (Early Stopping) |
| **early_stopping** | True | True | Early Stopping 활성화 (원본에도 있음) |
| **patience** | 5 | - | 5회 연속 개선 없으면 종료 |
| **val_ratio** | 0.1 | - | 10% validation split |

### 4.6 실험 도구

#### ExperimentTracker (`src/utils/experiment_tracker.py`)
```python
tracker = ExperimentTracker(base_dir='outputs/')

# Step 1: 학습 후 Score 저장
tracker.log_training(model, preprocess, dataset, scores, training_time)

# Step 2: 후처리 결과 기록
tracker.log_evaluation(model, preprocess, postprocess, dataset, metrics)

# 보고서용 테이블 생성
report = tracker.generate_report()  # 피벗 테이블, 비교표 자동 생성
```

#### AnomalyVisualizer (`src/evaluation/visualizer.py`)
```python
visualizer = AnomalyVisualizer(figsize=(14, 6))

# V1: Anomaly Score Plot
fig = visualizer.plot_anomaly_scores(scores, labels, threshold)

# V2: Binary Decision Plot (TP/FP/FN 색상 구분)
fig = visualizer.plot_binary_decision(predictions, labels)

# V3: 모델 비교 Bar Chart
fig = visualizer.plot_comparison(results, metric='pa_f1')

# V4: Score 분포 Histogram
fig = visualizer.plot_score_distribution(scores, labels)
```

---

## 5. 진행 체크리스트

### Step 0: 오픈소스 코드 분석 (선행 단계) ✅ 완료
> **목표**: 모델 코드를 이해하고 Framework 인터페이스 확정

#### 0.1 환경 및 리소스 준비
- [x] 프로젝트 환경 설정 (conda activate timeseries)
- [x] 디렉토리 구조 생성
- [x] [사용자] PSM/SWaT 데이터셋 다운로드
- [x] [사용자] DLinear 오픈소스 코드 `references/DLinear/`에 복사
- [x] [사용자] OmniAnomaly 오픈소스 코드 `references/OmniAnomaly/`에 복사

#### 0.2 DLinear 코드 분석
- [x] 모델 아키텍처 분석 (Moving Average → Trend/Seasonal 분해 → Linear)
- [x] 입출력 형식 파악 (`[B, seq_len, C]` → `[B, pred_len, C]`)
- [x] 필수 하이퍼파라미터 목록 작성 (seq_len, pred_len, enc_in, individual)
- [x] 사용 라이브러리 및 버전 확인 (PyTorch - 호환성 양호)
- [x] deprecated API 식별 (없음)

#### 0.3 OmniAnomaly 코드 분석
- [x] 모델 아키텍처 분석 (RNN + VAE + Planar Normalizing Flow)
- [x] 입출력 형식 파악 (`[B, window_len, x_dim]` → `log_prob`, `z`)
- [x] 필수 하이퍼파라미터 목록 작성 (x_dim, z_dim, window_length, rnn_num_hidden, nf_layers)
- [x] 사용 라이브러리 및 버전 확인 (TensorFlow 1.x + tfsnippet - PyTorch 변환 필요)
- [x] deprecated API 식별 (전체가 TF1 기반으로 재작성 필요)

#### 0.4 Framework 인터페이스 확정
- [x] 두 모델의 공통 입출력 패턴 도출 (슬라이딩 윈도우, 이상점수)
- [x] BaseModel 인터페이스 최종 확정 (fit, predict, get_anomaly_score)
- [x] Core 모듈 수정 계획 수립 (DLinear: 어댑터만, OmniAnomaly: PyTorch 재작성)

---

### Step 1: 데이터 분석 및 준비 (주차 1)
> **담당**: 사용자(데이터 다운로드) + Assistant(코드 구현)

- [x] DataLoader 클래스 구현 (`src/data/loader.py`)
- [x] DataAnalyzer 클래스 구현 (`src/data/analyzer.py`)
- [x] PSM 데이터 EDA 수행 (노트북 실행 완료)
- [x] SWaT 데이터 EDA 수행 (노트북 실행 완료)
- [x] 🎓 [사용자 학습] 정상성 검정 (ADF, KPSS) 코드 작성 ✅
- [x] 🎓 [사용자 학습] STL Decomposition 분석 수행 ✅
- [x] EDA 결과 시각화 (노트북 실행 완료)

---

### Step 2: Core Baseline 구축 (주차 2)
> **목표**: 오픈소스 기반 모델을 Framework에 통합

#### 2.1 전처리 모듈
- [x] BasePreprocessor 추상 클래스 구현 ✅
- [x] Scaler 클래스 구현 (MinMax, Standard) ✅
- [x] 🎓 [사용자 학습] EWMA Smoother 핵심 로직 구현 ✅
- [x] Detrender 클래스 구현 (MA, Differencing, STL) ✅

#### 2.2 모델 모듈 (Core 기반)
- [x] BaseModel 추상 클래스 구현 ✅
- [x] dlinear_core.py 작성 (원본 → 호환성 수정) ✅
- [x] omnianomaly_core.py 작성 (TF→PyTorch 변환) ✅
- [x] DLinear Wrapper 구현 ✅
- [x] 🎓 [사용자 학습] DLinear 이상 점수 계산 로직 구현 ✅
- [x] OmniAnomaly Wrapper 구현 ✅
- [x] 🎓 [사용자 학습] OmniAnomaly Reconstruction Loss 구현 ✅

#### 2.3 평가 모듈
- [x] 🎓 [사용자 학습] Point-wise F1 계산 로직 구현 ✅
- [x] 🎓 [사용자 학습] Point Adjustment F1 구현 ✅
- [x] 🎓 [사용자 학습] AUC Metrics (ROC-AUC, PR-AUC) 구현 ✅
- [x] Range-based Metrics 구현 ✅
- [x] TSADPipeline 구현 ✅

#### 2.4 Baseline 실험
- [x] 🎓 [사용자 학습] FixedThreshold 클래스 구현 (Sigma 방법) ✅
- [x] EWMAThreshold, AdaptiveThreshold 구현 ✅
- [ ] Baseline 실험 (MinMax + Fixed Threshold)
- [ ] Point-wise F1-score 측정 및 분석

---

### Step 3: 심화 실험 및 비교 분석 (주차 3-4)

#### 3.1 전처리 실험
- [ ] 전처리 비교 실험 (P1~P5)
- [ ] 결과 분석 및 최적 전처리 선정

#### 3.2 후처리 실험
- [ ] 🎓 [사용자 학습] POT Threshold 핵심 로직 구현
- [x] EWMA Threshold 구현 ✅
- [ ] Threshold 비교 실험 (T1~T4)

#### 3.3 Relabeling 실험
- [ ] 🎓 [사용자 학습] Point Adjustment 로직 구현
- [ ] Window Aggregation 구현
- [ ] Relabeling 실험 (T5~T6)

#### 3.4 시각화 및 분석
- [x] Visualizer 클래스 구현 ✅
- [x] Anomaly Score Plot 생성 ✅
- [x] 🎓 [사용자 학습] Binary Decision Plot (TP/FP/FN 마스크 구현) ✅
- [x] Score Distribution Plot 생성 ✅
- [ ] 실험 결과 정리

#### 3.5 실험 인프라
- [x] ExperimentTracker 클래스 구현 (자동 기록 시스템) ✅
- [x] Step 1 실험 스크립트 (`scripts/run_step1.py`) ✅
- [x] Step 2 실험 스크립트 (`scripts/run_step2.py`) ✅
- [x] 하이퍼파라미터 설정 파일 (`scripts/config.py`) ✅
- [x] Early Stopping 구현 (DLinear, OmniAnomaly) ✅

---

### Step 4: 최종 보고서 및 코드 정리 (주차 5)
- [ ] 실험 결과 종합 분석
- [ ] 🎓 [사용자 학습] 인사이트 도출 및 토론
- [ ] 보고서 작성 (10페이지 이내)
- [ ] 코드 정리 및 문서화
- [ ] 최종 검토

---

## 6. 기술 스택

### 6.1 필수 라이브러리
```
# requirements.txt
numpy>=1.21.0
pandas>=1.3.0
torch>=1.9.0
scikit-learn>=0.24.0
statsmodels>=0.12.0
matplotlib>=3.4.0
seaborn>=0.11.0
scipy>=1.7.0
tqdm>=4.61.0
PyYAML>=5.4.0
```

### 6.2 개발 환경
- Python: 3.8+
- 가상환경: conda (timeseries)
- GPU: CUDA 지원 권장

---

## 7. 참고 사항

### 7.1 코딩 규칙
- 객체지향 설계 원칙 준수
- PEP 8 스타일 가이드 준수
- Type Hints 사용
- Docstring 작성

### 7.2 주의사항
- 재사용 원칙 준수
- 난개발/일회성 파일 생성 금지
- 설계 문서 기반 개발
- 진행 시 체크리스트 업데이트

---

## 8. 학습 포인트 가이드 🎓

> 프로젝트 목적이 **학습**이므로, 핵심 개념을 직접 구현하며 이해합니다.

### 8.1 사용자 참여 구현 목록

| 단계 | 학습 주제 | 구현 내용 | 난이도 |
|------|----------|----------|--------|
| Step 1 | 시계열 정상성 | ADF/KPSS 검정 코드 작성 | ⭐⭐ |
| Step 1 | 시계열 분해 | STL Decomposition 수행 | ⭐⭐ |
| Step 2 | Smoothing | EWMA 핵심 로직 구현 | ⭐ |
| Step 2 | 이상 점수 (Prediction) | DLinear 예측 오차 계산 | ⭐⭐ |
| Step 2 | 이상 점수 (Reconstruction) | OmniAnomaly 복원 오차 계산 | ⭐⭐⭐ |
| Step 2 | 평가 지표 | Point-wise F1 계산 | ⭐⭐ |
| Step 3 | 극단값 이론 | POT Threshold 핵심 로직 | ⭐⭐⭐ |
| Step 3 | 후처리 | Point Adjustment 로직 | ⭐⭐ |
| Step 4 | 분석 | 실험 결과 인사이트 도출 | ⭐⭐ |

### 8.2 학습 진행 방식

```
[협업 패턴]
1. Assistant: 구조/틀 작성 + TODO(human) 마킹
2. 사용자: 핵심 로직 직접 구현
3. Assistant: 코드 리뷰 및 피드백
4. 함께: 결과 분석 및 토론
```

### 8.3 핵심 학습 질문

각 단계에서 스스로 답해볼 질문들:

#### 데이터 분석
- 왜 시계열 데이터에서 정상성(Stationarity)이 중요한가?
- Trend와 Seasonal 성분이 이상 탐지에 어떤 영향을 미치는가?

#### 모델 이해
- DLinear가 Trend/Seasonal을 분해하는 이유는?
- OmniAnomaly에서 Normalizing Flow의 역할은?
- Prediction-based와 Reconstruction-based의 근본적 차이는?

#### 평가 이해
- Point-wise F1과 PA F1의 차이점은?
- 왜 이상 탐지에서 PA(Point Adjustment)가 필요한가?

#### 후처리 이해
- Fixed Threshold의 한계는 무엇인가?
- POT(Peaks Over Threshold)가 어떻게 동적 임계값을 설정하는가?

---

## 9. 오픈소스 참조 링크

| 모델 | GitHub Repository | 논문 |
|------|-------------------|------|
| **DLinear** | https://github.com/cure-lab/LTSF-Linear | "Are Transformers Effective for Time Series Forecasting?" (AAAI 2023) |
| **OmniAnomaly** | https://github.com/NetManAIOps/OmniAnomaly | "Robust Anomaly Detection for Multivariate Time Series" (KDD 2019) |
