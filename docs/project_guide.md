## 🚀 시계열 이상 탐지(TSAD) 프로젝트 상세 설계안

요구사항, 리소스 제약, 데이터 특성을 모두 반영하여 **OmniAnomaly**와 **DLinear**를 핵심 모델로 하는 End-to-End TSAD 프로젝트의 상세 설계안을 다시 작성했습니다.

---

## 1. 프로젝트 개요 및 요구사항 정리

### 1.1. 최종 목표 및 산출물

* **최종 목표:** PSM/SWaT 데이터셋에서 **Prediction-based (DLinear)**와 **Reconstruction-based (OmniAnomaly)** 기법을 비교 분석하는 End-to-End 이상 탐지 파이프라인 구현.
* **제출 자료:**
    * **보고서:** Word 기준 표지 제외 10페이지 이내 (폰트 10pt 이상), 시각화 자료 및 실험 결과 필수 포함.
    * **실행 가능한 코드:** `ipynb` 또는 `.py` 파일.

### 1.2. 핵심 요구사항 요약

| 단계 | 요구사항 항목 | 선정 모델 및 전략 |
| :--- | :--- | :--- |
| **데이터** | PSM, SWaT 데이터 분석 | **다변량(Multivariate)** 특성 및 **정상성** 분석. |
| **파이프라인** | End-to-End 구현 | **전처리 → 모델 → 후처리/Thresholding → 결과** 단계 구현. |
| **모델 비교** | Prediction-based vs. Reconstruction-based | **DLinear** vs. **OmniAnomaly** 비교. |
| **전처리 실험** | 이동 평균 제거, 정규화, Smoothing | 각 모델에 미치는 영향 및 결과 비교 분석. |
| **후처리 실험** | Fixed vs. Adaptive Threshold, Relabeling (PA), Smoothing | 성능 지표(**PA F1-score**) 개선 효과 분석. |
| **평가/시각화** | Point-wise F1, PA F1, Score Plot, Binary Plot | 정량적 평가 및 시각적 인사이트 도출. |

---

## 2. 데이터 분석 및 전처리 설계

### 2.1. 데이터 분석 상세 계획 (PSM, SWaT)

| 분석 항목 | 상세 수행 내용 | 모델 선정/실험과의 연관성 |
| :--- | :--- | :--- |
| **기본 EDA** | 결측치, 이상치 레이블 구간 시각화. 데이터의 시간적 밀도 및 센서(변수) 개수 확인. | 다변량 데이터의 **센서 간 상호작용** 분석 필요성 강조. |
| **시계열 정상성 분석** | **ADF Test, KPSS Test**를 수행하여 통계적 정상성 여부 확인. | 비정상성일 경우, 차분(Differencing)의 필요성 보고서에 명시. |
| **Trend/Seasonal 분해** | STL Decomposition을 사용하여 Trend, Seasonal, Residual 요소를 시각화 및 분석. | **DLinear의 분해 기반 예측**과 **OmniAnomaly의 잠재 공간 학습**이 각 요소에 어떻게 반응하는지 예측.  |

### 2.2. 전처리 기법 상세 실험 설계

각 전처리 기법을 DLinear와 OmniAnomaly에 적용하여 성능 변화(F1-score)를 측정합니다.

* **정규화 비교:**
    * `MinMaxScaler` (0~1): VAE 기반 OmniAnomaly의 잠재 공간 학습에 안정성 제공.
    * `StandardScaler` (Z-Score): 이상 점수($\sigma$) 계산이 용이하고 DLinear의 선형 예측 정확도에 미치는 영향 분석.
* **Detrending/Smoothing 비교:**
    * **이동 평균 제거 (Detrending):** Trend 성분을 제거한 후, DLinear가 Trend를 재학습하는 방식과 OmniAnomaly가 순수 잔차(Residual) 패턴을 학습하는 방식 비교.
    * **EWMA Smoothing:** 데이터 노이즈를 감소시켜 Anomaly Score의 변동성을 줄이고 FP 감소 효과 분석.

---

## 3. 모델 선정 및 실험 구성

### 3.1. 핵심 모델별 역할 및 메커니즘

| 모델 | 방법론 | 이상 점수 산출 상세 | TSAD 적합성 (PSM/SWaT) |
| :--- | :--- | :--- | :--- |
| **OmniAnomaly** | **Reconstruction-based (VAE + NF)** | **1단계:** 복원 오차 ($|x - x_{rec}|$). **2단계:** 잠재 공간 내 확률 분포 오차 (Plane Normalizing Flow). | **다변량 변수 간의 복잡한 상관관계** 학습에 최적화되어, 구조적/상관관계 이상 탐지에 강력. |
| **DLinear** | **Prediction-based (Linear)** | Trend 및 Seasonal 요소 예측 후 예측 오차. | **리소스 효율성이 뛰어나고**, 선형적인 Trend 및 Seasonal 변화를 잘 포착하여 Temporal Anomaly 탐지에 효과적. |

### 3.2. 후처리 및 Thresholding 전략 상세 실험

이 단계는 **Point Adjustment F1-score**를 극대화하는 것을 목표로 합니다.

#### A. Thresholding 비교 실험

* **Fixed Threshold:** Anomaly Score의 $\mu + n\sigma$ 또는 상위 $k\%$ 기준으로 임계값 설정 (Baseline).
* **Adaptive Threshold (Dynamic):**
    * **POT (Peaks Over Threshold):** 극단값 이론(Extreme Value Theory)을 적용하여 동적으로 임계값을 설정.
    * **EWMA-based:** 이전 시점들의 Anomaly Score를 기반으로 변화하는 임계값을 설정.
    * **비교 분석:** 각 모델의 이상 점수 분포에 따라 어떤 Adaptive 전략이 더 효과적으로 FP를 완화하는지 분석.

#### B. Relabeling 및 Aggregation 실험

* **Window Aggregation (Temporal Smoothing):** 연속된 $N$개 시점 중 $M$개 이상이 이상치로 탐지되어야 최종 이상 이벤트로 확정.
* **Point Adjustment (Relabeling):** 탐지된 이상 시점 주변의 실제 이상 구간(Ground Truth)과 비교하여 평가 지표를 산출하는 로직 구현. PA를 적용했을 때와 안 했을 때의 F1-score 차이를 통해 Relabeling 전략의 중요성 강조. 

---

## 4. 평가 지표 및 시각화 계획

### 4.1. 평가 지표

| 지표명 | 수식/설명 | 활용 목적 |
| :--- | :--- | :--- |
| **Point-wise F1-score** | $$\frac{2 \cdot P_{pt} \cdot R_{pt}}{P_{pt} + R_{pt}}$$ (개별 시점 정확도) | 모델 자체의 기본적인 탐지 성능 비교. |
| **Point Adjustment (PA) F1-score** | True Anomaly Window 내 최소 1개의 TP를 포함 시 성공으로 간주. | 이상 이벤트 단위의 실질적인 탐지 성능을 측정. **최종 성능 지표**. |

### 4.2. 이상 탐지 시각화

* **시간 축 Anomaly Score Plot:** 실제 시계열 (변수 1~3개 선택)과 두 모델(DLinear, OmniAnomaly)의 Anomaly Score를 한 그래프에 시각화하여, 어떤 이상 유형(Point, Contextual, Collective)에 각 모델이 더 민감한지 비교 분석.
* **이진 결정(Binary Decision) Plot:** Ground Truth와 최적의 후처리 전략이 적용된 최종 탐지 결과(Binary Output)를 시간 축에 시각화하여, **FP/FN 발생 지점**을 명확히 제시.

---

## 5. 프로젝트 진행 단계 정의

| 단계 | 주요 내용 | 예상 기간 | 산출물 |
| :--- | :--- | :--- | :--- |
| **Step 1: 데이터 분석 및 준비** | 데이터 확보 및 EDA, 정상성 및 Trend/Seasonal 분석. 시계열 분해 시각화. | 주차 1 | EDA 결과 보고서 초안, 분석된 데이터 특성에 대한 모델 선정 근거. |
| **Step 2: Core Baseline 구축** | **OmniAnomaly** 및 **DLinear** 기본 파이프라인 구현 (MinMax Scaling, Fixed Threshold). Point-wise F1-score 측정. | 주차 2 | Baseline 코드 (`ipynb`), 초기 실험 결과 정리. |
| **Step 3: 심화 실험 및 비교 분석** | 전처리 기법(Smoothing, Detrending) 비교 분석. **Adaptive Threshold** 및 **Relabeling** 전략 적용 및 **PA F1-score** 측정. | 주차 3-4 | 모든 실험 결과(수치 및 시각화) 정리, 비교 분석 중간 보고서. |
| **Step 4: 최종 보고서 및 코드 정리** | 모든 실험 결과 요약 및 인사이트 도출. 보고서 요구사항(10p)에 맞춰 최종 보고서 완성. 코드 정리. | 주차 5 | **최종 보고서**, **실행 가능한 코드**. |

---

## 6. 최종 산출물 (보고서) 목차 구조

| 목차 | 내용 (요구사항 반영) |
| :--- | :--- |
| **1. 서론** | 프로젝트 목표, TSAD 문제 정의, **OmniAnomaly/DLinear 선정 근거**. |
| **2. 데이터 분석 및 특징** | PSM/SWaT EDA, 정상성 분석, **Trend/Seasonal/Residual 분해 시각화**. |
| **3. 이상 탐지 모델 및 파이프라인** | **DLinear** 및 **OmniAnomaly** 메커니즘 상세 설명 및 End-to-End 파이프라인 구조 제시. |
| **4. 실험 설계 및 결과 분석** | **4.1. 전처리 기법 비교 분석** 결과 및 인사이트. |
| | **4.2. Prediction-based vs. Reconstruction-based** 성능 비교 (OmniAnomaly 중심). |
| | **4.3. 후처리 및 Thresholding 전략** (Fixed vs. Adaptive, Relabeling) **PA F1-score** 개선 분석. |
| | **4.4. 이상 탐지 시각화** (Score Plot 및 Binary Decision Plot). |
| **5. 결론 및 기여** | 최적의 파이프라인 제안, 실험 결과 요약 및 **인사이트 작성**. |

## 기타 사항
객체지향 개발론 사용
재사용 원칙 준수, 난개발/일회성 .py 와 .md 파일 생성 금지 
설계 준수
각 진행시 설계문서에 진행여부 체크할것
가상환경은 conda activate timeseries 사용

