## 🔬 상세 실험 계획: OmniAnomaly vs. DLinear End-to-End 파이프라인 비교

제시된 요구사항을 충족하고, **전처리 변경 시 재학습의 필요성**과 **후처리 변경 시 재학습의 불필요성**을 반영한 **가장 효율적인 4단계 상세 실험 계획**입니다.

이 계획은 **전처리 조합($P$)당 모델을 1회만 학습**하고, 이후 해당 모델이 산출한 Anomaly Score를 기반으로 **6가지 후처리 전략($T1 \sim T6$)**을 빠르게 비교합니다.

---

## 1. Step 1: Baseline 구축 및 전처리 심화 실험 설계

이 단계에서는 전처리 전략을 변경하며 **모델을 새로 학습**시키고, 각 조합별로 **Anomaly Score를 산출 및 저장**하는 것이 목표입니다.

### 1.1. 전처리 기법 정의 및 조합

| ID | 전처리 기법 | 목표 |
| :--- | :--- | :--- |
| **P_BASE** | **정규화** (OmniAnomaly: MinMaxScaler, DLinear: StandardScaler) | 각 모델의 특성에 맞는 기본 정규화만 적용하여 Baseline Score 획득. |
| **P1** | P_BASE + **EWMA Smoothing** | 노이즈가 많은 산업 데이터에서 Smoothing이 패턴 학습에 미치는 영향 분석. |
| **P2** | P_BASE + **이동 평균 제거 (Detrending)** | Trend 성분을 제거하여 모델이 순수한 잔차/주기 패턴 학습에 집중할 수 있는지 확인. |
| **P3** | P_BASE + EWMA Smoothing + Detrending | 복합 전처리가 모델의 성능을 얼마나 개선시키는지 최종 확인. |

### 1.2. 실험 진행 순서 (모델별 독립 수행)

| 순서 | 모델 | 전처리 조합 (P_i) | 산출물 (저장) | 재학습 필요 여부 |
| :--- | :--- | :--- | :--- | :--- |
| **1** | **DLinear** | **P_BASE** | $S_{DL\_BASE}$ (DLinear Score) | O (최초 학습) |
| **2** | **DLinear** | **P1 (Smoothing)** | $S_{DL\_P1}$ | **O (재학습 필수)** |
| **3** | **DLinear** | **P2 (Detrending)** | $S_{DL\_P2}$ | **O (재학습 필수)** |
| **4** | **DLinear** | **P3 (복합)** | $S_{DL\_P3}$ | **O (재학습 필수)** |
| **5-8**| **OmniAnomaly**| **P\_BASE $\sim$ P3** | $S_{OA\_BASE} \sim S_{OA\_P3}$ | **O (각 조합별 재학습 필수)** |

---

## 2. Step 2: 후처리 및 Thresholding 전략 비교 실험

Step 1에서 저장된 **총 8가지 Anomaly Score**($S$) 파일을 불러와 **모델 학습 없이** 후처리 전략($T1 \sim T6$)을 적용하고 성능(PA F1)을 측정합니다.

### 2.1. 후처리 전략 정의

| ID | Threshold | Score Smoothing | Relabeling (Post-Filter) | 비교 목적 |
| :--- | :--- | :--- | :--- | :--- |
| **T1** | **Fixed (3$\sigma$ 또는 99th Percentile)** | **-** | **-** | Baseline (순수 Score의 성능) 측정. |
| **T2** | **Adaptive (EWMA)** | **-** | **-** | Fixed 대비 **동적 임계값** 적용 시 **FP 완화** 효과 분석. |
| **T3** | Fixed (3$\sigma$) | **EWMA** (Score Smoothing) | **-** | Anomaly Score 자체를 정규화하여 **Noise성 Anomaly Point** 제거 효과 확인. |
| **T4** | **Adaptive (POT)** | **-** | **-** | EWMA 기반 대비 **극단값 이론** 기반 Threshold의 성능 비교. |
| **T5** | **Best $T$ & $S$ of $T1 \sim T4$** | **Best $T$ & $S$ of $T1 \sim T4$** | **Window Aggregation (e.g., $W=5$, $k=3$)** | **Collective Anomaly** 탐지율 개선 및 최종 **Binary Decision** 노이즈 제거. |
| **T6** | **Best $T$ & $S$ of $T5$** | **Best $T$ & $S$ of $T5$** | **Point Adjustment (PA)** | Relabeling 전략의 최종 **PA F1-score 극대화** 효과 입증. |

### 2.2. 실험 진행 순서

1.  **점수 로드:** Step 1에서 저장된 $S_{DL\_BASE}$ 파일 로드.
2.  **후처리 루프:** 로드된 $S_{DL\_BASE}$에 대해 **T1부터 T6까지 순차적으로 후처리 로직을 적용**하고, PA F1-score를 측정하여 기록합니다.
3.  **반복:** 모든 $S_{DL\_P1} \sim S_{OA\_P3}$ 점수 파일에 대해 2번 과정을 반복합니다.

👉 **결과:** 총 $8 (\text{전처리 조합}) \times 6 (\text{후처리 전략}) = \mathbf{48}$ 가지의 성능 지표를 획득합니다.

---

## 3. Step 3: 최적 파이프라인 선정 및 최종 비교 분석

Step 2에서 획득한 48가지 조합의 성능을 분석하여 최종 결론을 도출합니다.

### 3.1. 최적 파이프라인 선정

* **DLinear 최적:** DLinear 모델이 달성한 **가장 높은 PA F1-score**를 기록한 전처리($P_{DL\_best}$)와 후처리($T_{DL\_best}$) 조합을 확정합니다.
* **OmniAnomaly 최적:** OmniAnomaly 모델이 달성한 **가장 높은 PA F1-score**를 기록한 전처리($P_{OA\_best}$)와 후처리($T_{OA\_best}$) 조합을 확정합니다.

### 3.2. 최종 성능 비교 및 시각화

| 비교 항목 | 상세 분석 내용 | 보고서 활용 |
| :--- | :--- | :--- |
| **성능 비교** | DLinear의 $PA F1_{best}$ vs. OmniAnomaly의 $PA F1_{best}$ | Bar Chart 및 상세 테이블. |
| **이상 유형 분석** | 최적 파이프라인을 특정 이상 이벤트 구간(예: SWaT의 공격 구간)에 적용했을 때의 **Anomaly Score Plot** 시각화. | **Prediction** 기반(DLinear)과 **Reconstruction** 기반(OmniAnomaly)이 어떤 유형의 이상에 민감하게 반응하는지 정성적으로 비교. |
| **전처리/후처리 인사이트** | **Detrending**이 DLinear($P_{DL\_best}$)에 긍정적이었는지, **Adaptive Threshold**가 OmniAnomaly($T_{OA\_best}$)의 FP 완화에 얼마나 기여했는지 등 **구체적인 수치 기반 인사이트** 도출. | 최종 보고서의 '결론 및 기여' 섹션. |

---

## 4. Step 4: 최종 산출물 정리

* **보고서:** Step 1~3의 모든 과정을 **시각화 자료**와 함께 정리하고, 특히 $T1 \sim T6$ 전략 변화에 따른 **PA F1-score의 개선 추이**를 보고서의 핵심 자료로 제시합니다.
* **코드:** **전처리 모듈, 모델 학습/추론 모듈, 후처리 평가 모듈($T1 \sim T6$ 로직 포함)**이 분리되어 쉽게 실행 가능한 구조로 제출합니다.