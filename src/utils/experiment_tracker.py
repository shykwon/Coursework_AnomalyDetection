# -*- coding: utf-8 -*-
"""
실험 결과 자동 기록 및 관리 시스템

🎓 [학습 포인트]
- Step 1 (전처리 + 학습): Score 저장 + 학습 정보 기록
- Step 2 (후처리): Score 로드 → 평가 → 결과 기록
- 보고서 작성 시 피벗 테이블, 비교표 자동 생성
"""

import os
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any

import numpy as np
import pandas as pd


class ExperimentTracker:
    """
    실험 결과 자동 기록 및 관리 클래스

    사용법:
    ```python
    tracker = ExperimentTracker(base_dir='outputs/')

    # Step 1: 학습 후 Score 저장
    tracker.log_training(
        model='DLinear',
        preprocess='P_MM',
        dataset='PSM',
        scores=anomaly_scores,
        training_time=120.5
    )

    # Step 2: 후처리 결과 기록
    tracker.log_evaluation(
        model='DLinear',
        preprocess='P_MM',
        postprocess='T1',
        dataset='PSM',
        metrics={'point_f1': 0.72, 'pa_f1': 0.85, ...}
    )

    # 보고서용 테이블 생성
    tables = tracker.generate_report()
    ```
    """

    def __init__(self, base_dir: str = 'outputs/'):
        """
        Args:
            base_dir: 결과 저장 기본 디렉토리
        """
        self.base_dir = Path(base_dir)
        self.scores_dir = self.base_dir / 'scores'
        self.results_dir = self.base_dir / 'results'
        self.figures_dir = self.base_dir / 'figures'
        self.logs_dir = self.base_dir / 'logs'

        # 디렉토리 생성
        for dir_path in [self.scores_dir, self.results_dir, self.figures_dir, self.logs_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)

        # 결과 파일 경로
        self.training_log_path = self.results_dir / 'training_log.csv'
        self.evaluation_log_path = self.results_dir / 'evaluation_results.csv'

        # 메모리 내 결과 저장
        self._training_records: List[Dict] = []
        self._evaluation_records: List[Dict] = []

        # 기존 결과 로드
        self._load_existing_results()

    def _load_existing_results(self):
        """기존 결과 파일 로드"""
        if self.training_log_path.exists():
            df = pd.read_csv(self.training_log_path)
            self._training_records = df.to_dict('records')

        if self.evaluation_log_path.exists():
            df = pd.read_csv(self.evaluation_log_path)
            self._evaluation_records = df.to_dict('records')

    # ==================== Step 1: 학습 기록 ====================

    def log_training(
        self,
        model: str,
        preprocess: str,
        dataset: str,
        scores: np.ndarray,
        training_time: float,
        config: Optional[Dict] = None,
        extra_info: Optional[Dict] = None
    ) -> str:
        """
        Step 1: 모델 학습 결과 기록 + Score 저장

        Args:
            model: 모델 이름 ('DLinear', 'OmniAnomaly')
            preprocess: 전처리 ID ('P_MM', 'P_STD', 'P_SM', 'P_DT')
            dataset: 데이터셋 ('PSM', 'SWaT')
            scores: Anomaly Score 배열
            training_time: 학습 시간 (초)
            config: 모델/전처리 설정 (optional)
            extra_info: 추가 정보 (optional)

        Returns:
            저장된 Score 파일 경로
        """
        # Score 파일명 생성
        score_filename = f"{model}_{preprocess}_{dataset}.npy"
        score_path = self.scores_dir / score_filename

        # Score 저장
        np.save(score_path, scores)

        # 학습 기록 생성
        record = {
            'exp_id': len(self._training_records) + 1,
            'model': model,
            'preprocess': preprocess,
            'dataset': dataset,
            'score_file': score_filename,
            'score_min': float(scores.min()),
            'score_max': float(scores.max()),
            'score_mean': float(scores.mean()),
            'score_std': float(scores.std()),
            'training_time_sec': training_time,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }

        # 추가 정보 병합
        if config:
            record['config'] = json.dumps(config)
        if extra_info:
            record.update(extra_info)

        # 기록 추가 및 저장
        self._training_records.append(record)
        self._save_training_log()

        print(f"✅ [Training] {model} + {preprocess} + {dataset}")
        print(f"   Score saved: {score_path}")
        print(f"   Training time: {training_time:.1f}s")

        return str(score_path)

    def _save_training_log(self):
        """학습 로그 CSV 저장"""
        df = pd.DataFrame(self._training_records)
        df.to_csv(self.training_log_path, index=False)

    # ==================== Step 2: 평가 기록 ====================

    def log_evaluation(
        self,
        model: str,
        preprocess: str,
        postprocess: str,
        dataset: str,
        metrics: Dict[str, float],
        threshold_value: Optional[float] = None,
        extra_info: Optional[Dict] = None
    ) -> None:
        """
        Step 2: 후처리 + 평가 결과 기록

        Args:
            model: 모델 이름
            preprocess: 전처리 ID
            postprocess: 후처리 ID ('T1', 'T2', 'T3', 'T4', 'T5')
            dataset: 데이터셋
            metrics: 평가 지표 {'point_f1': ..., 'pa_f1': ..., 'roc_auc': ..., 'pr_auc': ...}
            threshold_value: 사용된 임계값 (optional)
            extra_info: 추가 정보 (optional)
        """
        # 평가 기록 생성
        record = {
            'exp_id': len(self._evaluation_records) + 1,
            'model': model,
            'preprocess': preprocess,
            'postprocess': postprocess,
            'dataset': dataset,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }

        # 메트릭 추가
        for key, value in metrics.items():
            record[key] = value

        if threshold_value is not None:
            record['threshold'] = threshold_value

        if extra_info:
            record.update(extra_info)

        # 기록 추가 및 저장
        self._evaluation_records.append(record)
        self._save_evaluation_log()

        # 주요 지표 출력
        pa_f1 = metrics.get('pa_f1', metrics.get('f1', 'N/A'))
        print(f"✅ [Eval] {model} + {preprocess} + {postprocess}: PA F1 = {pa_f1}")

    def _save_evaluation_log(self):
        """평가 로그 CSV 저장"""
        df = pd.DataFrame(self._evaluation_records)
        df.to_csv(self.evaluation_log_path, index=False)

    # ==================== Score 관리 ====================

    def load_scores(self, model: str, preprocess: str, dataset: str) -> np.ndarray:
        """
        저장된 Score 로드

        Args:
            model: 모델 이름
            preprocess: 전처리 ID
            dataset: 데이터셋

        Returns:
            Anomaly Score 배열
        """
        score_filename = f"{model}_{preprocess}_{dataset}.npy"
        score_path = self.scores_dir / score_filename

        if not score_path.exists():
            raise FileNotFoundError(f"Score file not found: {score_path}")

        return np.load(score_path)

    def list_available_scores(self) -> List[Dict[str, str]]:
        """저장된 Score 파일 목록 반환"""
        scores = []
        for score_file in self.scores_dir.glob('*.npy'):
            # 파일명 형식: {model}_{preprocess}_{dataset}.npy
            # preprocess가 P_MM 처럼 _를 포함할 수 있으므로 rsplit 사용
            stem = score_file.stem
            parts = stem.rsplit('_', 1)  # 뒤에서부터 1번만 split → dataset 분리
            if len(parts) == 2:
                model_preprocess, dataset = parts
                # model과 preprocess 분리 (첫 번째 _에서)
                mp_parts = model_preprocess.split('_', 1)
                if len(mp_parts) == 2:
                    model, preprocess = mp_parts
                    scores.append({
                        'model': model,
                        'preprocess': preprocess,
                        'dataset': dataset,
                        'file': str(score_file)
                    })
        return scores

    # ==================== 보고서 생성 ====================

    def generate_report(self) -> Dict[str, pd.DataFrame]:
        """
        보고서용 테이블 생성

        Returns:
            dict with:
                - 'full': 전체 결과 DataFrame
                - 'preprocess_comparison': 전처리별 비교 (피벗)
                - 'postprocess_comparison': 후처리별 비교 (피벗)
                - 'model_comparison': 모델별 최고 성능
                - 'best_combinations': 최적 조합 Top 5
        """
        if not self._evaluation_records:
            print("⚠️ No evaluation results found.")
            return {}

        df = pd.DataFrame(self._evaluation_records)

        report = {
            'full': df
        }

        # 전처리별 비교 (PA F1 기준)
        if 'pa_f1' in df.columns:
            report['preprocess_comparison'] = df.pivot_table(
                index='preprocess',
                columns='model',
                values='pa_f1',
                aggfunc='max'
            )

            # 후처리별 비교
            report['postprocess_comparison'] = df.pivot_table(
                index='postprocess',
                columns='model',
                values='pa_f1',
                aggfunc='max'
            )

            # 모델별 최고 성능
            agg_dict = {
                'pa_f1': 'max',
                'point_f1': 'max',
            }
            # roc_auc, pr_auc는 있을 때만 추가
            if 'roc_auc' in df.columns:
                agg_dict['roc_auc'] = 'max'
            if 'pr_auc' in df.columns:
                agg_dict['pr_auc'] = 'max'

            report['model_comparison'] = df.groupby('model').agg(agg_dict).round(4)

            # 최적 조합 Top 5
            report['best_combinations'] = df.nlargest(5, 'pa_f1')[
                ['model', 'preprocess', 'postprocess', 'pa_f1', 'point_f1']
            ]

        return report

    def get_best(self, metric: str = 'pa_f1', model: Optional[str] = None) -> Dict:
        """
        최적 조합 반환

        Args:
            metric: 기준 지표 (default: 'pa_f1')
            model: 특정 모델만 필터링 (optional)

        Returns:
            최적 조합 정보
        """
        if not self._evaluation_records:
            return {}

        df = pd.DataFrame(self._evaluation_records)

        if model:
            df = df[df['model'] == model]

        if metric not in df.columns:
            print(f"⚠️ Metric '{metric}' not found.")
            return {}

        best_idx = df[metric].idxmax()
        return df.loc[best_idx].to_dict()

    def print_summary(self):
        """현재 실험 현황 출력"""
        print("=" * 60)
        print("📊 Experiment Summary")
        print("=" * 60)

        print(f"\n[Training Records] {len(self._training_records)} experiments")
        if self._training_records:
            df = pd.DataFrame(self._training_records)
            print(df[['model', 'preprocess', 'dataset', 'training_time_sec']].to_string(index=False))

        print(f"\n[Evaluation Records] {len(self._evaluation_records)} experiments")
        if self._evaluation_records:
            df = pd.DataFrame(self._evaluation_records)
            cols = ['model', 'preprocess', 'postprocess', 'pa_f1']
            cols = [c for c in cols if c in df.columns]
            print(df[cols].to_string(index=False))

        print("\n" + "=" * 60)

    def export_to_excel(self, filename: str = 'experiment_results.xlsx'):
        """
        Excel 파일로 내보내기 (보고서용)

        Args:
            filename: 출력 파일명
        """
        try:
            import openpyxl
        except ImportError:
            print("⚠️ openpyxl not installed. Exporting to CSV instead.")
            self._export_to_csv(filename.replace('.xlsx', ''))
            return

        filepath = self.results_dir / filename

        with pd.ExcelWriter(filepath, engine='openpyxl') as writer:
            # 전체 결과
            if self._evaluation_records:
                pd.DataFrame(self._evaluation_records).to_excel(
                    writer, sheet_name='Full Results', index=False
                )

            # 보고서 테이블
            report = self.generate_report()
            for name, df in report.items():
                if isinstance(df, pd.DataFrame) and not df.empty:
                    df.to_excel(writer, sheet_name=name[:31])  # Excel 시트명 31자 제한

            # 학습 로그
            if self._training_records:
                pd.DataFrame(self._training_records).to_excel(
                    writer, sheet_name='Training Log', index=False
                )

        print(f"✅ Exported to: {filepath}")

    def _export_to_csv(self, basename: str):
        """
        CSV 파일로 내보내기 (openpyxl 없을 때 대체용)

        Args:
            basename: 출력 파일 기본 이름 (확장자 제외)
        """
        # 전체 평가 결과
        if self._evaluation_records:
            eval_path = self.results_dir / f'{basename}_evaluation.csv'
            pd.DataFrame(self._evaluation_records).to_csv(eval_path, index=False)
            print(f"✅ Exported: {eval_path}")

        # 학습 로그
        if self._training_records:
            train_path = self.results_dir / f'{basename}_training.csv'
            pd.DataFrame(self._training_records).to_csv(train_path, index=False)
            print(f"✅ Exported: {train_path}")
