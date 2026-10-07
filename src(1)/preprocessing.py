import time
import numpy as np
from sklearn.decomposition import PCA


class DataPreprocessor:
    def __init__(self):
        self.seq_mean = None
        self.seq_std = None
        self.feat_mean = None
        self.feat_std = None
        self.pca_95 = None

    def fit(self, Xs_fit, Xf_fit):
        """학습 데이터(fit 16명)만 사용하여 통계 산출"""
        t0 = time.time()
        print("\n>> 전처리 표준화 통계 산출 및 PCA 피팅 중...")
        
        # 시계열 채널 표준화 통계 (N, T 축 통합)
        self.seq_mean = np.mean(Xs_fit, axis=(0, 1), keepdims=True)
        self.seq_std = np.std(Xs_fit, axis=(0, 1), keepdims=True)
        self.seq_std = np.maximum(self.seq_std, 1e-8)

        # 561 특징 열별 표준화 통계
        self.feat_mean = np.mean(Xf_fit, axis=0, keepdims=True)
        self.feat_std = np.std(Xf_fit, axis=0, keepdims=True)
        self.feat_std = np.maximum(self.feat_std, 1e-8)

        # PCA 100차원 피팅 (95% 분산 보존)
        Xf_fit_norm = (Xf_fit - self.feat_mean) / self.feat_std
        self.pca_95 = PCA(n_components=100, random_state=2026)
        self.pca_95.fit(Xf_fit_norm)

        pc1_2_var = np.sum(self.pca_95.explained_variance_ratio_[:2]) * 100
        total_var = np.sum(self.pca_95.explained_variance_ratio_) * 100

        print(f"✓ 전처리 피팅 완료 ({time.time() - t0:.2f}초)")
        print(f"   - PC1 + PC2 설명 분산    : {pc1_2_var:.2f}% (기대치: 약 58.35%)")
        print(f"   - 100개 주성분 누적 분산 : {total_var:.2f}% (기대치: 95% 이상)")

    def transform_sequence(self, Xs):
        return (Xs - self.seq_mean) / self.seq_std

    def transform_features(self, Xf):
        return (Xf - self.feat_mean) / self.feat_std

    def transform_pca(self, Xf):
        Xf_norm = self.transform_features(Xf)
        return self.pca_95.transform(Xf_norm)

    def save_stats(self, filepath="preprocess.npz"):
        np.savez(
            filepath,
            seq_mean=self.seq_mean,
            seq_std=self.seq_std,
            feat_mean=self.feat_mean,
            feat_std=self.feat_std,
            pca_components=self.pca_95.components_,
            pca_mean=self.pca_95.mean_,
        )
        print(f"  [파일 저장] {filepath}")

    def load_stats(self, filepath="preprocess.npz"):
        data = np.load(filepath)
        self.seq_mean = data["seq_mean"]
        self.seq_std = data["seq_std"]
        self.feat_mean = data["feat_mean"]
        self.feat_std = data["feat_std"]