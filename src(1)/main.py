import os
import sys
import time
import numpy as np

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
  sys.path.append(CURRENT_DIR)

# src. 없이 같은 폴더의 data, preprocessing을 직접 import
from data import load_dataset
from preprocessing import DataPreprocessor

DATA_DIR = r"C:\Users\wolah\Desktop\human+activity+recognition+using+smartphones\UCI HAR Dataset"


def save_mini_samples(dataset, save_path="mini_samples.npz"):
    """1주차 2일차 팀원 배포용 미니 샘플 저장 (학습 60개, 검증 30개)"""
    fit_idx, val_idx = [], []
    for c in range(6):
        fit_idx.extend(np.where(dataset["fit"]["y"] == c)[0][:10])
        val_idx.extend(np.where(dataset["val"]["y"] == c)[0][:5])

    np.savez_compressed(
        save_path,
        fit_Xs=dataset["fit"]["Xs"][fit_idx],
        fit_Xf=dataset["fit"]["Xf"][fit_idx],
        fit_y=dataset["fit"]["y"][fit_idx],
        fit_sample_id=dataset["fit"]["sample_id"][fit_idx],
        val_Xs=dataset["val"]["Xs"][val_idx],
        val_Xf=dataset["val"]["Xf"][val_idx],
        val_y=dataset["val"]["y"][val_idx],
        val_sample_id=dataset["val"]["sample_id"][val_idx],
    )
    print(f"  [파일 저장] {save_path} (팀원 즉시 공유용 샘플)")


def save_processed_cache(dataset, save_path="processed_data.npz"):
    """전체 데이터셋 압축 캐시 저장 (팀원들이 텍스트 파일 대신 1초 만에 로드 가능)"""
    np.savez_compressed(
        save_path,
        fit_Xs=dataset["fit"]["Xs"], fit_Xf=dataset["fit"]["Xf"],
        fit_y=dataset["fit"]["y"], fit_subject=dataset["fit"]["subject"], fit_sample_id=dataset["fit"]["sample_id"],
        val_Xs=dataset["val"]["Xs"], val_Xf=dataset["val"]["Xf"],
        val_y=dataset["val"]["y"], val_subject=dataset["val"]["subject"], val_sample_id=dataset["val"]["sample_id"],
        test_Xs=dataset["test"]["Xs"], test_Xf=dataset["test"]["Xf"],
        test_y=dataset["test"]["y"], test_subject=dataset["test"]["subject"], test_sample_id=dataset["test"]["sample_id"]
    )
    print(f"  [파일 저장] {save_path} (전체 데이터 캐시)")


def save_summary_report(dataset, save_path="data_summary.txt"):
    """데이터 분할 검증 리포트 작성[cite: 1]"""
    labels = ["걷기", "계단 오르기", "계단 내려가기", "앉기", "서기", "눕기"]
    with open(save_path, "w", encoding="utf-8") as f:
        f.write("=== 스마트폰 센서 행동인식 - 데이터 검증 리포트 ===\n\n")
        f.write(f"1. 학습(Fit) : {dataset['fit']['Xs'].shape} / 5,577개\n")
        f.write(f"2. 검증(Val) : {dataset['val']['Xs'].shape} / 1,775개 (피험자: 1, 6, 14, 21, 23)\n")
        f.write(f"3. 시험(Test): {dataset['test']['Xs'].shape} / 2,947개\n\n")
        f.write("4. 행동별 분포:\n")
        for idx, name in enumerate(labels):
            f.write(f"   [{idx}] {name:<10s}: fit={np.sum(dataset['fit']['y']==idx):4d} | val={np.sum(dataset['val']['y']==idx):4d} | test={np.sum(dataset['test']['y']==idx):4d}\n")
    print(f"  [파일 저장] {save_path}")


def main():
    start = time.time()
    
    # 1. 데이터 로드 및 3분할[cite: 1]
    dataset = load_dataset(DATA_DIR)

    # 2. 전처리 통계 피팅 (학습 데이터 기준)[cite: 1]
    preprocessor = DataPreprocessor()
    preprocessor.fit(dataset["fit"]["Xs"], dataset["fit"]["Xf"])

    # 3. 아웃풋 파일 생성 (중복 없이 main에서 일괄 저장)[cite: 1]
    print("\n>> 결과 산출물 파일 저장 중...")
    save_mini_samples(dataset, "mini_samples.npz")
    save_processed_cache(dataset, "processed_data.npz")
    preprocessor.save_stats("preprocess.npz")
    save_summary_report(dataset, "data_summary.txt")

    print("\n" + "=" * 65)
    print(f"✓ 파이프라인 전체 완료! (총 소요시간: {time.time() - start:.2f}초)")
    print("=" * 65)


if __name__ == "__main__":
    main()