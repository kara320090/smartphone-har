import os
import time
import numpy as np
from tqdm import tqdm

# 9개 센서 채널 순서 고정[cite: 1]
CHANNELS = [
    f"{base}_{axis}"
    for base in ["body_acc", "body_gyro", "total_acc"]
    for axis in ["x", "y", "z"]
]

# 검증 피험자 ID 고정 (5명)[cite: 1]
VAL_SUBJECTS = [1, 6, 14, 21, 23]


def read_txt_fast(file_path, expected_cols=None):
    """
    Pandas 의존성 없이 Python 내장 파서 + NumPy로 가장 빠르고 안전하게 읽습니다.
    (텍스트 파일의 공백/줄바꿈을 1차원 float 배열로 읽은 뒤 2차원으로 reshape)
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"[경로 오류] 파일이 존재하지 않습니다: {file_path}")
    
    # 텍스트 파일 전체를 읽어 공백 기준으로 숫자 파싱
    with open(file_path, "r", encoding="utf-8") as f:
        # np.fromstring/fromiter를 활용한 고속 변환
        data = np.fromstring(f.read(), sep=" ", dtype=np.float32)
        
    if expected_cols is not None:
        data = data.reshape(-1, expected_cols)
    return data


def read_signals(data_dir, split_type):
    """
    Inertial Signals 폴더 내 9개 채널 파일을 읽어 (N, 128, 9) 3D 배열을 구성합니다.[cite: 1]
    """
    signals = []
    signals_dir = os.path.join(data_dir, split_type, "Inertial Signals")
    
    print(f"\n>> [{split_type.upper()}] 9개 시계열 센서 파일 로드 시작...")
    # 터미널에 9개 파일 각각 진행 상황 표시
    for ch in tqdm(CHANNELS, desc=f"{split_type} 채널 로딩", ncols=80):
        file_path = os.path.join(signals_dir, f"{ch}_{split_type}.txt")
        sig = read_txt_fast(file_path, expected_cols=128)
        signals.append(sig)
        
    stacked = np.stack(signals, axis=-1)
    return stacked


def load_dataset(data_dir):
    """
    전체 HAR 데이터셋 로드 및 피험자 기준 분할[cite: 1]
    """
    t_start = time.time()
    print("=" * 65)
    print("  [UCI HAR 데이터셋 로딩 파이프라인 가동]")
    print(f"  기준 경로: {data_dir}")
    print("=" * 65)

    # 1. 시계열 데이터 로드 (train: 7352x128x9, test: 2947x128x9)[cite: 1]
    Xs_train_raw = read_signals(data_dir, "train")
    Xs_test = read_signals(data_dir, "test")

    # 2. 가공 특징(561개) 로드[cite: 1]
    print("\n>> 561개 가공 특징(X_train, X_test) 로드 중...")
    Xf_train_raw = read_txt_fast(os.path.join(data_dir, "train", "X_train.txt"), expected_cols=561)
    Xf_test = read_txt_fast(os.path.join(data_dir, "test", "X_test.txt"), expected_cols=561)

    # 3. 라벨(y) 및 피험자(subject) 로드[cite: 1]
    print(">> 라벨(y) 및 피험자 ID 로드 중...")
    y_train_raw = read_txt_fast(os.path.join(data_dir, "train", "y_train.txt")).astype(np.int64) - 1
    y_test = read_txt_fast(os.path.join(data_dir, "test", "y_test.txt")).astype(np.int64) - 1
    
    sub_train_raw = read_txt_fast(os.path.join(data_dir, "train", "subject_train.txt")).astype(np.int64)
    sub_test = read_txt_fast(os.path.join(data_dir, "test", "subject_test.txt")).astype(np.int64)

    # 4. sample_id 부여[cite: 1]
    sid_train_raw = np.array([f"train:{i:06d}" for i in range(len(y_train_raw))])
    sid_test = np.array([f"test:{i:06d}" for i in range(len(y_test))])

    # 5. 피험자 분할 마스크 (Train 16명 / Val 5명)[cite: 1]
    val_mask = np.isin(sub_train_raw, VAL_SUBJECTS)
    fit_mask = ~val_mask

    # 정합성 검증[cite: 1]
    assert np.sum(fit_mask) == 5577, f"학습 표본 수 오류: {np.sum(fit_mask)}"
    assert np.sum(val_mask) == 1775, f"검증 표본 수 오류: {np.sum(val_mask)}"
    assert len(y_test) == 2947, f"시험 표본 수 오류: {len(y_test)}"

    dataset = {
        "fit": {
            "Xs": Xs_train_raw[fit_mask],
            "Xf": Xf_train_raw[fit_mask],
            "y": y_train_raw[fit_mask],
            "subject": sub_train_raw[fit_mask],
            "sample_id": sid_train_raw[fit_mask]
        },
        "val": {
            "Xs": Xs_train_raw[val_mask],
            "Xf": Xf_train_raw[val_mask],
            "y": y_train_raw[val_mask],
            "subject": sub_train_raw[val_mask],
            "sample_id": sid_train_raw[val_mask]
        },
        "test": {
            "Xs": Xs_test,
            "Xf": Xf_test,
            "y": y_test,
            "subject": sub_test,
            "sample_id": sid_test
        }
    }

    print("\n" + "=" * 65)
    print(f"✓ 로딩 성공! (총 소요 시간: {time.time() - t_start:.2f}초)")
    print(f"  - 학습 데이터 (fit) : 시계열 {dataset['fit']['Xs'].shape} | 특징 {dataset['fit']['Xf'].shape}")
    print(f"  - 검증 데이터 (val) : 시계열 {dataset['val']['Xs'].shape} | 특징 {dataset['val']['Xf'].shape}")
    print(f"  - 시험 데이터 (test): 시계열 {dataset['test']['Xs'].shape} | 특징 {dataset['test']['Xf'].shape}")
    print("=" * 65)
    return dataset


# data.py 단독 실행 테스트용
if __name__ == "__main__":
    DATA_PATH = r"C:\Users\wolah\Desktop\human+activity+recognition+using+smartphones\UCI HAR Dataset"
    data = load_dataset(DATA_PATH)