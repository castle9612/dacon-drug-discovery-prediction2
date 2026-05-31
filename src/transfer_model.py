import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from pathlib import Path

# 경고 메시지 무시 설정
import warnings
warnings.filterwarnings('ignore')

DATA_DIR = Path("data")
OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

# Hugging Face Transformers 라이브러리
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from datasets import Dataset

# RDKit (SMILES 유효성 검사 및 정규화용)
from rdkit import Chem

# --- 0. 한글 폰트 설정 (matplotlib) ---
try:
    plt.rc('font', family='NanumBarunGothicOTF') 
    # 또는 폰트 파일 이름에 따라 'NanumBarunGothicOTF' 등일 수 있습니다.
    print("'NanumBarunGothic' 폰트 설정이 완료되었습니다.")
except Exception as e:
    print(f"폰트 설정 실패: {e}")
    print("Malgun Gothic을 기본 폰트로 설정합니다.")
    plt.rc('font', family='Malgun Gothic') # 윈도우 기본 폰트로 대체

plt.rcParams['axes.unicode_minus'] = False # 마이너스 폰트 깨짐 방지


# --- 1. 커스텀 평가지표 함수 정의 ---
def normalized_rmse(y_true, y_pred):
    """Normalized Root Mean Squared Error"""
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    # y_true의 범위가 0이 되는 경우를 방지하기 위해 작은 값(epsilon)을 더합니다.
    range_y = np.max(y_true) - np.min(y_true)
    return rmse / (range_y + 1e-8)

def pearson_correlation(y_true, y_pred):
    """Pearson Correlation Coefficient"""
    corr = np.corrcoef(y_true, y_pred)[0, 1]
    # NaN 값이 나올 경우 0으로 처리
    return np.nan_to_num(corr)

def competition_score(y_true, y_pred):
    """대회 평가지표 (NRMSE, Pearson)"""
    # NRMSE는 1을 넘지 않도록 clip
    nrmse = min(normalized_rmse(y_true, y_pred), 1.0)
    pearson = pearson_correlation(y_true, y_pred)
    return 0.5 * (1 - nrmse) + 0.5 * pearson


# --- 2. 데이터 로드 및 전처리 ---
try:
    train_df = pd.read_csv(DATA_DIR / 'train.csv')
    test_df = pd.read_csv(DATA_DIR / 'test.csv')
    submission_df = pd.read_csv(DATA_DIR / 'sample_submission.csv')
except FileNotFoundError:
    print("data/train.csv, data/test.csv, data/sample_submission.csv 파일이 있는지 확인해주세요.")
    # 예시 데이터 생성
    train_df = pd.DataFrame({'ID': [f'train_{i}' for i in range(100)], 'Canonical_Smiles': ['CCO', 'CCC', 'CCN', 'CCS'] * 25, 'Inhibition': np.random.rand(100) * 100})
    test_df = pd.DataFrame({'ID': [f'test_{i}' for i in range(50)], 'Canonical_Smiles': ['CC', 'CO', 'CN'] * 16 + ['CS', 'CF']})
    submission_df = pd.DataFrame({'ID': test_df['ID'], 'Inhibition': 0.0})

print("데이터 로드 완료")

# SMILES 정규화
train_df.dropna(subset=['Canonical_Smiles'], inplace=True)
train_df['Canonical_Smiles'] = train_df['Canonical_Smiles'].apply(lambda s: Chem.MolToSmiles(Chem.MolFromSmiles(s)) if Chem.MolFromSmiles(s) else None)
train_df.dropna(subset=['Canonical_Smiles'], inplace=True)

# 타겟 변수 스케일링
scaler = MinMaxScaler()
train_df['scaled_inhibition'] = scaler.fit_transform(train_df[['Inhibition']])


# --- 3. 사전 학습 모델 및 토크나이저 로드 ---
model_name = "DeepChem/ChemBERTa-77M-MLM"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=1, 
    problem_type="regression"
)
print("ChemBERTa 모델 및 토크나이저 로드 완료")


# --- 4. 데이터셋 준비 ---
train_texts, val_texts, train_labels_scaled, val_labels_scaled = train_test_split(
    train_df['Canonical_Smiles'].tolist(),
    train_df['scaled_inhibition'].tolist(),
    test_size=0.15,
    random_state=42
)

# 데이터셋 객체 생성 (코드는 이전과 동일)
train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=128)
val_encodings = tokenizer(val_texts, truncation=True, padding=True, max_length=128)
test_encodings = tokenizer(test_df['Canonical_Smiles'].tolist(), truncation=True, padding=True, max_length=128)

class InhibitionDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels=None):
        self.encodings = encodings
        self.labels = labels
    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        if self.labels:
            item['labels'] = torch.tensor(self.labels[idx], dtype=torch.float)
        return item
    def __len__(self):
        return len(self.encodings['input_ids'])

train_dataset = InhibitionDataset(train_encodings, train_labels_scaled)
val_dataset = InhibitionDataset(val_encodings, val_labels_scaled)
test_dataset = InhibitionDataset(test_encodings)


# --- 5. 모델 학습 (Fine-tuning) ---

# [수정됨] compute_metrics 함수에서 커스텀 평가지표 계산
def compute_metrics(p):
    preds_scaled = p.predictions.flatten()
    labels_scaled = p.label_ids.flatten()
    
    # 중요: 원래 스케일로 복원 후 평가지표 계산
    preds_original = scaler.inverse_transform(preds_scaled.reshape(-1, 1)).flatten()
    labels_original = scaler.inverse_transform(labels_scaled.reshape(-1, 1)).flatten()
    
    # 커스텀 지표 계산
    score = competition_score(labels_original, preds_original)
    nrmse = normalized_rmse(labels_original, preds_original)
    pearson = pearson_correlation(labels_original, preds_original)
    
    # Trainer에게 전달할 딕셔너리 생성
    return {
        "competition_score": score,
        "nrmse": nrmse,
        "pearson": pearson,
        "rmse": np.sqrt(mean_squared_error(labels_original, preds_original)) # 참고용으로 일반 RMSE도 추가
    }

# [수정됨] TrainingArguments에서 평가지표 기준 변경
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=100,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=32,
    learning_rate=1e-5,
    warmup_steps=50,
    weight_decay=0.1,
    logging_dir='./logs',
    logging_steps=50,
    evaluation_strategy="epoch",  # 매 에포크마다 평가
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="competition_score", # 최적 모델 선택 기준
    greater_is_better=True,                 # competition_score는 높을수록 좋음
    report_to="none"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics, # 수정한 compute_metrics 함수 전달
)

print("모델 파인튜닝 시작")
trainer.train()

# 학습이 끝나면, 각 에포크별로 계산된 평가지표를 history에서 확인할 수 있습니다.
print("\n--- 학습 과정 로그 ---")
print(pd.DataFrame(trainer.state.log_history))


# --- 6. 모델 성능 검증 및 시각화 ---
print("\n--- 모델 성능 검증 및 시각화 ---")
# (이하 코드는 이전과 동일)
val_predictions = trainer.predict(val_dataset)
val_preds_scaled = val_predictions.predictions.flatten()
y_val_original = scaler.inverse_transform(np.array(val_labels_scaled).reshape(-1, 1)).flatten()
y_val_pred_original = scaler.inverse_transform(val_preds_scaled.reshape(-1, 1)).flatten()

plt.figure(figsize=(10, 6))
plt.scatter(y_val_original, y_val_pred_original, alpha=0.5, label='예측값')
plt.plot([0, 100], [0, 100], 'r--', label='이상적인 예측 (y=x)')
plt.xlabel('실제 저해율 (%)', fontsize=14)
plt.ylabel('예측 저해율 (%)', fontsize=14)
plt.title('ChemBERTa 모델 검증 성능: 실제값 vs 예측값', fontsize=16)
plt.grid(True)
plt.legend()
plt.axis('equal')
plt.xlim(0, 100)
plt.ylim(0, 100)
plt.tight_layout()
plt.savefig('model_performance_transfer.png')
print("모델 성능 시각화 저장 완료: model_performance_transfer.png")

# ... (이하 제출 파일 생성 코드도 동일)
print("\n--- 최종 예측 및 제출 파일 생성 ---")
test_predictions = trainer.predict(test_dataset)
test_preds_scaled = test_predictions.predictions.flatten()
original_preds = scaler.inverse_transform(test_preds_scaled.reshape(-1, 1)).flatten()
final_preds = np.clip(original_preds, 0, 100)
submission_df['Inhibition'] = final_preds
submission_df.to_csv(OUTPUT_DIR / 'submission.csv', index=False)
print("submission.csv 파일 생성 완료!")
