# Drug Discovery Prediction

SMILES 기반 분자 구조 정보로 약물 활성도(`pIC50`, `Inhibition`)를 예측한 신약 개발 회귀 모델링 프로젝트입니다. RDKit 분자 지문/기술자, XGBoost, AutoGluon, ChemBERTa, GNN, MolCLR를 비교하며 분자 표현 방식과 모델 선택이 예측 성능에 미치는 영향을 실험했습니다.

## 프로젝트 개요

분자 예측 문제에서는 같은 SMILES라도 어떤 방식으로 수치화하느냐에 따라 모델 성능이 크게 달라집니다. 이 프로젝트는 단일 모델을 만드는 데서 끝내지 않고, fingerprint 기반 baseline부터 pretrained molecular language model, graph representation, distribution smoothing까지 단계적으로 실험하며 성능 개선 방향을 확인했습니다.

원본 데이터, 제출 파일, 학습 로그, checkpoint, 모델 weight는 저장소에 포함하지 않았습니다. 공개 저장소에는 재현 가능한 코드, 정리된 노트북, 결과 시각화 자료만 포함했습니다.

## 기술 스택

- Python, Jupyter Notebook
- pandas, NumPy, scikit-learn
- RDKit
- XGBoost, LightGBM, CatBoost, RandomForest
- AutoGluon
- PyTorch, PyTorch Geometric
- Hugging Face Transformers, ChemBERTa/RoBERTa 계열 SMILES encoder
- MolCLR
- Matplotlib, Seaborn

## 분석 목표

- SMILES 문자열을 분자 지문과 분자 기술자로 변환
- fingerprint/descriptor 기반 tree model을 baseline으로 설정
- ChemBERTa와 RoBERTa 계열 pretrained model을 회귀 문제에 적용
- GNN/MolCLR 기반 graph representation 실험
- LDS/FDS를 활용해 target 분포 불균형과 feature smoothing 효과 확인
- 실제값과 예측값 산점도, feature importance, smoothing 시각화로 결과 해석

## 성능 개선 과정

### 1. Fingerprint baseline

RDKit Morgan fingerprint를 생성하고 XGBoost 회귀 모델을 적용했습니다. 빠르게 학습 가능한 baseline을 먼저 만든 뒤, 이후 모델들의 성능을 비교하는 기준으로 사용했습니다.

### 2. Similarity matrix + autoencoder

Morgan fingerprint로 분자 간 similarity matrix를 만들고, autoencoder로 압축한 뒤 XGBoost에 입력했습니다. 고차원 fingerprint의 노이즈를 줄이면서 분자 간 유사도 정보를 보존하는 방향의 실험입니다.

구현 파일: `src/xgboost_baseline.py`

### 3. Fingerprint 크기 확장 및 추론 pipeline

128-bit fingerprint 실험 이후 더 큰 fingerprint 차원을 사용해 테스트 예측 pipeline을 구성했습니다. 생성되는 제출 파일과 예측 결과는 `outputs/`에 저장되며 Git에는 포함되지 않습니다.

구현 파일: `src/prediction_pipeline.py`

### 4. ChemBERTa / RoBERTa 기반 transfer learning

SMILES 문자열을 pretrained molecular language model에 입력하고 regression head를 붙여 `pIC50` 또는 `Inhibition`을 예측했습니다. Transformer embedding만 사용할 때의 한계와 descriptor 기반 모델 대비 안정성을 함께 확인했습니다.

관련 파일:

- `notebooks/legacy_transformer_finetuning.ipynb`
- `notebooks/chemberta_finetuning.ipynb`
- `src/transfer_model.py`

### 5. GNN / MolCLR representation

분자를 graph로 변환하고 GNN 및 MolCLR 기반 representation을 적용했습니다. 원자/결합 구조 정보를 반영하는 graph embedding이 tabular feature와 비교해 어떤 장단점을 갖는지 확인했습니다.

관련 파일:

- `notebooks/gnn_regression.ipynb`
- `notebooks/molclr_fds_experiment.ipynb`
- `src/molclr/`

### 6. LDS / FDS 기반 분포 보정

약물 활성도 데이터는 특정 target 구간에 샘플이 몰리고 희귀 구간은 부족한 경우가 많습니다. LDS(Label Distribution Smoothing)와 FDS(Feature Distribution Smoothing)를 적용해 target imbalance와 feature 통계 불안정을 완화하는 방향을 실험했습니다.

## 정량 지표 요약

| 실험 | 핵심 수치 | 해석 |
| --- | --- | --- |
| XGBoost baseline | 5-fold Competition Score: `0.7465`, `0.7136`, `0.7477`, `0.7038`, `0.7384` / 평균 `0.7300`, 표준편차 `0.0180` | fold 간 편차가 크지 않아 baseline으로 안정적입니다. |
| AutoGluon ensemble | leaderboard 기준 최고 `score_test = 0.9926` (`KNeighborsDist_BAG_L1`), 최고 `score_val = 0.7673` (`WeightedEnsemble_L3`) | 자동 모델 탐색과 stacking이 강한 후보군을 빠르게 찾았습니다. |
| GNN + LDS/FDS | 최고 Validation Score `0.4911` at epoch 3 / Train Loss `9.5136 -> 8.3279` | loss는 감소했지만 validation score는 초반 이후 정체되어 regularization이 중요했습니다. |
| ChemBERTa transfer | 예측값 대부분 약 `10~65%` 범위, 실제값은 `0~100%` 범위 | 고활성 구간에서 과소예측 경향이 보여 데이터 크기와 fine-tuning 안정성이 중요했습니다. |
| Feature importance | 상위 feature importance 약 `0.0185`, `0.0170`, `0.0160`, `0.0152`, `0.0144` | 특정 fingerprint/descriptor feature에 예측 기여도가 집중되었습니다. |

Competition Score는 `0.5 * (1 - NRMSE) + 0.5 * Pearson` 방식으로 계산했습니다.

## 결과 및 시각화

### XGBoost baseline 예측 결과

![XGBoost real vs predicted](assets/model_performance.png)

- 실제값/예측값 축 범위: 약 `0~100`
- 5-fold Competition Score 평균: `0.7300`
- 5-fold 표준편차: `0.0180`
- 산점도가 `y=x` 기준선 주변에 밀집해 baseline 모델이 전반적인 target scale을 잘 따라갑니다.
- 일부 구간에서는 기준선에서 벗어난 outlier가 있어, 후속 실험에서 feature 보강과 ensemble을 적용했습니다.

### AutoGluon ensemble 예측 결과

![AutoGluon real vs predicted](assets/autogluon_model_performance.png)

- AutoGluon 학습 데이터 행 수: `1,671`
- 입력 feature 수: `274`
- 최고 `score_test`: `0.9926` (`KNeighborsDist_BAG_L1`)
- 최고 `score_val`: `0.7673` (`WeightedEnsemble_L3`)
- `WeightedEnsemble_L2`는 `score_test = 0.9568`, `score_val = 0.7667`을 기록했습니다.
- 실제값이 높은 구간(`80~100`)에서는 일부 과소예측이 보이지만, 전체적으로 증가 방향은 유지됩니다.

### ChemBERTa transfer learning 결과

![ChemBERTa transfer model real vs predicted](assets/model_performance_transfer.png)

- 실제값 범위: 약 `0~100%`
- 예측값 범위: 대부분 약 `10~65%`
- 고활성 구간(`80~100%`)에서도 예측값이 `50~65%` 근처에 머무는 샘플이 있어 과소예측 경향이 나타났습니다.
- pretrained SMILES encoder는 representation 관점에서 유용하지만, 작은 데이터셋에서는 tree 기반 descriptor 모델보다 안정성이 떨어질 수 있음을 확인했습니다.

### Label Distribution Smoothing

![LDS effect visualization](assets/lds_effect_visualization.png)

- logit 변환 target 범위: 약 `-21~6`
- 원본 label distribution의 최빈 구간: `400개 이상`
- LDS 적용 후 effective distribution peak: 약 `180~190`
- 샘플이 몰린 구간의 영향력을 완화하고, 희귀 target 구간도 학습 손실에 반영되도록 조정했습니다.

### Feature Distribution Smoothing

![FDS effect visualization](assets/fds_effect_visualization.png)

- 시각화 기준 epoch: `2`
- 원본 feature mean 범위: 약 `2.4~4.9`
- FDS 적용 후 smoothed mean 범위: 약 `0.3~1.7`
- bucket 간 feature mean 변동폭이 줄어들어, target 구간별 feature 통계가 더 완만해지는 효과를 확인했습니다.

### Feature importance

![Feature importance](assets/feature_importance.png)

- 상위 feature id: `232`, `219`, `108`, `268`, `55`
- 상위 importance 값: 약 `0.0185`, `0.0170`, `0.0160`, `0.0152`, `0.0144`
- top 20 feature importance 범위: 약 `0.0078~0.0185`
- fingerprint/descriptor 기반 모델에서 일부 feature가 예측에 더 크게 기여하는 것을 확인했습니다.

## 폴더 구조

```text
.
|-- assets/
|   |-- autogluon_model_performance.png
|   |-- fds_effect_visualization.png
|   |-- feature_importance.png
|   |-- lds_effect_visualization.png
|   |-- model_performance.png
|   `-- model_performance_transfer.png
|-- docs/
|   |-- DATA.md
|   `-- THIRD_PARTY_NOTICES.md
|-- notebooks/
|   |-- chemberta_finetuning.ipynb
|   |-- fingerprint_descriptor_baseline.ipynb
|   |-- gnn_regression.ipynb
|   |-- legacy_transformer_finetuning.ipynb
|   `-- molclr_fds_experiment.ipynb
|-- src/
|   |-- molclr/
|   |-- my_metrics.py
|   |-- my_scorers.py
|   |-- prediction_pipeline.py
|   |-- transfer_model.py
|   `-- xgboost_baseline.py
|-- environment.yml
|-- requirements.txt
`-- README.md
```

## 데이터 구성

데이터 파일은 저장소에 포함하지 않습니다. 실행 시에는 다음 구조로 로컬에 배치합니다.

```text
data/
|-- train.csv
|-- test.csv
|-- sample_submission.csv
|-- ChEMBL_ASK1(IC50).csv
|-- Pubchem_ASK1.csv
`-- CAS_KPBMA_MAP3K5_IC50s.xlsx
```

Git에서 제외되는 항목:

- 원본 competition data
- 제출용 CSV
- TensorBoard logs
- checkpoint 및 model weight
- AutoGluon/CatBoost 산출물
- cached feature matrix

## 설치 방법

기본 Python 환경:

```bash
pip install -r requirements.txt
```

Conda 환경:

```bash
conda env create -f environment.yml
conda activate autogluon
```

PyTorch Geometric은 로컬 PyTorch/CUDA 버전에 맞는 wheel 설치가 필요할 수 있습니다.

## 실행 방법

먼저 private data를 `data/` 폴더에 배치합니다.

XGBoost autoencoder baseline:

```bash
python src/xgboost_baseline.py
```

예측 pipeline:

```bash
python src/prediction_pipeline.py
```

ChemBERTa transfer learning:

```bash
python src/transfer_model.py
```

노트북 실행:

```bash
jupyter notebook notebooks
```

MolCLR 실행:

```bash
cd src/molclr
python molclr.py
python finetune.py
```

## 주요 산출물

- XGBoost, AutoGluon, ChemBERTa real-vs-predicted 시각화
- LDS/FDS 기반 target 및 feature smoothing 시각화
- feature importance 기반 모델 해석 결과
- 공개 가능한 실험 노트북
- 데이터와 모델 weight를 제외한 재현용 코드

## 결론

분자 예측 문제에서는 RDKit fingerprint와 descriptor를 활용한 tree 기반 모델이 강력한 baseline으로 작동했습니다. XGBoost는 5-fold 평균 Competition Score `0.7300`을 기록했고, AutoGluon ensemble은 validation 기준 `0.7673` 수준의 score를 보여 자동화된 모델 탐색의 가능성을 확인했습니다.

반면 ChemBERTa 기반 transfer learning은 예측값이 중간 범위로 수렴하는 경향이 있어, 데이터 수가 충분하지 않거나 target 분포가 불균형할 때 pretrained model이 항상 더 좋은 결과를 보장하지는 않았습니다. GNN/MolCLR 실험은 graph representation의 가능성을 확인했지만, validation score가 초반 이후 정체되어 모델 복잡도와 regularization 조정이 필요했습니다.

최종적으로 이 프로젝트는 fingerprint/descriptor baseline, ensemble, transfer learning, graph learning, distribution smoothing을 한 흐름에서 비교하며 약물 활성 예측 모델의 개선 방향을 정리한 실험입니다.

## 개선 가능성

- 실험별 metric을 JSON/CSV로 자동 저장
- MLflow 또는 W&B 기반 실험 추적 도입
- scaffold split 기반 외부 검증 추가
- feature importance와 GNNExplainer를 함께 사용한 해석 강화
- ChemBERTa fine-tuning 시 learning rate, freezing 전략, target scaling 재검토

## Third-Party Notice

`src/molclr`의 MolCLR 구현은 분자 contrastive representation learning을 위한 공개 MolCLR 프로젝트를 기반으로 합니다. 자세한 내용은 `docs/THIRD_PARTY_NOTICES.md`와 `src/molclr/LICENSE`를 참고하세요.
