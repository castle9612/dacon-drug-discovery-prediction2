import pandas as pd
import numpy as np

import argparse
from yaml import parse
from tqdm import tqdm

import torch
import torch.nn as nn

from torch.utils.data import Dataset, DataLoader

import xgboost as xgb

from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem

from sklearn.model_selection import train_test_split
from sklearn.metrics import *
from sklearn.model_selection import KFold

import sys
from pathlib import Path
import warnings
warnings.simplefilter(action='ignore',category=FutureWarning)
parser = argparse.ArgumentParser()
parser.add_argument('--input-size', type=int, default=2083)
parser.add_argument('--batch-size', type=int, default=16)
parser.add_argument('--epochs', type=int, default=150)

args = parser.parse_args()

DATA_DIR = Path("data")
OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

#smile 차원조정
def smiles2morgan(s, nBits=4000, radius=2):
    try:
        mol = Chem.MolFromSmiles(s)
        features_vec = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=nBits)
        features = np.zeros((1,))
        DataStructs.ConvertToNumpyArray(features_vec, features)
    except:
        print('rdkit not found this smiles for morgan: ' + s + ' convert to all 0 features')
        features = np.zeros((nBits,))
    return features

def tanimoto_similarity(vector1, vector2):
    min_length = min(len(vector1), len(vector2))
    vector1 = vector1[:min_length]
    vector2 = vector2[:min_length]
    intersection = np.sum(vector1 * vector2)
    union = np.sum(vector1 + vector2) - intersection
    return intersection / union

def cal_russel(fp1, fp2):
    fp1 = ''.join(fp1.astype(str))
    fp1 = DataStructs.CreateFromBitString(fp1)
    fp2 = ''.join(fp2.astype(str))
    fp2 = DataStructs.CreateFromBitString(fp2)
    similarity = DataStructs.RusselSimilarity(fp1, fp2)
    return similarity

def euclidean_distance_similarity(vector1, vector2):
    distance = np.linalg.norm(vector1 - vector2)
    dimension = len(vector1)
    similarity = 1 - (distance / np.sqrt(dimension))
    return similarity

# similarity matrix 생성
def create_similarity_matrix(one_hot_vectors1, one_hot_vectors2, criterion):
    num_vectors1 = len(one_hot_vectors1)
    num_vectors2 = len(one_hot_vectors2)
    similarity_matrix = np.zeros((num_vectors1, num_vectors2))

    for i in tqdm(range(num_vectors1), desc="Calculating Similarity"):
        for j in range(num_vectors2):
            similarity_matrix[i, j] = criterion(one_hot_vectors1[i], one_hot_vectors2[j])

    return similarity_matrix

class medicineDataset(Dataset):
    def __init__(self, x):
        self.x = x
    
    def __len__(self):
        return self.x.shape[0]
    
    def __getitem__(self, i):
        x = self.x[i,:]
        return x

loss_mse = torch.nn.MSELoss()

dataset = pd.read_csv(DATA_DIR / 'train.csv')
print(dataset)
#set y(label)
y = dataset['pIC50']
 
smile_total_set = dataset['Smiles']

morgan_total = pd.DataFrame([smiles2morgan(s) for s in tqdm(smile_total_set)])
data_all = np.array(morgan_total)

# print("creating russel similarity matrix")
# similarity_matrix = create_similarity_matrix(data_all, data_all, tanimoto_similarity)
# print("Similarity Matrix:")
# # print(similarity_matrix)
# X = similarity_matrix
X = data_all
X = pd.DataFrame(X)
print(f"Shape of morgan_total: {data_all.shape}")

dataset1 = pd.read_csv(DATA_DIR / 'test.csv')
print(dataset1)
 
smile_total_set1 = dataset1['Smiles']

morgan_total1 = pd.DataFrame([smiles2morgan(s) for s in tqdm(smile_total_set1)])
data_all1 = np.array(morgan_total1)

# print("creating russel similarity matrix")
# similarity_matrix1 = create_similarity_matrix(data_all1, data_all1, tanimoto_similarity)
# print("Similarity Matrix:")
# print(similarity_matrix1)
# X1 = similarity_matrix1
X1 = data_all1

from sklearn.model_selection import KFold
k_fold = KFold(n_splits=5, shuffle=True, random_state=2024)

X_test = X1

lr_list = []
md_list = []
cb_list = []
l1_list = []

mse_list_train = []
mse_list = []


for fold, (train_idx, valid_idx) in enumerate(k_fold.split(X, y)):
    X_train, X_valid = X.iloc[train_idx,:],X.iloc[valid_idx,:]
    y_train, y_valid = y.iloc[train_idx], y.iloc[valid_idx]

    model = xgb.XGBRegressor(learning_rate=0.1, max_depth=5, colsample_bytree=0.7, alpha=0.5, objective='reg:squarederror',random_state=2024)
    model.fit(X_train, y_train)
    y_train_pred = model.predict(X_train)
    y_pred = model.predict(X_valid)

    print("===================================================")
    print("kfold = ", fold + 1)
    print(f"train mse: {mean_squared_error(y_train, y_train_pred)}")
    print(f"valid mse: {mean_squared_error(y_valid, y_pred)}")

test_pc = model.predict(X_test)

def pIC50_to_IC50(pic50_values):
    """Convert pIC50 values to IC50 (nM)."""
    return 10 ** (9 - pic50_values)

submit = pd.read_csv(DATA_DIR / 'sample_submission.csv')
submit['IC50_nM'] = pIC50_to_IC50(test_pc)
print(submit.head())

submit.to_csv(OUTPUT_DIR / 'xgboost_4096_ae.csv', index=False)

