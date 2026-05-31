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
def smiles2morgan(s, nBits=128, radius=2):
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

class AUTO_ENCODER(torch.nn.Module):
    def __init__(self,args):
        super(AUTO_ENCODER, self).__init__()

        self.args = args

        self.encoder = nn.Sequential(
            nn.Linear(1952, 1852),
            nn.LeakyReLU(),
            nn.Linear(1852, 1752),
            nn.LeakyReLU(),
            nn.Linear(1752, 1652),
            nn.LeakyReLU(),
            nn.Linear(1652, 1552),
            nn.LeakyReLU(),
            nn.Linear(1552, 1452),
            nn.LeakyReLU(),
            nn.Linear(1452, 1352),
            nn.LeakyReLU(),
            nn.Linear(1352, 1252),
            nn.LeakyReLU(),
            nn.Linear(1252, 1152),
            nn.LeakyReLU(),
            nn.Linear(1152, 1052),
            nn.LeakyReLU(),
            nn.Linear(1052, 952),
            nn.LeakyReLU(),
            nn.Linear(952, 852),
            nn.LeakyReLU(),
            nn.Linear(852, 752),
            nn.LeakyReLU(),
            nn.Linear(752, 652),
            nn.LeakyReLU(),
            nn.Linear(652, 552),
            nn.LeakyReLU(),
            nn.Linear(552, 452),
            nn.LeakyReLU(),
            nn.Linear(452, 352),
            nn.LeakyReLU(),
            nn.Linear(352, 252),
            nn.LeakyReLU(),
            nn.Linear(252, 152),
            nn.LeakyReLU(),
            nn.Linear(152, 128),
            # nn.LeakyReLU(),
            # nn.Linear(100, 64)
            )
        
        self.decoder = nn.Sequential(
            # nn.Linear(64, 100),
            # nn.LeakyReLU(),
            nn.Linear(128, 152),
            nn.LeakyReLU(),
            nn.Linear(152, 252),
            nn.LeakyReLU(),
            nn.Linear(252, 352),
            nn.LeakyReLU(),
            nn.Linear(352, 452),
            nn.LeakyReLU(),
            nn.Linear(452, 552),
            nn.LeakyReLU(),
            nn.Linear(552, 652),
            nn.LeakyReLU(),
            nn.Linear(652, 752),
            nn.LeakyReLU(),
            nn.Linear(752, 852),
            nn.LeakyReLU(),
            nn.Linear(852, 952),
            nn.LeakyReLU(),
            nn.Linear(952, 1052),
            nn.LeakyReLU(),
            nn.Linear(1052, 1152),
            nn.LeakyReLU(),
            nn.Linear(1152, 1252),
            nn.LeakyReLU(),
            nn.Linear(1252, 1352),
            nn.LeakyReLU(),
            nn.Linear(1352, 1452),
            nn.LeakyReLU(),
            nn.Linear(1452, 1552),
            nn.LeakyReLU(),
            nn.Linear(1552, 1652),
            nn.LeakyReLU(),
            nn.Linear(1652, 1752),
            nn.LeakyReLU(),
            nn.Linear(1752, 1852),
            nn.LeakyReLU(),
            nn.Linear(1852, 1952)
        )
    
    def forward(self, x):
        eco = self.encoder(x)
        dco = self.decoder(eco)

        return eco, dco
    
def train_model_auto(data_loader, model, criterion, optimizer):
    num_batches = len(data_loader)
    total_loss = 0
    model.train()

    for X in tqdm(data_loader):
        eco_output, dco_output = model(X)

        loss = criterion(dco_output, X)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
    
    avg_loss = total_loss / num_batches

    print(f"Train Loss: {avg_loss}")

loss_mse = torch.nn.MSELoss()

dataset = pd.read_csv(DATA_DIR / 'train.csv')
print(dataset)
#set y(label)
selected_label_data = dataset['pIC50']


label_data = selected_label_data

 
smile_total_set = dataset['Smiles']

morgan_total = pd.DataFrame([smiles2morgan(s) for s in tqdm(smile_total_set)])
data_all = np.array(morgan_total)

print("creating russel similarity matrix")
similarity_matrix = create_similarity_matrix(data_all, data_all, tanimoto_similarity)
print("Similarity Matrix:")
print(similarity_matrix)
X = similarity_matrix
X = torch.FloatTensor(X)


model_auto_fp = AUTO_ENCODER(args)
optimizer_auto_fp = torch.optim.Adam(model_auto_fp.parameters(), lr=1e-5)
dataset_auto_fp = medicineDataset(X)
dataloader_auto_fp = DataLoader(dataset_auto_fp, batch_size=args.batch_size, shuffle = False)

for epoch in range(5):
    print(f"Epoch {epoch + 1}\n ---------")
    train_model_auto(dataloader_auto_fp, model_auto_fp, loss_mse, optimizer_auto_fp)
    print("")

X_eco, X_dco = model_auto_fp(X)


from sklearn.ensemble import RandomForestClassifier

learning_rate = [0.000001,0.00001,0.0001,0.001,0.01,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
max_depth = [1,3,5,7,9,11,13,15,30]
colsample_bytree = [0.3,0.5,0.7,1.0]
lambda_l1 = [0.0, 0.1, 0.3, 0.5, 0.8]

mse_best_mse_list = []
mse_best_mse_list_train = []
mse_best_lr_list = []
mse_best_md_list = []
mse_best_cb_list = []
mse_best_l1_list = []

from sklearn.model_selection import KFold
k_fold = KFold(n_splits=5, shuffle=True, random_state=2024)


X_eco = X_eco.detach().numpy()
X_eco = pd.DataFrame(X_eco)
X_eco.columns = [f"encoded_{i}" for i in range(X_eco.shape[1])]
X = X_eco


lr_list = []
md_list = []
cb_list = []
l1_list = []

mse_list_train = []
mse_list = []

y = label_data
y = y.to_numpy()
X_final_train, X_final_test, y_final_train, y_final_test = train_test_split(X, y, test_size=0.2, random_state=2024)
print(len(X_final_train))
print(len(X_final_test))
for lr in learning_rate:
    for md in max_depth:
        for cb in colsample_bytree:
            for l1 in lambda_l1:
                y=pd.DataFrame(y)
                
                fold_result_mse = []
                fold_result_mse_train=[]
                for fold, (train_idx, valid_idx) in enumerate(k_fold.split(X_final_train, y_final_train)):
                    X_train, X_valid = X.iloc[train_idx,:],X.iloc[valid_idx,:]
                    y_train, y_valid = y.iloc[train_idx,:],y.iloc[valid_idx,:]

                    model = xgb.XGBRegressor(learning_rate=lr, max_depth=md, colsample_bytree=cb, alpha=l1, objective='reg:squarederror',random_state=2024)
                    model.fit(X_train, y_train)
                    y_train_pred = model.predict(X_train)
                    y_pred = model.predict(X_valid)
        
                    print("===================================================")
                    print("kfold = ", fold + 1)
                    print("learning_rate = ", lr)
                    print("max_depth = ", md)
                    print("colsample_bytree = ", cb)
                    print("lambda_l1 = ", l1)
                    print(f"train mse: {mean_squared_error(y_train, y_train_pred)}")
                    print(f"valid mse: {mean_squared_error(y_valid, y_pred)}")

                    fold_result_mse_train.append(mean_squared_error(y_train, y_train_pred))
                    fold_result_mse.append(mean_squared_error(y_valid, y_pred))
                mse_list_train.append((sum(fold_result_mse_train)/len(fold_result_mse_train)))
                mse_list.append(sum(fold_result_mse)/len(fold_result_mse))
                
                lr_list.append(lr)
                md_list.append(md)
                cb_list.append(cb)
                l1_list.append(l1)
                print(fold_result_mse)

arg_max_mse = np.argmin(mse_list)

result_list = []


mse_lr = lr_list[arg_max_mse]
mse_md = md_list[arg_max_mse]
mse_cb = cb_list[arg_max_mse]
mse_l1 = l1_list[arg_max_mse]

#X_final_train, y_final_train = ada.fit_resample(X_final_train, y_final_train)

final_model_mse = xgb.XGBRegressor(learning_rate=mse_lr, max_depth=mse_md, colsample_bytree=mse_cb, alpha=mse_l1, objective='reg:squarederror',random_state=2024)
final_model_mse.fit(X_final_train, y_final_train)

y_final_pred_mse = final_model_mse.predict(X_final_test)
mse_best_mse_list.append(mean_squared_error(y_final_test, y_final_pred_mse))

y_final_pred_mse_train = final_model_mse.predict(X_final_train)
mse_best_mse_list_train.append(mean_squared_error(y_final_train, y_final_pred_mse_train))

mse_best_lr_list.append(mse_lr)
mse_best_md_list.append(mse_md)
mse_best_cb_list.append(mse_cb)
mse_best_l1_list.append(mse_l1)

datadatadata = {'mse_score_train': mse_best_mse_list_train,
                'mse_score_test': mse_best_mse_list,
                'f1_learning_rate': mse_best_lr_list,
                'f1_max_depth': mse_best_md_list,
                'f1_colsample_bytree': mse_best_cb_list,
                'f1_lambdal1': mse_best_l1_list,
    }

file_name = 'only_smiles_128.csv'

for key, value in datadatadata.items():
    print(f"Length of {key}: {len(value)}")
# datadatadata.to_csv(file_name, index=False)
dfdfdf = pd.DataFrame(datadatadata)


dfdfdf.to_csv(OUTPUT_DIR / file_name, index=False)
