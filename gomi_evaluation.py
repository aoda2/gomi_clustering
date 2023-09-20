# imports
import pandas as pd
import numpy as np
import evaluation
from ast import literal_eval
from gomi_config import mode
from sklearn.ensemble import RandomForestClassifier
import joblib

# ファイル読み込み
gomi_embedding = pd.read_csv(f"output/0133_20230307_embedding_{mode}.csv")
# 文字列化されているカラムを ndarray 化
gomi_embedding["embedding"] = [np.array(literal_eval(d)) for d in gomi_embedding['embedding']]
# モデルのロード
model: RandomForestClassifier = joblib.load(f"output/0133_20230307_model_{mode}.joblib")

# モデルによる predict
gomi_embedding["predict_出し方"] =  model.predict(list(gomi_embedding["embedding"].values))
#gomi_embedding["predict_proba"] =  model.predict_proba(list(gomi_embedding["embedding"].values))

gomi_embedding[["品目名", "出し方", "predict_出し方"]].to_csv(f"output/0133_20230307_classcication_{mode}.csv")

# evaluation 実行
evaluation.evaluation(gomi_embedding)