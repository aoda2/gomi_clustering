# imports
import pandas as pd
import numpy as np
from models import OpenAIModel
from ast import literal_eval
from gomi_config import mode
from sklearn.ensemble import RandomForestClassifier
import joblib
from gomi_config import mode, context

# 入力テキスト
input = "枯れた花"

# モデル読み込み
model: RandomForestClassifier = joblib.load(f"output/0133_20230307_model_{mode}.joblib")

# OpenAI モデル生成
open_ai_model = OpenAIModel()
# embedding 取得
text = context[mode].format(input)
embedding = open_ai_model.convert_to_vector(text)

# モデルによる predict
predicted =  model.predict([embedding])
#gomi_embedding["predict_proba"] =  model.predict_proba(list(gomi_embedding["embedding"].values))

print(predicted)