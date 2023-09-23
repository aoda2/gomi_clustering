# imports
import pandas as pd
import numpy as np
from models import OpenAIModel
from ast import literal_eval
from gomi_config import mode
from gomi_config import mode, context
from sklearn.metrics.pairwise import cosine_similarity

# 入力テキスト
input = "3つの椅子"

# ファイル読み込み
gomi_embedding = pd.read_csv(f"output/0133_20230307_embedding_{mode}.csv")
# 文字列化されているカラムを ndarray 化
gomi_embedding["embedding"] = [np.array(literal_eval(d)) for d in gomi_embedding['embedding']]
# ベクトルのリストを取得し、NumPy配列に変換
category_vectors = np.array(gomi_embedding['embedding'].tolist())

# OpenAI モデル生成
open_ai_model = OpenAIModel()
# embedding 取得
text = context[mode].format(input)
embedding = open_ai_model.convert_to_vector(text)

# 入力テキストの embedding と全カテゴリベクトルのコサイン類似度を計算
cosine_similarities = cosine_similarity([np.array(embedding)], category_vectors)

# 最も類似度が高いカテゴリのインデックスを取得
predicted_category_index = np.argmax(cosine_similarities, axis=1)

predicted =  gomi_embedding['出し方'].iloc[predicted_category_index].values

print(predicted)
