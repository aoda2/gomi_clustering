import pandas as pd
from models import OpenAIModel
import time

# ゴミの出し方データの読み込み
gomi_df = pd.read_csv("input/0133_20230307.csv")
# デバッグ出力
print(gomi_df)

# モデル生成
open_ai_model = OpenAIModel()

# openai model を用いて embedding 化
#gomi_df["embedding"] = gomi_df["品目名"].apply(lambda x: open_ai_model.convert_to_vector(x))

embeddings = []
for i, r in gomi_df.iterrows():
    if i % 20 == 0:
        print(i)
    try:
        embedding = open_ai_model.convert_to_vector(r["品目名"])
        embeddings.append(embedding)
        time.sleep(0.1)
    except Exception as e:
        print(i)
        print(e)
        embeddings.append([])

gomi_df["embedding"] = embeddings

# デバッグ出力
print(gomi_df)

#ファイル出力
gomi_df.to_csv("output/0133_20230307_embedding.csv")
