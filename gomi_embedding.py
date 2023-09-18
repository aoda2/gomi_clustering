import pandas as pd
from models import OpenAIModel
import time
from gomi_config import mode, context

# ゴミの出し方データの読み込み
gomi_df = pd.read_csv("input/0133_20230307.csv")
# デバッグ出力
print(gomi_df)

# モデル生成
open_ai_model = OpenAIModel()


embeddings = []
for i, r in gomi_df.iterrows():
    # embedding 化する文字列を生成
    text = context[mode].format(r["品目名"])
    try:
        # openai model を用いて embedding 化
        embedding = open_ai_model.convert_to_vector(text)
        embeddings.append(embedding)
        time.sleep(0.05)
    except Exception as e:
        print(f"an error occured at {i}. {text}")
        print(e)
        embeddings.append([])

    # 進捗
    if i % 50 == 0:
        print(f"{i} done. {text}")

gomi_df["embedding"] = embeddings

# デバッグ出力
print(gomi_df)

#ファイル出力
gomi_df.to_csv(f"output/0133_20230307_embedding_{mode}.csv")
