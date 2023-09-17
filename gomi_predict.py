import ast
import predict
import evaluation
import numpy as np
import pandas as pd

# ファイル読み込み
gomi_embedding = pd.read_csv("output/0133_20230307_embedding.csv")
# 文字列化されているカラムを ndarray 化
gomi_embedding["embedding"] = [np.array(ast.literal_eval(d)) for d in gomi_embedding['embedding']]

gomi_embedding_predict = predict.predict_category(gomi_embedding)

gomi_embedding_predict[["品目名", "出し方", "predict_品目名", "predict_出し方", "cosine_similarity"]].to_csv("output/0133_20230307_predict.csv")

evaluation.evaluation(gomi_embedding_predict)