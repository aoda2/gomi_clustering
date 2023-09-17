import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

def predict_category(df_item: pd.DataFrame, df_category: pd.DataFrame) -> pd.DataFrame:
    # ベクトルのリストを取得し、NumPy配列に変換
    item_vectors = np.array(df_item['embedding'].tolist())
    category_vectors = np.array(df_category['embedding'].tolist())

    # 各ハッシュタグベクトルと全カテゴリベクトルのコサイン類似度を計算
    cosine_similarities = cosine_similarity(item_vectors, category_vectors)

    # 最も類似度が高いカテゴリのインデックスを取得
    predicted_category_indices = np.argmax(cosine_similarities, axis=1)

    # 最大類似度を取得
    max_similarities = np.max(cosine_similarities, axis=1)

    # 予測カテゴリを取得
    predicted_categories = df_category['出し方'].iloc[predicted_category_indices].values

    # df_itemに新しい列を追加
    df_item['predict_出し方'] = predicted_categories
    df_item['cosine_similarity'] = max_similarities

    return df_item