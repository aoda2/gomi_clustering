import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

def predict_category(df_item: pd.DataFrame) -> pd.DataFrame:
    # ベクトルのリストを取得し、NumPy配列に変換
    item_vectors = np.array(df_item['embedding'].tolist())

    # 各アイテムベクトルと全カテゴリベクトルのコサイン類似度を計算
    cosine_similarities = cosine_similarity(item_vectors, item_vectors)

    # 最も類似度が高いカテゴリのインデックスを取得 (argmax すると自分自身になるので 2番目を取得)
    predicted_category_indices = np.argsort(cosine_similarities, axis=1)[:, -2]

    # 最大類似度を取得 (max すると自分自身になるので 2番目を取得)
    max_similarities = np.sort(cosine_similarities, axis=1)[:, -2]

    # 最も似ている品目名を取得
    most_similar_item = df_item['品目名'].iloc[predicted_category_indices].values

    # 予測カテゴリを取得
    predicted_categories = df_item['出し方'].iloc[predicted_category_indices].values

    # df_itemに新しい列を追加
    df_item['predict_出し方'] = predicted_categories
    df_item['predict_品目名'] = most_similar_item
    df_item['cosine_similarity'] = max_similarities

    return df_item