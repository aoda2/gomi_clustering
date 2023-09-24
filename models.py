import csv
import openai
import os
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import seaborn as sns
import matplotlib.pyplot as plt
import japanize_matplotlib

class OpenAIModel:
    def __init__(self):
        openai.api_key = os.getenv("OPENAI_API_KEY")
        self.openai = openai
        
    def convert_to_vector(self, text):
        response = self.openai.Embedding.create(
          model="text-embedding-ada-002",
          input=text
        )
        
        # 消費したトークン数をCSVに記録
        filename = 'output/openai_usage_tokens.csv'
        file_exists = os.path.isfile(filename)
        
        with open(filename, mode='a') as file:
            writer = csv.writer(file)
            if not file_exists:
                writer.writerow(['Text', 'Total Tokens'])
            writer.writerow([text, response['usage']['total_tokens']])
        
        # vectorを返却
        return response['data'][0]['embedding']
    
def evaluation(df_predicted: pd.DataFrame):
    # 予測と実際のカテゴリを取得
    y_true = df_predicted['出し方']
    y_pred = df_predicted['predict_出し方']

    # 評価指標の計算
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='macro')
    recall = recall_score(y_true, y_pred, average='macro')
    f1 = f1_score(y_true, y_pred, average='macro')

    print(f'Accuracy: {accuracy:.2f}\nPrecision: {precision:.2f}\nRecall: {recall:.2f}\nF1 Score: {f1:.2f}')

    labels = list(df_predicted["出し方"].unique()) # ラベルの順序を指定

    # 混同行列の作成
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    cm = pd.DataFrame(data=cm, index=labels, columns=labels)
    plt.figure(figsize=(15, 10))
    sns.heatmap(cm, square=True, cbar=True, annot=True, cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()
#    plt.savefig('evaluation.png')


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