# imports
from models import OpenAIModel
from gomi_config import mode, context, sample_input
from elasticsearch import Elasticsearch

# 入力テキスト
input = sample_input


def cosine_similarity_sort(index_name, query_vector):

    # Elasticsearchの検索クエリを設定
    search_query = {
        "knn": [{
            "field": "embedding",
            "query_vector": query_vector,
            "k": 5,
            "num_candidates": 5000,
            "boost": 1.0
        # 複数書くこともできる
        # },{
        #     "field": "embedding",
        #     "query_vector": query_vector,
        #     "k": 5,
        #     "num_candidates": 5000,
        #     "boost": 1.0
        }],
        "size": 10,
        }

    # Elasticsearchに検索クエリを送信
    results = es.search(index=index_name, body=search_query)

    # クエリ結果を返す
    return results

# OpenAI モデル生成
open_ai_model = OpenAIModel()
# embedding 取得
text = context[mode].format(input)
embedding = open_ai_model.convert_to_vector(text)

# Elasticsearchクライアントの初期化
es = Elasticsearch([{'host': 'localhost', 'port': 9200, 'scheme': 'http'}])  # Elasticsearchのホストとポートに合わせて変更

# クエリ用のベクトルを設定
query_vector = embedding

# Elasticsearchのインデックス名を指定
index_name = f'gomi_index_{mode}'  # インデックス名を指定

# コサイン類似度に基づいてソートされた結果を取得
result = cosine_similarity_sort(index_name, query_vector)

# 結果を表示
for hit in result['hits']['hits']:
    print(f"Score: {hit['_score']}, Document: {hit['_source']['gomi_item_name']}, Category: {hit['_source']['gomi_category']}")

