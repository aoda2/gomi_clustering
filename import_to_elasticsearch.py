from elasticsearch import Elasticsearch
from gomi_config import mode
import json
from ast import literal_eval
import pandas as pd

es = Elasticsearch([{'host': 'localhost', 'port': 9200, 'scheme': 'http', 
                     'use_ssl': False}])  # Elasticsearchのホストとポートに合わせて変更

index_name = f'gomi_index_{mode}'  # インデックス名を指定

# インデックスを作成
index_mapping = {
    "mappings": {
        "properties": {
            "embedding": {
                "type": "dense_vector",
                "dims": 1536,
                "index": True,
                "similarity": "dot_product" 
            }
        }
    }
}

# インデックス削除
try:
    es.indices.delete(index=index_name)
except Exception as e:
    # 初回はないので
    print(e)

# インデックス作成
es.indices.create(index=index_name, body=index_mapping, ignore=400)

# データ読み込み
# ファイル読み込み
gomi_embedding = pd.read_csv(f"output/0133_20230307_embedding_{mode}.csv")
# 文字列化されているカラムを list 化
gomi_embedding["embedding"] = [literal_eval(d) for d in gomi_embedding['embedding']]

# Pandas DataFrameからJSONドキュメントに変換
def convert_to_json(row):
    doc = {
        'gomi_ID': row['ID'],  
        'gomi_prefix': str(row['頭文字']),  
        'gomi_item_name': str(row['品目名']), 
        'gomi_small_device': str(row['小型家電回収対象']), 
        'gomi_category': str(row['出し方']),  
        'gomi_category_detail': str(row['出し方のポイント']), 
        'embedding': row['embedding'] 
    }
    return json.dumps(doc)

# DataFrameの各行をJSONに変換
json_documents = gomi_embedding.apply(convert_to_json, axis=1)

# データを格納
for document in json_documents:
    es.index(index=index_name, body=document)