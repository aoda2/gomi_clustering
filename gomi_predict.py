import predict
import pandas as pd

gomi_embedding = pd.read_csv("output/0133_20230307_embedding.csv")

gomi_embedding_predict = predict.predict_category(gomi_embedding, gomi_embedding)

gomi_embedding_predict.to_csv("output/0133_20230307_predict.csv")