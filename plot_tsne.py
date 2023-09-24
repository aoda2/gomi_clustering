import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import time
from ast import literal_eval
from sklearn import manifold
from gomi_config import mode
import japanize_matplotlib


# ファイル読み込み
gomi_embedding = pd.read_csv(f"output/0133_20230307_embedding_{mode}.csv")
# 文字列化されているカラムを ndarray 化
gomi_embedding["embedding"] = [np.array(literal_eval(d)) for d in gomi_embedding['embedding']]
# ベクトルのリストを取得し、NumPy配列に変換
X = np.array(gomi_embedding['embedding'].tolist())
# カテゴリ
y = gomi_embedding["出し方"]
y_items = y.unique()

n_components = 2
perplexity = 75

start_time = time.time()
fig, ax = plt.subplots(figsize=(5,5))
tsne = manifold.TSNE(n_components=n_components, init='random', random_state=0, perplexity=perplexity)
Y = tsne.fit_transform(X)
for each_quality in y_items:
    c_plot_bool = y == each_quality # True/Falseのarrayを返す
    ax.scatter(Y[c_plot_bool, 0], Y[c_plot_bool, 1], label=each_quality)
end_time = time.time()
ax.legend(loc='upper left',bbox_to_anchor=(1, 1))
print("Time to plot is {:.2f} seconds.".format(end_time - start_time))
plt.show()