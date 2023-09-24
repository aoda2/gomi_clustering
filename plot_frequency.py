import pandas as pd
from gomi_config import mode, context
import matplotlib.pyplot as plt
import japanize_matplotlib

# ゴミの出し方データの読み込み
gomi_df = pd.read_csv("input/0133_20230307.csv")
# デバッグ出力
gomi_df["出し方"].value_counts().plot(kind="bar")

plt.show()