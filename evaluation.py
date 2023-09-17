from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import seaborn as sns
import matplotlib.pyplot as plt
import japanize_matplotlib
import pandas as pd

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