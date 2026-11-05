
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import sklearn
from IPython.display import display
from sklearn.datasets import load_iris, load_diabetes
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, ConfusionMatrixDisplay,
)

RANDOM_STATE = 42
# print('Python:', sys.version.split()[0])
# print('NumPy:', np.__version__, '| Pandas:', pd.__version__)
# print('Matplotlib:', matplotlib.__version__, '| scikit-learn:', sklearn.__version__)
# 数据集
iris = load_iris()
X = pd.DataFrame(iris.data, columns=iris.feature_names)
y = pd.Series(iris.target, name='target')
# print('\n--- 数据集基本情况 ---')
# print('样本数 x 特征数:', X.shape)
# print('类别:', list(iris.target_names))
# print('每列缺失值数量:')
# print(X.isna().sum())
# print('各类别样本数:')
# print(y.value_counts().sort_index())
# print('特征基本统计:')
# print(X.describe().round(2))
# 散点图
pairs = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
fig, axes = plt.subplots(2, 3, figsize=(15, 8))

for ax, (i, j) in zip(axes.ravel(), pairs):
    for cls in range(3):
        mask = (y == cls)
        ax.scatter(X.iloc[mask, i], X.iloc[mask, j],
                   label=iris.target_names[cls], edgecolors='k', linewidths=0.4)
    ax.set_xlabel(iris.feature_names[i])
    ax.set_ylabel(iris.feature_names[j])
    ax.legend()

# axes[0].set_title('Sepal length vs Sepal width')
# axes[1].set_title('Petal length vs Petal width')
# plt.tight_layout()
# plt.show()
# 留出法
# 留 20% 作最终测试集 (150 -> 120 + 30)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y)
# 剩下的 120 个再分 25% 作验证集 (120 -> 90 + 30)
X_train, X_val, y_train, y_val = train_test_split(
    X_train, y_train, test_size=0.25, random_state=RANDOM_STATE, stratify=y_train)

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
train_err = 1 - knn.score(X_train, y_train)
val_err = 1 - knn.score(X_val, y_val)
print('\n训练误差: %.4f' % train_err)
print('验证误差: %.4f' % val_err)
y_pred = knn.predict(X_test)
print('\n--- 测试集评估结果 ---')
print('Accuracy       : %.4f' % accuracy_score(y_test, y_pred))
print('Macro-Precision: %.4f' % precision_score(y_test, y_pred, average='macro'))
print('Macro-Recall   : %.4f' % recall_score(y_test, y_pred, average='macro'))
print('Macro-F1       : %.4f' % f1_score(y_test, y_pred, average='macro'))
cm = confusion_matrix(y_test, y_pred)
ConfusionMatrixDisplay(cm, display_labels=iris.target_names).plot(cmap='Blues')
plt.title('Confusion Matrix on Test Set (KNN, k=5)')
plt.show()

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
results = cross_validate(KNeighborsClassifier(n_neighbors=5), X, y,
                         cv=cv, scoring='accuracy')
scores = results['test_score']

print('\n--- 5 折交叉验证 (KNN, k=5) ---')
for i, s in enumerate(scores, 1):
    print('Fold %d accuracy: %.4f' % (i, s))
print('平均准确率: %.4f +/- %.4f' % (scores.mean(), scores.std()))