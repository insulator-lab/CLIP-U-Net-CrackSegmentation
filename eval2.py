import sys
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import sklearn

from sklearn.datasets import (
    load_diabetes, load_breast_cancer, load_iris, make_classification,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.dummy import DummyRegressor, DummyClassifier
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, ConfusionMatrixDisplay,
)

RANDOM_STATE = 42
print('Python:', sys.version.split()[0])
print('NumPy:', np.__version__, '| Pandas:', pd.__version__)
print('Matplotlib:', matplotlib.__version__, '| scikit-learn:', sklearn.__version__)

