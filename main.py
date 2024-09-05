import statistics
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import Normalizer
from sklearn import preprocessing


def update_app_purchase(n):
    for i in range(0, len(n)):
        s = ""
        if type(n[i]) == type(s):
            n[i] = n[i].split(",")
        n[i] = pd.to_numeric(n[i])
        n[i] = (np.rint(n[i])).astype(int)
        print(n[i])
        n[i] = statistics.mean(n[i])
        print(n[i])
    return n


# Loading data
data = pd.read_csv('games-regression-dataset.csv')
print(data.describe())

X = data.iloc[:, 0:17]
Y = data['Average User Rating']

# Split the data to training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.25, shuffle=True, random_state=10)

X_train["In-app Purchases"].fillna('0', inplace=True)
n = X_train["In-app Purchases"]
n = np.array(n)
print(n)
X_train["In-app Purchases"] = update_app_purchase(n)




