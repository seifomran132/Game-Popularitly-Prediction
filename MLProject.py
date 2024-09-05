import numpy as np
import pandas as pd


gamedata = pd.read_csv("games-regression-dataset.csv");



# Date

def dateEncoding(date):
    datelist = pd.to_datetime(date).dt.year.astype(str).str[2:]
    return datelist



# drop language nulls
gamedata.dropna(axis=0, inplace=True, subset=["Languages"])
# Languages
def EncodingLang(featuresLang):
    temp = list()
    for i in zip(range(featuresLang.shape[0]), featuresLang):
        y = i[1].replace(' ', '').replace(',', '')
        z = int(len(y) / 2)
        temp.append(z)
    featuresLang = pd.DataFrame(temp)
    return featuresLang

bef = gamedata["Languages"]
aft = EncodingLang(gamedata["Languages"])

gamedata["Languages"] = aft
# drop language nulls
gamedata.dropna(axis=0, inplace=True, subset=["Languages"])

X = gamedata.iloc[:, :-1]
y = gamedata.iloc[:, -1]



# Data Spliting
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25,shuffle=False,random_state=10)

X_train = pd.DataFrame(data=X_train)
X_test = pd.DataFrame(data=X_test)


# Missing Values



# 3749 null in subtitle
# 2039 null in In-app Purchase 
# 11 Null in language
# So we will drop subtitle and fill In-app purchase with zeros columns and drop the 11 null rows in language

X_train.drop(axis=1, columns="Subtitle", inplace=True)

# Sum all purchases

import statistics

X_train["In-app Purchases"].fillna('0', inplace=True)
inAppPurchases = X_train["In-app Purchases"]
inAppPurchases = np.array(inAppPurchases)

def update_app_purchase(n):
    for i in range(0, len(n)):
        s = ""
        if type(n[i]) == type(s):
            n[i] = n[i].split(",")
        n[i] = pd.to_numeric(n[i])
        n[i] = (np.rint(n[i])).astype(int)
        n[i] = statistics.mean(n[i])
    return n

X_train["In-app Purchases"] = update_app_purchase(inAppPurchases)




# Categorical Data





# Drop Unnecessary features
X_train.drop(axis=1, columns="URL", inplace=True)
X_train.drop(axis=1, columns="Name", inplace=True)
X_train.drop(axis=1, columns="ID", inplace=True)
X_train.drop(axis=1, columns="Icon URL", inplace=True)

# Feature Scaling
# Rating count, Price, Age Rating, Size
X_train["Age Rating"] = X_train["Age Rating"].str.replace("+", "")


import sklearn.preprocessing as pre
def normalize(X):
    myX = np.array(X)
    myX = myX.reshape(1, -1)
    normalized = pre.normalize(myX)
    normalized = normalized.T
    return normalized

X_train["Age Rating"] = normalize(X_train["Age Rating"])
X_train["User Rating Count"] = normalize(X_train["User Rating Count"])
X_train["Size"] = normalize(X_train["Size"])
X_train["Price"] = normalize(X_train["Price"])
X_train["In-app Purchases"] = normalize(X_train["In-app Purchases"])
X_train["Languages"] = normalize(X_train["Languages"])
X_train["Original Release Date"] = normalize(dateEncoding(X_train["Original Release Date"]))


# Temp
X_train.drop(axis=1, columns="Description", inplace=True)
X_train.drop(axis=1, columns="Developer", inplace=True)
X_train.drop(axis=1, columns="Primary Genre", inplace=True)
X_train.drop(axis=1, columns="Genres", inplace=True)
X_train.drop(axis=1, columns="Current Version Release Date", inplace=True)


# X_test

X_test["Original Release Date"] = normalize(dateEncoding(X_test["Original Release Date"]))


X_test.drop(axis=1, columns="Subtitle", inplace=True)
X_test.drop(axis=1, columns="URL", inplace=True)
X_test.drop(axis=1, columns="Name", inplace=True)
X_test.drop(axis=1, columns="ID", inplace=True)
X_test.drop(axis=1, columns="Icon URL", inplace=True)
X_test.drop(axis=1, columns="Description", inplace=True)
X_test.drop(axis=1, columns="Developer", inplace=True)
X_test.drop(axis=1, columns="Primary Genre", inplace=True)
X_test.drop(axis=1, columns="Genres", inplace=True)
X_test.drop(axis=1, columns="Current Version Release Date", inplace=True)

X_test["Age Rating"] = X_test["Age Rating"].str.replace("+", "")

X_test["In-app Purchases"].fillna('0', inplace=True)
inAppPurchases = X_test["In-app Purchases"]
inAppPurchases = np.array(inAppPurchases)
X_test["In-app Purchases"] = update_app_purchase(inAppPurchases)


X_test["Age Rating"] = normalize(X_test["Age Rating"])
X_test["User Rating Count"] = normalize(X_test["User Rating Count"])
X_test["Size"] = normalize(X_test["Size"])
X_test["Price"] = normalize(X_test["Price"])
X_test["In-app Purchases"] = normalize(X_test["In-app Purchases"])
X_test["Languages"] = normalize(X_test["Languages"])





#y_train = normalize(y_train)


import seaborn as sns
import matplotlib.pyplot as plt


corelation = gamedata.corr()
sns.heatmap(corelation, cmap=plt.cm.Reds, annot=True)
plt.show()


from sklearn import linear_model
from sklearn import metrics


cls = linear_model.LinearRegression()
X1=X_train
Y1=y_train
cls.fit(X1,Y1) #Fit method is used for fitting your training data into the model
prediction= cls.predict(X_test)
# plt.scatter(X1, Y1)
# plt.xlabel('SAT', fontsize = 20)
# plt.ylabel('GPA', fontsize = 20)
# plt.plot(X1, prediction, color='red', linewidth = 3)
# plt.show()
print('Co-efficient of linear regression',cls.coef_)
print('Intercept of linear regression model',cls.intercept_)
print('Mean Square Error', metrics.mean_squared_error(y_test, prediction))








