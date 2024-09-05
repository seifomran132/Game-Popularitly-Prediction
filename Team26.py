import numpy as np
import pandas as pd
import statistics
import sklearn.preprocessing as pre
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge, Lasso


gamedata = pd.read_csv("games-regression-dataset.csv");

gamedata.dropna(axis=0, subset=["Languages"], inplace=True)
gamedata.drop_duplicates(inplace=True)


X = gamedata.iloc[:, :-1]
y = gamedata.iloc[:, -1]

# Data Spliting
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True, random_state=10)

X_train = pd.DataFrame(data=X_train)
X_test = pd.DataFrame(data=X_test)


def calculateCorrelaction(xtrain, ytrain):
    corrdata = pd.DataFrame.copy(xtrain, deep=True)
    corrdata["Average User Rating"] = ytrain
    corelation = corrdata.corr()
    plt.figure(figsize=(15, 15))
    sns.heatmap(corelation, cmap=plt.cm.Reds, annot=True)
    plt.show()
    return corelation


class PreProcessing:
    data = pd.DataFrame()

    def __init__(self, data):
        self.data = data

    def removeNulls(self):
        nullCols = self.data.columns[self.data.isna().mean() > 0.3]
        self.data.drop(nullCols, axis=1, inplace=True)
        self.data.dropna(axis=0, inplace=True)

    def fillNulls(self, feature, value):
        self.data[feature].fillna(value, inplace=True)

    def dateEncoding(self):
        releaseDate = pd.to_datetime(self.data["Original Release Date"], dayfirst=True)
        releaseDatelist = pd.to_datetime(releaseDate).dt.year.astype(str).str[2:]
        updateDate = pd.to_datetime(self.data["Current Version Release Date"], dayfirst=True)
        updateDatelist = pd.to_datetime(updateDate).dt.year.astype(str).str[2:]

        self.data["Original Release Date"] = releaseDatelist
        self.data["Current Version Release Date"] = updateDatelist

    def update_app_purchase(self):
        n = self.data["In-app Purchases"]
        n = np.array(n)
        for i in range(0, len(n)):
            s = ""
            if type(n[i]) == type(s):
                n[i] = n[i].split(",")
            n[i] = pd.to_numeric(n[i])
            n[i] = (np.rint(n[i])).astype(int)
            n[i] = statistics.mean(n[i])
        self.data["In-app Purchases"] = n

    def setLanguageNumbers(self):
        featuresLang = self.data["Languages"]

        featuresLang = featuresLang.str.replace(", ", "")
        featuresLang = featuresLang.str.len() / 2

        self.data["Languages"] = featuresLang

    def setGenereNumbers(self):
        featuresLang = self.data["Genres"]

        featuresLang = featuresLang.str.split(", ")
        featuresLang = featuresLang.str.len()

        self.data["Genres"] = featuresLang

    def processAgeRating(self):
        self.data["Age Rating"] = self.data["Age Rating"].str.replace("+", "")

    

    def normalize(self, X):
        myX = np.array(X)
        myX = myX.reshape(1, -1)
        normalized = pre.normalize(myX)

        normalized = normalized.T
        return normalized

    def normalizeData(self):

        for col in self.data.columns:
            self.data[col] = self.normalize(self.data[col])

    def calculateGameAge(self):
        from datetime import date
        current_year = str(date.today().year)[-2:]
        series1 = int(current_year)
        series2 = self.data["Original Release Date"].astype(int)
        self.data["Game Age"] = series1 - series2
        self.data.drop(axis=1, columns="Original Release Date", inplace=True)
        #self.data.drop(axis=1, columns="Current Version Release Date", inplace=True)

    def encodeGeneres(self):
        dummy = pd.get_dummies(self.data['Primary Genre'], prefix='', prefix_sep='')

        for col in dummy.columns:
            self.data[col] = dummy[col]
            
        self.data.drop(axis=1, columns="Primary Genre", inplace=True)

    ###############
    def make_dummy_Frames(self):
        # change to array
        m = self.data["Languages"].astype(str)
        m = np.array(m)
        print(m)
        # splitting the list in each row
        for i in range(0, len(m)):
            m[i] = m[i].split(", ")
        print(m)
        # change to dataframe
        m = pd.DataFrame(m, columns=['Lang'])
        print(m)
        # make dummy variables
        k = pd.get_dummies(m.explode(['Lang'])).groupby(level=0).sum()
        # reset this column to be 1 because it is repeated many times
        k["Lang_ZH"] = k["Lang_ZH"].replace([2, 3, 4, 5, 6], 1)
        return k
    
    

    def encodeDeveloper(self):
        dummy = pd.get_dummies(self.data['Developer'], prefix='', prefix_sep='')

        for col in dummy.columns:
            self.data[col] = dummy[col]
        self.data.drop(axis=1, columns="Developer", inplace=True)

    def dropZerosCols(self):
        cols = self.data.columns

        for col in cols:
            zerosPercent = (self.data[col] == 0).sum() / len(self.data)
            if (zerosPercent > 0.90):
                self.data.drop(axis=1, columns=col, inplace=True)
                
    def dropUniqueCols(self):      
        cols = self.data.columns

        for col in cols:
            uniquePercent = (self.data[col].nunique()) / len(self.data)
            if (uniquePercent > 0.90):
                self.data.drop(axis=1, columns=col, inplace=True)
        

    def clean(self):
        self.fillNulls("In-app Purchases", '0')
        self.removeNulls()
        self.dateEncoding()
        self.update_app_purchase()
        self.setLanguageNumbers()
        self.calculateGameAge()
        self.processAgeRating()
        self.encodeGeneres()
        self.encodeDeveloper()
        self.setGenereNumbers()
        self.dropZerosCols()
        self.dropUniqueCols()
        self.normalizeData()
        return self.data


# Run PreProcessing Pipline on Train and Test Data Sets


def performPreProcessing_xtrain():
    train = PreProcessing(X_train)
    c = train.clean()
    return c



def performPreProcessing_xtest():
    test = PreProcessing(X_test)
    c1 = test.clean()
    return c1

# Polynomial Model


from sklearn import linear_model
from sklearn import metrics
from sklearn.preprocessing import PolynomialFeatures

y_test = np.array(y_test)




# Declare models

lr = linear_model.LinearRegression()
ridgeReg = Ridge(alpha=0.1)
lasso = Lasso(alpha=0.1)
poly_model = linear_model.LinearRegression()


def linearRegressionModel():
    # Fit the model

    lr.fit(X_train, y_train)
    yprediction = lr.predict(X_test)

    print("\nSimple Linear Regression Model----------------------------------\n")
    print('Co-efficient of linear regression', lr.coef_)
    print('Intercept of linear regression model', lr.intercept_)
    print('Mean Square Error', metrics.mean_squared_error(y_test, yprediction))
    print('Root Mean Square Error', np.sqrt(metrics.mean_squared_error(y_test, yprediction)))

    return yprediction



def ridgeRegressionModel():
    ridgeReg.fit(X_train, y_train)
    ypredridge = ridgeReg.predict(X_test)

    print("\nRidge Regression Model--------------------------------------------\n")
    print('Co-efficient of linear regression', ridgeReg.coef_)
    print('Intercept of linear regression model', ridgeReg.intercept_)
    print('Mean Square Error', metrics.mean_squared_error(y_test, ypredridge))
    print('Root Mean Square Error', np.sqrt(metrics.mean_squared_error(y_test, ypredridge)))

    return ypredridge


# Lasso regression model

def lassoRegressionModel():
    lasso.fit(X_train, y_train)
    ypredlasso = lasso.predict(X_test)

    print("\nLasso Regression Model--------------------------------------------\n")
    print('Co-efficient of linear regression', lasso.coef_)
    print('Intercept of linear regression model', lasso.intercept_)
    print('Mean Square Error', metrics.mean_squared_error(y_test, ypredlasso))
    print('Root Mean Square Error', np.sqrt(metrics.mean_squared_error(y_test, ypredlasso)))
 
    return ypredlasso




# Main

def selectTopFeatures(x, y):
    corrfeat = calculateCorrelaction(x, y)
    topfeatures = corrfeat.index[abs(corrfeat["Average User Rating"]) > 0.02]
    topfeatures = topfeatures.delete(-1)
    print(topfeatures)
    return topfeatures


def plotModel(X, Y, Pre):
    X = np.array(X)
    Y = np.array(Y)
    print(X.shape, Y.shape)
    for i in range(X.shape[1]):
        plt.scatter(X[:, i], Y)

    for i in range(X.shape[1]):
        plt.plot(X[:, i], Pre, color='red', linewidth=3)

    plt.show()



X_train = performPreProcessing_xtrain()
X_test = performPreProcessing_xtest()
topfet = selectTopFeatures(X_train, y_train)
X_train = X_train[topfet]
X_test = X_test[topfet]
ypred1 = linearRegressionModel()
plotModel(X_test, y_test, ypred1)
ypred3 = lassoRegressionModel()
plotModel(X_test, y_test, ypred3)
ypred4 = ridgeRegressionModel()
plotModel(X_test, y_test, ypred4)






