import numpy as np
import pandas as pd
import statistics
import sklearn.preprocessing as pre
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.feature_selection import SelectKBest, f_classif, chi2

import time, datetime

import warnings
warnings.filterwarnings("ignore")

gamedata = pd.read_csv("game1.csv");

X = gamedata.iloc[:, :-1]
y = gamedata.iloc[:, -1]

gamedata.drop_duplicates(inplace=True)

# Data Spliting
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False, random_state=10)

X_train = pd.DataFrame(data=X_train)
X_test = pd.DataFrame(data=X_test)

class PreProcessing:
    Xtrain = pd.DataFrame()
    Xtest = pd.DataFrame()
    trainFeatures = []
    langMod = 0
    scale = MinMaxScaler()
    selectedFeaturesNames = []

    def __init__(self, Xtrain, Xtest):
        self.Xtrain = Xtrain
        self.Xtest = Xtest

    def removeNulls(self, data):
        nullCols = data.columns[data.isna().mean() > 0.3]
        data.drop(nullCols, axis=1, inplace=True)
        #self.data.dropna(axis=0, inplace=True)

    def fillNulls(self, data, feature, value):
        data[feature].fillna(value, inplace=True)

    def dateEncoding(self, data):
#         releaseDate = pd.to_datetime(data["Original Release Date"], dayfirst=True)
#         releaseDatelist = pd.to_datetime(releaseDate).dt.year.astype(str).str[2:]
#         updateDate = pd.to_datetime(data["Current Version Release Date"], dayfirst=True)
#         updateDatelist = pd.to_datetime(updateDate).dt.year.astype(str).str[2:]

#         data["Original Release Date"] = releaseDatelist
#         data["Current Version Release Date"] = updateDatelist
        
        releaseDate = pd.to_datetime(data["Original Release Date"], dayfirst=True)
        release = pd.to_datetime(releaseDate).values.astype(np.int64) // 10 ** 9
        data["Original Release Date"] = release
        
        updateDate = pd.to_datetime(data["Current Version Release Date"], dayfirst=True)
        update = pd.to_datetime(updateDate).values.astype(np.int64) // 10 ** 9
        data["Current Version Release Date"] = update
        
    def update_app_purchase(self, data):
        n = data["In-app Purchases"]
        n = np.array(n)
        for i in range(0, len(n)):
            if type(n[i]) == type(""):
                n[i] = n[i].split(",")
            n[i] = pd.to_numeric(n[i])
            n[i] = (np.rint(n[i])).astype(int)
            n[i] = statistics.mean(n[i])
        data["In-app Purchases"] = n

    def setLanguageNumbers(self, data):
        featuresLang = data["Languages"]

        featuresLang = featuresLang.str.replace(", ", "")
        featuresLang = featuresLang.str.len() / 2

        data["Languages"] = featuresLang

    def setGenereNumbers(self, data):
        featuresLang = data["Genres"]

        featuresLang = featuresLang.str.split(", ")
        featuresLang = featuresLang.str.len()

        data["Genres"] = featuresLang
        
    def processAgeRating(self, data):
        data["Age Rating"] = data["Age Rating"].str.replace("+", "")
        data["Age Rating"] = data["Age Rating"].str.replace("+", "")
        
    def calculateGameAge(self, data):
        from datetime import date
        today = date.today()
        current_year = datetime.datetime.strptime(str(today),"%Y-%m-%d")
        timestamp = time.mktime(current_year.timetuple())
        series1 = int(timestamp)
        series2 = data["Original Release Date"].astype(int)
        data["Game Age"] = series1 - series2
        #self.data.drop(axis=1, columns="Current Version Release Date", inplace=True)

    def encodeGeneres(self, data):
        dummy = pd.get_dummies(data['Primary Genre'], prefix='', prefix_sep='')
        for col in dummy.columns:
            data[col] = dummy[col]
        data.drop(axis=1, columns="Primary Genre", inplace=True)
        
    def make_dummy_Frames(self, data):
        # change to array
        m = data["Languages"].astype(str)
        m = np.array(m)
        # splitting the list in each row
        for i in range(0, len(m)):
            m[i] = m[i].split(", ")
        # change to dataframe
        m = pd.DataFrame(m, columns=['Lang'])
        # make dummy variables
        k = pd.get_dummies(m.explode(['Lang'])).groupby(level=0).sum()
        # reset this column to be 1 because it is repeated many times
        k["Lang_ZH"] = k["Lang_ZH"].replace([2, 3, 4, 5, 6], 1)
        return k
    
    def makeGenereDummy(self, data):
        # change to array
        m = data["Genres"].astype(str)
        m = np.array(m)
        # splitting the list in each row
        for i in range(1, len(m)):
            m[i] = m[i].split(", ")
        # change to dataframe
        m = pd.DataFrame(m, columns=['Genres'])
        # make dummy variables
        k = pd.get_dummies(m.explode(['Genres'])).groupby(level=0).sum()
        for col in k.columns:
            data[col] = k[col].values
    
    
    def encodeDeveloper(self, data):
        dummy = pd.get_dummies(data['Developer'], prefix='', prefix_sep='')

        for col in dummy.columns:
            data[col] = dummy[col]
        data.drop(axis=1, columns="Developer", inplace=True)

    def dropZerosCols(self, data):
        cols = data.columns

        for col in cols:
            if (len(data) - (data[col] == 0).sum() < 15 ):
                data.drop(axis=1, columns=col, inplace=True)
                
    def dropUniqueCols(self, data):      
        cols = data.columns

        for col in cols:
            uniquePercent = (data[col].nunique()) / len(data)
            if (uniquePercent > 0.97 and col != "Size"):
                data.drop(axis=1, columns=col, inplace=True)
        
    def concatDummyVars(self, data, dummy):
        for col in dummy.columns:
            data[col] = dummy[col].values
        #data = pd.concat([self.Xtrain, dummy])
        
    def scalingFitTransform(self, data):
        cols = data.columns
        self.scale.fit(data)
        scaledData = self.scale.transform(data)
        scaled = pd.DataFrame(data=scaledData, columns=cols)
        return scaled
        
        
        
    def scalingTransform(self, test):
        cols = test.columns
        scaledTest = self.scale.transform(test)
        test = pd.DataFrame(data=scaledTest, columns=cols)
        return test
        
    def selectFeatures(self, data):
        

        featureSelector = SelectKBest(score_func=f_classif, k=38)
        topFeatures = featureSelector.fit_transform(data, y_train)
        selectedFeatures = featureSelector.get_support()
        for (s, a) in zip(selectedFeatures, self.trainFeatures):
            if(s == True):
                self.selectedFeaturesNames.append(a)
               

        topFeatures = pd.DataFrame(data=topFeatures)
        return topFeatures
    
    def mapSelectedFeatures(self, data):
        return data[self.selectedFeaturesNames]
        
        
        
    def cleanTrainData(self):
        ########
        print("-----------------------------------")
        print("PreProcessing Train")
        print("-----------------------------------")
        dummy_frames = self.make_dummy_Frames(self.Xtrain)
        self.concatDummyVars(self.Xtrain, dummy_frames)
        self.Xtrain.drop(columns=["Lang_nan"], inplace=True, axis=1)
        # print(type(dummy_frames))
        self.fillNulls(self.Xtrain, "In-app Purchases", '0')
        self.removeNulls(self.Xtrain)
        self.dateEncoding(self.Xtrain)
        self.encodeDeveloper(self.Xtrain)
        self.update_app_purchase(self.Xtrain)
        self.setLanguageNumbers(self.Xtrain)
        self.calculateGameAge(self.Xtrain)
        self.processAgeRating(self.Xtrain)
        self.encodeGeneres(self.Xtrain)
        self.makeGenereDummy(self.Xtrain)
        self.setGenereNumbers(self.Xtrain)
        #self.dropZerosCols(self.Xtrain)
        self.dropUniqueCols(self.Xtrain)
        self.langMod = self.Xtrain["Languages"].mode()
        self.fillNulls(self.Xtrain, "Languages", self.langMod[0])
        #self.Xtest["Languages"].fillna(axis=0, inplace=True, value=self.langMod[0])
        self.trainFeatures = self.Xtrain.columns
        scaledTrain = self.scalingFitTransform(self.Xtrain)

        return self.selectFeatures(scaledTrain)
        
        
    def chooseXTestCols(self):
        cols = self.Xtest.columns
        for col in cols:
            #print(self.trainFeatures.__contains__(col))
            if(self.trainFeatures.__contains__(col) == False):
                self.Xtest.drop(axis=1, columns=col, inplace=True)
        
        for feat in self.trainFeatures:
            if(cols.__contains__(feat) == False):
                self.Xtest[feat] = 0
        
    def cleanTestData(self):
        print("-----------------------------------")
        print("PreProcessing Test")
        print("-----------------------------------")
        dummy_frames = self.make_dummy_Frames(self.Xtest)
        self.concatDummyVars(self.Xtest, dummy_frames)
        #self.Xtest.drop(columns=["Lang_nan"], inplace=True, axis=1)
        self.fillNulls(self.Xtest, "In-app Purchases", '0')
        self.dateEncoding(self.Xtest)
        self.encodeDeveloper(self.Xtest)
        self.update_app_purchase(self.Xtest)
        self.setLanguageNumbers(self.Xtest)
        self.calculateGameAge(self.Xtest)
        self.processAgeRating(self.Xtest)
        self.encodeGeneres(self.Xtest)
        self.makeGenereDummy(self.Xtest)
        self.setGenereNumbers(self.Xtest)
        self.chooseXTestCols()
        self.fillNulls(self.Xtest, "Languages", self.langMod[0])
        #self.Xtest["Languages"].fillna(axis=0, inplace=True, value=self.langMod[0])
        xtestfeat = self.Xtest[self.trainFeatures]
        self.Xtest = xtestfeat
        scaledXtest = self.scalingTransform(self.Xtest)
        selectedXtest = self.mapSelectedFeatures(scaledXtest)
        return selectedXtest

preprocessing = PreProcessing(X_train, X_test)

topFeatures = None
testTopFeat = None







        

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint
import pickle




def saveModel(model, name):
    pickle.dump(model, open(name + '.pkl', 'wb'))

def logistcReg():
    classifier = LogisticRegression(solver='liblinear', multi_class='ovr')
    classifier.fit(topFeatures, y_train)
    ypred = classifier.predict(testTopFeat)
    print("-----------------------------------")
    print("Logistic Regression")
    print("-----------------------------------")
    print("Test Accuracy: " + str(round(classifier.score(testTopFeat, y_test) * 100, 2)))
    print("Train Accuracy: " + str(round(classifier.score(topFeatures, y_train) * 100, 2)))
    print("Confusion Matrix")
    cm = confusion_matrix(y_test, ypred)
    print(cm)
    print("-----------------------------------")
    print("Logistic Regression After Hyperparameter tuning")
    print("-----------------------------------")
    param_grid = {'C': [0.001, 0.01, 0.1, 1, 10, 100]}
    grid_search = GridSearchCV(classifier, param_grid, cv=5)
    grid_search.fit(topFeatures, y_train)
    
    print("Best value of C:", grid_search.best_params_['C'])
    print("Validation score:", grid_search.best_score_)
    print("Test Accuracy: " + str(round(grid_search.score(testTopFeat, y_test) * 100, 2)))
    print("Train Accuracy: " + str(round(grid_search.score(topFeatures, y_train) * 100, 2)))
    
    saveModel(grid_search, "logistic")



def KNNClassifier():
    knn = KNeighborsClassifier(n_neighbors=3)
    knn.fit(topFeatures, y_train)
    ypredknn = knn.predict(testTopFeat)
    print("-----------------------------------")
    print("KNN Classification")
    print("-----------------------------------")
    print("Test Accuracy: " + str(round(knn.score(testTopFeat, y_test) * 100, 2)))
    print("Train Accuracy: " + str(round(knn.score(topFeatures, y_train) * 100, 2)))
    print("Confusion Matrix")
    cm = confusion_matrix(y_test, ypredknn)
    print(cm)
    print("-----------------------------------")
    print("KNN Classification After Hyperparameter tuning")
    print("-----------------------------------")
    param_grid = {
        'n_neighbors': [1, 2, 3, 4, 5],
        'weights': ['uniform', 'distance'],
        'algorithm': ['ball_tree', 'kd_tree', 'brute']
    }
    knn_grid = GridSearchCV(
        estimator=knn,
        param_grid=param_grid,
        cv=5,
        verbose=2,
        n_jobs=-1
    )
    knn_grid.fit(topFeatures, y_train)
    print("Best Params" + str(knn_grid.best_params_))    
    print("Best Score" + str(round(knn_grid.best_score_ * 100, 2)))  
    ypredknnhyp = knn_grid.predict(testTopFeat)
    print("Test Accuracy: " + str(round(knn_grid.score(testTopFeat, y_test) * 100, 2)))
    print("Train Accuracy: " + str(round(knn_grid.score(topFeatures, y_train) * 100, 2)))
    print("Confusion Matrix")
    cm = confusion_matrix(y_test, ypredknnhyp)
    print(cm)
    
    saveModel(knn_grid, "knn")
    
def SVMClassifier():
    C=10
    SVM = svm.SVC(kernel='poly', degree=10, C=C)
    SVM.fit(topFeatures, y_train)
    ypredsvm = SVM.predict(testTopFeat)
    print("-----------------------------------")
    print("SVM Classification")
    print("-----------------------------------")
    print("Test Accuracy: " + str(round(SVM.score(testTopFeat, y_test) * 100, 2)))
    print("Train Accuracy: " + str(round(SVM.score(topFeatures, y_train) * 100, 2)))
    print("Confusion Matrix")
    cm = confusion_matrix(y_test, ypredsvm)
    print(cm)
    
    saveModel(SVM, "svm")



def DTreeClassifier():
    dt = DecisionTreeClassifier()
    dt.fit(topFeatures, y_train)
    ypreddt = dt.predict(testTopFeat)
    print("-----------------------------------")
    print("Decision Tree Classification")
    print("-----------------------------------")
    print("Test Accuracy: " + str(round(dt.score(testTopFeat, y_test) * 100, 2)))
    print("Train Accuracy: " + str(round(dt.score(topFeatures, y_train) * 100, 2)))
    print("Confusion Matrix")
    cm = confusion_matrix(y_test, ypreddt)
    print(cm)
    print("-----------------------------------")
    print("Decision Tree Classification After Hyperparameter tuning")
    print("-----------------------------------")
    param_grid = {'max_depth': [2, 3, 4, 5, 6, 7, 8], 'min_samples_split': [2, 3, 4, 5, 6, 7, 8]}
    dt_grid = GridSearchCV(dt, param_grid, cv=5)
    dt_grid.fit(topFeatures, y_train)
    ypreddthyp = dt_grid.predict(testTopFeat)
    print("Best Params" + str(dt_grid.best_params_))    
    print("Best Score" + str(round(dt_grid.best_score_ * 100, 2)))  
    print("Test Accuracy: " + str(round(dt_grid.score(testTopFeat, y_test) * 100, 2)))
    print("Train Accuracy: " + str(round(dt_grid.score(topFeatures, y_train) * 100, 2)))
    print("Confusion Matrix")
    cm = confusion_matrix(y_test, ypreddthyp)
    print(cm)
    
    saveModel(dt_grid, "dt")

# 'bootstrap': False, 'criterion': 'gini', 'max_depth': 15, 'max_features': 'log2', 'min_samples_leaf': 2, 'min_samples_split': 13, 'n_estimators': 39
def RFClassifier():
    rf = RandomForestClassifier(max_depth=(15), max_features="log2", min_samples_leaf=2, min_samples_split= 13, n_estimators = 39)
    rf.fit(topFeatures, y_train)
    ypredrf = rf.predict(testTopFeat)
    print("-----------------------------------")
    print("Random Forest Classification")
    print("-----------------------------------")
    print("Test Accuracy: " + str(round(rf.score(testTopFeat, y_test) * 100, 2)))
    print("Train Accuracy: " + str(round(rf.score(topFeatures, y_train) * 100, 2)))
    print("Confusion Matrix")
    cm = confusion_matrix(y_test, ypredrf)
    print(cm)
    print("-----------------------------------")
    print("Random Forest Classification after hyperparamter tuning")
    print("-----------------------------------")
    #param_grid = {'n_estimators': [50, 100, 150, 200], 'max_depth': [2, 3, 4, 5, 6, 7, 8], 'min_samples_split': [2, 3, 4, 5, 6, 7, 8]}
    
    param_grid = {
        'n_estimators': randint(10, 100),
        'max_features': ['auto', 'sqrt', 'log2'],
        'max_depth': [None] + list(range(5, 50, 5)),
        'min_samples_split': randint(2, 20),
        'min_samples_leaf': randint(1, 20),
        'bootstrap': [True, False],
        'criterion': ['gini', 'entropy']
    }
    
    rf_grid = RandomizedSearchCV(
        estimator=rf,
        param_distributions=param_grid,
        n_iter=100,
        cv=5,
        verbose=2,
        random_state=42,
        n_jobs=-1
    )
    rf_grid.fit(topFeatures, y_train)
    ypredrfhyp = rf_grid.predict(testTopFeat)
    print("Best Params" + str(rf_grid.best_params_))    
    print("Best Score" + str(round(rf_grid.best_score_ * 100, 2)))  
    print("Test Accuracy: " + str(round(rf_grid.score(testTopFeat, y_test) * 100, 2)))
    print("Train Accuracy: " + str(round(rf_grid.score(topFeatures, y_train) * 100, 2)))
    print("Confusion Matrix")
    cm = confusion_matrix(y_test, ypredrfhyp)
    print(cm)
    
    saveModel(rf_grid, "rf")



def loadModel(name, train , test, y):
    print("-----------------------------------")
    print("Loading Model " + name)
    print("-----------------------------------")
    pickled_model = pickle.load(open(name + '.pkl', 'rb'))
    ypredpickle = pickled_model.predict(test)
    print("Best Params" + str(pickled_model.best_params_))    
    print("Best Score" + str(round(pickled_model.best_score_ * 100, 2)))  
    print("Test Accuracy: " + str(round(pickled_model.score(test, y) * 100, 2)))
    #print("Train Accuracy: " + str(round(pickled_model.score(train, y_train) * 100, 2)))
    print("Confusion Matrix")
    cm = confusion_matrix(y, ypredpickle)
    print(cm)
    
def loadNativeModel(name):
    print("-----------------------------------")
    print("Loading Model " + name)
    print("-----------------------------------")
    pickled_model = pickle.load(open(name + '.pkl', 'rb'))
    ypredpickle = pickled_model.predict(testTopFeat)
    print("Test Accuracy: " + str(round(pickled_model.score(testTopFeat, y_test) * 100, 2)))
    print("Train Accuracy: " + str(round(pickled_model.score(topFeatures, y_train) * 100, 2)))
    print("Confusion Matrix")
    cm = confusion_matrix(y_test, ypredpickle)
    print(cm)
    

print("-----------------------------------")
print("Enter a choice")
print("1) Train Models")
print("2) Load Previously Trained Models")
print("3) Test New Data")
print("-----------------------------------")

choice = input();
if(choice == "1"):
    topFeatures = preprocessing.cleanTrainData()
    testTopFeat = preprocessing.cleanTestData(X_test)

    logistcReg()
    KNNClassifier()
    SVMClassifier()
    DTreeClassifier()
    RFClassifier()
    
elif (choice == "2"):
    topFeatures = preprocessing.cleanTrainData()
    testTopFeat = preprocessing.cleanTestData()

    loadModel("logistic", topFeatures, testTopFeat, y_test)
    loadModel("dt", topFeatures, testTopFeat, y_test)
    loadModel("knn", topFeatures, testTopFeat, y_test)
    loadModel("rf", topFeatures, testTopFeat, y_test)
    loadNativeModel("svm")
    
    
    
else:
    print("Enter filename")
    filename = input();
    testdata = pd.read_csv(filename);
    testdata.drop_duplicates(inplace=True)
    test = testdata.iloc[:, :-1]
    ytest = gamedata.iloc[:, -1]
    pr2 = PreProcessing(X_train, test)
    cleanedtrain = pr2.cleanTrainData()
    cleanedTest = pr2.cleanTestData()
    loadModel("rf", topFeatures, cleanedTest, ytest)






















