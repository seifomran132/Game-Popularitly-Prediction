import numpy as np
import pandas as pd
import statistics
import sklearn.preprocessing as pre
from sklearn.feature_selection import SelectPercentile
from sklearn.feature_selection import chi2, f_classif
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge, RidgeCV, Lasso
import pickle



gamedata = pd.read_csv("games-regression-dataset.csv");

gamedata.dropna(axis=0, subset=["Languages"], inplace=True)


X = gamedata.iloc[:, :-1]
y = gamedata.iloc[:, -1]

# Data Spliting
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2,shuffle=True,random_state=10)


X_train = pd.DataFrame(data=X_train)
X_test = pd.DataFrame(data=X_test)

def calculateCorrelaction(xtrain, ytrain):
    corrdata = pd.DataFrame.copy(xtrain, deep=True)
    corrdata["Average User Rating"] = ytrain
    corelation = corrdata.corr()
    plt.figure(figsize = (15, 15))
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
        releaseDate = self.data["Original Release Date"]
        releaseDatelist = pd.to_datetime(releaseDate).dt.year.astype(str).str[2:]
        updateDate = self.data["Current Version Release Date"]
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

    
    def dropCols(self):
        self.data.drop(axis=1, columns="URL", inplace=True)
        self.data.drop(axis=1, columns="Name", inplace=True)
        self.data.drop(axis=1, columns="ID", inplace=True)
        self.data.drop(axis=1, columns="Icon URL", inplace=True)
        self.data.drop(axis=1, columns="Description", inplace=True)
        self.data.drop(axis=1, columns="Developer", inplace=True)
        self.data.drop(axis=1, columns="Primary Genre", inplace=True)

     
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
        series1 = 24
        print(series1)
        series2 = self.data["Original Release Date"].astype(int)
        self.data["Game Age"] = series1 - series2
        self.data.drop(axis=1, columns="Original Release Date", inplace=True)
        self.data.drop(axis=1, columns="Current Version Release Date", inplace=True)
        
    def encodeGeneres(self):
        dummy = pd.get_dummies(self.data['Primary Genre'],prefix='',prefix_sep='')
        
        for col in dummy.columns:
            self.data[col] = dummy[col]
    
    def encodeDeveloper(self):
        dummy = pd.get_dummies(self.data['Developer'],prefix='',prefix_sep='')
        
        for col in dummy.columns:
            self.data[col] = dummy[col]
        

        
    
    def dropZerosCols(self):
        cols = self.data.columns
        
        for col in cols:
            zerosPercent = (self.data[col] == 0).sum() / len(self.data)
            if(zerosPercent > 0.90):
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
        # self.encodeDeveloper()
        self.setGenereNumbers()
        self.dropZerosCols()
        self.dropCols()    
        self.normalizeData()

 



# Run PreProcessing Pipline on Train and Test Data Sets



def performPreProcessing():
    train = PreProcessing(X_train)
    train.clean()

    test = PreProcessing(X_test)
    test.clean()



# X_train2 = X_train
# FeatureSelection = SelectPercentile(score_func = chi2, percentile=60) # score_func can = f_classif
# def featureSelection(myx, myy):
#     y_train2 = np.array(myy).astype(int)
#     myx = FeatureSelection.fit_transform(myx, y_train2)
#     return myx

# X_train = featureSelection(X_train, y_train)
# X_test = featureSelection(X_test, y_test)


# Polynomial Model


from sklearn import linear_model
from sklearn import metrics
from sklearn.preprocessing import PolynomialFeatures




y_test = np.array(y_test)


# Declare file names for save and load the model

linearfilename = 'multi_linear_model.sav'
ridgefilename = 'ridgereg.sav'
lassofilename = 'lassoreg.sav'
polyfilename = 'poly_model.sav'

# Declare models

lr = linear_model.LinearRegression()
ridgeReg = Ridge(alpha=0.1)
lasso = Lasso(alpha = 0.001)
poly_model = linear_model.LinearRegression()

def linearRegressionModel():
    
    # Fit the model
    
    lr.fit(X_train,y_train)
    yprediction= lr.predict(X_test)

    # Save the Model
    pickle.dump(lr, open(linearfilename, 'wb'))

    print("\nSimple Linear Regression Model----------------------------------\n")
    print('Co-efficient of linear regression',lr.coef_)
    print('Intercept of linear regression model',lr.intercept_)
    print('Mean Square Error', metrics.mean_squared_error(y_test, yprediction))
    print('Root Mean Square Error', np.sqrt(metrics.mean_squared_error(y_test, yprediction)))

    return yprediction



def loadLinearRegression():
    load_model = pickle.load(open(linearfilename, 'rb'))
    y_pred_pickle = load_model.predict(X_test)

    print("\nSimple Linear Regression Model (Saved)----------------------------------\n")
    print('Co-efficient of linear regression',load_model.coef_)
    print('Intercept of linear regression model',load_model.intercept_)
    print('Mean Square Error', metrics.mean_squared_error(y_test, y_pred_pickle))


poly_features = PolynomialFeatures(degree=2)
def polyRegressionModel():
    X_train_poly = poly_features.fit_transform(X_train)
    poly_model.fit(X_train_poly, y_train)
    ypredpoly = poly_model.predict(poly_features.transform(X_test))
    
    pickle.dump(poly_model, open(polyfilename, 'wb'))
    
    print("\nPolynomial Regression Model----------------------------------\n")
    print('Co-efficient of linear regression',poly_model.coef_)
    print('Intercept of linear regression model',poly_model.intercept_)
    print('Mean Square Error', metrics.mean_squared_error(y_test, ypredpoly))

def loadPolyModel():
    load_model = pickle.load(open(polyfilename, 'rb'))
    y_pred_poly_pickle = load_model.predict(poly_features.transform(X_test))   
    print("\nPolynomial Regression Model----------------------------------\n")
    print('Co-efficient of linear regression',poly_model.coef_)
    print('Intercept of linear regression model',poly_model.intercept_)
    print('Mean Square Error', metrics.mean_squared_error(y_test, y_pred_poly_pickle))




def ridgeRegressionModel():
    ridgeReg.fit(X_train,y_train)
    ypredridge = ridgeReg.predict(X_test)
    pickle.dump(ridgeReg, open(ridgefilename, 'wb'))
    
    
    print("\nRidge Regression Model--------------------------------------------\n")
    print('Co-efficient of linear regression',ridgeReg.coef_)
    print('Intercept of linear regression model',ridgeReg.intercept_)
    print('Mean Square Error', metrics.mean_squared_error(y_test, ypredridge))
    
    return ypredridge


def loadRidgeModel():
    load_model = pickle.load(open(ridgefilename, 'rb'))
    y_pred_ridge_pickle = load_model.predict(X_test)


    print("\nRidge Regression Model--------------------------------------------\n")
    print('Co-efficient of linear regression',load_model.coef_)
    print('Intercept of linear regression model',load_model.intercept_)
    print('Mean Square Error', metrics.mean_squared_error(y_test, y_pred_ridge_pickle))





def lassoRegressionModel():
    lasso.fit(X_train,y_train)
    ypredlasso = lasso.predict(X_test)
    pickle.dump(lasso, open(lassofilename, 'wb'))
    
    
    
    print("\nLasso Regression Model--------------------------------------------\n")
    print('Co-efficient of linear regression',lasso.coef_)
    print('Intercept of linear regression model',lasso.intercept_)
    print('Mean Square Error', metrics.mean_squared_error(y_test, ypredlasso))
    
    return ypredlasso


#Lasso regression model


def loadLassoModel():
    load_model = pickle.load(open(lassofilename, 'rb'))
    y_pred_lasso_pickle = load_model.predict(X_test)

    print("\nLasso Regression Model--------------------------------------------\n")
    print('Co-efficient of linear regression',load_model.coef_)
    print('Intercept of linear regression model',load_model.intercept_)
    print('Mean Square Error', metrics.mean_squared_error(y_test, y_pred_lasso_pickle))





# Main



choice = 0
print("\nPlease enter a choise\n");
print("1) Train Model\n2) Load Model");
choice = input()

print(choice)


def selectTopFeatures():
    corrfeat= calculateCorrelaction(X_train, y_train)
    topfeatures = corrfeat.index[abs(corrfeat["Average User Rating"]) > 0.02]
    topfeatures = topfeatures.delete(-1)
    return topfeatures


if(choice == "1"):
    performPreProcessing()
    topfet = selectTopFeatures()
    X_train = X_train[topfet]
    X_test = X_test[topfet]
    ypred1 = linearRegressionModel()
    polyRegressionModel()
    ypred2 = lassoRegressionModel()
    ypred3 = ridgeRegressionModel()
elif(choice == "2"):
    # performPreProcessing()
    test = PreProcessing(X_test)
    test.clean()
    
    loadLinearRegression()
    # loadPolyModel()
    loadLassoModel()
    loadRidgeModel()
    
    
def maual(X,Y,Pre):
    X=np.array(X)
    Y=np.array(Y)
    print(X.shape,Y.shape)
    for i in range(X.shape[1]):
        plt.scatter(X[:,i],Y)

    for i in range(X.shape[1]):
        plt.plot(X[:,i],Pre, color='red', linewidth=3)

    plt.show()


maual(X_test,y_test,ypred1)


















