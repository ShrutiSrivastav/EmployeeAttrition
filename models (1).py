# -*- coding: utf-8 -*-
"""
Created on Mon Jan 21 14:37:51 2019

@author: ACER
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt  
import seaborn as sns
from sklearn import model_selection
from sklearn import linear_model
from sklearn import preprocessing
from sklearn import utils
from sklearn import metrics
from sklearn.metrics import roc_curve,auc 
from sklearn import feature_selection
from sklearn import neighbors
from sklearn import tree
from sklearn import naive_bayes
from sklearn.model_selection import GridSearchCV

def modelstats(Xtrain,Xtest,ytrain,ytest):
    stats=[]
    modelnames=["LR","DecisionTree","KNN","NB"]
    models=list()
    models.append(linear_model.LogisticRegression())
    models.append(tree.DecisionTreeClassifier())
    models.append(neighbors.KNeighborsClassifier())
    models.append(naive_bayes.GaussianNB())
    for name,model in zip(modelnames,models):
        if name=="KNN":
            k=[l for l in range(11,33,2)]
            grid={"n_neighbors":k}
            grid_obj = GridSearchCV(estimator=model,param_grid=grid,scoring="f1")
            grid_fit =grid_obj.fit(Xtrain,ytrain)
            model = grid_fit.best_estimator_
            model.fit(Xtrain,ytrain)
            name=name+"("+str(grid_fit.best_params_["n_neighbors"])+")"
            print(grid_fit.best_params_)
        else:
            model.fit(Xtrain,ytrain)
        trainprediction=model.predict(Xtrain)
        testprediction=model.predict(Xtest)
        scores=list()
        scores.append(name+"-train")
        scores.append(metrics.accuracy_score(ytrain,trainprediction))
        scores.append(metrics.precision_score(ytrain,trainprediction))
        scores.append(metrics.recall_score(ytrain,trainprediction))
        scores.append(metrics.roc_auc_score(ytrain,trainprediction))
        stats.append(scores)
        scores=list()
        scores.append(name+"-test")
        scores.append(metrics.accuracy_score(ytest,testprediction))
        scores.append(metrics.precision_score(ytest,testprediction))
        scores.append(metrics.recall_score(ytest,testprediction))
        scores.append(metrics.roc_auc_score(ytest,testprediction))
        stats.append(scores)
    
    colnames=["MODELNAME","ACCURACY","PRECISION","RECALL","AUC"]
    return pd.DataFrame(stats,columns=colnames)

eadf=pd.read_csv("f:/datasets/EmployeeAttrition.csv")
eadf.info()
pd.set_option("display.max_columns",100)

le=preprocessing.LabelEncoder()
eadf["Attrition"]=le.fit_transform(eadf["Attrition"])
eadf["BusinessTravel"]=le.fit_transform(eadf["BusinessTravel"])
eadf["Department"]=le.fit_transform(eadf["Department"])
eadf["EducationField"]=le.fit_transform(eadf["EducationField"])
eadf["Gender"]=le.fit_transform(eadf["Gender"])
eadf["JobRole"]=le.fit_transform(eadf["JobRole"])
eadf["MaritalStatus"]=le.fit_transform(eadf["MaritalStatus"])
eadf["OverTime"]=le.fit_transform(eadf["OverTime"])

#-------------------------------------------------------------------------------
X = eadf.drop(["Attrition","EmployeeCount","Over18","StandardHours","EmployeeNumber"],axis=1)
y = eadf["Attrition"]

obj=feature_selection.SelectKBest(feature_selection.f_classif,k=12)
obj.fit(X,y)
obj.get_support()
cols=X.columns.values[obj.get_support()]
cols
X=eadf[cols]
X.info()
Xtrain,Xtest,ytrain,ytest = model_selection.train_test_split(X,y,
                            test_size=.4,random_state=42,stratify=y)

print(modelstats(Xtrain,Xtest,ytrain,ytest))      
X.columns.values.shape
X.info()

#---------------------------------------------------------------------------------
X = eadf.drop(["Attrition","EmployeeCount","Over18","StandardHours","EmployeeNumber"],axis=1)
y = eadf["Attrition"]

obj=feature_selection.SelectKBest(feature_selection.f_classif,k=12)
obj.fit(X,y)
obj.get_support()
cols=X.columns.values[obj.get_support()]
cols
X=eadf[cols]
X.info()
Xtrain,Xtest,ytrain,ytest = model_selection.train_test_split(X,y,test_size=.2,random_state=42,stratify=y)

print(modelstats(Xtrain,Xtest,ytrain,ytest))      
X.columns.values.shape
X.info()

#---------------------------------------------------------------------------------
X = eadf.drop(["Attrition","EmployeeCount","Over18","StandardHours","EmployeeNumber"],axis=1)
y = eadf["Attrition"]

obj=feature_selection.SelectKBest(feature_selection.f_classif,k=13)
obj.fit(X,y)
obj.get_support()
cols=X.columns.values[obj.get_support()]
cols
X=eadf[cols]
X.info()
Xtrain,Xtest,ytrain,ytest = model_selection.train_test_split(X,y,test_size=.2,random_state=42,stratify=y)

print(modelstats(Xtrain,Xtest,ytrain,ytest))      
X.columns.values.shape
X.info()

cols=["BusinessTravel","Department","EducationField","Gender","JobRole","MaritalStatus","OverTime"]
pref=["btravel","dept","edufield","g","jr","m","ot"]

Xohe=pd.get_dummies(X,columns=cols,prefix=pref)
X.head()
Xohe.columns.values.shape
Xohe.columns.values

Xtrain,Xtest,ytrain,ytest = model_selection.train_test_split(Xohe,y,test_size=.2,random_state=42,stratify=y)

selector=feature_selection.RFECV(estimator=linear_model.LogisticRegression(C=.1),cv=5,step=3,scoring="roc_auc")
selector.fit(Xtrain,ytrain)
Xohe.columns.values[selector.get_support()].shape
Xtrain1=Xtrain[Xohe.columns.values[selector.get_support()]]
Xtest1=Xtest[Xohe.columns.values[selector.get_support()]]

print(modelstats(Xtrain1,Xtest1,ytrain,ytest))

Xcont=eadf[["Attrition","DailyRate","MonthlyIncome","MonthlyRate"]]
sns.heatmap(Xcont.corr(),annot=True)
Xcont.corr()



print(modelstats(Xtrain,Xtest,ytrain,ytest))

cols=["BusinessTravel","Department","EducationField","Gender","JobRole","MaritalStatus","OverTime"]
pref=["btravel","dept","edufield","g","jr","m","ot"]

cl=["Education","EnvironmentSatisfaction","JobSatisfaction","PerformanceRating","RelationshipSatisfaction","WorkLifeBalance"]
pref=["edu","envsat","jobsat","prat","rs","wlb"]
Xohe=pd.get_dummies(X,columns=cl,prefix=pref)
X.head()
Xohe.columns.values.shape
Xohe.columns.values
X.info()

Xtrain,Xtest,ytrain,ytest = model_selection.train_test_split(Xohe,y,test_size=.2,random_state=42,stratify=y)

print(modelstats(Xtrain,Xtest,ytrain,ytest))
Xohe.columns.values.shape

#------------------------------------------------------------------------------
fpr=dict()
tpr=dict()
roc_auc=dict()
prec,recall,thres=metrics.precision_recall_curve(ytest,)
fpr,tpr,thres=roc_curve(ytest,testpredicted)
roc_auc=auc(fpr,tpr)
plt.figure()
lw=2
plt.plot(tpr,fpr,color="darkorange",lw=lw,label="ROC curve  (area=%0.2f)" % roc_auc)
plt.plot([0,1],[0,1],color="navy",lw=lw,linestyle="--")
plt.legend(loc="lower right")
plt.show()








