# -*- coding: utf-8 -*-
"""
Created on Wed Feb 23 19:29:50 2022

@author: User
"""
#Data Wrangling
import pandas as pd
df=pd.read_csv("C:/Users/User/Desktop/BC3409/Datasets/Credit Card Default II (balance).csv")

#Check for imbalance (no need SMOTE, good proportion of yes and no)
df.default.value_counts() 

#Remove NA
df = df.dropna()

#Remove non-numeric
for i in df.columns:
    df1 = pd.to_numeric(df[i], errors = "coerce")
    df = df[df1.notnull()] # make non-numeric null then remove null

#Remove outliers (3 rows)
import numpy as np
from scipy import stats
z_scores = stats.zscore(df.astype(np.float))  #zscore conversion need float
absz = np.abs(z_scores) #convert all to positive because the parity is not important
f1 = (absz < 3).all(axis=1) #3 is your choice, axis =1 means by columns, f is a flag
df=df[f1]

#Split X, Y
X=df.loc[:,["income","age","loan"]]
Y=df.loc[:,["default"]]

#Check collinearity
from statsmodels.stats.outliers_influence import variance_inflation_factor
vif_data = pd.DataFrame()
vif_data["feature"] = X.columns
vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(len(X.columns))]
print(vif_data)

#Visualise the data
df.hist()
import seaborn as sns
sns.heatmap(df.corr())

#normalization
from scipy import stats
import numpy as np 
pd.set_option('display.max_rows', 10)
for i in X.columns:
    X[i]=stats.zscore(X[i].astype(np.float))
print(X)

#Train test split
import numpy as np
np.random.seed(2)
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.3)


#Logistic Regression
"""Assumptions
1. binary logistic regression requires the dependent variable to be binary
2. observations independent of each other,should not come from repeated measurements or matched data.
3. little or no multicollinearity among the independent variables.  
4. independent variables are linearly related to the log odds
5. a large sample size: general guideline is minimum of 10 cases with the least frequent outcome for each independent variable in the model
"""
from sklearn import linear_model
model=linear_model.LogisticRegression()
model.fit(X_train,Y_train)
pred=model.predict(X_train)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y_train, pred)
print(cm)
accuracy = (cm[0,0]+cm[1,1])/sum(sum(cm))
print(accuracy)

pred = model.predict(X_test)
cm = confusion_matrix(Y_test, pred)
print(cm)
accuracy = (cm[0,0]+cm[1,1])/sum(sum(cm))
print(accuracy)

#Anova Table
import statsmodels.api as sm 
model=sm.Logit(Y,X)
result=model.fit()
print(result.summary2())

#Decision Tree
"""Assumptions
1. Based on attribute values records are distributed recursively
2. We use a statistical method for ordering attributes as a root node or the internal node
"""
from sklearn import tree
model = tree.DecisionTreeClassifier(max_depth=5)
model.fit(X_train, Y_train)
pred = model.predict(X_train)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y_train, pred)
print(cm)
accuracy = (cm[0,0] + cm[1,1])/(sum(sum(cm)))
print(accuracy)

pred = model.predict(X_test)
cm = confusion_matrix(Y_test, pred)
print(cm)
accuracy = (cm[0,0] + cm[1,1])/(sum(sum(cm)))
print(accuracy)

#Feature importance
fr = model.feature_importances_ 
print(fr)

#Plot tree
import matplotlib.pyplot as plt 
plt.subplots(figsize=(20, 10))
tree.plot_tree(model, fontsize=10)

#Optimisation using max depth
X_train1, X_test1, Y_train1, Y_test1 = train_test_split(X_train,Y_train)
for i in range(20):
    model = tree.DecisionTreeClassifier(max_depth=i+1)
    model.fit(X_train1, Y_train1)
    pred = model.predict(X_test1)
    cm = confusion_matrix(Y_test1, pred)
    accuracy = (cm[0,0] + cm[1,1])/(sum(sum(cm)))
    print(i)
    print(accuracy)
    print("========================") #optimal when i = 6 so use max_depth = 7
    
#Decision Tree using max depth of 7
from sklearn import tree
model = tree.DecisionTreeClassifier(max_depth=7)
model.fit(X_train, Y_train)
pred = model.predict(X_train)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y_train, pred)
print(cm)
accuracy = (cm[0,0] + cm[1,1])/(sum(sum(cm)))
print(accuracy)

pred = model.predict(X_test)
cm = confusion_matrix(Y_test, pred)
print(cm)
accuracy = (cm[0,0] + cm[1,1])/(sum(sum(cm)))
print(accuracy) #Test accuracy increased from 98% to 99% after optimisation

#Random Forest (using max depth 7 from optimisation)
"""Assumptions
1. No formal distributional assumptions
2. random forests are non-parametric, can handle skewed and multi-modal data as well as categorical data that are ordinal or non-ordinal.
"""
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(max_depth=7)
model.fit(X_train, Y_train)
pred = model.predict(X_train)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y_train, pred)
print(cm)
accuracy = (cm[0,0] + cm[1,1])/sum(sum(cm))
print(accuracy)

pred = model.predict(X_test)
cm = confusion_matrix(Y_test, pred)
print(cm)
accuracy = (cm[0,0] + cm[1,1])/sum(sum(cm))
print(accuracy)

#XG Boost
from sklearn.ensemble import GradientBoostingClassifier
model = GradientBoostingClassifier(max_depth=7)
model.fit(X_train, Y_train)
pred = model.predict(X_train)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y_train, pred)
print(cm)
accuracy = (cm[0,0] + cm[1,1])/sum(sum(cm))
print(accuracy)

pred = model.predict(X_test)
cm = confusion_matrix(Y_test, pred)
print(cm)
accuracy = (cm[0,0] + cm[1,1])/sum(sum(cm))
print(accuracy)

#Neural Network
"""Assumptions
1. Artificial Neurons are arranged in layers, which are sequentially arranged
2. Neurons within the same layer do not interact or communicate to each other
3. All inputs enter into the network through the input layer and passes through the output layer
4. All hidden layers at same level should have same activation function
5. Artificial neuron at consecutive layers are densely connected
6. Every inter-connected neural network has it’s own weight and biased associated with it
"""
from keras.models import Sequential
from keras.layers import Dense, Dropout
model=Sequential()
model.add(Dense(6, input_dim=3, activation="relu"))
model.add(Dropout(0.2))
model.add(Dense(4, activation="relu"))
model.add(Dropout(0.2))
model.add(Dense(1,activation="sigmoid"))
model.compile(loss="binary_crossentropy",metrics=["accuracy"])
model.fit(X_train, Y_train,batch_size=100,epochs=10)
model.evaluate(X_test,Y_test)
model.summary()
model.weights

#MLP Classifier
from sklearn.neural_network import MLPClassifier
model=MLPClassifier(solver="lbfgs",hidden_layer_sizes=(8,2)).fit(X_train,Y_train)
pred=model.predict(X_train)
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(Y_train,pred)
print(cm)
accuracy = (cm[0,0] + cm[1,1])/sum(sum(cm))
print(accuracy)


model=MLPClassifier(solver="lbfgs",hidden_layer_sizes=(8,2)).fit(X_test,Y_test)
pred=model.predict(X_test)
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(Y_test,pred)
print(cm)
accuracy = (cm[0,0] + cm[1,1])/sum(sum(cm))
print(accuracy)


import joblib
joblib.dump(model,"CCDefault")
model=joblib.load("CCDefault")

#Flask
pip install flask
from flask import Flask
app=Flask(__name__)

from flask import request, render_template

@app.route("/",methods=["GET","POST"])
def index():
    if request.method=="POST":
        income=request.form.get("income") 
        age=request.form.get("age")
        loan=request.form.get("loan")
        print(income,age,loan)
        model=load_model("CCDefault")
        pred=model.predict([[float(income),float(age),float(loan)]])
        s="The predicted default is : " + str(pred)
        return(render_template("index.html",result=s))
    else:
        return(render_template("index.html",result="2"))
    
if __name__=="__main__":
    app.run()    












