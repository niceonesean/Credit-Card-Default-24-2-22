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

from keras.models import Sequential
from keras.layers import Dense, Dropout
model=Sequential()
model.add(Dense(6, input_dim=3, activation="relu"))
model.add(Dropout(0.2))
model.add(Dense(4, activation="relu"))
model.add(Dropout(0.2))
model.add(Dense(1,activation="sigmoid"))
model.compile(loss="binary_crossentropy",metrics=["accuracy"])
model.fit(X_train, Y_train,batch_size=10,epochs=10)
model.evaluate(X_test,Y_test)
model.save("CCDefault")

#Flask
from flask import Flask
app=Flask(__name__)
from keras.models import load_model
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
        print(pred)
        pred=pred[0]
        s="The predicted default is : " + str(pred)
        return(render_template("index.html",result=s))
    else:
        return(render_template("index.html",result="2"))
    
if __name__=="__main__":
    app.run()    












