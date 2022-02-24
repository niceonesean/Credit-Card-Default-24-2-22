
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












