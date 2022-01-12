"""
@author: Kiran Elias
"""
import flask
import pickle
import numpy as np
from flask import Flask, render_template, request

app=Flask(__name__)
loaded_model = pickle.load(open("model.pkl","rb"))

@app.route('/')
@app.route('/index')

def index():
    return flask.render_template('index.html')

@app.route('/result',methods = ['POST'])
def result():     
   int_features= [int(x) for x in request.form.values()] 
   final_features=[np.array(int_features)]
   #print(final_features)
   loaded_model = pickle.load(open("model.pkl","rb"))
   prediction=loaded_model.predict(final_features)[0]
   if prediction==0:
       op="Low priced phone"
   elif prediction==1:
        op='Economy phone'
   elif prediction==2:
        op='Mid range phone'
   else:
       op='High priced phone'
        
   
   return render_template('result.html',prediction=op)

if __name__ == '__main__':
	app.run()
