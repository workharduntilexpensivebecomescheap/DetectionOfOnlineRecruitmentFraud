
#importing necessary libraries
import pandas as pd
import numpy as np
from flask import Flask,render_template,flash,request,flash, session
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score,classification_report,recall_score,f1_score
import nltk
from sklearn.feature_extraction.text import HashingVectorizer
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

app=Flask(__name__)



@app.route('/')
def index():
    return render_template('index.html')



@app.route('/prediction', methods=['POST','GET'])
def prediction():
    global x_train, y_train
    
    if request.method == "POST":
        f1 = request.form['text']
        print(f1)

        # Load data
        df = pd.read_csv("fake_job_postings.csv", encoding='latin-1')
        df = df.dropna()
        df = df[['description', 'fraudulent']]

        # Define features and target
        x = df['description']
        y = df['fraudulent']

        

        # Split data into training and testing sets
        x_train, x_test, y_train, y_test = train_test_split(x, y, stratify=y, test_size=0.2, random_state=42)

        # Vectorize text data
        hvectorizer = HashingVectorizer(n_features=52991, norm=None, alternate_sign=False)
        x_train_vectorized = hvectorizer.transform(x_train.ravel())

        # Train the model
        import pickle
        filename = 'XGBoost1.sav'
        model = pickle.load(open(filename, 'rb'))
        # Transform input text and make prediction
        # f1_vectorized = hvectorizer.transform([f1])
        result = model.predict(hvectorizer.transform([f1]))
        result = result[0]
        print(result)
        # Define response message
        if result == 0:
            msg = 'The Job Post is Genuine'
            return render_template('prediction.html', msg=msg)
        elif result == 1:
            msg = 'The job post is Fake'
            return render_template('prediction.html', msg=msg)

    return render_template('prediction.html')

if __name__ == '__main__':
    app.run(debug=True)
  













