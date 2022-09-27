from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)
model = pickle.load(open('trained_model.sav', 'rb')) 
labels = ['setosa', 'versicolor', 'virginica']

@app.route('/') #http://google/com/
def index():
    return render_template('index.html', info = 'Predicting type of Iris from give information. Types of iris: {}, {} and {}'.format(labels[0], labels[1], labels[2]))

@app.route('/predict', methods=['POST'])
def predict():
    features = [np.array([float(x) for x in request.form.values()])]
    prediction = model.predict(features)
    output = labels[prediction.item()]
    return render_template('index.html', prediction_text='Type of Iris is {}'.format(output))
    
    
app.run(port=5000, debug=True)