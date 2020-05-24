import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)
model = pickle.load(open('back_order.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('backorder.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    int_features = [x for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)

    output = prediction

    return render_template('backorder.html', prediction_text='Company Can full fill Order $ {}'.format(output))


if __name__ == "__main__":
    app.run(debug=True)
