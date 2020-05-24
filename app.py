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
    num_data=[i for i in int_features[:11] ]
    cat_data=[i for i in int_features[11:] ]

    for i,j in enumerate(cat_data):
        if j=='Yes':
            cat_data[i]=1
        else:
            cat_data[i]=0
        
    int_features=num_data+cat_data
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)

    if prediction==0:
        output='No'
    elif prediction==1:
        output='Yes'
        
        

    return render_template('backorder.html', prediction_text='Product actually went on backorder -> {}'.format(output))


if __name__ == "__main__":
    app.run(debug=True)
    
