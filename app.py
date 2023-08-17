import flask
import numpy as np
import pickle
from markupsafe import escape
from src import prepare_input
app = flask.Flask(__name__, template_folder='templates')

rainPredictor = pickle.load(open("model/models_trained.pkl", 'rb'))


@app.route('/')
def index():
    return(flask.render_template('index.html'))


@app.route('/predict', methods=['POST'])
def predict():
    # retrieve all features
    features = [x for x in flask.request.form.items()]
    final_features = prepare_input(dict(features), rainPredictor)

    # define the models
    logreg = rainPredictor['final_models']['logreg']

    # predict
    prediction = logreg.predict(final_features)

    # output
    output = {0: 'Tidak Hujan', 1: 'Hujan'}

    return flask.render_template('index.html', prediction_text=output[prediction[0]])


if __name__ == '__main__':
    app.run(debug=True)
