import pickle
import flask
from flask import Flask
from collections import Counter
import sklearn
from flask import request
import pandas as pd
from flask import Response

app = Flask(__name__)

MODEL = None

def preprocessing_predict(name):
    # Preparation

    alphabet = ['L', 'O', 'G', 'A', 'N', '-', 'B', 'C', 'D', 'E', 'F', 'H', 'I', 'J', 'K', 'M', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'À', 'Â', 'Ç', 'È', 'É', 'Ë', 'Î', 'Ï', 'Ô', 'Ù']
    def name_encoded(name):
        result = dict(Counter(list(name.upper())))
        for i in alphabet:
            if i not in result.keys():
                result[i] = 0
        return result

    X = pd.DataFrame([name_encoded(name)], columns=alphabet)

    # Prédiction
    sexe = MODEL.predict(X)
    return sexe[0]

@app.route("/predict")
def predict():
    name = request.args.get("name")
    prediction = preprocessing_predict(name)

    if prediction==1:
        sexe='Masculin'
    else:
        sexe = 'Feminin'

    response =  flask.make_response({
        "name": str(name),
        "classe predicte" : str(sexe)
    })
    #return flask.jsonify(response)

    #resp = Response(response=response,
    #                status=200,
    #                mimetype="application/json")

    response.mimetype = 'application/json'
    return response

if __name__ == "__main__":

    with open('model.saved', 'rb') as file:
        MODEL = pickle.load(file)


    app.run(host="127.0.0.1", port="5000")