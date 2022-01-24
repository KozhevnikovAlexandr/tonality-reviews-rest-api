import json
import pandas as pd
from flask import Flask, request
from preparing_data import load_model, prepare_data

app = Flask(__name__)


@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        text = json.loads(request.json)['text']
        model, vec = load_model()

        data = pd.DataFrame({'Text': [text]})
        data = prepare_data(data)

        vectorized_data = vec.transform(data["text_preprocessed"])
        data['answer'] = model.predict(vectorized_data)

        return data[['Text', 'answer']].to_json()


if __name__ == '__main__':
    app.run(debug=True)
