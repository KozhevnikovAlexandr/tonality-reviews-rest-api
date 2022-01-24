import json
import pandas as pd
import requests

URL = "http://127.0.0.1:5000/predict"
requests.get("http://127.0.0.1:5000/")
response = requests.post(URL, json='{"text":"Самый лучший и любимый банк на свете!!!"}')
response = json.loads(response.content)
response_df = pd.DataFrame(response, columns=['Text', 'Prediction'])
response_df.to_csv('response.csv', index=False)
