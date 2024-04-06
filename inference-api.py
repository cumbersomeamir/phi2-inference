from flask import Flask, request, jsonify
import requests

app = Flask(__name__)

API_URL = "https://api-inference.huggingface.co/models/microsoft/phi-2"
headers = {"Authorization": "Bearer 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"}

@app.route('generate', methods=['POST'])

def generate():
  payload = request.get_json()
  response = requests.post(API_URL, headers = headers,  json=payload)
  return jsonify(response.json())


if __name__ == "__main__":
  app.run(debug=True)
