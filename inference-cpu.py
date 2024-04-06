from flask import Flask, request, jsonify
import requests
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

app = Flask(__name__)

API_URL = "https://api-inference.huggingface.co/models/microsoft/phi-2"
headers = {"Authorization": "Bearer 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"}

# Load the model and tokenizer once
device = torch.device("cpu")
model = AutoModelForCausalLM.from_pretrained("microsoft/phi-2", torch_dtype="auto", trust_remote_code=True)
model.to(device)
tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-2", trust_remote_code=True)

@app.route('/generate', methods=['POST'])
def generate():
    payload = request.get_json()
    input_text = payload['inputs']

    # Encode the input text
    inputs = tokenizer(input_text, return_tensors="pt", return_attention_mask=False)
    inputs = inputs.to(device)

    # Generate the output
    outputs = model.generate(**inputs, max_length=200)
    generated_text = tokenizer.batch_decode(outputs)[0]

    return jsonify({'output': generated_text})

if __name__ == '__main__':
    app.run(debug=True)
