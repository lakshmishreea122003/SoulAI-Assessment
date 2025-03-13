from flask import Flask, jsonify, request
import numpy as np
import pickle
import boto3
from PIL import Image
import io
from flask_httpauth import HTTPBasicAuth

app = Flask(__name__)
auth = HTTPBasicAuth()

# 🔐 Secure credentials (Store them safely)
USER_CREDENTIALS = {
    "admin": "password123",
    "user1": "mypassword"
}

# Authentication function
@auth.verify_password
def verify_password(username, password):
    if username in USER_CREDENTIALS and USER_CREDENTIALS[username] == password:
        return username
    return None

# AWS S3 Details
AWS_BUCKET_NAME = "soulai-assessment"
MODEL_FILE_KEY = "model.pkl"

# Initialize S3 client
s3 = boto3.client('s3')

# Function to download model from S3
def download_model_from_s3(bucket_name, key):
    try:
        response = s3.get_object(Bucket=bucket_name, Key=key)
        return response['Body'].read()
    except Exception as e:
        print(f"Error downloading model from S3: {e}")
        return None

# Load model
model_data = download_model_from_s3(AWS_BUCKET_NAME, MODEL_FILE_KEY)
if model_data:
    model = pickle.loads(model_data)  # Deserialize model
else:
    model = None
    print("❌ Failed to load model from S3.")

# CIFAR-10 class names
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

@app.route("/", methods=["GET"])
def hello():
    return jsonify({"message": "Hello, World!"})

@app.route("/predict", methods=["POST"])
@auth.login_required  # 🔒 Require authentication for predictions
def predict_api():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    image = Image.open(io.BytesIO(file.read()))
    image = image.convert("RGB")
    image = image.resize((32, 32))
    image = np.array(image) / 255.0
    image = image.reshape(1, 32, 32, 3)

    if model is None:
        return jsonify({"error": "Model not loaded"}), 500
    
    predictions = model.predict(image)
    predicted_class = class_names[np.argmax(predictions)]

    return jsonify({"predicted_class": predicted_class, "confidence": float(np.max(predictions))})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
