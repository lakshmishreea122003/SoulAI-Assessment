from flask import Flask, jsonify, request
import numpy as np
import pickle
import boto3
from PIL import Image
import io
from flask_httpauth import HTTPBasicAuth

app = Flask(__name__)
auth = HTTPBasicAuth()

# Sample credentials (Replace these with secure ones)
USER_CREDENTIALS = {
    "admin": "password123",
    "user1": "mypassword"
}

# AWS S3 Details
AWS_BUCKET_NAME = "soulai-assessment"
MODEL_FILE_KEY = "model.pkl"
IMAGE_FILE_KEY = "test_image.jpg"

# Initialize S3 client
s3 = boto3.client('s3', aws_access_key_id='Enter', aws_secret_access_key='Enter')

# Function to download from S3
def download_from_s3(bucket_name, key):
    try:
        response = s3.get_object(Bucket=bucket_name, Key=key)
        return response['Body'].read()
    except Exception as e:
        print(f"Error downloading {key} from S3: {e}")
        return None

# Load model from S3
model_data = download_from_s3(AWS_BUCKET_NAME, MODEL_FILE_KEY)
if model_data:
    model = pickle.loads(model_data)
else:
    model = None
    print("❌ Failed to load model from S3.")

# CIFAR-10 class names
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

# Authentication function
@auth.verify_password
def verify_password(username, password):
    if username in USER_CREDENTIALS and USER_CREDENTIALS[username] == password:
        return username
    return None

@app.route('/', methods=['GET'])
def hello():
    return jsonify({"message": "Hello, World!"})

@app.route('/predictSt', methods=['GET'])
@auth.login_required
def predict():
    if not model:
        return jsonify({"error": "Model not loaded"}), 500

    # Load image from S3
    image_data = download_from_s3(AWS_BUCKET_NAME, IMAGE_FILE_KEY)
    if not image_data:
        return jsonify({"error": "Failed to load image from S3"}), 500

    # Convert image bytes to PIL Image
    image = Image.open(io.BytesIO(image_data))
    image = image.resize((32, 32))
    image = np.array(image) / 255.0
    image = image.reshape(1, 32, 32, 3)

    # Make prediction
    prediction = model.predict(image)
    predicted_class = np.argmax(prediction)

    return jsonify({"predicted_class": class_names[predicted_class]})

# This is the api endpoint that takes image data as input to give the prediction
@app.route('/predict', methods=['POST'])
def predict_api():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    file = request.files['file']
    image = Image.open(io.BytesIO(file.read()))  # Read image
    image = image.convert("RGB")  # Convert to RGB (ensures 3 channels)
    image = image.resize((32, 32))  
    image = np.array(image) / 255.0  
    image = image.reshape(1, 32, 32, 3)
    predictions = model.predict(image)  # Get predictions
    predicted_class = class_names[np.argmax(predictions)]  # Get class label
    return jsonify({"predicted_class": predicted_class, "confidence": float(np.max(predictions))})


