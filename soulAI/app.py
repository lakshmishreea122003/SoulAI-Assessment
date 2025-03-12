# from flask import Flask, request, jsonify
# import pickle
# import numpy as np
# from PIL import Image
# import io
# import base64
# from werkzeug.security import check_password_hash, generate_password_hash
# from tensorflow.keras.models import Sequential
# import tensorflow as tf
# from tensorflow.keras import datasets, layers, models

# # print(tf.__version__)


# app = Flask(__name__)

# # Load your trained model
# model = pickle.load(open('D:/full-stack-rag/soulAI/model.pkl', 'rb'))

# # Set a basic username and password for authentication
# USERNAME = "admin"
# PASSWORD_HASH = generate_password_hash("password123")  # Hashed password


# # Function to preprocess the image
# def preprocess_image(image):
#     image = image.resize((224, 224))  # Resize to model input size
#     image = np.array(image) / 255.0   # Normalize
#     image = image.reshape(1, 224, 224, 3)  # Reshape for prediction
#     return image

# # Route for prediction
# @app.route('/predict')
# def predict():
#     # Basic authentication
#     # auth = request.authorization
#     # if not auth or not check_password_hash(PASSWORD_HASH, auth.password):
#     #     return jsonify({"error": "Unauthorized"}), 401

#     # # Check if image is sent
#     file = 'D:/full-stack-rag/soulAI/test_image.jpg'
#     # if file not in request.files:
#     #     return jsonify({"error": "No image uploaded"}), 400

#     # file = request.files['file']
#     image = Image.open(file)

#     # Preprocess the image
#     processed_image = preprocess_image(image)

#     # Make prediction
#     prediction = model.predict(processed_image)
#     predicted_class = np.argmax(prediction)

#     # Return prediction result
#     response = {
#         "predicted_class": int(predicted_class)
#     }
#     # response = {"Hello World"}
#     return jsonify(response)


# # Health check route
# @app.route('/', methods=['GET'])
# def health():
#     return jsonify({"status": "running"})


# if __name__ == '__main__':
#     app.run(host='0.0.0.0', port=5000)










####################### 2
from flask import Flask, jsonify
import numpy as np
from PIL import Image
import tensorflow as tf
import pickle

app = Flask(__name__)

# ✅ Load your trained model
# model = tf.keras.models.load_model('model.h5')
model = pickle.load('model.h5')

# ✅ Function to preprocess the image
def preprocess_image(image):
    image = image.resize((224, 224))  # Resize to model input size
    image = np.array(image) / 255.0   # Normalize
    image = image.reshape(1, 224, 224, 3)  # Reshape for prediction
    return image

# ✅ Root endpoint (Health Check)
@app.route('/', methods=['GET'])
def health():
    return jsonify({"status": "running"})

# ✅ Prediction endpoint (Hardcoded Image Path)
@app.route('/predict', methods=['GET'])
def predict():
    # ✅ Hardcoded file path
    file_path = 'D:/full-stack-rag/soulAI/test_image.jpg'

    # ✅ Open the image
    image = Image.open(file_path)

    # ✅ Preprocess the image
    processed_image = preprocess_image(image)

    # ✅ Make prediction
    prediction = model.predict(processed_image)
    predicted_class = np.argmax(prediction)

    # ✅ CIFAR-10 class names
    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                   'dog', 'frog', 'horse', 'ship', 'truck']

    # ✅ Return prediction result
    response = {
        "predicted_class": class_names[predicted_class]
    }
    return jsonify(response)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
