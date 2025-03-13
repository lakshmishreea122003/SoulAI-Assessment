import streamlit as st
import requests
from PIL import Image
import io

# Flask API endpoint
API_URL = "http://3.80.131.5:5000/predict"  # Change to EC2 public IP when deployed

def authenticate(username, password):
    return requests.auth.HTTPBasicAuth(username, password)

def predict(image, username, password):
    try:
        img_bytes = io.BytesIO()
        image.save(img_bytes, format='PNG')
        img_bytes = img_bytes.getvalue()
        
        files = {"file": ("image.png", img_bytes, "image/png")}
        auth = authenticate(username, password)
        response = requests.post(API_URL, files=files, auth=auth)
        
        if response.status_code == 200:
            return response.json()
        else:
            return {"error": f"Error {response.status_code}: {response.text}"}
    except Exception as e:
        return {"error": str(e)}

# Streamlit UI
st.title("Image Classification App 🖼️")
st.write("Upload an image and get a prediction!")

# User Authentication
username = st.text_input("Username", "", type="default")
password = st.text_input("Password", "", type="password")

# File Uploader
uploaded_file = st.file_uploader("Choose an image...", type=["png", "jpg", "jpeg"])

if uploaded_file and username and password:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    
    if st.button("Predict 🚀"):
        result = predict(image, username, password)
        if "error" in result:
            st.error(result["error"])
        else:
            st.success(f"Prediction: {result['predicted_class']} (Confidence: {result['confidence']:.2f})")
