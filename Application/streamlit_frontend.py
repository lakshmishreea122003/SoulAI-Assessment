import streamlit as st
import boto3
import requests
from requests.auth import HTTPBasicAuth

# S3 Configuration
BUCKET_NAME = "soulai-assessment"
S3_FILE_PATH = "test_image.jpg"

# Flask API URL
FLASK_API_URL = "https://8080-lakshmishre-soulaiasses-od2krypankz.ws-us118.gitpod.io/predict"

# Function to upload file to S3
def upload_to_s3(file):
    """Uploads a file to AWS S3."""
    try:
        s3 = boto3.client('s3')
        s3.upload_fileobj(file, BUCKET_NAME, S3_FILE_PATH)
        return True
    except Exception as e:
        st.error(f"⚠️ S3 Upload Failed: {e}")
        return False

# Streamlit UI
st.title("🔍 Image Classifier with Authentication")
st.markdown("Upload an image, authenticate, and get predictions.")

# User authentication inputs
st.subheader("🔑 Authentication")
username = st.text_input("Username", value="", type="default")
password = st.text_input("Password", value="", type="password")

# File uploader
st.subheader("📤 Upload Image")
uploaded_file = st.file_uploader("Choose an image", type=["jpg", "png", "jpeg"])

# Process only if an image is uploaded
if uploaded_file:
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)

    # Upload to S3
    if upload_to_s3(uploaded_file):
        st.success("✅ File successfully uploaded to S3!")

        # Ensure credentials are provided
        if username and password:
            st.subheader("🔎 Getting Prediction...")
            try:
                # Call Flask API with Basic Authentication
                response = requests.get(FLASK_API_URL, auth=HTTPBasicAuth(username, password))

                if response.status_code == 200:
                    result = response.json()
                    st.success(f"🎯 **Predicted Class:** {result['predicted_class']}")
                elif response.status_code == 401:
                    st.error("🚫 Unauthorized! Please check your username and password.")
                else:
                    st.error(f"⚠️ Error: {response.status_code} - {response.text}")

            except requests.exceptions.RequestException as e:
                st.error(f"⚠️ API Request Failed: {e}")

        else:
            st.warning("⚠️ Please enter your username and password to proceed.")
