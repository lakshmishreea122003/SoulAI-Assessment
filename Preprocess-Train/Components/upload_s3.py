import boto3

# AWS S3 details
AWS_BUCKET_NAME = "soulai-assessment"
MODEL_FILE_PATH = 'D:/full-stack-rag/practice/model.pkl'  # Change to the path of your model file
IMAGE_FILE_PATH = 'D:/full-stack-rag/practice/test_image.jpg'
# Initialize S3 client
s3 = boto3.client('s3')

def upload_to_s3(file_name, bucket, object_name):
    """Upload a file to an S3 bucket"""
    try:
        s3.upload_file(file_name, bucket, object_name)
        print(f"Upload successful: {file_name} -> s3://{bucket}/{object_name}")
    except Exception as e:
        print(f"Error uploading {file_name}: {e}")

# Upload model and image
upload_to_s3(MODEL_FILE_PATH, AWS_BUCKET_NAME, "models/model.pkl")
upload_to_s3(IMAGE_FILE_PATH, AWS_BUCKET_NAME, "images/image.jpg")
