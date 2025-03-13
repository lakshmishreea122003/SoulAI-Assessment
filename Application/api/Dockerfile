# Use Python 3.10 base image
FROM python:3.10

# Set the working directory
WORKDIR /app

# Copy requirements.txt and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire app code
COPY . .

# Expose port 8080 for Flask
EXPOSE 8080

# Run the Flask app
CMD ["python", "f_api.py"]
