# Image Classification (Soul AI Assessment)
## Build and Deploy an Image Classification Model
This project involves training the ResNet50 model on CIFAR-10 dataset for image classification task. 
<br>Tech-stack: TensorFlow, Numpy, Matplotlib, Seaborn, Scikit-learn, Flask, Streamlit, AWS(EC2,S3), Docker, Grad-CAM, Postman, pickle

### Part 1 - EDA (Exploratory Data Analysis)
[This Jupyter Notebook](https://github.com/lakshmishreea122003/SoulAI-Assessment/blob/main/EDA/EDA_CIFAR_10.ipynb) performs Exploratory Data Analysis (EDA) on the CIFAR-10 dataset, analyzing its structure, distribution, and image properties.
<br>Tech Stack: TensorFlow, Keras, Matplotlib, Seaborn, NumPy

This part involves the following steps:
- 1.1. Shape of the Data – Check dataset dimensions and structure.
- 1.2. Visualize Images – Display sample images from each class.
- 1.3. Class Distribution – Analyze the number of images per class.
- 1.4. Pixel Value Distribution – Examine intensity variations.
- 1.5. Image Size & Aspect Ratio – Identify unique shapes and plot aspect ratio distribution.
- 1.6. Corrupt/Blank Images Check – Detect anomalies in the dataset.
- 1.7 Mean & Std of Pixel Values – Compute dataset-wide pixel statistics.


### Part 2 - Data Preprocessing and Training
[This directory](https://github.com/lakshmishreea122003/SoulAI-Assessment/tree/main/Preprocess-Train) demonstrates the data preprocessing of CIFAR-10 dataset and Training of ResNet50 model.
'Run '
<br>Tech Stack: TensorFlow, Keras, NumPy

The Data Preprocessing part involves the follwoing:
- Load CIFAR-10 data – Split into train, validation, and test sets.
- Normalize Images – Scale pixel values to [0,1] for better model performance.
- Resize Images – Convert from (32,32,3) to (224,224,3) for ResNet50.
- One-Hot Encode Labels – Convert categorical labels for model compatibility.

The Data Augmentation part involves the follwoing:
- Rotation Range (15°) – Rotates images randomly up to 15 degrees for variability.
- Width Shift (±10%) – Shifts images horizontally to enhance robustness.
- Height Shift (±10%) – Shifts images vertically to improve generalization.
- Horizontal Flip – Flips images randomly to learn symmetrical patterns.
- Zoom Range (±20%) – Randomly zooms in/out to simulate different perspectives.
- Batch Processing (32 images) – Augments images in batches for efficient training.

The Model Training Steps (ResNet50) part involves the follwoing:
- Pretrained ResNet50 Model – Leverages ImageNet weights for the base model to speed up training and improve accuracy.
- Custom Fully Connected Layers – Adds two fully connected layers (512 and 256 units) for CIFAR-10 classification.
- Softmax Output – A final dense layer with 10 units for multi-class classification, using the softmax activation function.
- Freezing Base Model – Freezes the ResNet50 layers to only train the newly added custom layers.
- Optimizer & Loss Function – Uses Adam optimizer and categorical crossentropy for multi-class classification.
- Early Stopping – Monitors validation loss to stop training early and restore the best model when no improvements are observed.
- Learning Rate Scheduling – Reduces the learning rate by a factor of 0.5 when validation loss plateaus for better convergence.
- Training with Augmentation – Uses the augmented training data for better generalization.
- Epochs & Batch Size – Configurable number of epochs (default 10) and batch size (default 32).
- Save Model – Store the trained model as a .pkl file.
- Upload to AWS S3 – Ensure cloud accessibility.

### Part 3 - Model Evaluation
[This Jupyter Notebook](https://github.com/lakshmishreea122003/SoulAI-Assessment/blob/main/Test/Testing.ipynb) involves testing the model on test data of CIFAR-10.
<br>Tech Stack: TensorFlow, Keras, NumPy
- Accuracy – Overall correct predictions percentage.
- Precision – Proportion of correct positive predictions.
- Recall – Proportion of actual positives identified correctly.
- Confusion Matrix – Shows true vs. predicted class distributions for deeper analysis.
- Grad-CAM – Visualizes regions of an image contributing most to the model's prediction.

### Part 4 - App Development
[This directory](https://github.com/lakshmishreea122003/SoulAI-Assessment/tree/main/Application) has the dockerised application built using Flask, Streamlit, Docker, AWS, pickle, Postman
The following are the features of the app:
- /predict endpoint takes an image as input and returns the predicted class.
- Implemented basic authentication for API access.
- Tested the API using Postman.
- Streamlit for image classification UI.
- Containerize the application using Docker.
- Deployed the app on AWS EC2.
- Model is located in AWS S3.

## This app has 2 parts
- Flask API (Deployed on EC2):
  ### [Link](http://3.80.131.5:5000)
  The Flask API is dockerized and deployed on an AWS EC2 instance. It provides a "/predict" endpoint that accepts an image and returns the predicted class.
  ![Flask API- Postman](https://github.com/lakshmishreea122003/SoulAI-Assessment/blob/main/API.jpg)

  
- Streamlit App (Runs Locally): The Streamlit app connects to the Flask API, allowing users to Enter username & password for authentication, Upload an image and Receive the predicted class from the API.
  ![Streamlit UI](https://github.com/lakshmishreea122003/SoulAI-Assessment/blob/main/StreamlitUI.jpg)


  
## Steps to run Appication(Streamlit app with flask API endpoint) [code](https://github.com/lakshmishreea122003/SoulAI-Assessment/tree/main/Application/streamlitApp)
- Clone the Repository
  ``` bash
  git clone https://github.com/lakshmishreea122003/SoulAI-Assessment.git
  cd SoulAI-Assessment/Application/streamlitApp
  ```
- Install Dependencies
  ``` bash
  pip install -r requirements.txt
  ```
- Run the Streamlit App
  ``` bash
  streamlit run streamlit_app.py
  ```
  
 ## Running the Flask API on EC2 (For Reference) [code](https://github.com/lakshmishreea122003/SoulAI-Assessment/tree/main/Application/api)
 - SSH into EC2
   ``` bash
   ssh -i your-key.pem ubuntu@<your-ec2-public-ip>
   ```
 - Clone and Build the Docker Image
   ``` bash
   git clone https://github.com/lakshmishreea122003/SoulAI-Assessment.git
   cd SoulAI-Assessment/Application/api
   docker build -t flask-app .
   ```
  - Run the Docker Container
    ``` bash
    docker run -d -p 5000:5000 flask-app
    ``` 
  - API should now be accessible at: http://your-ec2-public-ip:5000/
 ### Note: When running the flask app, make sure to upload the model.pkl to AWS S3 bucket using [upload_data.py](https://github.com/lakshmishreea122003/SoulAI-Assessment/blob/main/Preprocess-Train/Components/upload_s3.py)





