import pickle

from Components.loadData import load_data  
from Components.preprocess_methods import normalization, resize, categorical, data_aug
from Components.model_build import train_resnet50

# 1. load CIFAR-10 data 
train_images, train_labels, test_images, test_labels, val_images, val_labels = load_data()

# 2. Class names in CIFAR-10
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer','dog', 'frog', 'horse', 'ship', 'truck']

# 3. Normalize the Images
# Since pixel values range from 0 to 255, normalize them to [0,1] for faster convergence:
train_images, val_images, test_images = normalization(train_images, val_images, test_images)

# 4. Resize
# The ResNet50 model requires the input image shape to be (224, 224, 3). But the CIFAR-10 samples have a shape of (32,32,3).Thus we have to resize the input image data.
train_images, val_images, test_images = resize(train_images, val_images, test_images)

# One-hot encode labels to be trained by ResNet
train_labels, val_labels, test_labels = categorical(train_labels,val_labels,test_labels)

# 5. Data Augmentation
train_generator = data_aug(train_images, train_labels)

# train ResNet50
model, history = train_resnet50(train_images, train_labels, val_images, val_labels,train_generator)

# store model as pkl file
file_path = 'D:/full-stack-rag/Assessment/resnet50_model.pkl'
with open(file_path, 'wb') as f:
    pickle.dump(model, f)