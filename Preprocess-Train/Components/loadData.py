from tensorflow.keras import datasets, layers, models


def load_data():
    print("loading data")
    # Load the CIFAR-10 dataset
    (train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()

    # Split Training Set into Training & Validation The original dataset has: 50,000 training images 10,000 test images We'll use 80% for training and 20% for validation.
    # Define split ratio
    validation_ratio = 0.2
    # Compute validation set size
    num_train = int(train_images.shape[0] * (1 - validation_ratio))
    # Split training data into train & validation sets
    train_images, val_images = train_images[:num_train], train_images[num_train:]
    train_labels, val_labels = train_labels[:num_train], train_labels[num_train:]
    print("data loaded")
    return train_images, train_labels, test_images, test_labels, val_images, val_labels