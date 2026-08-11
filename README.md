🐱🐶 Cat vs Dog Image Classifier using CNN

A Convolutional Neural Network (CNN) project that classifies images aseither Cat or Dog using TensorFlow/Keras.

📌 Project Overview

This project demonstrates an end-to-end image classification workflow:

Downloading the Cat vs Dog dataset using kagglehub

Loading images using TensorFlow's image_dataset_from_directory

Normalizing image pixel values

Building a CNN from scratch

Using Batch Normalization

Using Dropout to reduce overfitting

Training the model for multiple epochs

Evaluating training and validation performance

Visualizing accuracy and loss

Testing the trained model on real Cat and Dog images

📂 Dataset

The project uses the Kaggle Dogs vs Cats dataset:

salader/dogsvscats

The dataset is downloaded directly into the Colab runtime using:

import kagglehub

path = kagglehub.dataset_download("salader/dogsvscats")

The relevant directory structure is:

dogsvscats/
├── train/
│   ├── cats/
│   └── dogs/
├── test/
│   ├── cats/
│   └── dogs/
└── catsvsdogs/

The dataset is not stored in the GitHub repository.

🛠️ Technologies Used

Python

TensorFlow

Keras

OpenCV

NumPy

Matplotlib

KaggleHub

Google Colab

🧠 CNN Architecture

The model consists of three convolutional blocks followed by fullyconnected layers.

Input: 256 × 256 × 3

Conv2D (32 filters, 3×3)
BatchNormalization
MaxPooling2D

Conv2D (64 filters, 3×3)
BatchNormalization
MaxPooling2D

Conv2D (128 filters, 3×3)
BatchNormalization
MaxPooling2D

Flatten

Dense (128)
Dropout (0.1)

Dense (64)
Dropout (0.1)

Dense (1, sigmoid)

Output

The final sigmoid layer produces a value between 0 and 1.

0 → Cat 🐱
1 → Dog 🐶

🔄 Data Preprocessing

Images are resized to:

256 × 256 pixels

Pixel values are normalized from:

0–255

to:

0–1

using:

def process(image, label):
    image = tf.cast(image / 255., tf.float32)
    return image, label

🏋️ Model Training

The model is compiled using:

model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

Training is performed for 10 epochs:

history = model.fit(
    train_ds,
    epochs=10,
    validation_data=validation_ds
)

📊 Performance Visualization

Training and validation accuracy are plotted to observe modelperformance and the gap between training and validation results.

Training and validation loss are also plotted to monitor learning andpossible overfitting.

Batch Normalization and Dropout were used to improve generalization andreduce the training-validation gap.

🧪 Testing with Real Images

The notebook also tests the trained model on external images such as:

/content/dog.png
/content/cat.png

Images are:

Loaded using OpenCV

Converted from BGR to RGB

Resized to 256 × 256

Reshaped into a batch of one image

Passed to the trained CNN

Example:

test_img = cv2.imread("/content/dog.png")
test_img = cv2.cvtColor(test_img, cv2.COLOR_BGR2RGB)
test_img = cv2.resize(test_img, (256, 256))

test_input = test_img.reshape((1, 256, 256, 3))
prediction = model.predict(test_input)

⚠️ Important

Because the training pipeline normalizes images using image / 255,real test images should also be normalized before prediction:

test_input = test_input / 255.0

Otherwise, the model receives inputs on a different scale from the datait was trained on.

🚀 How to Run

1. Open the notebook

Open:

cat_dog_image_classifier.ipynb

in Google Colab.

2. Install KaggleHub

!pip install -q kagglehub

3. Download the dataset

import kagglehub

path = kagglehub.dataset_download("salader/dogsvscats")

4. Run the notebook cells

Run the cells in order to:

Load the dataset

Preprocess the images

Build the CNN

Train the model

Plot accuracy/loss

Test real images

📁 Suggested Repository Structure

Cat-Dog-Image-Classifier/
│
├── cat_dog_image_classifier.ipynb
├── README.md
└── .gitignore

The dataset should not be uploaded to GitHub.

🎯 Learning Outcomes

This project provides practical experience with:

Image classification

CNN architecture

Convolution and pooling

Batch Normalization

Dropout

Binary classification

Image preprocessing

Model training and validation

Overfitting analysis

Real-image inference

TensorFlow/Keras

👨‍💻 Project

Cat vs Dog Image Classifier

Built using Python, TensorFlow/Keras, OpenCV and Google Colab.