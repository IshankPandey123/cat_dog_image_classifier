# 🐱🐶 Cat vs Dog Image Classifier using CNN

A complete Convolutional Neural Network (CNN) project that classifies images as either a **Cat** or a **Dog** using **TensorFlow/Keras**.

---

## 📌 Project Overview

This project demonstrates an end-to-end image classification workflow:

* Downloading the Cat vs Dog dataset using **kagglehub**.
* Loading images using TensorFlow's `image_dataset_from_directory`.
* Normalizing image pixel values for stable training.
* Building a custom **CNN architecture** from scratch.
* Applying **Batch Normalization** and **Dropout** to improve generalization and reduce overfitting.
* Training and evaluating model performance over multiple epochs.
* Visualizing accuracy and loss metrics.
* Testing the trained model on real-world external images.

---

## 📂 Dataset

The project uses the Kaggle **Dogs vs Cats** dataset (`salader/dogsvscats`).

The dataset is downloaded directly into the Colab runtime using:

```python
import kagglehub

path = kagglehub.dataset_download("salader/dogsvscats")

```

### Directory Structure

```text
dogsvscats/
├── train/
│   ├── cats/
│   └── dogs/
├── test/
│   ├── cats/
│   └── dogs/
└── catsvsdogs/

```

*(Note: The dataset is large and should not be stored in the GitHub repository.)*

---

## 🛠️ Technologies Used

* **Python**
* **TensorFlow** & **Keras**
* **OpenCV**
* **NumPy**
* **Matplotlib**
* **KaggleHub**
* **Google Colab**

---

## 🧠 CNN Architecture

The model consists of three convolutional blocks followed by dense layers:

* **Input:** $256 \times 256 \times 3$
* **Block 1:** Conv2D (32 filters, $3 \times 3$) $\rightarrow$ BatchNormalization $\rightarrow$ MaxPooling2D
* **Block 2:** Conv2D (64 filters, $3 \times 3$) $\rightarrow$ BatchNormalization $\rightarrow$ MaxPooling2D
* **Block 3:** Conv2D (128 filters, $3 \times 3$) $\rightarrow$ BatchNormalization $\rightarrow$ MaxPooling2D
* **Flatten Layer**
* **Dense Layer (128 units)** $\rightarrow$ Dropout ($0.1$)
* **Dense Layer (64 units)** $\rightarrow$ Dropout ($0.1$)
* **Output Layer:** Dense (1 unit, `sigmoid`)

> **Note:** The final sigmoid layer outputs a value between $0$ and $1$, where:
> * $0 \rightarrow$ **Cat** 🐱
> * $1 \rightarrow$ **Dog** 🐶
> 
> 

---

## 🔄 Data Preprocessing

* Images are resized to **$256 \times 256$ pixels**.
* Pixel values are normalized from **$0–255$** to **$0–1$**:

```python
def process(image, label):
    image = tf.cast(image / 255., tf.float32)
    return image, label

```

---

## 🏋️ Model Training

The model is compiled and trained using the following configuration:

```python
model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

history = model.fit(
    train_ds,
    epochs=10,
    validation_data=validation_ds
)

```

---

## 📊 Performance Visualization

* Training and validation accuracy are plotted to observe performance and check the gap between splits.
* Training and validation loss curves monitor learning progress and potential overfitting.

---

## 🧪 Testing with Real Images

External images can be tested by preprocessing them identically to the training pipeline:

```python
test_img = cv2.imread("/content/dog.png")
test_img = cv2.cvtColor(test_img, cv2.COLOR_BGR2RGB)
test_img = cv2.resize(test_img, (256, 256))

test_input = test_img.reshape((1, 256, 256, 3)) / 255.0
prediction = model.predict(test_input)

```

> ⚠️ **Important:** Real test images **must** be normalized (`/ 255.0`) before prediction to match the scale the model was trained on.

---

## 🚀 How to Run

1. Open the notebook `cat_dog_image_classifier.ipynb` in **Google Colab**.
2. Install KaggleHub:
```bash
!pip install -q kagglehub

```


3. Run the cells sequentially to download data, preprocess, build, train, and test the model.

---

## 📁 Suggested Repository Structure

```text
Cat-Dog-Image-Classifier/
│
├── cat_dog_image_classifier.ipynb
├── README.md
└── .gitignore

```

---

## 🎯 Learning Outcomes

* Practical experience with end-to-end **image classification**.
* Understanding **CNN architecture**, convolution, and pooling layers.
* Implementing **Batch Normalization** and **Dropout** for regularization.
* Managing image data pipelines, preprocessing, and real-world model inference.
