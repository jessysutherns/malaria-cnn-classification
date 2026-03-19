# 🧬 Malaria Cell Image Classification using CNNs

> Deep learning model for detecting malaria-infected blood cells using transfer learning (MobileNetV2).

---

## 📌 Overview

This project uses deep learning to classify blood smear images as **Parasitized** or **Uninfected**.

It leverages transfer learning with MobileNetV2 to build an efficient and accurate medical image classification model.

---

## 🎯 Objectives

* Detect malaria-infected cells from microscopy images
* Build a complete machine learning pipeline
* Apply transfer learning for improved performance
* Evaluate model performance using standard metrics

---

## 🚀 Key Features

* Transfer learning using MobileNetV2
* Image data augmentation
* Binary classification (infected vs uninfected)
* Evaluation using precision, recall, and F1-score
* Prediction system for classifying new images

---

## 🧠 Model

* **MobileNetV2 (Transfer Learning)**
* Pretrained on ImageNet
* Fine-tuned for binary classification

---

## 📊 Results

* **Accuracy:** ~92%
* **F1-score:** ~0.92
* **Dataset size:** ~40,000 images

![Model Results](outputs/result.png)

### Confusion Matrix

```id="d5l3g7"
[[2235  384]
 [  58 2560]]
```

---

## 📂 Dataset Structure

Train / Validation / Test split:

* Parasitized
* Uninfected

---

## ⚙️ Tech Stack

* Python
* TensorFlow / Keras
* NumPy
* Scikit-learn
* Matplotlib

---

## 📁 Dataset

NIH Malaria Dataset:
https://www.kaggle.com/datasets/iarunava/cell-images-for-detecting-malaria

⚠️ Dataset is not included in this repository due to size.

---

## 🚀 How to Run

```bash id="9h9u6n"
pip install -r requirements.txt
python main.py
```

---

## 🔍 Prediction

Run prediction on a new image:

```bash id="t0i7vc"
python predict.py --image "path_to_image"
```

Example output:

```id="h2t0fa"
Prediction: Uninfected (97.57%)
```

---

## 🧪 Example

Run prediction on sample image:

```bash id="j9s4k2"
python predict.py --image "samples/test.png"
```

Example output:

```id="4o5c9c"
Prediction: Parasitized (97%)
```

---

## 📂 Project Structure

malaria-cnn-classification/
│
├── main.py
├── predict.py
├── split_data.py
├── requirements.txt
├── README.md
├── .gitignore
├── outputs/
│   └── result.png
└── samples/
└── test.png

---

## 💡 Future Improvements

* Add ResNet50 model for comparison
* Improve accuracy with hyperparameter tuning
* Deploy model using Streamlit or Flask
* Add interactive UI for predictions

---

## 👤 Author

GitHub: https://github.com/jessysutherns
