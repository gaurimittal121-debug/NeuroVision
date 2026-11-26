# 🧠 NeuroVision – Brain Tumor Classification

## 📌 Project Overview
NeuroVision is a Machine Learning-based web application that classifies brain MRI images into four categories using **Histogram of Oriented Gradients (HOG)** for feature extraction and a **Support Vector Machine (SVM)** for classification.

## 🎯 Objectives
- Classify brain MRI images into four categories.
- Provide an easy-to-use web interface.
- Display prediction results and confidence information.
- Perform basic image analysis to estimate bright regions.

## 🧠 Tumor Categories
1. Glioma
2. Meningioma
3. No Tumor
4. Pituitary

## 🛠️ Technologies Used
- Python
- OpenCV
- NumPy
- Scikit-learn
- Scikit-image
- Streamlit
- Joblib

## ⚙️ Methodology
1. MRI image upload
2. Image preprocessing and resizing
3. HOG feature extraction
4. SVM model prediction
5. Display of classification result
6. Basic bright-region analysis

## 📂 Dataset
The dataset contains brain MRI images belonging to four categories.

**Google Drive Dataset:**  
[Access Dataset](https://drive.google.com/drive/folders/1TmDCDNzvUcoIU8XZel-gXeywTo7aH4Ui?usp=drive_link)

## 💾 Trained Model
The trained SVM model is stored on Google Drive.

[Download SVM Model](https://drive.google.com/file/d/1bHS0i4vFPIqQPTYi-jlFxjtO2wMafPQ_/view?usp=sharing)

## 🚀 How to Run the Project

1. Clone the repository:
   `git clone https://github.com/gaurimittal121-debug/NeuroVision.git`

2. Open the project folder:
   `cd NeuroVision`

3. Install dependencies:
   `pip install -r requirements.txt`

4. Run the application:
   `streamlit run app.py`

## 🌐 GitHub Repository
https://github.com/gaurimittal121-debug/NeuroVision

## ⚠️ Disclaimer
NeuroVision is an academic machine learning project developed for educational purposes. Its predictions and image analysis are experimental and must not be used as a substitute for professional medical diagnosis.
