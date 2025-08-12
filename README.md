# 🛳 Titanic Survival Prediction App

A Streamlit-based web application that explores the Titanic dataset, visualizes key patterns, and predicts passenger survival using a trained **Random Forest** model.

## 📌 Project Overview
This project demonstrates a complete **machine learning workflow** — from data preprocessing and model training to interactive deployment on **Streamlit Cloud**.  
The app allows users to:
- Explore and filter the Titanic dataset.
- View key visualizations.
- Enter passenger details to get survival predictions.
- See model performance metrics and evaluation plots.

---

## 📂 Project Structure
project/
├── app.py # Main Streamlit app
├── requirements.txt # Python dependencies
├── model.pkl # Trained ML model
├── feature_names.pkl # Model's expected feature names
├── data/
│ └── titanic.csv # Dataset
├── notebooks/
│ └── model_training.ipynb # Jupyter notebook for training
└── README.md # Project documentation
---

## 📊 Dataset Description
The dataset used is the [Titanic dataset from Kaggle](https://www.kaggle.com/c/titanic), which contains information about passengers such as:
- **Pclass**: Passenger class (1 = Upper, 2 = Middle, 3 = Lower)
- **Sex**: Gender
- **Age**: Age in years
- **SibSp**: Number of siblings/spouses aboard
- **Parch**: Number of parents/children aboard
- **Fare**: Ticket fare
- **Embarked**: Port of embarkation (C, Q, S)
- **Survived**: Target variable (0 = Did not survive, 1 = Survived)

---

## ⚙️ Features of the App
1. **Home Page**  
   - Overview of the dataset with basic stats.

2. **Data Exploration**  
   - Interactive table filtering by passenger attributes.

3. **Visualisations**  
   - Survival count plot.
   - Survival by passenger class.
   - Age distribution by survival.
   - Correlation heatmap.
   - Histograms for numeric features.

4. **Model Prediction**  
   - User inputs passenger details using widgets.
   - Returns prediction and survival probability.

5. **Model Performance**  
   - Classification report.
   - Confusion matrix heatmap.
   - ROC curve with AUC score.

---

## 🚀 How to Run Locally
### **1. Clone the repository**
```bash
git clone https://github.com/your-username/titanic-survival-prediction.git
cd titanic-survival-prediction
