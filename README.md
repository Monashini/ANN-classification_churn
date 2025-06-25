# ANN-classification_churn

## 📊 Customer Churn Prediction using ANN

This project builds an **Artificial Neural Network (ANN)** to classify whether a customer will churn or not. It includes **data preprocessing, model training, serialization (pickling), and prediction**, all structured for both experimentation and deployment.


### 🚀 Features Implemented

+ Clean preprocessing pipeline
+ Feature scaling & encoding
+ANN training with Keras
+ Evaluation (accuracy, confusion matrix)
+ Model pickling with `joblib`
+ Streamlit app for real-time prediction


### 📁 Project Structure

```bash
.
├── app.py                  # Streamlit app for live prediction
├── churn_data.csv          # Dataset
├── ANN_model.ipynb         # Full notebook with training workflow
├── requirements.txt        # Dependencies
├── model.pkl               # Saved model (pickled)
├── preprocessor.pkl        # Saved preprocessing pipeline
└── README.md               # This file
```



### 🔍 Problem Statement

**Goal**: Predict whether a customer will churn (leave the bank) using structured data with features like geography, credit score, age, balance, and tenure.



### 🧹 Data Preprocessing

* Dropped irrelevant columns like `RowNumber`, `CustomerId`, `Surname`
* Handled categorical variables using:

  * Label Encoding (e.g. Gender)
  * One-Hot Encoding (e.g. Geography)
* Feature scaling with **StandardScaler**
* Created and saved a **preprocessing pipeline** (`preprocessor.pkl`)



### 🧠 Model Training

* ANN built using **Keras Sequential API**
* Architecture:

  * Input Layer
  * Hidden Layers with ReLU activation
  * Output Layer with Sigmoid activation (binary classification)
* Loss function: `binary_crossentropy`
* Optimizer: `Adam`
* Evaluation: Accuracy, Confusion Matrix



### 💾 Serialization (Pickling)

* Model saved as `model.pkl` using `joblib`
*  used in the Streamlit app for **live predictions**



### 🖥️ Streamlit App

Interactive UI where users can input customer details and get an instant prediction:

* Uses the **pickled model and scaler**
* Outputs: `Churn` or `No Churn` based on input

Run it with:

```bash
streamlit run app.py
```



### 🧪 How to Run

#### 1. Clone the repo

```bash
git clone https://github.com/Monashini/ANN-classification_churn.git
cd ANN-classification_churn
```

#### 2. Install dependencies

```bash
pip install -r requirements.txt
```

#### 3. Run the app

```bash
streamlit run app.py
```


### 📈 Future Work (Optional)

* Add SHAP or LIME explainability
* Train on larger datasets
* Enable batch prediction from CSV uploads
* Save training logs & charts via TensorBoard


### 👩‍💻 Author

**Monashini**
*AI/ML Undergrad | Streamlit Enthusiast | ANN Builder*

