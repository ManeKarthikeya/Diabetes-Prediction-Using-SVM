# Diabetes Prediction using Support Vector Machine (SVM)

## 📖 Project Overview

This project implements a Machine Learning classifier to predict the likelihood of diabetes in patients based on diagnostic measurements. Using the PIMA Indians Diabetes Dataset, a Support Vector Machine (SVM) model is trained to classify whether a person is diabetic or non-diabetic.

This is a classic binary classification problem in the healthcare domain, demonstrating how machine learning can be used for early medical diagnosis and risk assessment.

## 🧠 Algorithm Used

- **Support Vector Machine (SVM)**: A powerful supervised learning model used for classification tasks. The linear kernel is employed in this implementation.
- **StandardScaler**: Used for feature standardization to ensure all input features are on the same scale, which is crucial for SVM performance.

## 📊 About the Dataset

The PIMA Indians Diabetes Dataset contains several medical predictor variables and one target variable.

### Features (Input Variables):

- **Pregnancies**: Number of times pregnant
- **Glucose**: Plasma glucose concentration
- **BloodPressure**: Diastolic blood pressure (mm Hg)
- **SkinThickness**: Triceps skin fold thickness (mm)
- **Insulin**: 2-Hour serum insulin (mu U/ml)
- **BMI**: Body mass index (weight in kg/(height in m)^2)
- **DiabetesPedigreeFunction**: Diabetes pedigree function
- **Age**: Age in years

### Target Variable (Output):

- **Outcome**: 0 (Non-Diabetic) or 1 (Diabetic)

- **Dataset Size**: 768 samples with 8 features each.

## 🛠️ Implementation Steps

The code follows a complete machine learning pipeline:

1. **Import Dependencies**: Uses pandas, numpy, and scikit-learn.
2. **Data Loading & Exploration**: Loads the dataset and performs exploratory data analysis (EDA) to understand data distribution and characteristics.
3. **Data Preprocessing**: Separates features and labels, and applies feature standardization using StandardScaler.
4. **Train-Test Split**: Splits the data into 80% training and 20% testing sets with stratification to maintain class distribution.
5. **Model Training**: An SVM classifier with linear kernel is trained on the standardized training data.
6. **Model Evaluation**: The model's performance is evaluated using accuracy scores on both training and test datasets.
7. **Prediction System**: Implements a complete pipeline to make predictions on new patient data.

## 📈 Results

The model's performance is evaluated based on accuracy metrics:

- **Training Accuracy**: Measures how well the model learned from the training data
- **Test Accuracy**: Indicates how well the model generalizes to unseen data

*(Note: The actual accuracy percentages will be displayed when you run the code with your dataset.)*

## 🚀 How to Run this Project

1. **Clone the repository**:
```bash
git clone https://github.com/YOUR_USERNAME/YOUR_REPOSITORY_NAME.git
```

2. **Navigate to the project directory**.

3. **Install required dependencies**:
```bash
pip install numpy pandas scikit-learn
```

4. **Download the dataset and update the path**:
   - Obtain the PIMA Diabetes Dataset (commonly available as diabetes.csv)
   - Update the file path in the code: `diabetes_dataset = pd.read_csv('path/to/your/diabetes.csv')`

5. **Run the Python script**:
```bash
python diabetes_prediction_(svm).py
```

## 💡 Making a Prediction

To predict diabetes risk for a new patient, provide the 8 medical parameters in the correct order to the `input_data` variable:

```python
# Example format: (Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age)
input_data = (5, 166, 72, 19, 175, 25.8, 0.587, 51)
```

The system will output either:
- **'The person is not diabetic'** (Prediction: 0)
- **'The person is diabetic'** (Prediction: 1)

## 📁 Repository Structure

```
├── diabetes_prediction_(svm).py    # Main Python script
├── diabetes.csv                    # Dataset file
└── README.md                       # Project documentation (this file)
```
