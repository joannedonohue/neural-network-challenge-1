 # Student Loan Risk Prediction with Deep Learning
Predict the likelihood that an applicant will repay their student loans using neural networks

## Project Overview
This project focuses on predicting the likelihood of student loan repayment success using a Deep Neural Network (DNN) model built with TensorFlow's Keras library. By analyzing student demographic, academic, and financial data, we aim to predict whether a student will successfully repay their loan. Additionally, we explore the idea of building a recommendation system for suggesting optimal loan products based on student profiles.

## Data Source
The dataset used in this project is sourced from student-loans.csv. It contains 1,599 records with various features, including:

Student Attributes: Payment history, GPA ranking, financial aid score, etc.
Academic Factors: Study major code, time to completion, cohort ranking.
Loan Factors: Total loan score, credit ranking (target variable).

Observations:
The dataset includes a binary credit_ranking variable (0 = loan not repaid, 1 = loan repaid).
Out of 1,599 records, 53% indicate successful repayment, while 47% indicate failure.

## Project Steps

### Step 1: Data Preprocessing
Loaded the dataset into a Pandas DataFrame.
Explored data types and value counts to understand the structure and balance of the dataset.
Defined the target (y) as the credit_ranking column.
Defined the feature set (X) as all other columns excluding credit_ranking.

#### Key Observations:

The dataset is relatively balanced with a slight majority in successful loan repayments (53%).
The features include numerical data which can directly be used after scaling.

### Step 2: Splitting and Scaling the Data
Split the data into training (75%) and testing (25%) sets using train_test_split with a fixed random state for reproducibility.
Used StandardScaler to standardize the feature set for improved model performance.

### Model Development
### Step 3: Building the Neural Network
Model Type: Feedforward Sequential Neural Network.

### Architecture:

Input Layer: 11 input nodes (one for each feature).
Hidden Layers:
Layer 1: 6 neurons with ReLU activation.
Layer 2: 3 neurons with ReLU activation.
Output Layer: 1 neuron with sigmoid activation for binary classification.

### Model Compilation:

Loss Function: binary_crossentropy (suitable for binary classification).
Optimizer: adam (adaptive learning rate).
Evaluation Metric: accuracy.

### Model Summary:

|Layer (type)         |    Output Shape         | Param # | 
| ------------------- | ----------------------- | --------|
|dense_1 (Dense)      |    (None, 6)            |  72     |
|dense_2 (Dense)      |    (None, 3)            |  21     |
|dense_3 (Dense)      |    (None, 1)            |  4      |

Total params: 97

### Step 4: Model Training
Trained the model over 50 epochs.
Observed an increase in accuracy as epochs progressed, reaching a peak around epoch 45.

### Model Evaluation
### Step 5: Model Performance on Test Data
Evaluated the model on the test set:
Loss: 0.5086, Accuracy: 0.7550
Accuracy: 75.5%
Loss: 0.51 (lower is better)

#### Observations:

The model performs reasonably well with a 75% accuracy rate.
The loss value indicates that there is still room for optimization.

### Step 6: Classification Report
Generated a classification report to understand precision, recall, and F1-score:

            precision    recall  f1-score   support
         0       0.72      0.78      0.75       188
         1       0.79      0.74      0.76       212
  accuracy                           0.76       400
 macro avg       0.76      0.76      0.75       400
weighted avg     0.76      0.76      0.76       400


**Key Insights**:
- The model has better precision for identifying successful repayments (Class 1) but better recall for identifying failures (Class 0).
- Overall accuracy of 76% is a decent start, though improvements are possible.


### Model Saving and Deployment
- The trained model was saved to `student_loans.keras` for future use.
- Reloaded the saved model to make predictions on new, unseen data.

---

### Future Project: Building a Recommendation System

#### 1. Data Collection
- For a recommendation system, collect the following:
- **Student Demographics**: Age, income, school type.
- **Academic Background**: GPA, field of study, future job prospects.
- **Financial Data**: Existing debt, family income.
- **Loan Features**: Interest rates, repayment terms.

#### 2. Filtering Method
- **Content-Based Filtering**:
- Leverage student profiles to match with loan features such as interest rates, repayment terms, and loan amounts.
- Suitable for structured data with rich item attributes.

#### 3. Challenges
- **Data Privacy and Security**: Handling sensitive information requires stringent safeguards to protect student data.
- **Bias and Fairness**: Ensuring that the model does not perpetuate biases that could unfairly disadvantage certain student groups.

---

### **Conclusion**
The project demonstrates how a deep learning model can be used to predict student loan repayment risk. The next step would be to refine the model for better accuracy and potentially integrate a recommendation system that guides students toward the best loan options based on their profiles.

---

### How to Run This Project
1. **Clone the repository**:
 ```bash
 git clone https://github.com/yourusername/student-loan-risk.git
```

2. **Install dependencies**:
pip install pandas tensorflow scikit-learn

3. **Run the Jupyter Notebook**:
Open student_loan_risk.ipynb and run all cells to see the analysis and model training steps.


### Author
This project was developed as part of a machine learning course to explore predictive modeling and recommendation systems in the context of student loans.

Feel free to contribute!
