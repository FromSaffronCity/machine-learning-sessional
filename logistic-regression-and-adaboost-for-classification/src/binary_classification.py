# Logistic Regression and AdaBoost for Binary Classification

# Import Statements

# NumPy Documentation: https://numpy.org/doc/  
# pandas Documentation: https://pandas.pydata.org/docs/

import numpy as np
import pandas as pd

# sklearn.impute.SimpleImputer: https://scikit-learn.org/stable/modules/generated/sklearn.impute.SimpleImputer.html
# sklearn.preprocessing.StandardScaler: https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html
# sklearn.model_selection.train_test_split: https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# ref: https://stackoverflow.com/questions/21494489/what-does-numpy-random-seed0-do
np.random.seed(1)

# ref: https://stackoverflow.com/questions/14861891/runtimewarning-invalid-value-encountered-in-divide/54364060
# ref: https://numpy.org/doc/stable/reference/generated/numpy.seterr.html
np.seterr(divide='ignore', invalid='ignore')

# ref: https://stackoverflow.com/questions/20625582/how-to-deal-with-settingwithcopywarning-in-pandas
pd.options.mode.chained_assignment = None

# Datasets Preprocessing

# Reference:  
# Preprocessing data: https://scikit-learn.org/stable/modules/preprocessing.html  
# Imputation of missing values: https://scikit-learn.org/stable/modules/impute.html  
# Easy Guide to Data Preprocessing in Python: https://www.kdnuggets.com/2020/07/easy-guide-data-preprocessing-python.html  
# Data preprocessing: https://cs.ccsu.edu/~markov/ccsu_courses/DataMining-3.html

# Datasets:  
# Telco Customer Churn: https://www.kaggle.com/blastchar/telco-customer-churn  
# Adult Salary Scale: https://archive.ics.uci.edu/ml/datasets/adult  
# Credit Card Fraud Detection: https://www.kaggle.com/mlg-ulb/creditcardfraud

# Telco Customer Churn Dataset Preprocessing

def telco_customer_churn_dataset_preprocessing():
    data_frame = pd.read_csv('./datasets/telco-customer-churn.csv')
    
    # imputing missing values in specific columns
    numeric_imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
    
    data_frame['TotalCharges'].replace({' ': np.nan}, inplace=True)
    data_frame['TotalCharges'] = numeric_imputer.fit_transform(data_frame[['TotalCharges']])
    
    # dropping unnecessary columns
    data_frame.drop(['customerID'], inplace=True, axis=1)
    
    # modifying values in specific columns
    data_frame['MultipleLines'].replace({'No phone service': 'No'}, inplace=True)
    
    for key in ['OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies']:
        data_frame[key].replace({'No internet service': 'No'}, inplace=True)
    
    data_frame['Churn'].replace({'Yes': 1, 'No': 0}, inplace=True)
    
    # encoding categorical features
    data_frame = pd.get_dummies(
        data_frame, 
        columns=[
            'gender', 'SeniorCitizen', 'Partner', 'Dependents', 
            'PhoneService', 'MultipleLines', 'InternetService', 'OnlineSecurity', 
            'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 
            'StreamingMovies', 'Contract', 'PaperlessBilling', 'PaymentMethod'
        ]
    )
    
    # separating target column from data_frame
    data_frame_target = data_frame['Churn']
    data_frame.drop(['Churn'], inplace=True, axis=1)
    
    # standardizing specific columns in data_frame
    standard_scaler = StandardScaler()
    
    for key in ['tenure', 'MonthlyCharges', 'TotalCharges']:
        data_frame[key] = standard_scaler.fit_transform(data_frame[[key]])
    
    return data_frame.to_numpy(), data_frame_target.to_numpy()

# Adult Salary Scale Dataset Preprocessing

def adult_dataset_preprocessing():
    train_data_frame = pd.read_csv('./datasets/adult-train.csv')
    test_data_frame = pd.read_csv('./datasets/adult-test.csv')
    
    # modifying values in specific columns
    train_data_frame['salary-scale'].replace({' >50K': 1, ' <=50K': 0}, inplace=True)
    test_data_frame['salary-scale'].replace({' >50K.': 1, ' <=50K.': 0}, inplace=True)
    
    # concatenating train_data_frame and test_data_frame
    # ref: https://www.kaggle.com/c/house-prices-advanced-regression-techniques/discussion/60425
    train_data_frame['is_train'] = 1
    test_data_frame['is_train'] = 0
    data_frame = pd.concat([train_data_frame, test_data_frame], ignore_index=True)
    
    # imputing missing values in specific columns
    categorical_imputer = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
    
    for key in ['workclass', 'occupation', 'native-country']:
        data_frame[key].replace({' ?': np.nan}, inplace=True)
        data_frame[key] = categorical_imputer.fit_transform(data_frame[[key]])
    
    # encoding categorical features
    data_frame = pd.get_dummies(
        data_frame, 
        columns=['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'native-country']
    )
    data_frame.drop(['native-country_ Holand-Netherlands'], inplace=True, axis=1)
    
    # separating train_data_frame and test_data_frame from data_frame
    train_data_frame = data_frame[data_frame['is_train'] == 1]
    test_data_frame = data_frame[data_frame['is_train'] == 0]
    
    train_data_frame.drop(['is_train'], inplace=True, axis=1)
    test_data_frame.drop(['is_train'], inplace=True, axis=1)
    
    # separating target column from data_frames
    train_data_frame_target = train_data_frame['salary-scale']
    train_data_frame.drop(['salary-scale'], inplace=True, axis=1)
    
    test_data_frame_target = test_data_frame['salary-scale']
    test_data_frame.drop(['salary-scale'], inplace=True, axis=1)
    
    # standardizing specific columns in data_frame
    standard_scaler = StandardScaler()
    
    for key in ['age', 'fnlwgt', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week']:
        train_data_frame[key] = standard_scaler.fit_transform(train_data_frame[[key]])
        test_data_frame[key] = standard_scaler.fit_transform(test_data_frame[[key]])   
    
    return (
        train_data_frame.to_numpy(), 
        train_data_frame_target.to_numpy(), 
        test_data_frame.to_numpy(), 
        test_data_frame_target.to_numpy()
    )

# Credit Card Fraud Detection Dataset Preprocessing

def credit_card_fraud_detection_dataset_preprocessing(use_smaller_subset=False):
    data_frame = pd.read_csv('./datasets/credit-card-fraud-detection.csv')
    
    # separating data samples based on value in 'Class' column
    data_frame_0 = data_frame[data_frame['Class'] == 0]
    data_frame_1 = data_frame[data_frame['Class'] == 1]
    
    # splitting data_frame_0 and data_frame_1 into training and testing sets
    train_data_frame_0 = data_frame_0.sample(frac=0.8, random_state=1)
    test_data_frame_0 = data_frame_0.drop(train_data_frame_0.index)
    
    if use_smaller_subset:
        train_data_frame_0 = train_data_frame_0.sample(n=16000, random_state=1)
        test_data_frame_0 = test_data_frame_0.sample(n=4000, random_state=1)
    
    train_data_frame_1 = data_frame_1.sample(frac=0.8, random_state=1)
    test_data_frame_1 = data_frame_1.drop(train_data_frame_1.index)
    
    # concatenating train_data_frame_0 and train_data_frame_1, test_data_frame_0 and test_data_frame_1
    train_data_frame = pd.concat([train_data_frame_0, train_data_frame_1], ignore_index=True).sample(frac=1, random_state=1)
    test_data_frame = pd.concat([test_data_frame_0, test_data_frame_1], ignore_index=True).sample(frac=1, random_state=1)
    
    # separating target columns from data_frames
    train_data_frame_target = train_data_frame['Class']
    train_data_frame.drop(['Class'], inplace=True, axis=1)
    
    test_data_frame_target = test_data_frame['Class']
    test_data_frame.drop(['Class'], inplace=True, axis=1)
    
    # standardizing specific columns in data_frames
    standard_scaler = StandardScaler()
    
    for key in list(train_data_frame.columns):
        train_data_frame[key] = standard_scaler.fit_transform(train_data_frame[[key]])
        
    for key in list(test_data_frame.columns):
        test_data_frame[key] = standard_scaler.fit_transform(test_data_frame[[key]])
    
    return (
        train_data_frame.to_numpy(), 
        train_data_frame_target.to_numpy(), 
        test_data_frame.to_numpy(), 
        test_data_frame_target.to_numpy()
    )

# Logistic Regression Implementation

# Reference: https://towardsdatascience.com/logistic-regression-from-scratch-in-python-ec66603592e2

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def tanh(z):
    return (np.exp(z) - np.exp(-z)) / (np.exp(z) + np.exp(-z))

def mean_squared_error(y_true, y_predicted):
    return np.mean((y_true - y_predicted) ** 2)

def gradients(X, y_true, y_predicted, is_weak_learning=False):
    num_samples = X.shape[0]
    
    # calculating gradients of loss with respect to weights w
    if is_weak_learning:
        # using tanh as logistic function
        dw = np.dot(X.T, (y_true - y_predicted) * (1 - y_predicted ** 2)) / num_samples
    else:
        # using sigmoid as logistic function
        dw = np.dot(X.T, (y_true - y_predicted) * y_predicted * (1 - y_predicted)) / num_samples
    
    return dw

def normalize(X):
    # standardizing to have zero mean and unit variance
    return (X - X.mean(axis=0)) / X.std(axis=0)

def train(X, y_true, epochs, learning_rate, report_loss=False, is_weak_learning=False, early_stopping_threshold=0):
    num_samples, num_features = X.shape
    
    # initializing weights w to zero
    w = np.zeros((num_features + 1, 1))
    
    # normalizing inputs X
    X = normalize(X)
    
    # augmenting dummy input attribute 1 to each row of X
    X = np.concatenate((X, np.ones((num_samples, 1))), axis=1)
    
    # reshaping target y_true
    y_true = y_true.reshape(num_samples, 1)
    
    # training loop
    for epoch in range(epochs):
        # calculating hypotheses
        if is_weak_learning:
            y_predicted = (1 + tanh(np.dot(X, w))) / 2
        else:
            y_predicted = sigmoid(np.dot(X, w))
        
        # calculating gradients of loss with respect to weights w
        dw = gradients(X, y_true, y_predicted, is_weak_learning=is_weak_learning)
        
        # gradient descent: updating parameters weights w
        w = w + learning_rate * dw
        
        # calculating MSE loss in this epoch
        if is_weak_learning:
            loss = mean_squared_error(y_true, (1 + tanh(np.dot(X, w))) / 2)
        else:
            loss = mean_squared_error(y_true, sigmoid(np.dot(X, w)))
        
        # reporting MSE loss in this epoch
        if report_loss:
            print(f'Epoch {epoch + 1}: MSE Loss = {loss}')
        
        # early termination of gradient descent
        if loss <= early_stopping_threshold:
            break
    
    return w

def predict(X, w, is_weak_learning=False):
    num_samples = X.shape[0]
    
    # normalizing inputs X
    X = normalize(X)
    
    # augmenting dummy input attribute 1 to each row of X
    X = np.concatenate((X, np.ones((num_samples, 1))), axis=1)
    
    # calculating hypotheses
    if is_weak_learning:
        y_predicted = (1 + tanh(np.dot(X, w))) / 2
    else:
        y_predicted = sigmoid(np.dot(X, w))
    
    # determining and storing predictions
    predictions = [1 if y_pred >= 0.5 else 0 for y_pred in y_predicted]
    
    return np.array(predictions).reshape(num_samples, 1)

# AdaBoost Implementation

def adaptive_boosting(X, y_true, num_boosting_rounds, report_accuracy=False):
    num_samples, num_features = X.shape
    
    # initializing local variables
    example_weights = np.full(num_samples, 1 / num_samples)
    hypotheses = []
    hypothesis_weights = []
    
    # boosting loop
    for k in range(num_boosting_rounds):
        # resampling input examples
        examples = np.concatenate((X, y_true), axis=1)
        
        # ref: https://stackoverflow.com/questions/14262654/numpy-get-random-set-of-rows-from-2d-array
        # ref: https://numpy.org/doc/stable/reference/random/generated/numpy.random.choice.html
        data = examples[np.random.choice(num_samples, size=num_samples, replace=True, p=example_weights)]
        
        data_X = data[:, :num_features]
        data_y_true = data[:, -1:]
        
        # getting hypothesis from a learning algorithm
        w = train(
            data_X, 
            data_y_true, 
            epochs=1000, 
            learning_rate=0.01, 
            report_loss=False, 
            is_weak_learning=True, 
            early_stopping_threshold=0.2
        )
        
        # predicting target values with hypothesis
        y_predicted = predict(X, w, is_weak_learning=True)
        
        # reporting accuracy of hypothesis
        if report_accuracy:
            print(np.sum(y_true == y_predicted) / num_samples)
        
        # calculating error for hypothesis
        error = 0
        
        for i in range(num_samples):
            error = error + (example_weights[i] if y_true[i] != y_predicted[i] else 0)
        
        if error > 0.5:
            continue
        else:
            hypotheses.append(w)
        
        # updating example_weights
        for i in range(num_samples):
            example_weights[i] = example_weights[i] * (error / (1 - error) if y_true[i] == y_predicted[i] else 1)
        
        example_weights = example_weights / example_weights.sum()
        
        # updating hypothesis_weights
        hypothesis_weights.append(np.log2((1 - error) / error))
    
    return hypotheses, np.array(hypothesis_weights).reshape(len(hypotheses), 1)

def weighted_majority_predict(X, hypotheses, hypothesis_weights):
    num_samples = X.shape[0]
    num_hypotheses = len(hypotheses)
    
    # normalizing inputs X
    X = normalize(X)
    
    # augmenting dummy input attribute 1 to each row of X
    X = np.concatenate((X, np.ones((num_samples, 1))), axis=1)
    
    # calculating hypotheses
    y_predicteds = []
    
    for i in range(num_hypotheses):
        y_predicted = (1 + tanh(np.dot(X, hypotheses[i]))) / 2
        y_predicteds.append([1 if y_pred >= 0.5 else -1 for y_pred in y_predicted])
        
    y_predicteds = np.array(y_predicteds)
    
    # calculating weighted majority hypothesis and storing predictions
    weighted_majority_hypothesis = np.dot(y_predicteds.T, hypothesis_weights)
    predictions = [1 if y_pred >= 0 else 0 for y_pred in weighted_majority_hypothesis]
    
    return np.array(predictions).reshape(num_samples, 1)

# Performance Evaluation

# Reference: https://en.wikipedia.org/wiki/Confusion_matrix

def performance_evaluation(y_true, y_predicted):
    num_samples = y_true.shape[0]
    
    # initializing confusion matrix outcomes
    true_positive = 0
    false_negative = 0
    true_negative = 0
    false_positive = 0
    
    # calculating and storing confusion matrix outcomes
    for i in range(num_samples):
        if y_true[i] == 1:
            if y_true[i] == y_predicted[i]:
                true_positive = true_positive + 1
            else:
                false_negative = false_negative + 1
        elif y_true[i] == 0:
            if y_true[i] == y_predicted[i]:
                true_negative = true_negative + 1
            else:
                false_positive = false_positive + 1
    
    # calculating and storing performance measures
    accuracy = (true_positive + true_negative) / (true_positive + false_negative + true_negative + false_positive)
    sensitivity = true_positive / (true_positive + false_negative)
    specificity = true_negative / (true_negative + false_positive)
    precision = true_positive / (true_positive + false_positive)
    false_discovery_rate = false_positive / (true_positive + false_positive)
    f1_score = 2 * sensitivity * precision / (sensitivity + precision)
    
    return accuracy, sensitivity, specificity, precision, false_discovery_rate, f1_score

# Extracting Dataset Features & Targets and Splitting Datasets into Training & Testing Sets  

# Reference: https://developers.google.com/machine-learning/crash-course/training-and-test-sets/splitting-data

# Telco Customer Churn Dataset

telco_customer_churn_features, telco_customer_churn_target = telco_customer_churn_dataset_preprocessing()

(
    train_churn_features, 
    test_churn_features, 
    train_churn_target, 
    test_churn_target
) = train_test_split(telco_customer_churn_features, telco_customer_churn_target, test_size=0.2, random_state=1)

train_churn_target = train_churn_target.reshape(train_churn_target.shape[0], 1)
test_churn_target = test_churn_target.reshape(test_churn_target.shape[0], 1)

# Adult Salary Scale Dataset

(train_salary_features, train_salary_target, test_salary_features, test_salary_target) = adult_dataset_preprocessing()

train_salary_target = train_salary_target.reshape(train_salary_target.shape[0], 1)
test_salary_target = test_salary_target.reshape(test_salary_target.shape[0], 1)

# Credit Card Fraud Detection Dataset (Entire)

(
    train_fraud_features, 
    train_fraud_target, 
    test_fraud_features, 
    test_fraud_target
) = credit_card_fraud_detection_dataset_preprocessing(use_smaller_subset=False)

train_fraud_target = train_fraud_target.reshape(train_fraud_target.shape[0], 1)
test_fraud_target = test_fraud_target.reshape(test_fraud_target.shape[0], 1)

# Credit Card Fraud Detection Dataset (Subsampled)

(
    train_fraud_sub_features, 
    train_fraud_sub_target, 
    test_fraud_sub_features, 
    test_fraud_sub_target
) = credit_card_fraud_detection_dataset_preprocessing(use_smaller_subset=True)

train_fraud_sub_target = train_fraud_sub_target.reshape(train_fraud_sub_target.shape[0], 1)
test_fraud_sub_target = test_fraud_sub_target.reshape(test_fraud_sub_target.shape[0], 1)

# Performance Measurement

# Logistic Regression with sigmoid

# Telco Customer Churn Dataset

w = train(
    train_churn_features, 
    train_churn_target, 
    epochs=1000, 
    learning_rate=0.01, 
    report_loss=False, 
    is_weak_learning=False, 
    early_stopping_threshold=0
)

(
    accuracy, 
    sensitivity, 
    specificity, 
    precision, 
    false_discovery_rate, 
    f1_score
) = performance_evaluation(train_churn_target, predict(train_churn_features, w, is_weak_learning=False))

print(
    f'Logistic Regression with sigmoid for Telco Customer Churn Dataset: Train\n'
    f'Accuracy: {accuracy}\n'
    f'Sensitivity: {sensitivity}\n'
    f'Specificity: {specificity}\n'
    f'Precision: {precision}\n'
    f'False Discovery Rate: {false_discovery_rate}\n'
    f'F1 Score: {f1_score}\n'
)

(
    accuracy, 
    sensitivity, 
    specificity, 
    precision, 
    false_discovery_rate, 
    f1_score
) = performance_evaluation(test_churn_target, predict(test_churn_features, w, is_weak_learning=False))

print(
    f'Logistic Regression with sigmoid for Telco Customer Churn Dataset: Test\n'
    f'Accuracy: {accuracy}\n'
    f'Sensitivity: {sensitivity}\n'
    f'Specificity: {specificity}\n'
    f'Precision: {precision}\n'
    f'False Discovery Rate: {false_discovery_rate}\n'
    f'F1 Score: {f1_score}\n'
)

# Adult Salary Scale Dataset

w = train(
    train_salary_features, 
    train_salary_target, 
    epochs=1000, 
    learning_rate=0.01, 
    report_loss=False, 
    is_weak_learning=False, 
    early_stopping_threshold=0
)

(
    accuracy, 
    sensitivity, 
    specificity, 
    precision, 
    false_discovery_rate, 
    f1_score
) = performance_evaluation(train_salary_target, predict(train_salary_features, w, is_weak_learning=False))

print(
    f'Logistic Regression with sigmoid for Adult Salary Scale Dataset: Train\n'
    f'Accuracy: {accuracy}\n'
    f'Sensitivity: {sensitivity}\n'
    f'Specificity: {specificity}\n'
    f'Precision: {precision}\n'
    f'False Discovery Rate: {false_discovery_rate}\n'
    f'F1 Score: {f1_score}\n'
)

(
    accuracy, 
    sensitivity, 
    specificity, 
    precision, 
    false_discovery_rate, 
    f1_score
) = performance_evaluation(test_salary_target, predict(test_salary_features, w, is_weak_learning=False))

print(
    f'Logistic Regression with sigmoid for Adult Salary Scale Dataset: Test\n'
    f'Accuracy: {accuracy}\n'
    f'Sensitivity: {sensitivity}\n'
    f'Specificity: {specificity}\n'
    f'Precision: {precision}\n'
    f'False Discovery Rate: {false_discovery_rate}\n'
    f'F1 Score: {f1_score}\n'
)

# Credit Card Fraud Detection Dataset (Entire)

w = train(
    train_fraud_features, 
    train_fraud_target, 
    epochs=1000, 
    learning_rate=0.01, 
    report_loss=False, 
    is_weak_learning=False, 
    early_stopping_threshold=0
)

(
    accuracy, 
    sensitivity, 
    specificity, 
    precision, 
    false_discovery_rate, 
    f1_score
) = performance_evaluation(train_fraud_target, predict(train_fraud_features, w, is_weak_learning=False))

print(
    f'Logistic Regression with sigmoid for Credit Card Fraud Detection Dataset (Entire): Train\n'
    f'Accuracy: {accuracy}\n'
    f'Sensitivity: {sensitivity}\n'
    f'Specificity: {specificity}\n'
    f'Precision: {precision}\n'
    f'False Discovery Rate: {false_discovery_rate}\n'
    f'F1 Score: {f1_score}\n'
)

(
    accuracy, 
    sensitivity, 
    specificity, 
    precision, 
    false_discovery_rate, 
    f1_score
) = performance_evaluation(test_fraud_target, predict(test_fraud_features, w, is_weak_learning=False))

print(
    f'Logistic Regression with sigmoid for Credit Card Fraud Detection Dataset (Entire): Test\n'
    f'Accuracy: {accuracy}\n'
    f'Sensitivity: {sensitivity}\n'
    f'Specificity: {specificity}\n'
    f'Precision: {precision}\n'
    f'False Discovery Rate: {false_discovery_rate}\n'
    f'F1 Score: {f1_score}\n'
)

# Credit Card Fraud Detection Dataset (Subsampled)

w = train(
    train_fraud_sub_features, 
    train_fraud_sub_target, 
    epochs=1000, 
    learning_rate=0.01, 
    report_loss=False, 
    is_weak_learning=False, 
    early_stopping_threshold=0
)

(
    accuracy, 
    sensitivity, 
    specificity, 
    precision, 
    false_discovery_rate, 
    f1_score
) = performance_evaluation(train_fraud_sub_target, predict(train_fraud_sub_features, w, is_weak_learning=False))

print(
    f'Logistic Regression with sigmoid for Credit Card Fraud Detection Dataset (Subsampled): Train\n'
    f'Accuracy: {accuracy}\n'
    f'Sensitivity: {sensitivity}\n'
    f'Specificity: {specificity}\n'
    f'Precision: {precision}\n'
    f'False Discovery Rate: {false_discovery_rate}\n'
    f'F1 Score: {f1_score}\n'
)

(
    accuracy, 
    sensitivity, 
    specificity, 
    precision, 
    false_discovery_rate, 
    f1_score
) = performance_evaluation(test_fraud_sub_target, predict(test_fraud_sub_features, w, is_weak_learning=False))

print(
    f'Logistic Regression with sigmoid for Credit Card Fraud Detection Dataset (Subsampled): Test\n'
    f'Accuracy: {accuracy}\n'
    f'Sensitivity: {sensitivity}\n'
    f'Specificity: {specificity}\n'
    f'Precision: {precision}\n'
    f'False Discovery Rate: {false_discovery_rate}\n'
    f'F1 Score: {f1_score}\n'
)

# Logistic Regression with tanh

# Telco Customer Churn Dataset

w = train(
    train_churn_features, 
    train_churn_target, 
    epochs=1000, 
    learning_rate=0.01, 
    report_loss=False, 
    is_weak_learning=True, 
    early_stopping_threshold=0
)

(
    accuracy, 
    sensitivity, 
    specificity, 
    precision, 
    false_discovery_rate, 
    f1_score
) = performance_evaluation(train_churn_target, predict(train_churn_features, w, is_weak_learning=True))

print(
    f'Logistic Regression with tanh for Telco Customer Churn Dataset: Train\n'
    f'Accuracy: {accuracy}\n'
    f'Sensitivity: {sensitivity}\n'
    f'Specificity: {specificity}\n'
    f'Precision: {precision}\n'
    f'False Discovery Rate: {false_discovery_rate}\n'
    f'F1 Score: {f1_score}\n'
)

(
    accuracy, 
    sensitivity, 
    specificity, 
    precision, 
    false_discovery_rate, 
    f1_score
) = performance_evaluation(test_churn_target, predict(test_churn_features, w, is_weak_learning=True))

print(
    f'Logistic Regression with tanh for Telco Customer Churn Dataset: Test\n'
    f'Accuracy: {accuracy}\n'
    f'Sensitivity: {sensitivity}\n'
    f'Specificity: {specificity}\n'
    f'Precision: {precision}\n'
    f'False Discovery Rate: {false_discovery_rate}\n'
    f'F1 Score: {f1_score}\n'
)

# Adult Salary Scale Dataset

w = train(
    train_salary_features, 
    train_salary_target, 
    epochs=1000, 
    learning_rate=0.01, 
    report_loss=False, 
    is_weak_learning=True, 
    early_stopping_threshold=0
)

(
    accuracy, 
    sensitivity, 
    specificity, 
    precision, 
    false_discovery_rate, 
    f1_score
) = performance_evaluation(train_salary_target, predict(train_salary_features, w, is_weak_learning=True))

print(
    f'Logistic Regression with tanh for Adult Salary Scale Dataset: Train\n'
    f'Accuracy: {accuracy}\n'
    f'Sensitivity: {sensitivity}\n'
    f'Specificity: {specificity}\n'
    f'Precision: {precision}\n'
    f'False Discovery Rate: {false_discovery_rate}\n'
    f'F1 Score: {f1_score}\n'
)

(
    accuracy, 
    sensitivity, 
    specificity, 
    precision, 
    false_discovery_rate, 
    f1_score
) = performance_evaluation(test_salary_target, predict(test_salary_features, w, is_weak_learning=True))

print(
    f'Logistic Regression with tanh for Adult Salary Scale Dataset: Test\n'
    f'Accuracy: {accuracy}\n'
    f'Sensitivity: {sensitivity}\n'
    f'Specificity: {specificity}\n'
    f'Precision: {precision}\n'
    f'False Discovery Rate: {false_discovery_rate}\n'
    f'F1 Score: {f1_score}\n'
)

# Credit Card Fraud Detection Dataset (Entire)

w = train(
    train_fraud_features, 
    train_fraud_target, 
    epochs=1000, 
    learning_rate=0.01, 
    report_loss=False, 
    is_weak_learning=True, 
    early_stopping_threshold=0
)

(
    accuracy, 
    sensitivity, 
    specificity, 
    precision, 
    false_discovery_rate, 
    f1_score
) = performance_evaluation(train_fraud_target, predict(train_fraud_features, w, is_weak_learning=True))

print(
    f'Logistic Regression with tanh for Credit Card Fraud Detection Dataset (Entire): Train\n'
    f'Accuracy: {accuracy}\n'
    f'Sensitivity: {sensitivity}\n'
    f'Specificity: {specificity}\n'
    f'Precision: {precision}\n'
    f'False Discovery Rate: {false_discovery_rate}\n'
    f'F1 Score: {f1_score}\n'
)

(
    accuracy, 
    sensitivity, 
    specificity, 
    precision, 
    false_discovery_rate, 
    f1_score
) = performance_evaluation(test_fraud_target, predict(test_fraud_features, w, is_weak_learning=True))

print(
    f'Logistic Regression with tanh for Credit Card Fraud Detection Dataset (Entire): Test\n'
    f'Accuracy: {accuracy}\n'
    f'Sensitivity: {sensitivity}\n'
    f'Specificity: {specificity}\n'
    f'Precision: {precision}\n'
    f'False Discovery Rate: {false_discovery_rate}\n'
    f'F1 Score: {f1_score}\n'
)

# Credit Card Fraud Detection Dataset (Subsampled)

w = train(
    train_fraud_sub_features, 
    train_fraud_sub_target, 
    epochs=1000, 
    learning_rate=0.01, 
    report_loss=False, 
    is_weak_learning=True, 
    early_stopping_threshold=0
)

(
    accuracy, 
    sensitivity, 
    specificity, 
    precision, 
    false_discovery_rate, 
    f1_score
) = performance_evaluation(train_fraud_sub_target, predict(train_fraud_sub_features, w, is_weak_learning=True))

print(
    f'Logistic Regression with tanh for Credit Card Fraud Detection Dataset (Subsampled): Train\n'
    f'Accuracy: {accuracy}\n'
    f'Sensitivity: {sensitivity}\n'
    f'Specificity: {specificity}\n'
    f'Precision: {precision}\n'
    f'False Discovery Rate: {false_discovery_rate}\n'
    f'F1 Score: {f1_score}\n'
)

(
    accuracy, 
    sensitivity, 
    specificity, 
    precision, 
    false_discovery_rate, 
    f1_score
) = performance_evaluation(test_fraud_sub_target, predict(test_fraud_sub_features, w, is_weak_learning=True))

print(
    f'Logistic Regression with tanh for Credit Card Fraud Detection Dataset (Subsampled): Test\n'
    f'Accuracy: {accuracy}\n'
    f'Sensitivity: {sensitivity}\n'
    f'Specificity: {specificity}\n'
    f'Precision: {precision}\n'
    f'False Discovery Rate: {false_discovery_rate}\n'
    f'F1 Score: {f1_score}\n'
)

# AdaBoost

# Telco Customer Churn Dataset

for K in range(5, 25, 5):
    (
        hypotheses, 
        hypothesis_weights
    ) = adaptive_boosting(train_churn_features, train_churn_target, num_boosting_rounds=K, report_accuracy=False)
    
    (
        accuracy, 
        sensitivity, 
        specificity, 
        precision, 
        false_discovery_rate, 
        f1_score
    ) = performance_evaluation(
        train_churn_target, 
        weighted_majority_predict(train_churn_features, hypotheses, hypothesis_weights)
    )

    print(f'AdaBoost ({K} -> {len(hypotheses)} hypotheses) for Telco Customer Churn Dataset: Train\nAccuracy: {accuracy}\n')
    
    (
        accuracy, 
        sensitivity, 
        specificity, 
        precision, 
        false_discovery_rate, 
        f1_score
    ) = performance_evaluation(
        test_churn_target, 
        weighted_majority_predict(test_churn_features, hypotheses, hypothesis_weights)
    )

    print(f'AdaBoost ({K} -> {len(hypotheses)} hypotheses) for Telco Customer Churn Dataset: Test\nAccuracy: {accuracy}\n')

# Adult Salary Scale Dataset

for K in range(5, 25, 5):
    (
        hypotheses, 
        hypothesis_weights
    ) = adaptive_boosting(train_salary_features, train_salary_target, num_boosting_rounds=K, report_accuracy=False)
    
    (
        accuracy, 
        sensitivity, 
        specificity, 
        precision, 
        false_discovery_rate, 
        f1_score
    ) = performance_evaluation(
        train_salary_target, 
        weighted_majority_predict(train_salary_features, hypotheses, hypothesis_weights)
    )

    print(f'AdaBoost ({K} -> {len(hypotheses)} hypotheses) for Adult Salary Scale Dataset: Train\nAccuracy: {accuracy}\n')
    
    (
        accuracy, 
        sensitivity, 
        specificity, 
        precision, 
        false_discovery_rate, 
        f1_score
    ) = performance_evaluation(
        test_salary_target, 
        weighted_majority_predict(test_salary_features, hypotheses, hypothesis_weights)
    )

    print(f'AdaBoost ({K} -> {len(hypotheses)} hypotheses) for Adult Salary Scale Dataset: Test\nAccuracy: {accuracy}\n')

# Credit Card Fraud Detection Dataset (Subsampled)

for K in range(5, 25, 5):
    (
        hypotheses, 
        hypothesis_weights
    ) = adaptive_boosting(train_fraud_sub_features, train_fraud_sub_target, num_boosting_rounds=K, report_accuracy=False)
    
    (
        accuracy, 
        sensitivity, 
        specificity, 
        precision, 
        false_discovery_rate, 
        f1_score
    ) = performance_evaluation(
        train_fraud_sub_target, 
        weighted_majority_predict(train_fraud_sub_features, hypotheses, hypothesis_weights)
    )

    print(
        f'AdaBoost ({K} -> {len(hypotheses)} hypotheses) for Credit Card Fraud Detection Dataset (Subsampled): Train\n'
        f'Accuracy: {accuracy}\n'
    )
    
    (
        accuracy, 
        sensitivity, 
        specificity, 
        precision, 
        false_discovery_rate, 
        f1_score
    ) = performance_evaluation(
        test_fraud_sub_target, 
        weighted_majority_predict(test_fraud_sub_features, hypotheses, hypothesis_weights)
    )

    print(
        f'AdaBoost ({K} -> {len(hypotheses)} hypotheses) for Credit Card Fraud Detection Dataset (Subsampled): Test\n'
        f'Accuracy: {accuracy}\n'
    )
