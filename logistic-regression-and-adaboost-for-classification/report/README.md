## Offline-1: Logistic Regression and AdaBoost for Classification  

This report is prepared by Ajmain Yasar Ahmed Sahil **(Student ID: 1605023)**.  



### Getting Started  

You have to follow along the below steps in order to run the script for both training and testing purposes.  

1. Make sure that you have below Python modules installed beforehand.  
   - `numpy` can be installed with `pip install numpy`  
   - `pandas` can be installed with `pip install pandas`  
   - `scikit-learn` can be installed with `pip install scikit-learn`  
2. In your workspace folder (where the script is located), create a folder named `datasets` and place all the `.csv` datasets file inside it.  
3. Type `python <script_name>.py` to run the script.  



### Performance Evaluation  

#### Logistic Regression on Telco Customer Churn Dataset  

| Performance Measure    | Training `sigmoid` | Test `sigmoid` | Training `tanh` | Test `tanh` |
| ---------------------- | ------------------ | -------------- | --------------- | ----------- |
| Accuracy               | 0.7781             | 0.7778         | 0.8015          | 0.8012      |
| TPR/Sensitivity/Recall | 0.7061             | 0.7557         | 0.5575          | 0.5890      |
| TNR/Specificity        | 0.8047             | 0.7851         | 0.8918          | 0.8708      |
| PPV/Precision          | 0.5721             | 0.5356         | 0.6558          | 0.5994      |
| False Discovery Rate   | 0.4278             | 0.4643         | 0.3441          | 0.4005      |
| F1 Score               | 0.6321             | 0.6269         | 0.6027          | 0.5942      |



#### Logistic Regression on Adult Salary Scale Dataset  

| Performance Measure    | Training `sigmoid` | Test `sigmoid` | Training `tanh` | Test `tanh` |
| ---------------------- | ------------------ | -------------- | --------------- | ----------- |
| Accuracy               | 0.8200             | 0.8195         | 0.8441          | 0.8450      |
| TPR/Sensitivity/Recall | 0.7448             | 0.7433         | 0.5883          | 0.5852      |
| TNR/Specificity        | 0.8439             | 0.8431         | 0.9253          | 0.9253      |
| PPV/Precision          | 0.6021             | 0.5943         | 0.7141          | 0.7080      |
| False Discovery Rate   | 0.3978             | 0.4056         | 0.2858          | 0.2919      |
| F1 Score               | 0.6659             | 0.6605         | 0.6451          | 0.6408      |



#### Logistic Regression on Credit Card Fraud Detection Dataset (Entire)  

| Performance Measure    | Training `sigmoid` | Test `sigmoid` | Training `tanh` | Test `tanh` |
| ---------------------- | ------------------ | -------------- | --------------- | ----------- |
| Accuracy               | 0.9988             | 0.9989         | 0.9989          | 0.9990      |
| TPR/Sensitivity/Recall | 0.4238             | 0.4285         | 0.4974          | 0.4897      |
| TNR/Specificity        | 0.9998             | 0.9998         | 0.9998          | 0.9998      |
| PPV/Precision          | 0.8308             | 0.8750         | 0.8448          | 0.8888      |
| False Discovery Rate   | 0.1691             | 0.1250         | 0.1551          | 0.1111      |
| F1 Score               | 0.5613             | 0.5753         | 0.6261          | 0.6315      |



#### Logistic Regression on Credit Card Fraud Detection Dataset (Subsampled)  

| Performance Measure    | Training `sigmoid` | Test `sigmoid` | Training `tanh` | Test `tanh` |
| ---------------------- | ------------------ | -------------- | --------------- | ----------- |
| Accuracy               | 0.9951             | 0.9943         | 0.9951          | 0.9943      |
| TPR/Sensitivity/Recall | 0.8045             | 0.7857         | 0.8045          | 0.7857      |
| TNR/Specificity        | 0.9998             | 0.9995         | 0.9998          | 0.9995      |
| PPV/Precision          | 0.9906             | 0.9746         | 0.9906          | 0.9746      |
| False Discovery Rate   | 0.0093             | 0.0253         | 0.0093          | 0.0253      |
| F1 Score               | 0.8879             | 0.8701         | 0.8879          | 0.8701      |



#### AdaBoost Accuracy on Telco Customer Churn Dataset  

| Number of Boosting Rounds | Training | Test   |
| ------------------------- | -------- | ------ |
| 5 (5 Hypotheses)          | 0.7823   | 0.7870 |
| 10 (10 Hypotheses)        | 0.7933   | 0.7849 |
| 15 (15 Hypotheses)        | 0.7875   | 0.7927 |
| 20 (20 Hypotheses)        | 0.7783   | 0.7842 |



#### AdaBoost Accuracy on Adult Salary Scale Dataset  

| Number of Boosting Rounds | Training | Test   |
| ------------------------- | -------- | ------ |
| 5 (5 Hypotheses)          | 0.8378   | 0.8364 |
| 10 (10 Hypotheses)        | 0.8385   | 0.8374 |
| 15 (15 Hypotheses)        | 0.8401   | 0.8368 |
| 20 (15 Hypotheses)        | 0.8372   | 0.8370 |



#### AdaBoost Accuracy on Credit Card Fraud Detection Dataset (Subsampled)  

| Number of Boosting Rounds | Training | Test   |
| ------------------------- | -------- | ------ |
| 5 (5 Hypotheses)          | 0.9951   | 0.9943 |
| 10 (9 Hypotheses)         | 0.9951   | 0.9943 |
| 15 (11 Hypotheses)        | 0.9951   | 0.9943 |
| 20 (13 Hypotheses)        | 0.9951   | 0.9943 |

