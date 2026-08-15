**a. Problem Statement:**

The objective of this assignment is to develop and compare multiple machine learning classification models for breast cancer diagnosis.



The task is to classify breast cancer cases into two classes based on numerical features extracted from breast cell nuclei measurements. Five machine learning classification algorithms are implemented and evaluated:



Logistic Regression

Decision Tree

K-Nearest Neighbors (KNN)

Gaussian Naive Bayes

Random Forest



The models are evaluated using Accuracy, AUC, Precision, Recall, F1 Score, and Matthews Correlation Coefficient (MCC). The models are then compared to identify the best-performing model on the selected test dataset.



The final models are deployed through a Streamlit application that allows users to upload test data, select a machine learning model, view evaluation metrics, confusion matrix, classification report, and prediction results.



**b. Dataset Description:**

The Breast Cancer dataset contains 569 observations and 30 numerical input features related to characteristics of cell nuclei obtained from breast tissue samples.



The target variable is diagnosis, which represents the breast cancer classification.



For this assignment, the dataset was divided into training and testing data:



Total samples: 569

Training samples: 455

Testing samples: 114

Input features: 30

Target variable: diagnosis



The test dataset used by the Streamlit application contains 114 samples and 31 columns, consisting of the 30 input features and the target column.



The features include measurements such as radius, texture, perimeter, area, smoothness, compactness, concavity, concave points, symmetry, and fractal dimension.



**c. GitHub Repository Link**: https://github.com/sadhve/ML-Assignment-2.git



**d. Comparison Table with the models used**



|Sl.No|ML Model Name|Accuracy|AUC|Precision|Recall|F1|MCC|
|-|-|-|-|-|-|-|-|
|1|Logistic Regression|0.9649|0.9960|0.9750|0.9286|0.9512| 0.9245|
|2|Decision Tree|0.9298| 0.9246|0.9048|0.9048|0.9048|0.8492|
|3|kNN|0.9561|0.9823|0.9744|0.9048|0.9383|0.9058|
|4|Naive Bayes|0.9386|0.9934|**1.0000**|0.8333|0.9091|0.8715|
|5|Random Forest (Ensemble)|**0.9737**|**0.9929**|**1.0000**|**0.9286**|**0.9630**|**0.9442**|





**Observation/Analysis**



|ML Model Name|Observation about Model Performance|
|-|-|
|Logistic Regression|Logistic Regression achieved an accuracy of 96.49% and the highest AUC of 0.9960 among the five models. It also achieved a strong F1 Score of 0.9512 and MCC of 0.9245. The model performed very well overall, particularly in distinguishing between the two classes.|
|Decision Tree|Decision Tree achieved an accuracy of 92.98%, which was the lowest among the five models. Its AUC was 0.9246, and its F1 Score was 0.9048. Although the model performed reasonably well, it was less effective than the other models on this test dataset.|
|kNN|KNN achieved an accuracy of 95.61% and an AUC of 0.9823. Its precision was 0.9744, while its recall was 0.9048. The model performed well after feature scaling, but its overall performance was slightly below Logistic Regression and Random Forest.|
|Naive Bayes|Gaussian Naive Bayes achieved an accuracy of 93.86% and a high AUC of 0.9934. It achieved 100% precision, but its recall was comparatively lower at 0.8333. This indicates that while its positive predictions were highly precise, it missed more positive cases than the better-performing models.|
|Random Forest (Ensemble)|Random Forest achieved the highest accuracy of 97.37%, along with an F1 Score of 0.9630 and the highest MCC of 0.9442. It also achieved 100% precision and 92.86% recall. Overall, Random Forest provided the strongest balanced performance among the five models.|
|Overall Winner |Based on the evaluation results, **Random Forest** is the overall best-performing model for this dataset.|











