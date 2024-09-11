# Diabetes Prediction Using Machine Learning Classification Models

## Project Overview

This project focuses on developing a machine learning model to predict whether a patient is likely to have diabetes based on various demographic and medical features. The main goal is to build an accurate classification model that meets the performance benchmark of at least 75% accuracy, through the data extraction, cleaning, exploratory analysis, model training, and optimization.

## Dataset

The dataset used in this project is publicly available on [Kaggle](https://www.kaggle.com/datasets/priyamchoksi/100000-diabetes-clinical-dataset), titled "100000 Diabetes Clinical Dataset." It contains 100,000 clinical records of patients, including medical history and demographic information relevant to diabetes diagnosis.

- **Features**:
  - **year**: The year of data collection.
  - **gender**: Whether the patient is Male or Female.
  - **age**: Age of the patient in years.
  - **location**: Geographic location of the patient in the USA.
  - **race**: Four specified ethnic groups and "others"  (e.g., African-American, Asian).
  - **hypertension**: Whether the patient has a history of high blood pressure.
  - **heart_disease**: Whether the patient has a history of heart disease.
  - **smoking_history**: Patient's smoking habits.
  - **bmi**: Body Mass Index.
  - **hbA1c_level**: Hemoglobin A1C level, average blood sugar levels over time, usually given as a percentage.
  - **blood_glucose_level**: Blood glucose levels, is given at a specific moment in time, usually measured in milligrams per deciliter (mg/dL).
  - **diabetes**: Target variable (0: No diabetes, 1: Diabetes).

## License

The dataset is available under the [CC BY-SA 4.0 License](https://creativecommons.org/licenses/by-sa/4.0/), allowing others to remix, adapt, and build upon the data even for commercial purposes, as long as they credit the original creator and license their new creations under the same terms.

Priyam Choksi, as the owner of the kaggle account, declares that the dataset is released under MIT License.

# Methodology

### Data Preprocessing

The data underwent a thorough cleaning process, including:
- Checking for missing values.
- Encoding of categorical variables.
- Renaming columns for clarity.
- Handling duplicates.

### Exploratory Data Analysis (EDA)

EDA was conducted to understand the distribution of variables and identify correlations between features and the target variable (diabetes). Some key findings include:
- A higher BMI and HbA1c level are strongly associated with the presence of diabetes.
- Certain demographic factors such as age and smoking history also show a notable correlation with the target variable.
- Visualizations such as histograms, correlation matrices, and box plots were used to highlight these trends.

### Models Selection

- Logistic Regression
- RandomForest
- Decision Tree
- LightGBM
- XgBoost
- AdaBoost

### Model Performance

- **Accuracy**: Models achieved an accuracy over than 97%.
- **Other Metrics**: Precision, recall, and F1-score were calculated to evaluate the model's performance further.

### Other Feature Engineering
- Scaling numerical features like age and BMI to ensure they contribute equally to the model.
- Check Medical Indicator Features based on medical threshold, like Blood Glucose Category.
- Combining Risk Factors: heart Disease and Hypertension.
- Regrouped similar values of Smoking History.
- clustering for BMI, HbA1c level, and blood glucose level.

### Model Optimization
- Tested multiple models while comparing evaluation metrics.
- Iterated through features to find the optimal feature set.
- Applied weighting to LogReg model.
- Applied SMOTE to all models.
- Evaluation Metrics: Train/Test Accuracy, Recall, F1 Score, MCC, ROC-AUC, confusion matrix.
- Key Metric: ROC-AUC.
- Hyperparameter Tuning on best two models.

## Scoring Metric Comparison
- Logistic Regression (Full Feature) has the best overall performance with the highest accuracy (97%) and a strong balance between recall, F1, and MCC. Its high ROC-AUC score also indicates excellent discriminatory power.
- LightGBM (Tuned) is slightly behind the Full Feature model in most metrics, but it's still very competitive.
- Logistic Regression (Baseline) has strong accuracy, an excellent MCC, and a competitive F1 score, making it a solid choice despite being a baseline model.
- Logistic Regression (Tuned), despite its high recall, struggles with precision (88%), as indicated by the low F1 score and MCC, which makes it less favorable unless recall is the primary concern for the project.

## Results and Findings

The final model showed a high level of predictive accuracy for diabetes, with the most significant factors being age, BMI, HbA1c levels, and blood glucose levels. These findings suggest that machine learning models can be effectively used to assist in the early detection of diabetes based on easily measurable medical parameters.

Successfully achieved the goal of built multiple models to predict diabetes with over 75% accuracy using this dataset. Strong performance by logistic regression model suggests a linear relationship between features and the target variable, also suggests low complexity in the dataset and the importance of scoring metric in tuning.

## Future Work

Given more time, the following improvements could be considered:
-  Explore more complex models like neural networks or continue to tune.
- Use additional features or external datasets to improve prediction accuracy. More complex or real-life datasets.
- Next Steps: Test the model on different populations to assess generalizability.

## How to Run the Project

1. Clone the repository.
2. Install the required dependencies listed below.
3. Run the Jupyter notebook `analysis.ipynb` to perform the data preprocessing, EDA, visualization, and base-model test.
4. Run the Jupyter notebook script `model testing.ipynb` for various models training, evaluations, hyperparameter tuning, and final model.
5. Additional files and a folder with the visualizations summary are also available.

## Dependencies

- Python
- Pandas
- Matplotlib
- Numpy
- Seaborn
- Sklearn
- Imblearn
- scipy
- Jupyter Notebook

## Contributors

- Michael Leston
- Giselle Gomez

## Acknowledgments

We would like to thank our Instructor Joe Palaia and our Teaching Assistant Yuyang Zhong, also to our fellow classmates. To all of you, thank you for yor guidance and support throughout this course and the projects. 

