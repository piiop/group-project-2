# Diabetes Prediction Using Machine Learning Classification Models

## Project Overview

This project focuses on developing a machine learning model to predict whether a patient is likely to have diabetes based on various demographic and medical features. The primary goal is to build an accurate classification model that meets the performance benchmark of at least 75% accuracy, through the data extraction, cleaning, exploratory analysis, model training, and optimization.

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

## Data Preprocessing and Exploration

The data underwent a thorough cleaning process, including:
- **Handling missing values**: There were no missing values in the dataset, so no imputation was needed.
- **One-hot encoding**: Categorical variables such as gender, location, and smoking history were encoded using one-hot encoding to make them suitable for the machine learning model.

### Exploratory Data Analysis (EDA)

EDA was conducted to understand the distribution of variables and identify correlations between features and the target variable (diabetes). Some key findings include:
- A higher BMI and HbA1c level are strongly associated with the presence of diabetes.
- Certain demographic factors such as age and smoking history also show a notable correlation with the target variable.
- Visualizations such as histograms, correlation matrices, and box plots were used to highlight these trends.

## Model Implementation

### Model Selection

A **Random Forest Classifier** was chosen for this task due to its robustness and ability to handle both categorical and numerical features effectively. The model was trained on 80% of the data, with 20% reserved for testing.

### Model Performance

- **Accuracy**: The baseline model achieved an accuracy of [insert accuracy value].
- **Other Metrics**: Precision, recall, and F1-score were calculated to evaluate the model's performance further.

## Model Optimization

To improve the model's performance, hyperparameter tuning was performed. Key optimizations included:
- Adjusting the number of trees in the forest.
- Modifying the depth of the trees.
- Cross-validation techniques to assess model stability.

After tuning, the model's accuracy improved to [insert improved accuracy], exceeding the project goal of 75%.

## Results and Findings

The final model showed a high level of predictive accuracy for diabetes, with the most significant factors being age, BMI, HbA1c levels, and blood glucose levels. These findings suggest that machine learning models can be effectively used to assist in the early detection of diabetes based on easily measurable medical parameters.

## Future Work

Given more time, the following improvements could be considered:
- **Data augmentation**: Additional features or data from external sources could be incorporated to improve the model’s predictive power.
- **Model exploration**: Other models such as XGBoost or neural networks could be explored for potentially better performance.
- **Deployment**: The model could be deployed in a healthcare environment as a decision-support tool.

## How to Run the Project

1. Clone the repository.
2. Ensure you have all the necessary dependencies installed (see below).
3. Run the Jupyter notebook `analysis.ipynb` to perform data preprocessing, model training, and evaluation.
4. Alternatively, you can run the Python script `train_model.py` to train and evaluate the model directly.

## Dependencies

- Python 3.x
- Pandas
- Matplotlib
- Numpy
- Seaborn
- Sklearn
- imblearn
- Jupyter Notebook

## Contributors

- Michael Leston
- Giselle Gomez

## Acknowledgments

We would like to thank our Instructor Joe Palaia and our Teaching Assistant Yuyang, also to our fellow classmates. To all of you, thank you for yor guidance and support throughout this course and the projects. 

